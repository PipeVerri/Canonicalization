import json
import bisect
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.SeqRecord import SeqRecord
from intervaltree import IntervalTree
import src.utils as utils
import src.utils.train

log = src.utils.train.get_logger(__name__)

#########################
# Genomic files parsing #
#########################

def read_fasta_into_records_dict(fasta_path):
    records = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        records[record.id] = record.seq
    return records

def parse_ncbi_assembly(jsonl_path):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():  
                data.append(json.loads(line))
    return data

def build_ncbi_to_ucsc_id(assembly_path, training_genome):
    ncbi_to_training_id = {}
    ncbi_assembly = parse_ncbi_assembly(assembly_path)
    for assembly in ncbi_assembly:
        if assembly["ucscStyleName"] in training_genome:
            ncbi_to_training_id[assembly["refseqAccession"]] = assembly["ucscStyleName"]
    return ncbi_to_training_id


#########################
# Biopython Seq Parsing #
#########################

def reverse(seq, start, end):
    seq[start:end] = seq[start:end][::-1]

def complement(seq, start, end):
    seq[start:end] = seq[start:end].complement()

def save_seqs_dict(seqs_dict, save_path):
    records = [
        SeqRecord(seq, id=name, description="")
        for name, seq in seqs_dict.items()
    ]
    SeqIO.write(records, save_path, "fasta")

########
# Misc #
########

def intervaltree_to_tuples(interval_tree):
    return [(interval.begin, interval.end) for interval in interval_tree]

def merge_segment_list(segments_list):
    if not segments_list:
        return []
    merged_segments = [segments_list[0]]
    
    for current_start, current_end in segments_list[1:]:
        last_start, last_end = merged_segments[-1]
        if current_start <= last_end:  
            merged_segments[-1] = (last_start, max(last_end, current_end))  
        else:
            merged_segments.append((current_start, current_end))  
    
    return merged_segments

##############
# Main logic #
##############

def parse_transcribed_features_from_gff(df, ucsc_id, 
                                        annotations_level,
                                        include_cdjv_segments,
                                        include_pseudogenes):
    assert type(annotations_level) == int and annotations_level <= 3 and annotations_level >= 1

    df = df.copy() # Copy the df to not overwrite the original
    df = df[df["Chromosome"] == ucsc_id].copy() # Filter the annotations for the target chromosome
    assert len(df) > 0 # Check that the chromosome actually exists
    
    df = df[df["level"].astype(int) <= annotations_level] # Filter the annotations for the desired level
    df = df[df["Feature"] == "transcript"] # Filter all the non-transcript features
    
    # ~ inverts the index, that's why the na is inverted
    # If the user wants to exclude na transcript_types, it's taken care of when dropping the na columns in the beggining
    if not include_pseudogenes:
        df = df[~df["transcript_type"].str.contains("pseudogene", na=False)]
    if not include_cdjv_segments:
        df = df[~df["transcript_type"].str.contains("IG_C", na=False)]
        df = df[~df["transcript_type"].str.contains("IG_D", na=False)]
        df = df[~df["transcript_type"].str.contains("IG_J", na=False)]
        df = df[~df["transcript_type"].str.contains("IG_V", na=False)]
    
    # Build the positive and negative segments list
    positive_segments = []
    negative_segments = []

    for _, row in df.iterrows():
        # The strand for each transcribed feature is the coding strand(opposite of the template)
        # Pyranges v0 converts the 1-based index of the GFF to 0-based (https://pyranges.readthedocs.io/en/v0/tutorial.html)
        if row["Strand"] == '+':
            bisect.insort_left(positive_segments, (row["Start"], row["End"]))
        elif row["Strand"] == '-':
            bisect.insort_left(negative_segments, (row["Start"], row["End"]))
        else:
            print(row)
            print(type(row))
            raise ValueError(f"Strand value {row['Strand']} not recognized")

    positive_segments = merge_segment_list(positive_segments)
    negative_segments = merge_segment_list(negative_segments)
    
    # Ignore chrM because it's so small in humans it's reasonable that it only has genes on one strand
    if len(positive_segments) == 0 and ucsc_id != "chrM":
        log.warning(f"positive_segments has length 0 on {ucsc_id}")
    if len(negative_segments) == 0 and ucsc_id != "chrM":
        log.warning(f"negtative segments has length 0 on {ucsc_id}")
     
    return positive_segments, negative_segments

def canonicalize_genome(training_genome, training_genome_canonicalization_segments, transformation):
    results = {}
    for training_genome_id, segments in training_genome_canonicalization_segments.items():
        canonicalied = MutableSeq(training_genome[training_genome_id])
        
        for (start, end) in segments:
            if transformation == "reverse":
                reverse(canonicalied, start, end)
            if transformation == "complement":
                complement(canonicalied, start, end)
            if transformation == "rc":
                reverse(canonicalied, start, end)
                complement(canonicalied, start, end)

        results[training_genome_id] = canonicalied

    return results

def decide_which_strand_should_canonicalize(positive_template_segments, negative_template_segments):
    negative_template_intervaltree = IntervalTree.from_tuples(negative_template_segments)
    keep_positive_strand_cost = 0
    keep_negative_strand_cost = 0

    for (start, end) in positive_template_segments:
        overlapping_templates = intervaltree_to_tuples(negative_template_intervaltree.overlap(start, end))
        keep_positive_strand_cost += sum(seg_end - seg_start for seg_start, seg_end in overlapping_templates)
        keep_negative_strand_cost += end - start
        
    # Decide which segments to canonicalize. Return the segments to canonicalize
    if keep_positive_strand_cost < keep_negative_strand_cost:
        return positive_template_segments
    else:
        return negative_template_segments

#################################
# NT Benchmark canonicalization #
#################################

HISTONE_TASKS = ("H3K4me3", "H3K27ac", "H3K27me3", "H3K4me1", "H3K36me3", "H3K9me3", "H3K9ac", "H3K4me2", "H4K20me1", "H2AFZ")
ENHANCERS_TASKS = ("enhancers", "enhancers_types")
PROMOTER_TASKS = ("promoter_all", "promoter_tata", "promoter_no_tata")
SPLICING_TASKS = ("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor")

def canonicalize_nt_benchmark_fasta(fasta_path, aligner, gencode_transcribed_segments_intervaltree, transformation, save_path, partial_seq_canonicalization, orient_promoters):
    records = read_fasta_into_records_dict(fasta_path)
    task_name = fasta_path.stem.replace("_train", "").replace("_test", "")
    
    if task_name in SPLICING_TASKS or (not orient_promoters and task_name in PROMOTER_TASKS): # On the splicing tasks, the record is already on the coding strand
        save_seqs_dict(records, save_path)
        return
    
    canonicalized_records = {}
    unaligned_records = 0
    aligned_records_without_annotation = 0
    
    for id in records:
        # Align the record with the reference genome, and take the primary alignment
        seq_str = str(records[id])
        hits = aligner.map(seq_str)
        
        # Find primary alignment
        primary_alignment = None
        for hit in hits:
            if hit.is_primary:
                primary_alignment = hit
                break

        if primary_alignment is None:
            canonicalized_records[id] = records[id]
            unaligned_records += 1
            continue

        # If the primary alignment isn't on an annotated chromosome(like an unalocated segment), see if there's another alignment with the same mapq
        if not primary_alignment.ctg in gencode_transcribed_segments_intervaltree:
            found_annotated_hit = False

            for hit in hits:
                if hit.mapq >= primary_alignment.mapq and hit.ctg in gencode_transcribed_segments_intervaltree:
                    primary_alignment = hit
                    found_annotated_hit = True
                    break
            
            if not found_annotated_hit:
                canonicalized_records[id] = records[id]
                aligned_records_without_annotation += 1
                continue

        # Check if seq falls inside a transcribed segment or not
        alignment_start = primary_alignment.r_st
        alignment_end = primary_alignment.r_en
        
        if primary_alignment.strand == 1:
            segments_to_check = gencode_transcribed_segments_intervaltree[primary_alignment.ctg]["positive"]
        elif primary_alignment.strand == -1:
            segments_to_check = gencode_transcribed_segments_intervaltree[primary_alignment.ctg]["positive"]
        else:
            raise ValueError(f"Primary alignment had strand {primary_alignment.strand} on {task_name} id {id}")
        
        transcribed_segments_that_contain_seq = intervaltree_to_tuples(segments_to_check.overlap(alignment_start, alignment_end))
        
        canonicalized_seq = MutableSeq(records[id])
        if partial_seq_canonicalization:
            # Canonicalize seq using the segments
            for segment in transcribed_segments_that_contain_seq:
                # Index the segment relative to the seq's position
                segment_start = max(0, segment[0] - alignment_start)
                segment_end = min(len(seq_str), segment[1] - alignment_end)
                
                if transformation == "reverse":
                    canonicalized_seq = reverse(canonicalized_seq, segment_start, segment_end)
                elif transformation == "complement":
                    canonicalized_seq = complement(canonicalized_seq, segment_start, segment_end)
                elif transformation == "rc":
                    canonicalized_seq = reverse(complement(canonicalized_seq, segment_start, segment_end), segment_start, segment_end)
                else:
                    raise ValueError(f"{transformation} not recognized")
        else:
            # Check if more than 50% of seq should be canonicalized
            segments_length = 0
            for segment in transcribed_segments_that_contain_seq:
                segment_start = max(0, segment[0] - alignment_start)
                segment_end = min(len(seq_str), segment[1] - alignment_end)
                segments_length += segment_end - segment_start
            
            # in-place modifications to seq
            if segments_length > len(seq_str) / 2:
                if transformation == "reverse":
                    reverse(canonicalized_seq, 0, len(seq_str))
                elif transformation == "complement":
                    complement(canonicalized_seq, 0, len(seq_str))
                elif transformation == "rc":
                    reverse(canonicalized_seq, 0, len(seq_str))
                    complement(canonicalized_seq, 0, len(seq_str))
                else:
                    raise ValueError(f"{transformation} not recognized")
                
        canonicalized_records[id] = canonicalized_seq
        
        # There's no need for canonicalizing the labels because:
        # - On histone tasks, histones interact with double stranded DNA, the label is the same for the RC
        # - Enhancers (seem like) they're mostly bidirectional, and even if they aren't, I can't know if taking a non-enhancer sequence's RC suddenly turns it into an enhancer
        # - The promoters aren't transcribed
        # - The splicing seqs are already on the coding strand
    
    # Save the canonicalized_record
    save_seqs_dict(canonicalized_records, save_path)
    unaligned_percentage = round((unaligned_records * 100) / len(records.keys()), 1)
    aligned_without_annotation_percentage = round((aligned_records_without_annotation * 100) / len(records.keys()), 1)

    if unaligned_percentage > 0.05 or aligned_without_annotation_percentage > 0.05:
        log.warning(f"Task {task_name} has {unaligned_percentage + aligned_without_annotation_percentage}% records that couldn't be processed\n" + \
                    f"  {unaligned_percentage}% unaligned records\n" + \
                    f"  {aligned_without_annotation_percentage}% algined without annotations on the gff")