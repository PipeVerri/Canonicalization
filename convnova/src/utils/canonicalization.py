import json
import bisect
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.SeqRecord import SeqRecord
from intervaltree import IntervalTree
import pronto
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

def build_ncbi_to_training_id(assembly_path, training_genome):
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

def parse_transcribed_features_from_gff(df, reference_id, 
                                        annotations_level=2,
                                        include_cdjv_segments=True,
                                        include_pseudogenes=True):
    
    assert type(annotations_level) == int and annotations_level <= 3 and annotations_level >= 1

    df = df.copy() # Copy the df to not overwrite the original
    df = df[df["Chromosome"] == reference_id].copy() # Filter the annotations for the target chromosome
    df = df[df["level"] <= annotations_level] # Filter the annotations for the desired level
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
        if row["Strand"] == 1:
            bisect.insort_left(positive_segments, (row["Start"], row["End"]))
        elif row["Strand"] == -1:
            bisect.insort_left(negative_segments, (row["Start"], row["End"]))

    positive_segments = merge_segment_list(positive_segments)
    negative_segments = merge_segment_list(negative_segments)
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


#################################
# NT Benchmark canonicalization #
#################################

def canonicalize_nt_benchmark_fasta(fasta_path, aligner, training_genome_canonicalization_intervaltree, transformation, save_path, partial_seq_canonicalization):
    records = read_fasta_into_records_dict(fasta_path)
    
    canonicalized_records = {}
    for id in records:
        # Align the record with the reference genome
        seq = str(records[id])
        best_alignment = next((h for h in aligner.map(seq) if h.is_primary), None)
        if best_alignment is None or best_alignment.mapq < 40:
            raise ValueError(f"Couldn't find good alignment for {id}")
    
        # Check if seq should be canonicalized or not
        alignment_start = best_alignment.r_st
        alignment_end = best_alignment.r_en
        overlapping_segments = intervaltree_to_tuples(training_genome_canonicalization_intervaltree.overlap(alignment_start, alignment_end))
        
        canonicalized_seq = MutableSeq(records[id])
        if partial_seq_canonicalization:
            # Canonicalize seq using the segments
            for segment in overlapping_segments:
                # Index the segment relative to the seq's position
                segment_start = max(0, segment[0] - alignment_start)
                segment_end = min(len(seq), segment[1] - alignment_end)
                
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
            for segment in overlapping_segments:
                segment_start = max(0, segment[0] - alignment_start)
                segment_end = min(len(seq), segment[1] - alignment_end)
                segments_length += segment_end - segment_start
            
            if segments_length > len(seq) / 2:
                if transformation == "reverse":
                    canonicalized_seq = reverse(canonicalized_seq, 0, len(seq))
                elif transformation == "complement":
                    canonicalized_seq = complement(canonicalized_seq, 0, len(seq))
                elif transformation == "rc":
                    canonicalized_seq = reverse(complement(canonicalized_seq, 0, len(seq)), 0, len(seq))
                else:
                    raise ValueError(f"{transformation} not recognized")
        
        raise ValueError("Y canonicalization needed")
        
        canonicalized_records[id] = canonicalized_seq
    
    # Save the canonicalized_record
    save_seqs_dict(canonicalized_records, save_path)

def canonicalize_nt_benchmark_label(label, task):
    HISTONE_TASKS = ("H3K4me3", "H3K27ac", "H3K27me3", "H3K4me1", "H3K36me3", "H3K9me3", "H3K9ac", "H3K4me2", "H4K20me1", "H2AFZ")
    ENHANCERS_TASKS = ("enhancers", "enhancers_types")
    PROMOTER_TASKS = ("promoter_all", "promoter_tata", "promoter_no_tata")
    SPLICING_TASKS = ("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor")
    
    if task in HISTONE_TASKS:
        return label # Histones interact with double stranded DNA, directionality doesn't matter
    elif task in ENHANCERS_TASKS:
        return label # Most enhancers seem to be bidirectional
    elif task in PROMOTER_TASKS:
        return invert_binary_label(label) # Promoters are directional
    elif task in SPLICING_TASKS:
        raise ValueError("Unimplemented")

def invert_binary_label(label):
    if label == 1:
        return 0
    elif label == 0:
        return 1
    else:
        raise ValueError("Invalid binary label")