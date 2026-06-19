import json
import bisect
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.SeqRecord import SeqRecord
from intervaltree import IntervalTree
import src.utils as utils
import src.utils.train
import numpy as np

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

def transform_sequence(seq, start, end, transformation):
    if transformation == "reverse":
        reverse(seq, start, end)
    elif transformation == "complement":
        complement(seq, start, end)
    elif transformation == "rc":
        reverse(seq, start, end)
        complement(seq, start, end)
    else:
        raise ValueError(f"{transformation} not recognized")

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
    for training_genome_id, transformation_params in training_genome_canonicalization_segments.items():
        canonicalied = MutableSeq(training_genome[training_genome_id])
        
        for (start, end) in transformation_params["segments"]:
            transform_sequence(canonicalied, start, end, transformation)

        if transformation_params["should_rc_at_end"]:
            reverse(canonicalied, 0, len(canonicalied))
            complement(canonicalied, 0, len(canonicalied))

        results[training_genome_id] = canonicalied

    return results

def calculate_overlap_length(start, end, overlap_start, overlap_end):
    return max(0, min(end, overlap_end) - max(start, overlap_start))

def get_branch_segments_to_canonicalize_and_cost(to_orient, correctly_oriented_intervaltree, orient_all_segments, partial_canonicalization_works):
    cost = 0
    to_canonicalize = to_orient if orient_all_segments else []

    for (start, end) in to_orient:
        overlaps = intervaltree_to_tuples(correctly_oriented_intervaltree.overlap(start, end))
        ignore_segment_cost = end - start
        if partial_canonicalization_works:
            orient_segment_cost = sum(calculate_overlap_length(start, end, overlap_start, overlap_end) for (overlap_start, overlap_end) in overlaps)
        else:
            orient_segment_cost = sum(overlap_end - overlap_start for (overlap_start, overlap_end) in overlaps)
        
        if orient_all_segments:
            cost += orient_segment_cost
        else:
            if ignore_segment_cost < orient_segment_cost:
                cost += ignore_segment_cost
            else:
                cost += orient_segment_cost
                to_canonicalize.append((start, end))

    return to_canonicalize, cost

def generate_to_canonicalize_and_should_rc_at_end(positive_coding_segments, negative_coding_segments, orient_all_segments, partial_canonicalization_works):
    positive_coding_intervaltree = IntervalTree.from_tuples(positive_coding_segments)
    negative_coding_intervaltree = IntervalTree.from_tuples(negative_coding_segments)

    keep_positive_to_orient, keep_positive_cost = get_branch_segments_to_canonicalize_and_cost(
        negative_coding_segments, positive_coding_intervaltree, 
        orient_all_segments, partial_canonicalization_works
    )
    keep_negative_to_orient, keep_negative_cost = get_branch_segments_to_canonicalize_and_cost(
        positive_coding_segments, negative_coding_intervaltree,
        orient_all_segments, partial_canonicalization_works
    )

    if keep_positive_cost <= keep_negative_cost:
        return keep_positive_to_orient, False
    else:
        return keep_negative_to_orient, True


##################
# Control signal #
##################

def save_canonicalization_control_signal(training_genome, training_genome_to_canonicalize, out_dir):
    for ucsc_id in training_genome_to_canonicalize:
        if training_genome_to_canonicalize[ucsc_id]["should_rc_at_end"]:
            control_signal = np.ones(len(training_genome[ucsc_id]))
            segment_value = 0
        else:
            control_signal = np.zeros(len(training_genome[ucsc_id]))
            segment_value = 1
        
        for (start, end) in training_genome_to_canonicalize[ucsc_id]["segments"]:
            control_signal[start:end] = segment_value

        np.save(str(out_dir / f"{ucsc_id}.npy"), control_signal)


#################################
# NT Benchmark canonicalization #
#################################

HISTONE_TASKS = ("H3K4me3", "H3K27ac", "H3K27me3", "H3K4me1", "H3K36me3", "H3K9me3", "H3K9ac", "H3K4me2", "H4K20me1", "H2AFZ")
ENHANCERS_TASKS = ("enhancers", "enhancers_types")
PROMOTER_TASKS = ("promoter_all", "promoter_tata", "promoter_no_tata")
SPLICING_TASKS = ("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor")

def find_primary_alignment(hits):
    for hit in hits:
        if hit.is_primary:
            return hit
    return None

def find_annotated_alignment(hits, primary_alignment_mapq, gencode_transcribed_segments_intervaltree):
    for hit in hits:
        if hit.mapq >= primary_alignment_mapq and hit.ctg in gencode_transcribed_segments_intervaltree:
            return hit
    return None

def get_transcribed_segments_that_contain_seq(alignment, gencode_transcribed_segments_intervaltree):
    if alignment.strand == 1:
        segments_to_check = gencode_transcribed_segments_intervaltree[alignment.ctg]["positive"]
    elif alignment.strand == -1:
        segments_to_check = gencode_transcribed_segments_intervaltree[alignment.ctg]["negative"]
    else:
        return None

    return intervaltree_to_tuples(segments_to_check.overlap(alignment.r_st, alignment.r_en))

def align_transcript_position_with_seq(alignment, segment, chromosome_length):
    segment_start = max(0, segment[0] - alignment.r_st)
    segment_end = min(chromosome_length, segment[1] - alignment.r_st)
    return segment_start, segment_end

def save_control_signals(control_signals, save_dir, fasta_path):
    np.savez(str(save_dir / f"{fasta_path.stem}.npz"), **control_signals)

def complete_control_signals(control_signals, records):
    for id in records:
        if not id in control_signals:
            control_signals[id] = np.zeros(len(records[id]))

def canonicalize_nt_benchmark_fasta(
        fasta_path, 
        aligner, 
        gencode_transcribed_segments_intervaltree, 
        transformation, 
        partial_seq_canonicalization, 
        orient_promoters, 
        save_dir,
    ):
    records = read_fasta_into_records_dict(fasta_path)
    task_name = fasta_path.stem.replace("_train", "").replace("_test", "")
    save_file_path = save_dir / f"{fasta_path.stem}.fasta"
    
    # I don't know if the splicing tasks were taken from the template or coding strand
    # Even though promoters aren't transcribed, they can be inside a "transcript" feature if there's an ATSS(alternate transcription start site)
    # Orienting the promoters may break them, because they're inherently directional.
    if not orient_promoters and task_name in PROMOTER_TASKS:
        save_seqs_dict(records, save_file_path)
        control_signals = {id: np.zeros(len(records[id])) for id in records} # Save a control signal with all zeros
        save_control_signals(control_signals, save_dir, fasta_path)
        return
    
    canonicalized_records = {}
    control_signals = {}

    unaligned_records = 0
    aligned_records_without_annotation = 0
    for id in records:
        # Align the record with the reference genome, and take the primary alignment
        seq_str = str(records[id])
        hits = aligner.map(seq_str)
        
        # Find primary alignment
        alignment = find_primary_alignment(hits)

        if alignment is None:
            canonicalized_records[id] = records[id]
            unaligned_records += 1
            continue

        # If the primary alignment isn't on an annotated chromosome(like an unalocated segment), see if there's another alignment with the same mapq
        if not alignment.ctg in gencode_transcribed_segments_intervaltree:
            annotated_alignment = find_annotated_alignment(hits, alignment.mapq, gencode_transcribed_segments_intervaltree)
            if annotated_alignment is None:
                canonicalized_records[id] = records[id]
                aligned_records_without_annotation += 1
                continue
            else:
                alignment = annotated_alignment

        # Check if seq falls inside a transcribed segment or not
        transcribed_segments_that_contain_seq = get_transcribed_segments_that_contain_seq(alignment, gencode_transcribed_segments_intervaltree)
        if transcribed_segments_that_contain_seq is None:
            ValueError(f"Primary alignment had strand {alignment.strand} on {task_name} id {id}")

        canonicalized_seq = MutableSeq(records[id])
        control_signal = np.zeros(len(seq_str))

        if partial_seq_canonicalization:
            # Canonicalize seq using the segments
            for segment in transcribed_segments_that_contain_seq:
                # Index the segment relative to the seq's position
                segment_start, segment_end = align_transcript_position_with_seq(alignment, segment, len(seq_str))
                transform_sequence(canonicalized_seq, segment_start, segment_end, transformation)            
                control_signal[segment_start:segment_end] = 1
        else:
            # Check if more than 50% of seq should be canonicalized
            segments_length = 0
            for segment in transcribed_segments_that_contain_seq:
                segment_start, segment_end = align_transcript_position_with_seq(alignment, segment, len(seq_str))
                segments_length += segment_end - segment_start
            
            # in-place modifications to seq
            if segments_length > len(seq_str) / 2:
                transform_sequence(canonicalized_seq, 0, len(seq_str), transformation)
                control_signal[0:len(seq_str)] = 1
                
        canonicalized_records[id] = canonicalized_seq
        control_signals[id] = control_signal
        
        # There's no need for canonicalizing the labels because:
        # - On histone tasks, histones interact with double stranded DNA, the label is the same for the RC
        # - Enhancers (seem like) they're mostly bidirectional, and even if they aren't, I can't know if taking a non-enhancer sequence's RC suddenly turns it into an enhancer
        # - The promoters aren't transcribed
        # - The splicing seqs are already on the coding strand
    
    # Save the canonicalized_record
    save_seqs_dict(canonicalized_records, save_file_path)
    unaligned_percentage = round((unaligned_records * 100) / len(records.keys()), 1)
    aligned_without_annotation_percentage = round((aligned_records_without_annotation * 100) / len(records.keys()), 1)

    # Save the control signal
    complete_control_signals(control_signals, records) # Set the unprocessed records' control signals to 0s
    save_control_signals(control_signals, save_dir, fasta_path)

    if unaligned_percentage > 0.05 or aligned_without_annotation_percentage > 0.05:
        log.warning(f"Task {task_name} has {unaligned_percentage + aligned_without_annotation_percentage}% records that couldn't be processed\n" + \
                    f"  {unaligned_percentage}% unaligned records\n" + \
                    f"  {aligned_without_annotation_percentage}% algined without annotations on the gff")