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
import pyranges as pr
import mappy as mp

log = src.utils.train.get_logger(__name__)


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
    
    if unaligned_percentage + aligned_without_annotation_percentage > 50:
        return True
    else:
        return False