from intervaltree import IntervalTree
from Bio.Seq import MutableSeq
import numpy as np
import math
from .utils import intervaltree_to_tuples, transform_sequence
from .parsed_genome import ParsedGenome

####################################
# Decide canonicalization segments #
####################################

def calculate_segment_set_destruction_cost(segment_set: set):
    return sum(end - start for (start, end) in segment_set)

def generate_canonicalization_segments_and_overlap_windows(positive_coding_segments, negative_coding_segments, overlapped_segments_break_entirely, partial_canonicalization_works):
    positive_coding_intervaltree = IntervalTree.from_tuples(positive_coding_segments)
    negative_coding_intervaltree = IntervalTree.from_tuples(negative_coding_segments)

    segments_to_flip = []
    overlap_windows_to_flip = []
    already_handled_negative_segments = set()

    for (start, end) in negative_coding_segments:
        # This check is neccesary when doing the overlap window orientation, when multiple segments at a time are oriented
        if (start, end) in already_handled_negative_segments:
            continue
        already_handled_negative_segments.add((start, end))

        overlaps = positive_coding_intervaltree.overlap(start, end)
        if len(overlaps) == 0 or partial_canonicalization_works:
            # The upper bound for the overlap length is the segment length
            # When doing partial canonicalization, it's always better or at least the same orienting the segment than not
            segments_to_flip.append((start, end))
        else:
            if overlapped_segments_break_entirely:
                # It's better to ignore(break) B than break A and B, so in this case, the segments that overlap are skipped
                pass
            else:
                # Recursively build the overlap window
                overlap_window_negative_segments = {(start, end)}
                overlap_window_positive_segments = set()
                build_overlap_window_segments(
                    start,
                    end,
                    "negative",
                    overlap_window_positive_segments,
                    overlap_window_negative_segments,
                    positive_coding_intervaltree,
                    negative_coding_intervaltree
                )

                # Always add the segments to the set, no matter if I'm destroying the positive or negative segments. I already handled them, no need to re-handle them
                for (start, end) in overlap_window_negative_segments:
                    already_handled_negative_segments.add((start, end))

                # Check if it's better to break the + segments or break the - segments
                if calculate_segment_set_destruction_cost(overlap_window_positive_segments) < calculate_segment_set_destruction_cost(overlap_window_negative_segments):
                    # To destroy the positive segments means flipping, not keeping, the negative segments
                    for (start, end) in overlap_window_negative_segments:
                        segments_to_flip.append((start, end))
                else:
                    for (start, end) in overlap_window_positive_segments:
                        segments_to_flip.append((start, end))
                    
                    # Now get the overlap window bounds
                    overlap_window_start = math.inf
                    overlap_window_end = 0
                    for (start, end) in overlap_window_positive_segments.union(overlap_window_negative_segments):
                        overlap_window_start = min(start, overlap_window_start)
                        overlap_window_end = max(end, overlap_window_end)

                    overlap_windows_to_flip.append((overlap_window_start, overlap_window_end))

    return segments_to_flip, overlap_windows_to_flip

def build_overlap_window_segments(
        current_segment_start: int, 
        current_segment_end: int, 
        current_segment_strand: str, 
        overlap_window_positive_segments: set,
        overlap_window_negative_segments: set,
        positive_segments_intervaltree: IntervalTree,
        negative_segments_intervaltree: IntervalTree
    ):

    if current_segment_strand == "positive":
        oposite_strand = negative_segments_intervaltree
        oposite_segments_set = overlap_window_negative_segments
        overlapping_segments_strand = "negative"
    elif current_segment_strand == "negative":
        oposite_strand = positive_segments_intervaltree
        oposite_segments_set = overlap_window_positive_segments
        overlapping_segments_strand = "positive"
    else:
        raise ValueError(f"Strand {current_segment_strand} not recognized")

    for overlapping_segment in oposite_strand.overlap(current_segment_start, current_segment_end):
        start = overlapping_segment[0]
        end = overlapping_segment[1]
        if (start, end) in oposite_segments_set:
            continue

        oposite_segments_set.add((start, end))
        build_overlap_window_segments(
            start, 
            end, 
            overlapping_segments_strand, 
            overlap_window_positive_segments, 
            overlap_window_negative_segments,
            positive_segments_intervaltree,
            negative_segments_intervaltree
        )

#################################
# Whole genome canonicalization #
#################################

def canonicalize_genome(training_genome, training_genome_canonicalization_segments, transformation):
    results = {}
    for training_genome_id, transformation_params in training_genome_canonicalization_segments.items():
        canonicalized = MutableSeq(training_genome[training_genome_id])
        
        for (start, end) in transformation_params["segments"]:
            transform_sequence(canonicalized, start, end, transformation)

        for (start, end) in transformation_params["overlap_windows"]:
            transform_sequence(canonicalized, start, end, transformation)

        results[training_genome_id] = canonicalized

    return results

def save_canonicalization_control_signal(training_genome, training_genome_to_canonicalize, out_dir):
    for ucsc_id in training_genome_to_canonicalize:
        control_signal = np.zeros(len(training_genome[ucsc_id]))
        
        for (start, end) in training_genome_to_canonicalize[ucsc_id]["segments"]:
            control_signal[start:end] = 1

        for (start, end) in training_genome_to_canonicalize[ucsc_id]["overlap_windows"]:
            control_signal[start:end] = 0 if control_signal[start] == 1 else 1 # Invert the first operation, even though it isn't truly representative

        np.save(str(out_dir / f"{ucsc_id}.npy"), control_signal)

def save_raw_control_signal(training_genome, annotations: ParsedGenome, out_dir):
    for ucsc_id in annotations.genome_transcribed_segments:
        control_signal = np.zeros(len(training_genome[ucsc_id]))

        # I know it isn't right to do this with overlapping segments, but it's for a simple ablation
        for segment in annotations.genome_transcribed_segments[ucsc_id]["positive"]:
            control_signal[segment[0]:segment[1]] = 1
        for segment in annotations.genome_transcribed_segments[ucsc_id]["negative"]:
            control_signal[segment[0]:segment[1]] = -1

        np.save(str(out_dir / f"{ucsc_id}.npy"), control_signal)