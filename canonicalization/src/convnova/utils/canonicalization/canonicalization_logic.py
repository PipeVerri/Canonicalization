from intervaltree import IntervalTree
from Bio.Seq import MutableSeq
import numpy as np
from .utils import intervaltree_to_tuples, transform_sequence

####################################
# Decide canonicalization segments #
####################################

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

def calculate_overlap_length(start, end, overlap_start, overlap_end):
    return max(0, min(end, overlap_end) - max(start, overlap_start))


#################################
# Whole genome canonicalization #
#################################

def canonicalize_genome(training_genome, training_genome_canonicalization_segments, transformation):
    results = {}
    for training_genome_id, transformation_params in training_genome_canonicalization_segments.items():
        canonicalized = MutableSeq(training_genome[training_genome_id])
        
        for (start, end) in transformation_params["segments"]:
            transform_sequence(canonicalized, start, end, transformation)

        if transformation_params["should_rc_at_end"]:
            transform_sequence(canonicalized, 0, len(canonicalized), "rc")

        results[training_genome_id] = canonicalized

    return results

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