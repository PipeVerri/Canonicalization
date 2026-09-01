from intervaltree import IntervalTree
from Bio.Seq import MutableSeq
import numpy as np
import math
from .utils import transform_sequence
from .parsed_genome import ParsedGenome
from dataclasses import dataclass
from pathlib import Path

####################################
# Decide canonicalization segments #
####################################

@dataclass(frozen=True)
class CanonicalizationParams:
    segments: list
    conflict_windows: list
    should_rc_at_end: bool

def calculate_segment_set_destruction_cost(segment_set: set):
    return sum(end - start for (start, end) in segment_set)

def calculate_cost_of_bringing_everything_to_strand(origin_strand: list[tuple[int, int]], dest_strand_intervaltree: IntervalTree):
    total_cost = 0
    for (start, end) in origin_strand:
        overlaps = dest_strand_intervaltree.overlap(start, end)
        for overlap_segment in overlaps:
            total_cost += overlap_segment[1] - overlap_segment[0]
    return total_cost

def generate_canonicalization_params(
        positive_coding_segments: list[tuple[int, int]], 
        negative_coding_segments: list[tuple[int, int]],
        overlap_transformation_behavior: str,
        orient_all_segments: bool,
        keep_segments_order: bool
    ) -> CanonicalizationParams:
    if not overlap_transformation_behavior in ("partial_break", "full_break", "both_break"):
        raise ValueError(f"Overlap handle mode {overlap_transformation_behavior} not recognized")

    positive_coding_intervaltree = IntervalTree.from_tuples(positive_coding_segments)
    negative_coding_intervaltree = IntervalTree.from_tuples(negative_coding_segments)

    segments_to_flip = []
    conflict_windows_to_flip = []
    already_handled_segments = set()

    # This is a special case that can't be processed in the way that the rest of the cases are
    if overlap_transformation_behavior == "full_break" and orient_all_segments and keep_segments_order:
        # Forcing the segments order means having to do a compromise and put them all in the same strand
        # For each strand, check the cost of bringing everything into that strand
        bring_to_positive_strand_cost = calculate_cost_of_bringing_everything_to_strand(
            origin_strand=negative_coding_segments,
            dest_strand_intervaltree=positive_coding_intervaltree
        )
        bring_to_negative_strand_cost = calculate_cost_of_bringing_everything_to_strand(
            origin_strand=positive_coding_segments,
            dest_strand_intervaltree=negative_coding_intervaltree
        )
        if bring_to_positive_strand_cost <= bring_to_negative_strand_cost:
            return CanonicalizationParams(
                segments=negative_coding_segments,
                conflict_windows=[],
                should_rc_at_end=False
            )
        else:
            return CanonicalizationParams(
                segments=positive_coding_segments,
                conflict_windows=[],
                should_rc_at_end=True
            )

    # For the explanations, A is a positive segment that overlaps with B, a negative segment
    for (start, end) in negative_coding_segments:
        # Sometimes I handle multiple segments in one iteration, and this is done to not re-process them again
        if (start, end) in already_handled_segments:
            continue
        already_handled_segments.add((start, end))
        
        overlaps = positive_coding_intervaltree.overlap(start, end)
        if len(overlaps) > 0:
            if overlap_transformation_behavior == "partial_break":
                # No matter the rest of the params, the cost of not orienting a segment is >= orienting it because it's overlap lenght <= segment length
                segments_to_flip.append((start, end))
            elif overlap_transformation_behavior == "both_break":
                # If both segments A and B break when being oriented, it's better to only break B instead of breaking A + B
                continue
            elif overlap_transformation_behavior == "full_break":
                conflict_window_positive_segments = set()
                conflict_window_negative_segments = {(start, end)}
                build_conflict_windows(
                    start, end, "-", 
                    conflict_window_positive_segments=conflict_window_positive_segments, 
                    conflict_window_negative_segments=conflict_window_negative_segments,
                    positive_segments_intervaltree=positive_coding_intervaltree,
                    negative_segments_intervaltree=negative_coding_intervaltree
                )

                # Add the negative segments to the already_handled set, that should be done in both cases
                for conflict_segment in conflict_window_negative_segments:
                    already_handled_segments.add((conflict_segment[0], conflict_segment[1]))
                
                # The cost will be used for both logic paths
                save_positive_segments_cost = calculate_segment_set_destruction_cost(conflict_window_negative_segments) # Saving the positive costs destroying the negative
                save_negative_segments_cost = calculate_segment_set_destruction_cost(conflict_window_positive_segments)

                if not orient_all_segments:
                    # If I'm not forced to orient all segments, I have to choose wether to save A(and ignore B), or save B(flip it and destroy A)
                    # If there are more than two segments(A+, B-,C-), then maybe it's better to save B and C than to save A, I can't infer that by only looking at A and B.
                    # That's why I'm building the conflict window first
                    
                    if save_negative_segments_cost < save_positive_segments_cost:
                        for negative_conflict_segment in conflict_window_negative_segments:
                            segments_to_flip.append((negative_conflict_segment[0], negative_conflict_segment[1]))
                    else:
                        # If it's cheaper to save the positive segments, just ignore the negative ones
                        pass
                else:
                    if keep_segments_order:
                        raise RuntimeError("This case should have been handled above")
                    else:
                        # The same logic as the not_orient_all_segments case, but if it's cheaper to save the positives, then take their RC and then take the RC of the whole conflict window
                        if save_negative_segments_cost < save_positive_segments_cost:
                            for negative_conflict_segment in conflict_window_negative_segments:
                                segments_to_flip.append((negative_conflict_segment[0], negative_conflict_segment[1]))
                        else:
                            # Bring the positive segments into the negative strand
                            for positive_conflict_segment in conflict_window_positive_segments:
                                segments_to_flip.append((positive_conflict_segment[0], positive_conflict_segment[1]))
                            
                            # Then, take the RC of the whole conflicts window
                            conflict_window_start = math.inf
                            conflict_window_end = 0
                            for conflict_segment in conflict_window_negative_segments.union(conflict_window_negative_segments):
                                conflict_window_start = min(conflict_window_start, conflict_segment[0])
                                conflict_window_end = max(conflict_window_end, conflict_segment[1])
                            conflict_windows_to_flip.append((conflict_window_start, conflict_window_end))
        else:
            # If there aren't any overlaps, just bring it
            segments_to_flip.append((start, end))

    return CanonicalizationParams(
        segments=segments_to_flip,
        conflict_windows=conflict_windows_to_flip,
        should_rc_at_end=False
    )


def build_conflict_windows(
        current_segment_start: int, 
        current_segment_end: int, 
        current_segment_strand: str, 
        conflict_window_positive_segments: set,
        conflict_window_negative_segments: set,
        positive_segments_intervaltree: IntervalTree,
        negative_segments_intervaltree: IntervalTree
    ):

    if current_segment_strand == "positive":
        oposite_strand = negative_segments_intervaltree
        oposite_segments_set = conflict_window_negative_segments
        overlapping_segments_strand = "negative"
    elif current_segment_strand == "negative":
        oposite_strand = positive_segments_intervaltree
        oposite_segments_set = conflict_window_positive_segments
        overlapping_segments_strand = "positive"
    else:
        raise ValueError(f"Strand {current_segment_strand} not recognized")

    for overlapping_segment in oposite_strand.overlap(current_segment_start, current_segment_end):
        start = overlapping_segment[0]
        end = overlapping_segment[1]
        if (start, end) in oposite_segments_set:
            continue

        oposite_segments_set.add((start, end))
        build_conflict_windows(
            start, 
            end, 
            overlapping_segments_strand, 
            conflict_window_positive_segments, 
            conflict_window_negative_segments,
            positive_segments_intervaltree,
            negative_segments_intervaltree
        )

#################################
# Whole genome canonicalization #
#################################

def canonicalize_genome(training_genome, training_genome_canonicalization_params: dict[str, CanonicalizationParams], transformation):
    results = {}
    for training_genome_id, transformation_params in training_genome_canonicalization_params.items():
        canonicalized = MutableSeq(training_genome[training_genome_id])
        
        for (start, end) in transformation_params.segments:
            transform_sequence(canonicalized, start, end, transformation)

        for (start, end) in transformation_params.conflict_windows:
            transform_sequence(canonicalized, start, end, transformation)

        if transformation_params.should_rc_at_end:
            transform_sequence(canonicalized, 0, len(canonicalized), transformation)

        results[training_genome_id] = canonicalized

    return results

def save_canonicalization_control_signal(training_genome, training_genome_to_canonicalize: dict[str, CanonicalizationParams], out_dir: Path):
    for ucsc_id in training_genome_to_canonicalize:
        control_signal = np.zeros(len(training_genome[ucsc_id]))
        
        for (start, end) in training_genome_to_canonicalize[ucsc_id].segments:
            control_signal[start:end] = 1

        for (start, end) in training_genome_to_canonicalize[ucsc_id].conflict_windows:
            control_signal[start:end] = 0 if control_signal[start] == 1 else 1 # Invert the first operation, even though it isn't truly representative

        if training_genome_to_canonicalize[ucsc_id].should_rc_at_end:
            mask = control_signal == 1
            control_signal[control_signal == 0] = 1
            control_signal[mask] = 0

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