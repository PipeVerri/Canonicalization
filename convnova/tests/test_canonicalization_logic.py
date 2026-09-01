from src.utils.canonicalization import (
    calculate_segment_set_destruction_cost,
    generate_canonicalization_segments_and_overlap_windows,
    build_overlap_window_segments,
    canonicalize_genome,
)
from Bio.Seq import Seq
from intervaltree import IntervalTree


def test_calculate_segment_set_destruction_cost():
    assert calculate_segment_set_destruction_cost(set()) == 0
    assert calculate_segment_set_destruction_cost({(0, 10)}) == 10
    assert calculate_segment_set_destruction_cost({(0, 10), (20, 25)}) == 15


def test_build_overlap_window_segments_no_overlap():
    positive_tree = IntervalTree.from_tuples([(100, 110)])
    negative_tree = IntervalTree.from_tuples([(0, 10)])

    overlap_window_positive_segments = set()
    overlap_window_negative_segments = {(0, 10)}
    build_overlap_window_segments(
        0, 10, "negative",
        overlap_window_positive_segments,
        overlap_window_negative_segments,
        positive_tree,
        negative_tree,
    )

    assert overlap_window_negative_segments == {(0, 10)}
    assert overlap_window_positive_segments == set()


def test_build_overlap_window_segments_follows_chain_across_multiple_segments():
    # neg1 (0,10) overlaps pos1 (5,20) overlaps neg2 (18,30) overlaps pos2 (25,28):
    # a chain of alternating-strand segments that all end up in the same window.
    neg1, pos1, neg2, pos2 = (0, 10), (5, 20), (18, 30), (25, 28)
    positive_tree = IntervalTree.from_tuples([pos1, pos2])
    negative_tree = IntervalTree.from_tuples([neg1, neg2])

    overlap_window_positive_segments = set()
    overlap_window_negative_segments = {neg1}
    build_overlap_window_segments(
        neg1[0], neg1[1], "negative",
        overlap_window_positive_segments,
        overlap_window_negative_segments,
        positive_tree,
        negative_tree,
    )

    assert overlap_window_negative_segments == {neg1, neg2}
    assert overlap_window_positive_segments == {pos1, pos2}


def test_build_overlap_window_segments_does_not_revisit_segments():
    # neg1 and pos1 mutually overlap each other; without a visited check this would
    # recurse between them forever.
    neg1, pos1 = (0, 10), (5, 15)
    positive_tree = IntervalTree.from_tuples([pos1])
    negative_tree = IntervalTree.from_tuples([neg1])

    overlap_window_positive_segments = set()
    overlap_window_negative_segments = {neg1}
    build_overlap_window_segments(
        neg1[0], neg1[1], "negative",
        overlap_window_positive_segments,
        overlap_window_negative_segments,
        positive_tree,
        negative_tree,
    )

    assert overlap_window_negative_segments == {neg1}
    assert overlap_window_positive_segments == {pos1}


def test_generate_segments_no_overlap_flips_negative_directly():
    positive_segments = [(50, 60)]
    negative_segments = [(0, 10)]

    to_flip, overlap_windows = generate_canonicalization_segments_and_overlap_windows(
        positive_segments, negative_segments, "full"
    )

    assert to_flip == [(0, 10)]
    assert overlap_windows == []


def test_generate_segments_partial_canonicalization_flips_despite_overlap():
    positive_segments = [(5, 30)]
    negative_segments = [(0, 10)]

    to_flip, overlap_windows = generate_canonicalization_segments_and_overlap_windows(
        positive_segments, negative_segments, "partial"
    )

    assert to_flip == [(0, 10)]
    assert overlap_windows == []


def test_generate_segments_break_entirely_skips_overlapping_segment():
    positive_segments = [(5, 30)]
    negative_segments = [(0, 10)]

    to_flip, overlap_windows = generate_canonicalization_segments_and_overlap_windows(
        positive_segments, negative_segments, "skip"
    )

    assert to_flip == []
    assert overlap_windows == []


def test_generate_segments_overlap_window_destroys_cheaper_negative_side():
    positive_segments = [(3, 20)]
    negative_segments = [(0, 5), (15, 18)]
    
    to_flip, overlap_windows = generate_canonicalization_segments_and_overlap_windows(
        positive_segments, negative_segments, "full"
    )

    assert to_flip == [(3, 20)]
    assert overlap_windows == [(0, 20)]


def test_generate_segments_overlap_window_destroys_cheaper_positive_side():
    positive_segments = [(5, 20), (25, 28)] # 18 nucleotides
    negative_segments = [(0, 10), (18, 30)] # 22 nucleotides

    to_flip, overlap_windows = generate_canonicalization_segments_and_overlap_windows(
        positive_segments, negative_segments, "partial"
    )

    assert sorted(to_flip) == [(0, 10), (18, 30)]
    assert overlap_windows == []


def test_generate_segments_multiple_independent_overlap_groups():
    positive_segments = [(3, 20), (205, 210)] # 17 nucleotides on the first window, 5 nucleotides on the second
    negative_segments = [(0, 5), (15, 18), (200, 230)] # 8 nucleotides on the first window, 30 nucleotides on the second

    to_flip, overlap_windows = generate_canonicalization_segments_and_overlap_windows(
        positive_segments, negative_segments, "full"
    )

    assert sorted(to_flip) == [(3, 20), (200, 230)]
    assert overlap_windows == [(0, 20)]


def test_generate_segments_empty_inputs():
    to_flip, overlap_windows = generate_canonicalization_segments_and_overlap_windows(
        [], [], "partial"
    )
    assert to_flip == []
    assert overlap_windows == []

def test_canonicalize_genome_applies_segments_and_overlap_windows():
    training_genome = {"chr1": Seq("AAAACCCCGGGGTTTT")} # 4 of each
    canonicalization_segments = {
        "chr1": {"segments": [(0, 4)], "overlap_windows": [(4, 8)]}
    }

    result = canonicalize_genome(training_genome, canonicalization_segments, "complement")

    assert str(result["chr1"]) == "TTTTGGGGGGGGTTTT"
