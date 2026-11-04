from src.utils.canonicalization import (
    calculate_segment_set_destruction_cost,
    calculate_cost_of_bringing_everything_to_strand,
)
from intervaltree import IntervalTree


def test_calculate_segment_set_destruction_cost_empty():
    assert calculate_segment_set_destruction_cost(set()) == 0


def test_calculate_segment_set_destruction_cost_single_segment():
    assert calculate_segment_set_destruction_cost({(0, 10)}) == 10


def test_calculate_segment_set_destruction_cost_multiple_segments():
    assert calculate_segment_set_destruction_cost({(0, 10), (20, 25)}) == 15


def test_calculate_cost_of_bringing_everything_to_strand_no_overlap():
    dest_tree = IntervalTree.from_tuples([(100, 110)])
    assert calculate_cost_of_bringing_everything_to_strand([(0, 10)], dest_tree) == 0


def test_calculate_cost_of_bringing_everything_to_strand_counts_full_overlapping_segment_length():
    # The cost is the length of the whole overlapping destination segment, not just
    # the intersection length - a single 5-nucleotide origin segment overlapping a
    # 50-nucleotide destination segment costs 50, not 5.
    dest_tree = IntervalTree.from_tuples([(0, 50)])
    assert calculate_cost_of_bringing_everything_to_strand([(20, 25)], dest_tree) == 50


def test_calculate_cost_of_bringing_everything_to_strand_sums_across_origin_segments():
    dest_tree = IntervalTree.from_tuples([(0, 5), (10, 15), (20, 25)])
    origin_segments = [(0, 5), (10, 15), (20, 25)]
    assert calculate_cost_of_bringing_everything_to_strand(origin_segments, dest_tree) == 15


def test_calculate_cost_of_bringing_everything_to_strand_sums_across_multiple_overlaps_of_one_segment():
    # A single origin segment spanning several destination segments is charged for
    # every one of them.
    dest_tree = IntervalTree.from_tuples([(0, 5), (10, 15), (20, 25)])
    assert calculate_cost_of_bringing_everything_to_strand([(0, 25)], dest_tree) == 15
