from src.utils.canonicalization import build_conflict_windows
from intervaltree import IntervalTree
import pytest


def test_build_conflict_windows_no_overlap():
    positive_tree = IntervalTree.from_tuples([(100, 110)])
    negative_tree = IntervalTree.from_tuples([(0, 10)])

    conflict_window_positive_segments = set()
    conflict_window_negative_segments = {(0, 10)}
    build_conflict_windows(
        0, 10, "negative",
        conflict_window_positive_segments,
        conflict_window_negative_segments,
        positive_tree,
        negative_tree,
    )

    assert conflict_window_negative_segments == {(0, 10)}
    assert conflict_window_positive_segments == set()


def test_build_conflict_windows_follows_chain_across_multiple_segments():
    # neg1 (0,10) overlaps pos1 (5,20) overlaps neg2 (18,30) overlaps pos2 (25,28):
    # a chain of alternating-strand segments that all end up in the same window.
    neg1, pos1, neg2, pos2 = (0, 10), (5, 20), (18, 30), (25, 28)
    positive_tree = IntervalTree.from_tuples([pos1, pos2])
    negative_tree = IntervalTree.from_tuples([neg1, neg2])

    conflict_window_positive_segments = set()
    conflict_window_negative_segments = {neg1}
    build_conflict_windows(
        neg1[0], neg1[1], "negative",
        conflict_window_positive_segments,
        conflict_window_negative_segments,
        positive_tree,
        negative_tree,
    )

    assert conflict_window_negative_segments == {neg1, neg2}
    assert conflict_window_positive_segments == {pos1, pos2}


def test_build_conflict_windows_can_start_from_the_positive_strand():
    # The chain-following logic is symmetric: starting the walk from a positive
    # segment must reach the overlapping negative segment the same way.
    pos1, neg1 = (0, 10), (5, 20)
    positive_tree = IntervalTree.from_tuples([pos1])
    negative_tree = IntervalTree.from_tuples([neg1])

    conflict_window_positive_segments = {pos1}
    conflict_window_negative_segments = set()
    build_conflict_windows(
        pos1[0], pos1[1], "positive",
        conflict_window_positive_segments,
        conflict_window_negative_segments,
        positive_tree,
        negative_tree,
    )

    assert conflict_window_positive_segments == {pos1}
    assert conflict_window_negative_segments == {neg1}


def test_build_conflict_windows_does_not_revisit_segments():
    # neg1 and pos1 mutually overlap each other; without a visited check this would
    # recurse between them forever.
    neg1, pos1 = (0, 10), (5, 15)
    positive_tree = IntervalTree.from_tuples([pos1])
    negative_tree = IntervalTree.from_tuples([neg1])

    conflict_window_positive_segments = set()
    conflict_window_negative_segments = {neg1}
    build_conflict_windows(
        neg1[0], neg1[1], "negative",
        conflict_window_positive_segments,
        conflict_window_negative_segments,
        positive_tree,
        negative_tree,
    )

    assert conflict_window_negative_segments == {neg1}
    assert conflict_window_positive_segments == {pos1}


def test_build_conflict_windows_rejects_unrecognized_strand():
    with pytest.raises(ValueError):
        build_conflict_windows(0, 10, "sideways", set(), set(), IntervalTree(), IntervalTree())
