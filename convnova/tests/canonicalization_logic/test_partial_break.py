from src.utils.canonicalization import generate_canonicalization_params
import pytest


@pytest.mark.parametrize("orient_all_segments", [True, False])
@pytest.mark.parametrize("keep_segments_order", [True, False])
def test_overlapping_negative_segment_is_always_flipped_regardless_of_flags(orient_all_segments, keep_segments_order):
    # partial_break always flips an overlapping negative segment: the cost of
    # leaving it unoriented is never lower than the cost of orienting it, since an
    # overlap's length can't exceed the segment's own length. orient_all_segments
    # and keep_segments_order have no effect on this decision.
    result = generate_canonicalization_params(
        [(5, 30)], [(0, 10)], "partial_break", orient_all_segments, keep_segments_order
    )

    assert result.segments == [(0, 10)]
    assert result.conflict_windows == []
    assert result.should_rc_at_end is False


def test_multiple_overlapping_negative_segments_are_all_flipped():
    result = generate_canonicalization_params(
        [(5, 30), (40, 60)], [(0, 10), (35, 50)], "partial_break", False, False
    )

    assert sorted(result.segments) == [(0, 10), (35, 50)]


def test_fully_contained_negative_segment_is_flipped():
    result = generate_canonicalization_params(
        [(0, 100)], [(20, 30)], "partial_break", False, False
    )

    assert result.segments == [(20, 30)]


def test_mix_of_overlapping_and_non_overlapping_segments():
    result = generate_canonicalization_params(
        [(5, 30)], [(0, 10), (200, 210)], "partial_break", False, False
    )

    assert sorted(result.segments) == [(0, 10), (200, 210)]
