from src.utils.canonicalization import generate_canonicalization_params
import pytest


@pytest.mark.parametrize("orient_all_segments", [True, False])
@pytest.mark.parametrize("keep_segments_order", [True, False])
def test_overlapping_negative_segment_is_always_skipped_regardless_of_flags(orient_all_segments, keep_segments_order):
    # both_break leaves overlapping segments untouched on both strands: it's
    # cheaper to only break the negative segment's canonicalization than to break
    # both it and the positive segment it conflicts with. orient_all_segments and
    # keep_segments_order have no effect on this decision.
    result = generate_canonicalization_params(
        [(5, 30)], [(0, 10)], "both_break", orient_all_segments, keep_segments_order
    )

    assert result.segments == []
    assert result.conflict_windows == []
    assert result.should_rc_at_end is False


def test_multiple_overlapping_negative_segments_are_all_skipped():
    result = generate_canonicalization_params(
        [(5, 30), (40, 60)], [(0, 10), (35, 50)], "both_break", False, False
    )

    assert result.segments == []


def test_non_overlapping_segments_are_still_flipped_alongside_skipped_ones():
    result = generate_canonicalization_params(
        [(5, 30)], [(0, 10), (100, 110)], "both_break", False, False
    )

    assert result.segments == [(100, 110)]
