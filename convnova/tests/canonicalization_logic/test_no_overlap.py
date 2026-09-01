from src.utils.canonicalization import generate_canonicalization_params
import pytest


@pytest.mark.parametrize("behavior", ["partial_break", "both_break", "full_break"])
@pytest.mark.parametrize("orient_all_segments", [True, False])
@pytest.mark.parametrize("keep_segments_order", [True, False])
def test_non_overlapping_negative_segment_is_always_flipped_directly(behavior, orient_all_segments, keep_segments_order):
    # When a negative segment has no positive counterpart to conflict with, it gets
    # brought over as-is no matter which overlap-handling behavior or flag
    # combination is requested.
    result = generate_canonicalization_params(
        [(50, 60)], [(0, 10)], behavior, orient_all_segments, keep_segments_order
    )

    assert result.segments == [(0, 10)]
    assert result.conflict_windows == []
    assert result.should_rc_at_end is False


@pytest.mark.parametrize("behavior", ["partial_break", "both_break", "full_break"])
def test_empty_inputs_produce_empty_result(behavior):
    result = generate_canonicalization_params([], [], behavior, False, False)

    assert result.segments == []
    assert result.conflict_windows == []
    assert result.should_rc_at_end is False


def test_multiple_non_overlapping_negative_segments_are_all_flipped():
    result = generate_canonicalization_params(
        [(200, 210)], [(0, 10), (50, 60)], "full_break", False, False
    )

    assert sorted(result.segments) == [(0, 10), (50, 60)]
    assert result.conflict_windows == []
