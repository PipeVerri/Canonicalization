from src.utils.canonicalization import generate_canonicalization_params
import pytest


# both_break ignores orient_all_segments. With keep_segments_order=False, each
# connected conflict window is flipped independently when the total length of its
# positive segments is no greater than the total length of its negative segments.


@pytest.mark.parametrize("orient_all_segments", [True, False])
def test_free_order_skips_conflict_window_when_flipping_costs_more(orient_all_segments):
    # Flipping would destroy a 25 nt positive segment to save a 10 nt negative one.
    positive = [(5, 30)]
    negative = [(0, 10), (100, 110)]

    result = generate_canonicalization_params(
        positive, negative, "both_break", orient_all_segments, False
    )

    assert result.segments == [(100, 110)]
    assert result.conflict_windows == []
    assert result.should_transform_at_end is False


@pytest.mark.parametrize("orient_all_segments", [True, False])
def test_free_order_flips_conflict_window_when_cheaper(orient_all_segments):
    # Flipping costs the 15 nt positive segment instead of the 30 nt negative one.
    result = generate_canonicalization_params(
        [(5, 20)], [(0, 30)], "both_break", orient_all_segments, False
    )

    assert result.segments == []
    assert result.conflict_windows == [(0, 30)]
    assert result.should_transform_at_end is False


def test_free_order_tie_prefers_to_flip_the_conflict_window():
    result = generate_canonicalization_params(
        [(0, 10)], [(0, 10)], "both_break", False, False
    )

    assert result.segments == []
    assert result.conflict_windows == []
    assert result.should_transform_at_end is False


def test_free_order_flips_a_chained_conflict_window_only_once():
    # neg1 (0,10) overlaps pos1 (5,20), which overlaps neg2 (18,30),
    # which in turn overlaps pos2 (25,28). All four form one window.
    positive = [(5, 20), (25, 28)]  # 15 + 3 = 18 nt flipping cost
    negative = [(0, 10), (18, 30)]  # 10 + 12 = 22 nt ignoring cost

    result = generate_canonicalization_params(
        positive, negative, "both_break", False, False
    )

    assert result.segments == []
    assert result.conflict_windows == [(0, 30)]
    assert result.should_transform_at_end is False


def test_free_order_resolves_independent_conflict_windows_separately():
    # First window: flipping costs 17 nt vs 8 nt to ignore, so leave it alone.
    # Second window: flipping costs 5 nt vs 30 nt to ignore, so flip it.
    positive = [(3, 20), (205, 210)]
    negative = [(0, 5), (15, 18), (200, 230), (300, 310)]

    result = generate_canonicalization_params(
        positive, negative, "both_break", False, False
    )

    assert result.segments == [(300, 310)]
    assert result.conflict_windows == [(200, 230)]
    assert result.should_transform_at_end is False


# With keep_segments_order=True, the decision is chromosome-wide instead of per
# conflict window. Only non-overlapping segments are transformed directly; when
# the negative strand wins, the whole sequence is reverse-complemented at the end.


@pytest.mark.parametrize("orient_all_segments", [True, False])
def test_keep_order_prefers_positive_strand_when_cheaper(orient_all_segments):
    positive = [(5, 20), (25, 28)]
    negative = [(0, 10), (18, 30), (100, 110)]
    # Bringing negative segments to the positive strand costs 33, vs 34 in the
    # other direction. Only the non-overlapping negative segment is flipped.

    result = generate_canonicalization_params(
        positive, negative, "both_break", orient_all_segments, True
    )

    assert result.segments == [(100, 110)]
    assert result.conflict_windows == []
    assert result.should_transform_at_end is False


def test_keep_order_prefers_negative_strand_when_cheaper():
    positive = [(0, 50), (100, 110)]
    negative = [(0, 5), (10, 15), (20, 25), (200, 210)]
    # Bringing negative segments to the positive strand costs 150, vs 15 in the
    # other direction. Only the non-overlapping positive segment is flipped before
    # the whole-sequence reverse complement.

    result = generate_canonicalization_params(
        positive, negative, "both_break", False, True
    )

    assert result.segments == [(100, 110)]
    assert result.conflict_windows == []
    assert result.should_transform_at_end is True


def test_keep_order_tie_prefers_the_positive_strand():
    result = generate_canonicalization_params(
        [(0, 10)], [(0, 10), (100, 110)], "both_break", False, True
    )

    assert result.segments == [(100, 110)]
    assert result.conflict_windows == []
    assert result.should_transform_at_end is False
