from src.utils.canonicalization import generate_canonicalization_params

# full_break + orient_all_segments=True + keep_segments_order=True is a special case
# handled before the per-segment loop: forcing both "orient everything" and "keep
# relative order" means every segment must end up on the same strand, so the whole
# genome's segments are compared by the total cost of moving them to each strand
# and the cheaper strand wins outright - there's no per-conflict-window reasoning.


def test_moves_everything_to_positive_strand_when_cheaper():
    positive = [(5, 20), (25, 28)] # 18 nucleotides
    negative = [(0, 10), (18, 30)] # 22 nucleotides
    # bring negative->positive costs 33, bring positive->negative costs 34
    # (see test_cost_helpers for how this full-overlapping-segment-length cost is computed)

    result = generate_canonicalization_params(positive, negative, "full_break", True, True)

    assert sorted(result.segments) == [(0, 10), (18, 30)]
    assert result.conflict_windows == []
    assert result.should_transform_at_end is False


def test_moves_everything_to_negative_strand_when_cheaper():
    # A single small positive segment overlapped by many small negative segments:
    # bringing the many small negatives onto the positive segment is charged the
    # full length of that positive segment each time, making it far more expensive
    # than bringing the one positive segment onto the negative strand.
    positive = [(0, 50)]
    negative = [(0, 5), (10, 15), (20, 25)]

    result = generate_canonicalization_params(positive, negative, "full_break", True, True)

    assert result.segments == [(0, 50)]
    assert result.conflict_windows == []
    assert result.should_transform_at_end is True


def test_tie_prefers_the_positive_strand():
    # The positive strand is the one resolved for ties
    result = generate_canonicalization_params([(0, 10)], [(0, 10)], "full_break", True, True)

    assert result.segments == [(0, 10)]
    assert result.should_transform_at_end is False


def test_takes_this_path_even_when_no_segments_overlap():
    result = generate_canonicalization_params([(50, 60)], [(0, 10)], "full_break", True, True)

    assert result.segments == [(0, 10)]
    assert result.conflict_windows == []
    assert result.should_transform_at_end is False
