from src.utils.canonicalization import generate_canonicalization_params

# full_break + orient_all_segments=True + keep_segments_order=False: every conflict
# window must be resolved (nothing is left mixed-strand), but segments don't need to
# stay in their original relative order. When it's cheaper to keep the negative
# segments, they're flipped exactly like the not-orient-all-segments case. When it's
# cheaper to keep the positive segments, those are flipped into the negative strand
# *and* the whole conflict window (spanning both the positive and negative segments
# involved) is queued to be reverse-complemented, undoing the local flip within the
# window's positive segments so the window's original overlap is preserved on the
# negative strand instead.


def test_flips_negative_segments_when_cheaper_and_adds_no_window():
    positive = [(5, 20), (25, 28)]  # 18 nucleotides
    negative = [(0, 10), (18, 30)]  # 22 nucleotides

    result = generate_canonicalization_params(positive, negative, "full_break", True, False)

    assert sorted(result.segments) == [(0, 10), (18, 30)]
    assert result.conflict_windows == []


def test_flips_positive_segments_and_queues_the_conflict_window_when_cheaper():
    positive = [(3, 20)]  # conflicting negative cost = 8
    negative = [(0, 5), (15, 18)]  # conflicting positive cost = 17

    result = generate_canonicalization_params(positive, negative, "full_break", True, False)

    assert result.segments == [(3, 20)]
    assert result.conflict_windows == [(0, 20)]


def test_conflict_window_spans_the_full_extent_of_both_strands():
    # The positive segment (0, 100) extends well past the negative segment (40, 60)
    # it conflicts with; the queued window must cover the whole conflict, not just
    # the narrower negative segment's bounds.
    result = generate_canonicalization_params([(0, 100)], [(40, 60)], "full_break", True, False)

    assert result.segments == [(0, 100)]
    assert result.conflict_windows == [(0, 100)]


def test_multiple_independent_conflict_windows_are_each_resolved_on_their_own_terms():
    positive = [(3, 20), (205, 210)]
    negative = [(0, 5), (15, 18), (200, 230)]

    result = generate_canonicalization_params(positive, negative, "full_break", True, False)

    assert result.segments == [(3, 20), (200, 230)]
    assert result.conflict_windows == [(0, 20)]
