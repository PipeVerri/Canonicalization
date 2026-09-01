from src.utils.canonicalization import generate_canonicalization_params

# full_break + orient_all_segments=False: for each conflict window, only flip the
# negative segments if that's cheaper than leaving the window untouched. Unlike the
# forced-order case, this decision is made per conflict window using the
# intersection-length cost (see calculate_segment_set_destruction_cost), and
# keep_segments_order has no effect here since it's only read when
# orient_all_segments is True.


def test_flips_negative_segments_when_cheaper_to_destroy_positive():
    positive = [(5, 20), (25, 28)]  # 15 + 3 = 18 nucleotides
    negative = [(0, 10), (18, 30)]  # 10 + 12 = 22 nucleotides

    result = generate_canonicalization_params(positive, negative, "full_break", False, False)

    assert sorted(result.segments) == [(0, 10), (18, 30)]
    assert result.conflict_windows == []


def test_ignores_negative_segments_when_cheaper_to_destroy_negative():
    positive = [(3, 20)]  # conflicting negative cost = 5 + 3 = 8
    negative = [(0, 5), (15, 18)]  # conflicting positive cost = 17

    result = generate_canonicalization_params(positive, negative, "full_break", False, False)

    assert result.segments == []
    assert result.conflict_windows == []


def test_tie_prefers_to_ignore_the_negative_segments():
    result = generate_canonicalization_params([(0, 10)], [(0, 10)], "full_break", False, False)

    assert result.segments == []


def test_multiple_independent_conflict_windows_are_resolved_separately():
    # first window: positive 17nt vs negative 8nt -> ignore negative
    # second window: positive 5nt vs negative 30nt -> flip negative
    positive = [(3, 20), (205, 210)]
    negative = [(0, 5), (15, 18), (200, 230)]

    result = generate_canonicalization_params(positive, negative, "full_break", False, False)

    assert result.segments == [(200, 230)]
    assert result.conflict_windows == []


def test_keep_segments_order_flag_is_ignored_when_not_orienting_all_segments():
    positive = [(5, 20), (25, 28)]
    negative = [(0, 10), (18, 30)]

    with_order = generate_canonicalization_params(positive, negative, "full_break", False, True)
    without_order = generate_canonicalization_params(positive, negative, "full_break", False, False)

    assert with_order == without_order
