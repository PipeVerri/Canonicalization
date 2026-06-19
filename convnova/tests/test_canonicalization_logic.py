from src.utils.canonicalization import (
    merge_segment_list, 
    generate_to_canonicalize_and_should_rc_at_end, 
    canonicalize_genome,
    get_branch_segments_to_canonicalize_and_cost,
    calculate_overlap_length
)
from Bio.Seq import Seq
from intervaltree import IntervalTree

def test_merge_segment_list():
    segments_list = []
    merged = merge_segment_list(segments_list)
    assert merged == []

    segments_list = [(0, 10)]
    merged = merge_segment_list(segments_list)
    assert merged == [(0, 10)]

    segments_list = [(0, 10), (10, 20)]
    merged = merge_segment_list(segments_list)
    assert merged == [(0, 20)]

    # I know the segments are already sorted with bisect.insort, so there's no need to iteratively merge multiple segments.
    segments_list = [(0, 20), (10, 50), (25, 40)]
    merged = merge_segment_list(segments_list)
    assert merged == [(0, 50)]

def test_calculate_overlap_length():
    # Sin overlap
    assert calculate_overlap_length(0, 5, 5, 10) == 0   # adyacentes
    assert calculate_overlap_length(0, 3, 5, 10) == 0   # separados
    assert calculate_overlap_length(5, 10, 0, 3) == 0   # orden inverso, sin overlap

    # Overlap parcial
    assert calculate_overlap_length(0, 5, 3, 8) == 2    # [3, 5)
    assert calculate_overlap_length(3, 8, 0, 5) == 2    # simétrico

    # Uno contiene al otro
    assert calculate_overlap_length(0, 10, 2, 6) == 4   # [2, 6)
    assert calculate_overlap_length(2, 6, 0, 10) == 4   # interior es el primero

    # Idénticos
    assert calculate_overlap_length(3, 7, 3, 7) == 4

    # Overlap de 1
    assert calculate_overlap_length(0, 5, 4, 9) == 1    # [4, 5)


def test_get_branch_segments_to_canonicalize_and_cost():
    # Scenario 1: orient_all_segments = True, partial_canonicalization_works = True
    to_orient = [(10, 20)]
    correctly_oriented = IntervalTree.from_tuples([(15, 25)])
    to_canonicalize, cost = get_branch_segments_to_canonicalize_and_cost(
        to_orient, correctly_oriented, orient_all_segments=True, partial_canonicalization_works=True
    )
    # Overlap is [15, 20) -> length 5. Since orient_all_segments is True, we must canonicalize.
    assert to_canonicalize == [(10, 20)]
    assert cost == 5

    # Scenario 2: orient_all_segments = True, partial_canonicalization_works = False
    to_canonicalize, cost = get_branch_segments_to_canonicalize_and_cost(
        to_orient, correctly_oriented, orient_all_segments=True, partial_canonicalization_works=False
    )
    # Overlap interval is (15, 25) -> length 10. Since orient_all_segments is True, we must canonicalize.
    assert to_canonicalize == [(10, 20)]
    assert cost == 10

    # Scenario 3: orient_all_segments = False, partial_canonicalization_works = False
    # Case A: ignore_segment_cost (10) < orient_segment_cost (20) -> decide to ignore segment
    to_orient = [(15, 25)]  # length 10
    correctly_oriented = IntervalTree.from_tuples([(10, 30)])  # overlap interval is (10, 30) with length 20
    to_canonicalize, cost = get_branch_segments_to_canonicalize_and_cost(
        to_orient, correctly_oriented, orient_all_segments=False, partial_canonicalization_works=False
    )
    assert to_canonicalize == []
    assert cost == 10  # ignore_segment_cost is added to cost

    # Case B: ignore_segment_cost (10) >= orient_segment_cost (4) -> decide to orient segment
    to_orient = [(10, 20)]  # length 10
    correctly_oriented = IntervalTree.from_tuples([(18, 22)])  # overlap interval is (18, 22) with length 4
    to_canonicalize, cost = get_branch_segments_to_canonicalize_and_cost(
        to_orient, correctly_oriented, orient_all_segments=False, partial_canonicalization_works=False
    )
    assert to_canonicalize == [(10, 20)]
    assert cost == 4  # orient_segment_cost is added to cost

    # Scenario 4: Multiple segments
    to_orient = [(10, 20), (30, 40)]
    correctly_oriented = IntervalTree.from_tuples([(15, 25), (35, 38)])
    # With orient_all_segments = True, partial_canonicalization_works = True
    # Segment 1: (10, 20) overlaps (15, 25) -> overlap length 5
    # Segment 2: (30, 40) overlaps (35, 38) -> overlap length 3
    to_canonicalize, cost = get_branch_segments_to_canonicalize_and_cost(
        to_orient, correctly_oriented, orient_all_segments=True, partial_canonicalization_works=True
    )
    assert to_canonicalize == [(10, 20), (30, 40)]
    assert cost == 8


def test_generate_to_canonicalize_and_should_rc_at_end():
    # Case 1: keep_positive_cost < keep_negative_cost -> should_rc_at_end = False
    # positive_coding_segments = [(15, 25)]
    # negative_coding_segments = [(10, 30)]
    # We use orient_all_segments = True, partial_canonicalization_works = False to trigger asymmetry.
    # keep_positive_cost (orient negative [(10, 30)], correct positive [(15, 25)]):
    #   overlap: (15, 25) -> length 10. Cost = 10.
    # keep_negative_cost (orient positive [(15, 25)], correct negative [(10, 30)]):
    #   overlap: (10, 30) -> length 20. Cost = 20.
    # keep_positive_cost (10) < keep_negative_cost (20) -> should_rc_at_end = False, to_canonicalize = keep_positive_to_orient = [(10, 30)]
    pos_segments = [(15, 25)]
    neg_segments = [(10, 30)]
    to_canonicalize, should_rc = generate_to_canonicalize_and_should_rc_at_end(
        pos_segments, neg_segments, orient_all_segments=True, partial_canonicalization_works=False
    )
    assert to_canonicalize == [(10, 30)]
    assert should_rc is False

    # Case 2: keep_positive_cost >= keep_negative_cost -> should_rc_at_end = True
    # positive_coding_segments = [(10, 30)]
    # negative_coding_segments = [(15, 25)]
    # keep_positive_cost (orient negative [(15, 25)], correct positive [(10, 30)]):
    #   overlap: (10, 30) -> length 20. Cost = 20.
    # keep_negative_cost (orient positive [(10, 30)], correct negative [(15, 25)]):
    #   overlap: (15, 25) -> length 10. Cost = 10.
    # keep_positive_cost (20) >= keep_negative_cost (10) -> should_rc_at_end = True, to_canonicalize = keep_negative_to_orient = [(10, 30)]
    pos_segments = [(10, 30)]
    neg_segments = [(15, 25)]
    to_canonicalize, should_rc = generate_to_canonicalize_and_should_rc_at_end(
        pos_segments, neg_segments, orient_all_segments=True, partial_canonicalization_works=False
    )
    assert to_canonicalize == [(10, 30)]
    assert should_rc is True


