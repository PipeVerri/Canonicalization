from intervaltree import IntervalTree
from scripts.canonicalize_genomes import intervaltree_to_tuples

segments = [(0, 10), (20, 30), (40, 50), (60, 80)]
segment_index = IntervalTree.from_tuples(segments)

def test_find_overlaps():
    assert intervaltree_to_tuples(segment_index.overlap(5, 15)) == [(0, 10)]
    assert intervaltree_to_tuples(segment_index.overlap(25, 35)) == [(20, 30)]
    assert intervaltree_to_tuples(segment_index.overlap(45, 55)) == [(40, 50)]
    assert intervaltree_to_tuples(segment_index.overlap(15, 25)) == [(20, 30)]
    assert intervaltree_to_tuples(segment_index.overlap(35, 45)) == [(40, 50)]
    assert sorted(intervaltree_to_tuples(segment_index.overlap(0, 50))) == [(0, 10), (20, 30), (40, 50)]
    assert sorted(intervaltree_to_tuples(segment_index.overlap(-10, 70))) == segments