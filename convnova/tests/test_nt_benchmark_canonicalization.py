import pytest
from unittest.mock import MagicMock
from intervaltree import IntervalTree
from src.utils.canonicalization import (
    find_primary_alignment,
    find_annotated_alignment,
    get_transcribed_segments_that_contain_seq,
    align_transcript_position_with_seq,
    intervaltree_to_tuples
)

class MockHit:
    def __init__(self, is_primary, mapq, ctg, strand=1, r_st=0, r_en=100):
        self.is_primary = is_primary
        self.mapq = mapq
        self.ctg = ctg
        self.strand = strand
        self.r_st = r_st
        self.r_en = r_en

def test_find_primary_alignment():
    hit1 = MockHit(is_primary=False, mapq=60, ctg="chr1")
    hit2 = MockHit(is_primary=True, mapq=60, ctg="chr1")
    hit3 = MockHit(is_primary=False, mapq=60, ctg="chr2")
    
    hits = [hit1, hit2, hit3]
    assert find_primary_alignment(hits) == hit2
    
    hits_no_primary = [hit1, hit3]
    assert find_primary_alignment(hits_no_primary) is None

def test_find_annotated_alignment():
    intervaltree_dict = {"chr1": {}, "chr2": {}}
    
    hit1 = MockHit(is_primary=True, mapq=30, ctg="chr3") # Not in intervaltree
    hit2 = MockHit(is_primary=False, mapq=60, ctg="chr1") # In intervaltree, high mapq
    hit3 = MockHit(is_primary=False, mapq=20, ctg="chr2") # In intervaltree, low mapq
    
    hits = [hit1, hit2, hit3]
    # Should find hit2 because it has mapq >= 30 and its ctg is in intervaltree_dict
    assert find_annotated_alignment(hits, 30, intervaltree_dict) == hit2
    
    # Should return None if no hit matches criteria
    assert find_annotated_alignment([hit1, hit3], 30, intervaltree_dict) is None

def test_get_transcribed_segments_that_contain_seq():
    pos_tree = IntervalTree.from_tuples([(10, 20), (50, 70)])
    neg_tree = IntervalTree.from_tuples([(30, 40)])
    
    intervaltree_dict = {
        "chr1": {
            "positive": pos_tree,
            "negative": neg_tree
        }
    }
    
    # Positive strand alignment overlapping (10, 20)
    hit_pos = MockHit(is_primary=True, mapq=60, ctg="chr1", strand=1, r_st=15, r_en=25)
    segments = get_transcribed_segments_that_contain_seq(hit_pos, intervaltree_dict)
    assert segments == [(10, 20)]
    
    # Negative strand alignment overlapping (30, 40)
    hit_neg = MockHit(is_primary=True, mapq=60, ctg="chr1", strand=-1, r_st=25, r_en=35)
    segments = get_transcribed_segments_that_contain_seq(hit_neg, intervaltree_dict)
    assert segments == [(30, 40)]
    
    # Alignment with no overlap
    hit_no_overlap = MockHit(is_primary=True, mapq=60, ctg="chr1", strand=1, r_st=100, r_en=200)
    segments = get_transcribed_segments_that_contain_seq(hit_no_overlap, intervaltree_dict)
    assert segments == []

def test_align_transcript_position_with_seq():
    # Alignment starts at 100 on reference
    hit = MockHit(is_primary=True, mapq=60, ctg="chr1", r_st=100, r_en=200)
    
    # Segment (120, 150) relative to reference
    # Should be (20, 50) relative to sequence
    segment = (120, 150)
    start, end = align_transcript_position_with_seq(hit, segment, 100)
    assert (start, end) == (20, 50)
    
    # Segment (80, 120) - partially before alignment
    # Should be (0, 20)
    segment = (80, 120)
    start, end = align_transcript_position_with_seq(hit, segment, 100)
    assert (start, end) == (0, 20)
    
    # Segment (180, 220) - partially after alignment
    # Should be (80, 100)
    segment = (180, 220)
    start, end = align_transcript_position_with_seq(hit, segment, 100)
    assert (start, end) == (80, 100)
