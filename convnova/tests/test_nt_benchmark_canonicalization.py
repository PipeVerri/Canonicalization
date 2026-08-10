import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from intervaltree import IntervalTree
from src.utils.canonicalization import (
    ParsedGenome,
    find_primary_hit,
    get_alignment_transcribed_segments,
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

def test_find_primary_hit():
    # `find_primary_alignment` was renamed to `find_primary_hit`; behavior is unchanged.
    hit1 = MockHit(is_primary=False, mapq=60, ctg="chr1")
    hit2 = MockHit(is_primary=True, mapq=60, ctg="chr1")
    hit3 = MockHit(is_primary=False, mapq=60, ctg="chr2")

    hits = [hit1, hit2, hit3]
    assert find_primary_hit(hits) == hit2

    hits_no_primary = [hit1, hit3]
    assert find_primary_hit(hits_no_primary) is None

def test_find_annotated_hit():
    # `find_annotated_alignment(hits, mapq, intervaltree_dict)` is now the private
    # `ParsedGenome._find_annotated_hit(self, hits, mapq)`, reading the annotation dict off
    # `self.genome_transcribed_segments` instead of taking it as an argument. The selection
    # logic itself (mapq >= threshold and ctg present) is unchanged, so we bind it to a
    # lightweight stand-in object rather than constructing a full ParsedGenome.
    intervaltree_dict = {"chr1": {}, "chr2": {}}
    fake_genome = SimpleNamespace(genome_transcribed_segments=intervaltree_dict)

    hit1 = MockHit(is_primary=True, mapq=30, ctg="chr3") # Not in intervaltree
    hit2 = MockHit(is_primary=False, mapq=60, ctg="chr1") # In intervaltree, high mapq
    hit3 = MockHit(is_primary=False, mapq=20, ctg="chr2") # In intervaltree, low mapq

    hits = [hit1, hit2, hit3]
    # Should find hit2 because it has mapq >= 30 and its ctg is in intervaltree_dict
    assert ParsedGenome._find_annotated_hit(fake_genome, hits, 30) == hit2

    # Should return None if no hit matches criteria
    assert ParsedGenome._find_annotated_hit(fake_genome, [hit1, hit3], 30) is None

def test_get_alignment_transcribed_segments():
    # `get_transcribed_segments_that_contain_seq(hit, intervaltree_dict)` is now
    # `get_alignment_transcribed_segments(alignment, reference_genome)`, taking a
    # ParsedGenome-like object exposing `genome_transcribed_segments` instead of the
    # raw dict. Overlap/strand logic is unchanged.
    pos_tree = IntervalTree.from_tuples([(10, 20), (50, 70)])
    neg_tree = IntervalTree.from_tuples([(30, 40)])

    intervaltree_dict = {
        "chr1": {
            "positive": pos_tree,
            "negative": neg_tree
        }
    }
    fake_genome = SimpleNamespace(genome_transcribed_segments=intervaltree_dict)

    # Positive strand alignment overlapping (10, 20)
    hit_pos = MockHit(is_primary=True, mapq=60, ctg="chr1", strand=1, r_st=15, r_en=25)
    segments = get_alignment_transcribed_segments(hit_pos, fake_genome)
    assert segments == [(10, 20)]

    # Negative strand alignment overlapping (30, 40)
    hit_neg = MockHit(is_primary=True, mapq=60, ctg="chr1", strand=-1, r_st=25, r_en=35)
    segments = get_alignment_transcribed_segments(hit_neg, fake_genome)
    assert segments == [(30, 40)]

    # Alignment with no overlap
    hit_no_overlap = MockHit(is_primary=True, mapq=60, ctg="chr1", strand=1, r_st=100, r_en=200)
    segments = get_alignment_transcribed_segments(hit_no_overlap, fake_genome)
    assert segments == []

def test_align_transcript_position_with_seq():
    # `align_transcript_position_with_seq(hit, segment, chromosome_length)` is now the
    # instance method `ParsedGenome.align_transcript_position_with_seq(self, alignment, segment,
    # seq_length)`. `seq_length` is the length of the local read/sequence being canonicalized
    # (the caller in downstream_tasks.py passes `len(seq)`), used to clamp segment_end into
    # the sequence's own coordinate space. Same cases as before, with the length passed explicitly.
    fake_genome = SimpleNamespace()

    # Alignment starts at 100 on reference
    hit = MockHit(is_primary=True, mapq=60, ctg="chr1", r_st=100, r_en=200)
    seq_length = 100

    # Segment (120, 150) relative to reference
    # Should be (20, 50) relative to sequence
    segment = (120, 150)
    start, end = ParsedGenome.align_transcript_position_with_seq(fake_genome, hit, segment, seq_length)
    assert (start, end) == (20, 50)

    # Segment (80, 120) - partially before alignment
    # Should be (0, 20)
    segment = (80, 120)
    start, end = ParsedGenome.align_transcript_position_with_seq(fake_genome, hit, segment, seq_length)
    assert (start, end) == (0, 20)

    # Segment (180, 220) - partially after alignment
    # Should be (80, 100)
    segment = (180, 220)
    start, end = ParsedGenome.align_transcript_position_with_seq(fake_genome, hit, segment, seq_length)
    assert (start, end) == (80, 100)
