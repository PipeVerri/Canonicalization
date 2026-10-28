import pytest
import random
import mappy as mp
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock
from Bio.Seq import Seq
from intervaltree import IntervalTree
from src.utils.canonicalization import (
    CanonicalizedDownstreamTask,
    ParsedGenome,
    find_primary_hit,
    get_alignment_transcribed_segments,
    intervaltree_to_tuples
)

class MockHit:
    def __init__(self, is_primary, mapq, ctg, strand=1, r_st=0, r_en=100, q_st=0, q_en=None):
        self.is_primary = is_primary
        self.mapq = mapq
        self.ctg = ctg
        self.strand = strand
        self.r_st = r_st
        self.r_en = r_en
        self.q_st = q_st
        self.q_en = q_st + r_en - r_st if q_en is None else q_en

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

@pytest.mark.parametrize(
    "strand, start, end, expected",
    [
        pytest.param(1, 15, 25, [], id="plus-query-same-strand"),
        pytest.param(-1, 15, 25, [(10, 20)], id="minus-query-opposite-strand"),
        pytest.param(1, 25, 35, [(30, 40)], id="plus-query-opposite-strand"),
        pytest.param(-1, 25, 35, [], id="minus-query-same-strand"),
        pytest.param(1, 0, 100, [(30, 40)], id="plus-query-both-strands-present"),
        pytest.param(-1, 0, 100, [(10, 20), (50, 70)], id="minus-query-both-strands-present"),
        pytest.param(1, 100, 200, [], id="plus-query-no-overlap"),
        pytest.param(-1, 100, 200, [], id="minus-query-no-overlap"),
    ],
)
def test_get_alignment_transcribed_segments(strand, start, end, expected):
    # CANON-04: select opposite-strand coding annotations, because those are the
    # features requiring RC to appear in the query's positive transcriptional orientation.
    pos_tree = IntervalTree.from_tuples([(10, 20), (50, 70)])
    neg_tree = IntervalTree.from_tuples([(30, 40)])

    intervaltree_dict = {
        "chr1": {
            "positive": pos_tree,
            "negative": neg_tree
        }
    }
    fake_genome = SimpleNamespace(genome_transcribed_segments=intervaltree_dict)

    hit = MockHit(is_primary=True, mapq=60, ctg="chr1", strand=strand, r_st=start, r_en=end)
    assert sorted(get_alignment_transcribed_segments(hit, fake_genome)) == expected


@pytest.fixture
def downstream_canonicalizer():
    # Bypass filesystem setup and atexit handlers; exercise the real record processing.
    canonicalizer = object.__new__(CanonicalizedDownstreamTask)
    canonicalizer.config = SimpleNamespace(
        transformation="rc",
        partial_downstream_canonicalization=False,
    )
    return canonicalizer


@pytest.fixture
def strand_reference_genome():
    # Keep the actual coordinate-projection method without needing an index or GFF.
    genome = object.__new__(ParsedGenome)
    genome.genome_transcribed_segments = {
        "chr1": {"positive": IntervalTree(), "negative": IntervalTree()}
    }
    return genome


@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial"])
@pytest.mark.parametrize(
    "alignment_strand, transcript_strand, expected_sequence, transformed",
    [
        pytest.param(1, "negative", "ACGGTT", True, id="plus-query-negative-transcript"),
        pytest.param(-1, "positive", "ACGGTT", True, id="minus-query-positive-transcript"),
        pytest.param(1, "positive", "AACCGT", False, id="plus-query-positive-transcript"),
        pytest.param(-1, "negative", "AACCGT", False, id="minus-query-negative-transcript"),
    ],
)
def test_downstream_rc_orients_only_opposite_strand_transcripts(
    downstream_canonicalizer, strand_reference_genome, partial,
    alignment_strand, transcript_strand, expected_sequence, transformed,
):
    # Contract-level regression for issue 11.4: assert actual DNA and controls,
    # not just which annotation bucket the selector returns.
    sequence = Seq("AACCGT")  # Not an RC palindrome: its RC is ACGGTT.
    strand_reference_genome.genome_transcribed_segments["chr1"][transcript_strand].addi(100, 106)
    downstream_canonicalizer.config.partial_downstream_canonicalization = partial
    hit = MockHit(True, 60, "chr1", strand=alignment_strand, r_st=100, r_en=106)

    canonicalized, control, _ = downstream_canonicalizer._canonicalize_record(
        sequence, hit, strand_reference_genome,
    )

    assert str(canonicalized) == expected_sequence
    np.testing.assert_array_equal(control, np.full(len(sequence), int(transformed)))
    assert str(sequence) == "AACCGT"


@pytest.mark.parametrize(
    "alignment_strand, transcript_strand, expected_sequence, expected_control",
    [
        pytest.param(1, "negative", "GGTTTTGT", [1, 1, 1, 1, 1, 1, 0, 0], id="plus-query"),
        pytest.param(-1, "positive", "AAACGGTT", [0, 0, 1, 1, 1, 1, 1, 1], id="minus-query"),
    ],
)
def test_partial_downstream_rc_uses_opposite_strand_overlap_in_query_coordinates(
    downstream_canonicalizer, strand_reference_genome,
    alignment_strand, transcript_strand, expected_sequence, expected_control,
):
    sequence = Seq("AAAACCGT")
    strand_reference_genome.genome_transcribed_segments["chr1"][transcript_strand].addi(100, 106)
    downstream_canonicalizer.config.partial_downstream_canonicalization = True
    hit = MockHit(True, 60, "chr1", strand=alignment_strand, r_st=100, r_en=108)

    canonicalized, control, _ = downstream_canonicalizer._canonicalize_record(
        sequence, hit, strand_reference_genome,
    )

    assert str(canonicalized) == expected_sequence
    np.testing.assert_array_equal(control, expected_control)

@pytest.mark.parametrize(
    "strand, q_st, q_en, seq_length, segment, expected",
    [
        pytest.param(1, 0, 100, 100, (120, 150), (20, 50), id="plus-full-query"),
        pytest.param(1, 0, 100, 100, (80, 120), (0, 20), id="plus-clamp-start"),
        pytest.param(1, 0, 100, 100, (180, 220), (80, 100), id="plus-clamp-end"),
        pytest.param(-1, 0, 100, 100, (120, 150), (50, 80), id="minus-full-query"),
        pytest.param(-1, 0, 100, 100, (80, 120), (80, 100), id="minus-clamp-end"),
        pytest.param(-1, 0, 100, 100, (180, 220), (0, 20), id="minus-clamp-start"),
        # Reference [100, 200) aligns to query [20, 120), with 30 trailing bases.
        # Unequal leading/trailing clipping distinguishes q_en from seq_length.
        pytest.param(1, 20, 120, 150, (120, 150), (40, 70), id="plus-clipped-query"),
        pytest.param(-1, 20, 120, 150, (120, 150), (70, 100), id="minus-clipped-query"),
        pytest.param(1, 20, 120, 150, (100, 110), (20, 30), id="plus-reference-start"),
        pytest.param(-1, 20, 120, 150, (100, 110), (110, 120), id="minus-reference-start"),
        pytest.param(1, 20, 120, 150, (190, 200), (110, 120), id="plus-reference-end"),
        pytest.param(-1, 20, 120, 150, (190, 200), (20, 30), id="minus-reference-end"),
    ],
)
def test_align_transcript_position_with_seq(strand, q_st, q_en, seq_length, segment, expected):
    # All alignments are gap-free; coordinates are zero-based, end-exclusive.
    hit = MockHit(
        True, 60, "chr1", strand=strand, r_st=100, r_en=200, q_st=q_st, q_en=q_en,
    )
    assert ParsedGenome.align_transcript_position_with_seq(
        SimpleNamespace(), hit, segment, seq_length,
    ) == expected


@pytest.mark.parametrize("strand", [1, -1], ids=["plus", "minus"])
@pytest.mark.parametrize("trailing_bases", [0, 30], ids=["leading-clip", "both-ends-clipped"])
def test_align_transcript_position_with_real_clipped_alignment(strand, trailing_bases):
    # Exercise real mappy coordinates, not just the mock's interpretation of them.
    rng = random.Random(1729)
    reference = "".join(rng.choices("ACGT", k=3000))
    aligned_sequence = reference[1000:1100]
    if strand == -1:
        aligned_sequence = mp.revcomp(aligned_sequence)
    query = "N" * 20 + aligned_sequence + "N" * trailing_bases
    aligner = mp.Aligner(seq=reference)
    hit = find_primary_hit(aligner.map(query))

    assert hit is not None
    assert hit.strand == strand
    assert (hit.r_st, hit.r_en) == (1000, 1100)
    assert (hit.q_st, hit.q_en) == (20, 120)
    assert hit.cigar_str == "100M"

    # Reference [1002, 1012) occurs at these known positions in the constructed query.
    expected = (22, 32) if strand == 1 else (108, 118)
    annotated_sequence = reference[1002:1012]
    if strand == -1:
        annotated_sequence = mp.revcomp(annotated_sequence)
    assert query[expected[0]:expected[1]] == annotated_sequence
    assert ParsedGenome.align_transcript_position_with_seq(
        SimpleNamespace(), hit, (1002, 1012), len(query),
    ) == expected
