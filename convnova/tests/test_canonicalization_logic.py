from src.utils.canonicalization import merge_segment_list, should_canonicalize_negative_strand, canonicalize_genome
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

def test_should_canonicalize_segment():
    correctly_oriented_segments = [(0, 10), (20, 30), (40, 50)]
    # Test with whole segment overlap check
    assert should_canonicalize_negative_strand([(5, 17)], correctly_oriented_segments) == True
    assert should_canonicalize_negative_strand([(5, 15)], correctly_oriented_segments) == False
    assert should_canonicalize_negative_strand([(5, 25)], correctly_oriented_segments) == False
    assert should_canonicalize_negative_strand([(5, 26)], correctly_oriented_segments) == True
    assert should_canonicalize_negative_strand([(0, 45)], correctly_oriented_segments) == True

def test_canonicalize_genome():
    # This is a basic test to check if the function runs without errors and produces expected results.
    training_genome = {"genome1": Seq("ACGTACGTAC" + "TTTTTTTTTT" + "ACGTACGTAC")}
    
    canonicalization = {
        "genome1": {
            "segments": [(0, 10), (20, 30)],
            "should_rc_at_end": False
        }
    }

    reverse = canonicalize_genome(
        training_genome, 
        canonicalization,
        "reverse"
    )["genome1"]
    complement = canonicalize_genome(
        training_genome, 
        canonicalization,
        "complement"
    )["genome1"]
    rc = canonicalize_genome(
        training_genome, 
        canonicalization,
        "rc"
    )["genome1"]

    assert str(reverse)    == "CATGCATGCA" + "TTTTTTTTTT" + "CATGCATGCA"
    assert str(complement) == "TGCATGCATG" + "TTTTTTTTTT" + "TGCATGCATG"
    assert str(rc)         == "GTACGTACGT" + "TTTTTTTTTT" + "GTACGTACGT"

    canonicalization = {
        "genome1": {
            "segments": [],
            "should_rc_at_end": True
        }
    }
    canon = canonicalize_genome(training_genome, canonicalization, "rc")
    #print(str(canon))
    #print(str(training_genome["genome1"].reverse_complement()))
    assert str(canon["genome1"]) == str(training_genome["genome1"].reverse_complement())