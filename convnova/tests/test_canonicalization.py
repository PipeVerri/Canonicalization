from scripts.canonicalize_genomes import merge_segment_list, parse_record_into_segments_and_skew, should_canonicalize_segment, canonicalize_genome
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
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

def test_parse_record_into_segments_and_skew():
    training_genome = {"genome1": Seq("TTTT" * 25)}
    record = SeqRecord(Seq(""), id="record1")
    
    feature1 = SeqFeature(FeatureLocation(0, 10, strand=1))
    feature2 = SeqFeature(FeatureLocation(10, 20, strand=-1))
    feature3 = SeqFeature(FeatureLocation(20, 30, strand=1))
    
    record.features = [feature1, feature2, feature3]
    
    positive_segments, negative_segments, total_skew = parse_record_into_segments_and_skew(record, training_genome, "genome1")
    
    assert positive_segments == [(0, 10), (20, 30)]
    assert negative_segments == [(10, 20)]
    assert total_skew == 10 # 10 - 10 + 10

def test_should_canonicalize_segment():
    correctly_oriented_segments = IntervalTree.from_tuples([(0, 10), (20, 30), (40, 50)])
    # Test with whole segment overlap check
    assert should_canonicalize_segment(5, 17, correctly_oriented_segments, overlap_check_whole_segment=True) == True
    assert should_canonicalize_segment(5, 15, correctly_oriented_segments, overlap_check_whole_segment=True) == False
    assert should_canonicalize_segment(5, 25, correctly_oriented_segments, overlap_check_whole_segment=True) == False
    assert should_canonicalize_segment(5, 26, correctly_oriented_segments, overlap_check_whole_segment=True) == True
    assert should_canonicalize_segment(0, 45, correctly_oriented_segments, overlap_check_whole_segment=True) == True
    # Test with partial segment overlap check
    assert should_canonicalize_segment(5, 15, correctly_oriented_segments, overlap_check_whole_segment=False) == True

def test_canonicalize_genome():
    # This is a very basic test to check if the function runs without errors.
    training_genome = {"genome1": Seq("ACGTACGTAC" + "TTTTTTTTTT" + "ACGTACGTAC")}
    record = SeqRecord(Seq(""), id="record1")
    
    feature1 = SeqFeature(FeatureLocation(0, 10, strand=1))
    feature2 = SeqFeature(FeatureLocation(10, 20, strand=-1))
    feature3 = SeqFeature(FeatureLocation(20, 30, strand=1))
    
    record.features = [feature1, feature2, feature3]
    
    positive_segments, negative_segments, total_skew = parse_record_into_segments_and_skew(record, training_genome, "genome1")
    canonicalization = {
        "genome1": {
            "to_orient": positive_segments,
            "correctly_oriented": IntervalTree.from_tuples(negative_segments),
        }
    }

    reversed_record, complemented_record, rc_record = canonicalize_genome(training_genome, canonicalization)
    #print(str(reversed_record["genome1"]))

    assert str(reversed_record["genome1"]) == "CATGCATGCA" + "TTTTTTTTTT" + "CATGCATGCA"
    assert str(complemented_record["genome1"]) == "TGCATGCATG" + "TTTTTTTTTT" + "TGCATGCATG"
    assert str(rc_record["genome1"])           == "GTACGTACGT" + "TTTTTTTTTT" + "GTACGTACGT"