import pytest
from src.utils.canonicalization import parse_record_into_segments_and_skew, is_feature_type_transcribed, load_ontology
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from pathlib import Path
import os

@pytest.fixture(scope="module")
def ontology():
    # Setup PROJECT_ROOT if not present
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].absolute())
    
    ontology_path = Path(os.environ["PROJECT_ROOT"]) / "so.clean.obo"
    return load_ontology(str(ontology_path))

def test_is_feature_type_transcribed(ontology):
    # Basic tests for transcription detection using pronto/Sequence Ontology
    assert is_feature_type_transcribed("mRNA", ontology, orient_cdjv_segments=False) == True
    assert is_feature_type_transcribed("transcript", ontology, orient_cdjv_segments=False) == True
    assert is_feature_type_transcribed("primary_transcript", ontology, orient_cdjv_segments=False) == True
    assert is_feature_type_transcribed("gene", ontology, orient_cdjv_segments=False) == False
    assert is_feature_type_transcribed("exon", ontology, orient_cdjv_segments=False) == False
    assert is_feature_type_transcribed("lnc_RNA", ontology, orient_cdjv_segments=False) == True
    
    # CDJV segments logic
    assert is_feature_type_transcribed("V_gene_segment", ontology, orient_cdjv_segments=True) == True
    assert is_feature_type_transcribed("V_gene_segment", ontology, orient_cdjv_segments=False) == False
    
    # Pseudogene logic
    assert is_feature_type_transcribed("pseudogene", ontology, orient_whole_pseudogene=True) == True
    assert is_feature_type_transcribed("pseudogene", ontology, orient_whole_pseudogene=False) == False

def test_parse_record_into_segments_and_skew_sequential_no_overlap(ontology):
    training_genome = {"genome1": Seq("A" * 100)}
    record = SeqRecord(Seq(""), id="record1")
    
    # Create overlapping and contiguous features
    f1 = SeqFeature(FeatureLocation(0, 10, strand=1), type="mRNA")
    f2 = SeqFeature(FeatureLocation(5, 15, strand=1), type="mRNA")
    f3 = SeqFeature(FeatureLocation(15, 20, strand=1), type="mRNA")
    f4 = SeqFeature(FeatureLocation(30, 40, strand=1), type="mRNA")
    
    f5 = SeqFeature(FeatureLocation(50, 60, strand=-1), type="mRNA")
    f6 = SeqFeature(FeatureLocation(55, 65, strand=-1), type="mRNA")
    
    features = [f1, f2, f3, f4, f5, f6]
    for f in features:
        f.sub_features = []
    
    record.features = features
    
    # Pass explicit defaults to be independent of file config
    positive_segments, negative_segments, total_skew = parse_record_into_segments_and_skew(
        record, training_genome, "genome1", ontology,
        orient_pseudogene_exons=False,
        raise_error_if_encounter=(),
        orient_whole_pseudogene=False,
        orient_cdjv_segments=True
    )
    
    # Expected positive: (0, 20), (30, 40)
    assert positive_segments == [(0, 20), (30, 40)]
    # Expected negative: (50, 65)
    assert negative_segments == [(50, 65)]
    
    # Verify sequential and no overlap (internal consistency of each list)
    def check_segments_properties(segments, label):
        for i in range(len(segments)):
            start, end = segments[i]
            assert start < end, f"{label} segment {i} ({start}, {end}) has invalid range"
            if i > 0:
                prev_start, prev_end = segments[i-1]
                assert prev_end < start, f"{label} segments overlap or are not merged: {segments[i-1]} and {segments[i]}"
            
    check_segments_properties(positive_segments, "Positive")
    check_segments_properties(negative_segments, "Negative")

def test_parse_record_into_segments_and_skew(ontology):
    training_genome = {"genome1": Seq("GGGGGGGGGG" + "CCCCCCCCCC" + "GGGGGGGGGG")}
    record = SeqRecord(Seq(""), id="record1")
    
    feature1 = SeqFeature(FeatureLocation(0, 10, strand=1), type="mRNA")
    feature2 = SeqFeature(FeatureLocation(10, 20, strand=-1), type="mRNA")
    feature3 = SeqFeature(FeatureLocation(20, 30, strand=1), type="mRNA")
    feature4 = SeqFeature(FeatureLocation(30, 40, strand=1), type="gene")
    
    record.features = [feature1, feature2, feature3, feature4]
    for f in record.features:
        f.sub_features = []
    
    positive_segments, negative_segments, total_skew = parse_record_into_segments_and_skew(
        record, training_genome, "genome1", ontology,
        orient_cdjv_segments=False,
        
    )
    
    assert positive_segments == [(0, 10), (20, 30)]
    assert negative_segments == [(10, 20)]
    assert total_skew == 30 

def test_parse_record_into_segments_and_skew_with_subfeatures(ontology):
    training_genome = {"genome1": Seq("GGGGGGGGGG" * 3)}
    record = SeqRecord(Seq(""), id="record1")
    
    sub_feature = SeqFeature(FeatureLocation(0, 10, strand=1), type="mRNA")
    parent_feature = SeqFeature(FeatureLocation(0, 10, strand=1), type="gene")
    parent_feature.sub_features = [sub_feature]
    
    record.features = [parent_feature]
    
    positive_segments, negative_segments, total_skew = parse_record_into_segments_and_skew(
        record, training_genome, "genome1", ontology,
        orient_cdjv_segments=False
    )
    
    assert positive_segments == [(0, 10)]
    assert total_skew == 10
