from scripts.canonicalize_genomes import parse_record_into_segments_and_skew, is_feature_type_transcribed
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation

def test_is_feature_type_transcribed():
    # Basic tests for transcription detection using pronto/Sequence Ontology
    assert is_feature_type_transcribed("mRNA") == True
    assert is_feature_type_transcribed("transcript") == True
    assert is_feature_type_transcribed("primary_transcript") == True
    assert is_feature_type_transcribed("gene") == False # Even though a part of a gene is usually transcribed, the whole gene isn't
    assert is_feature_type_transcribed("exon") == False # Exons are parts, usually not the main transcribed feature we track here
    assert is_feature_type_transcribed("lnc_RNA") == True # lnc_RNA doesn't show up, but lncRNA does

def test_parse_record_into_segments_and_skew():
    # Use Gs and Cs to create skew
    # + strand: 10 Gs -> skew = 10
    # - strand: 10 Gs (which are Cs on + strand) -> skew = -(-10) = 10? No.
    # Script logic:
    # if strand == 1: total_skew += G + T - C - A
    # if strand == -1: total_skew += C + A - G - T
    
    # Feature 1 (+): 10 Gs -> skew += 10
    # Feature 2 (-): 10 Gs on - strand (Cs on + strand). 
    #   feature_seq = training_genome[location.start:location.end] = "CCCCCCCCCC"
    #   total_skew += C(10) + A(0) - G(0) - T(0) = 10
    # Feature 3 (+): 10 Gs -> skew += 10
    
    training_genome = {"genome1": Seq("GGGGGGGGGG" + "CCCCCCCCCC" + "GGGGGGGGGG")}
    record = SeqRecord(Seq(""), id="record1")
    
    feature1 = SeqFeature(FeatureLocation(0, 10, strand=1), type="mRNA")
    feature2 = SeqFeature(FeatureLocation(10, 20, strand=-1), type="mRNA")
    feature3 = SeqFeature(FeatureLocation(20, 30, strand=1), type="mRNA")
    # Non-transcribed feature should be ignored
    feature4 = SeqFeature(FeatureLocation(30, 40, strand=1), type="gene")
    
    record.features = [feature1, feature2, feature3, feature4]
    for f in record.features:
        f.sub_features = []
    
    positive_segments, negative_segments, total_skew = parse_record_into_segments_and_skew(record, training_genome, "genome1")
    
    assert positive_segments == [(0, 10), (20, 30)]
    assert negative_segments == [(10, 20)]
    assert total_skew == 30 # 10 + 10 + 10

def test_parse_record_into_segments_and_skew_with_subfeatures():
    # Test recursive exploration
    training_genome = {"genome1": Seq("GGGGGGGGGG" * 3)}
    record = SeqRecord(Seq(""), id="record1")
    
    # Top level is gene (not transcribed), but has mRNA subfeature (transcribed)
    sub_feature = SeqFeature(FeatureLocation(0, 10, strand=1), type="mRNA")
    parent_feature = SeqFeature(FeatureLocation(0, 10, strand=1), type="gene")
    parent_feature.sub_features = [sub_feature]
    print(parent_feature.sub_features)
    
    record.features = [parent_feature]
    
    positive_segments, negative_segments, total_skew = parse_record_into_segments_and_skew(record, training_genome, "genome1")
    
    assert positive_segments == [(0, 10)]
    assert total_skew == 10
