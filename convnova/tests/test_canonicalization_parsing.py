import pytest
import pandas as pd
from Bio.Seq import Seq, MutableSeq
from src.utils.canonicalization import (
    merge_segment_list,
    parse_transcribed_features_from_gff,
)

def test_merge_segment_list():
    # Empty list
    assert merge_segment_list([]) == []
    
    # Non-overlapping segments (already sorted)
    segments = [(0, 10), (20, 30), (40, 50)]
    assert merge_segment_list(segments) == segments
    
    # Overlapping segments
    segments = [(0, 10), (5, 15), (20, 30)]
    assert merge_segment_list(segments) == [(0, 15), (20, 30)]
    
    # Contiguous segments
    segments = [(0, 10), (10, 20)]
    assert merge_segment_list(segments) == [(0, 20)]
    
    # Nested segments
    segments = [(0, 100), (10, 20), (30, 40)]
    assert merge_segment_list(segments) == [(0, 100)]

def test_parse_transcribed_features_from_gff():
    data = {
        "Chromosome": ["chr1", "chr1", "chr1", "chr1", "chr1", "chr2"],
        "Start": [0, 20, 40, 60, 80, 0],
        "End": [10, 30, 50, 70, 90, 10],
        "Strand": ["+", "-", "+", "-", "-", "+"],
        "Feature": ["transcript", "transcript", "transcript", "gene", "transcript", "transcript"],
        "level": [1, 2, 3, 1, 1, 1],
        "transcript_type": ["mRNA", "mRNA", "pseudogene", "mRNA", "mRNA", "mRNA"]
    }
    df = pd.DataFrame(data)
    
    # Test basic filtering (chr1, level 3, all included)
    # Pos: 0-10 (mRNA, lvl 1), 40-50 (pseudogene, lvl 3)
    # Neg: 20-30 (mRNA, lvl 2), 80-90 (mRNA, lvl 1)
    # 60-70 is 'gene' -> excluded
    pos, neg = parse_transcribed_features_from_gff(
        df, "chr1", annotations_level=3, include_cdjv_segments=True, include_pseudogenes=True
    )
    assert pos == [(0, 10), (40, 50)]
    assert neg == [(20, 30), (80, 90)]
    
    # Test level filtering (level 2)
    # 40-50 is level 3 -> excluded
    pos, neg = parse_transcribed_features_from_gff(
        df, "chr1", annotations_level=2, include_cdjv_segments=True, include_pseudogenes=True
    )
    assert pos == [(0, 10)]
    assert neg == [(20, 30), (80, 90)]
    
    # Test pseudogene exclusion
    # 40-50 is pseudogene -> excluded
    pos, neg = parse_transcribed_features_from_gff(
        df, "chr1", annotations_level=3, include_cdjv_segments=True, include_pseudogenes=False
    )
    assert pos == [(0, 10)]
    assert neg == [(20, 30), (80, 90)]

    # Test CDJV exclusion
    data_cdjv = data.copy()
    data_cdjv["transcript_type"] = ["mRNA", "IG_V_gene", "mRNA", "mRNA", "mRNA", "mRNA"]
    df_cdjv = pd.DataFrame(data_cdjv)
    # 20-30 is IG_V_gene -> excluded
    pos, neg = parse_transcribed_features_from_gff(
        df_cdjv, "chr1", annotations_level=3, include_cdjv_segments=False, include_pseudogenes=True
    )
    assert neg == [(80, 90)]