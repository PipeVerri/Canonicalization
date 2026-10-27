import pytest
import pandas as pd
from types import SimpleNamespace
from Bio.Seq import Seq
from src.utils.canonicalization import (
    ParsedGenome,
    canonicalize_genome,
    CanonicalizationParams
)

# `parse_transcribed_features_from_gff` used to be a single free function that both
# filtered the annotations (by level/feature/pseudogene/cdjv) and split them into
# positive/negative segments. It's now split across two ParsedGenome instance methods:
# `_filter_gencode_transcribed_features` (filtering) and `_parse_transcribed_features_from_gff`
# (splitting). Neither needs a fully constructed ParsedGenome (which requires a genome index,
# a GFF file and an ontology on disk), so we bind them to a lightweight stand-in object instead.
def parse_transcribed_features_from_gff(df, ucsc_id, annotations_level, include_cdjv_segments, include_pseudogenes):
    fake_self = SimpleNamespace(
        canonicalization_config=SimpleNamespace(
            annotations_level=annotations_level,
            include_cdjv_segments=include_cdjv_segments,
            include_pseudogenes=include_pseudogenes
        )
    )
    filtered_df = ParsedGenome._filter_gencode_transcribed_features(fake_self, df)
    return ParsedGenome._parse_transcribed_features_from_gff(fake_self, filtered_df, ucsc_id)


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


def test_canonicalize_genome():
    # This is a basic test to check if the function runs without errors and produces expected results.
    training_genome = {"genome1": Seq("ACGTACGTAC" + "TTTTTTTTTT" + "ACGTACGTAC")}

    canonicalization = {
        "genome1":
            CanonicalizationParams(
                segments=[(0, 10), (20, 30)],
                conflict_windows=[],
                should_transform_at_end=False
            )
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
        "genome1": CanonicalizationParams(
            segments=[],
            conflict_windows=[],
            should_transform_at_end=True
        )
    }
    canon = canonicalize_genome(training_genome, canonicalization, "rc")
    #print(str(canon))
    #print(str(training_genome["genome1"].reverse_complemshould_rc_at_endent()))
    assert str(canon["genome1"]) == str(training_genome["genome1"].reverse_complement())
