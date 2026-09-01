from src.utils.canonicalization import canonicalize_genome, CanonicalizationParams
from Bio.Seq import Seq


def test_applies_segments_and_conflict_windows():
    training_genome = {"chr1": Seq("AAAACCCCGGGGTTTT")}  # 4 of each
    canonicalization_params = {
        "chr1": CanonicalizationParams(segments=[(0, 4)], conflict_windows=[(4, 8)], should_rc_at_end=False)
    }

    result = canonicalize_genome(training_genome, canonicalization_params, "complement")

    assert str(result["chr1"]) == "TTTTGGGGGGGGTTTT"


def test_should_rc_at_end_applies_the_given_transformation_to_the_whole_sequence():
    # should_rc_at_end doesn't hardcode reverse-complement: it re-applies whatever
    # transformation was passed in, over the whole sequence, after the segments and
    # conflict windows have already been transformed.
    training_genome = {"chr1": Seq("AAAACCCCGGGGTTTT")}
    canonicalization_params = {
        "chr1": CanonicalizationParams(segments=[(0, 4)], conflict_windows=[], should_rc_at_end=True)
    }

    result = canonicalize_genome(training_genome, canonicalization_params, "complement")

    assert str(result["chr1"]) == "AAAAGGGGCCCCAAAA"


def test_conflict_windows_only():
    training_genome = {"chr1": Seq("AAAACCCC")}
    canonicalization_params = {
        "chr1": CanonicalizationParams(segments=[], conflict_windows=[(0, 8)], should_rc_at_end=False)
    }

    result = canonicalize_genome(training_genome, canonicalization_params, "rc")

    assert str(result["chr1"]) == "GGGGTTTT"


def test_each_genome_id_is_transformed_independently():
    training_genome = {"a": Seq("AAAA"), "b": Seq("TTTT")}
    canonicalization_params = {
        "a": CanonicalizationParams(segments=[(0, 4)], conflict_windows=[], should_rc_at_end=False),
        "b": CanonicalizationParams(segments=[], conflict_windows=[], should_rc_at_end=False),
    }

    result = canonicalize_genome(training_genome, canonicalization_params, "complement")

    assert str(result["a"]) == "TTTT"
    assert str(result["b"]) == "TTTT"
