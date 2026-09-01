from src.utils.canonicalization import generate_canonicalization_params
import pytest


@pytest.mark.parametrize("bad_behavior", ["full", "partial", "skip", "", "FULL_BREAK"])
def test_generate_canonicalization_params_rejects_unrecognized_behavior(bad_behavior):
    with pytest.raises(ValueError):
        generate_canonicalization_params([(0, 10)], [(5, 15)], bad_behavior, False, False)
