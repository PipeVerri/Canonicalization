from collections import namedtuple
from pathlib import Path
from src.utils.canonicalization import (
    NTBenchmarkCanonicalizer,
    GenomicBenchmarkCanonicalizer,
    BendBenchmarkCanonicalizer,
    DeepseaCanonicalizer
)

def test_downstream_tasks_initialization():
    # Mock init of the canonicalizers
    MockConfig = namedtuple("MockConfig", ["input_dir", "output_dir"])
    data_dir = Path(__file__).parents[1] / "data"

    nt_benchmark_conf = MockConfig(data_dir / "nucleotide_transformer", data_dir / "nucleotide_transformer_parsed")
    genomic_benchmark_conf = MockConfig(data_dir / "genomic_benchmark", data_dir / "genomic_benchmark_parsed")
    bend_benchmark_conf = MockConfig(data_dir / "bend_benchmark", data_dir / "bend_benchmark_parsed")
    deepsea_conf = MockConfig(data_dir / "deepsea", data_dir / "deepsea_parsed")
    
    NTBenchmarkCanonicalizer({}, nt_benchmark_conf)
    GenomicBenchmarkCanonicalizer({}, genomic_benchmark_conf)
    BendBenchmarkCanonicalizer({}, bend_benchmark_conf)
    DeepseaCanonicalizer({}, deepsea_conf)