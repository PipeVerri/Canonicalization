from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from Bio.Seq import Seq
from hydra import compose, initialize_config_dir
from intervaltree import IntervalTree

from src.utils.canonicalization import downstream_tasks as downstream
from src.utils.canonicalization.parsed_genome import AlignmentError, ParsedGenome


@pytest.fixture
def canonicalization_config(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        return compose(config_name="canonicalization/default").canonicalization


@pytest.mark.parametrize(
    "canonicalizer_class, family, task_name, max_length, count",
    [
        (downstream.NTBenchmarkCanonicalizer, "nt_benchmark", "enhancers", 400, 18),
        (downstream.GenomicBenchmarkCanonicalizer, "genomic_benchmark", "human_ensembl_regulatory", 802, 10),
        (downstream.BendBenchmarkCanonicalizer, "bend_benchmark", "gene_finding", 16384, 4),
        (downstream.DeepseaCanonicalizer, "deepsea", "deepsea", 1000, 1),
    ],
)
def test_downstream_tasks_initialization(
    tmp_path, monkeypatch, canonicalization_config,
    canonicalizer_class, family, task_name, max_length, count,
):
    task_config = canonicalization_config[family]
    task_config.input_dir = str(tmp_path / "input")
    task_config.output_dir = str(tmp_path / "output")
    Path(task_config.input_dir).mkdir()
    monkeypatch.setattr(downstream.atexit, "register", lambda *args: None)

    canonicalizer = canonicalizer_class(canonicalization_config, task_config)

    assert len(canonicalizer.tasks_list) == count
    assert len(list(canonicalizer.tasks_list)) == count  # Must remain reusable.
    assert next(t for t in canonicalizer.tasks_list if t.task_name == task_name).max_length == max_length
    assert len(canonicalizer.has_downstream_task_been_processed) == count
    assert task_name in canonicalizer.unaligned_tasks_counter


@pytest.fixture
def canonicalizer():
    canonicalizer = object.__new__(downstream.NTBenchmarkCanonicalizer)
    canonicalizer.config = SimpleNamespace(transformation="rc", partial_downstream_canonicalization=False)
    canonicalizer.task_config = SimpleNamespace(skip_tasks=[])
    canonicalizer.unaligned_tasks_counter = {
        "enhancers": {"unaligned": 0, "aligned_without_annotations": 0, "total_tasks": 0}
    }
    return canonicalizer


def reference_genome(strand=1, annotated=True):
    genome = object.__new__(ParsedGenome)
    genome.genome_transcribed_segments = {
        "chr1": {"positive": IntervalTree(), "negative": IntervalTree()}
    }
    if annotated:
        genome.genome_transcribed_segments["chr1"]["negative" if strand == 1 else "positive"].addi(100, 104)
    hit = SimpleNamespace(ctg="chr1", strand=strand, r_st=100, r_en=104, q_st=0, q_en=4)
    genome.find_annotated_alignment = Mock(return_value=(hit, None))
    return genome


@pytest.mark.parametrize("strand", [1, -1])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("transformation, expected", [("rc", "CGTT"), ("reverse", "GCAA"), ("complement", "TTGC")])
def test_crops_before_alignment_and_transformation(canonicalizer, strand, partial, transformation, expected):
    canonicalizer.config.partial_downstream_canonicalization = partial
    canonicalizer.config.transformation = transformation
    task = downstream.DownstreamTask("enhancers", "hg38", max_length=4)
    genome = reference_genome(strand)

    sequence, control, raw, _ = canonicalizer.align_and_canonicalize_record(Seq("AACGTT"), task, genome)

    genome.find_annotated_alignment.assert_called_once_with("AACG")
    assert str(sequence) == expected
    np.testing.assert_array_equal(control, np.ones(4))
    np.testing.assert_array_equal(raw, -np.ones(4))
    assert canonicalizer.unaligned_tasks_counter["enhancers"]["total_tasks"] == 1


@pytest.mark.parametrize("mode", ["unaligned", "unannotated", "skipped"])
def test_unchanged_records_and_controls_keep_same_prefix(canonicalizer, mode):
    task = downstream.DownstreamTask("enhancers", "hg38", max_length=4)
    genome = reference_genome(annotated=mode != "unannotated")
    if mode == "unaligned":
        genome.find_annotated_alignment.return_value = (None, AlignmentError.PrimaryAlignmentNotFound)
    if mode == "skipped":
        canonicalizer.task_config.skip_tasks = ["enhancers"]

    record_id, sequence, control, raw = canonicalizer._process_record(task, "example", Seq("AACGTT"), genome)

    assert record_id == "example"
    assert str(sequence) == "AACG"
    np.testing.assert_array_equal(control, np.zeros(4))
    assert len(raw) == 4


def test_annotations_outside_retained_prefix_do_not_affect_orientation(canonicalizer):
    genome = reference_genome(annotated=False)
    genome.genome_transcribed_segments["chr1"]["negative"].addi(104, 110)
    # BEND's known-coordinate path can still be given the original full alignment.
    hit = SimpleNamespace(ctg="chr1", strand=1, r_st=100, r_en=110, q_st=0, q_en=10)

    sequence, control, raw = canonicalizer._canonicalize_record(Seq("AACGTTTTTT"), hit, genome, 4)

    assert str(sequence) == "AACG"
    np.testing.assert_array_equal(control, np.zeros(4))
    np.testing.assert_array_equal(raw, np.zeros(4))


@pytest.mark.parametrize("skip", [False, True])
def test_genomic_worker_passes_task_and_crops_skipped_records(tmp_path, canonicalizer, skip):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "sample.txt").write_text("AACGTT")
    task = downstream.DownstreamTask("enhancers", "hg38", max_length=4)
    cls = downstream.GenomicBenchmarkCanonicalizer
    cls._init_worker(task, canonicalizer, reference_genome(), input_dir, output_dir)
    try:
        control, raw, status = cls._process_single_file(("sample.txt", skip))
    finally:
        cls._worker_state_dict.clear()

    assert (output_dir / "sample.txt").read_text() == ("AACG" if skip else "CGTT")
    assert len(control) == len(raw) == 4
    assert status is None


def test_deepsea_worker_uses_task_max_length(canonicalizer):
    cls = downstream.DeepseaCanonicalizer
    cls._worker_state_dict.update({
        "task": downstream.DownstreamTask("enhancers", "hg38", max_length=4),
        "canonicalizer": canonicalizer,
        "reference_genome": reference_genome(),
        "x": np.array([[0, 0, 1, 2, 3, 3]]),  # AACGTT
    })
    try:
        sequence, control, raw, status = cls._canonicalize_sequence(0)
    finally:
        cls._worker_state_dict.clear()

    np.testing.assert_array_equal(sequence, [1, 2, 3, 3])  # CGTT
    assert len(sequence) == len(control) == len(raw) == 4
    assert status is None
