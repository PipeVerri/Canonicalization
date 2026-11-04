import numpy as np
import pandas as pd
import pytest
import torch
from intervaltree import IntervalTree
from types import MethodType, SimpleNamespace

from src.dataloaders.datasets.bend_benchmark_dataset import BendBenchmarkDataset
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.genomics import BendBenchmark
from src.utils.canonicalization import (
    BendBenchmarkCanonicalizer,
    DownstreamTask,
    ParsedGenome,
)
from src.utils.bend_benchmark_h5 import (
    BEND_H5_FILENAME,
    read_bend_split,
    read_bend_task,
    write_bend_h5,
)


def _tokenizer(padding_side="right"):
    return CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=16,
        add_special_tokens=False,
        padding_side=padding_side,
    )


def test_shared_h5_round_trip_preserves_long_labels_and_other_tasks(tmp_path):
    h5_path = tmp_path / BEND_H5_FILENAME
    long_label = np.arange(1200, dtype=np.int16) % 9
    gene_finding = pd.DataFrame({
        "sequence": ["A" * len(long_label), "C"],
        "label": [long_label, np.array([8])],
        "chromosome": ["chr1", "chr1"],
        "start": [0, 2000],
        "end": [1200, 2001],
        "split": ["train", "valid"],
    })
    cpg = pd.DataFrame({
        "sequence": ["AC"],
        "label": [[1, 0, 1, 0, 0, 0, 0]],
        "chromosome": ["chr2"],
        "start": [10],
        "end": [12],
        "split": ["test"],
    })

    write_bend_h5(h5_path, {"gene_finding": gene_finding, "cpg_methylation": cpg}, mode="w")

    gene_train = read_bend_split(h5_path, "gene_finding", "train")
    assert gene_train["sequence"][0] == "A" * len(long_label)
    assert np.array_equal(gene_train["label"][0], long_label)
    assert gene_train["label"][0].dtype == np.int16

    all_gene_rows = read_bend_task(h5_path, "gene_finding")
    assert list(all_gene_rows["split"]) == ["train", "valid"]
    assert np.array_equal(all_gene_rows.iloc[0]["label"], long_label)

    updated_cpg = cpg.copy()
    updated_cpg["sequence"] = ["TG"]
    write_bend_h5(h5_path, {"cpg_methylation": updated_cpg}, mode="a")

    assert read_bend_split(h5_path, "cpg_methylation", "test")["sequence"][0] == "TG"
    assert np.array_equal(
        read_bend_split(h5_path, "gene_finding", "train")["label"][0],
        long_label,
    )


def test_datamodule_loads_shared_h5_and_preserves_gene_labels(tmp_path):
    frame = pd.DataFrame({
        "sequence": ["AC", "G", "T"],
        "label": [np.array([1, 2]), np.array([3]), np.array([4])],
        "split": ["train", "valid", "test"],
    })
    write_bend_h5(tmp_path / BEND_H5_FILENAME, {"gene_finding": frame}, mode="w")
    datamodule = BendBenchmark(
        dataset_name="gene_finding",
        dataset_dir=tmp_path,
        control_signals_dir=tmp_path,
        max_length=5,
        d_output=9,
        use_padding=True,
        padding_side="right",
        return_mask=True,
        use_tokenizer=False,
        num_workers=0,
    )

    datamodule.setup()
    seq_ids, labels, extra = datamodule.dataset_train[0]

    assert torch.equal(seq_ids, torch.tensor([0, 1, 4, 4, 4]))
    assert torch.equal(labels, torch.tensor([1, 2, -100, -100, -100]))
    assert torch.equal(extra["mask"], torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool))
    assert len(datamodule.dataset_val) == 1
    assert len(datamodule.dataset_test) == 1


def test_cpg_dataset_reads_multilabel_vector_from_shared_h5(tmp_path):
    h5_path = tmp_path / BEND_H5_FILENAME
    frame = pd.DataFrame({
        "sequence": ["AC"],
        "label": [[1, 0, 1, 0, 0, 1, 0]],
        "split": ["train"],
    })
    write_bend_h5(h5_path, {"cpg_methylation": frame}, mode="w")
    dataset = BendBenchmarkDataset(
        split="train",
        max_length=2,
        dataset_name="cpg_methylation",
        d_output=7,
        tokenizer=_tokenizer(),
        tokenizer_name="char",
        use_padding=True,
        return_mask=True,
        use_tokenizer=False,
        h5_path=h5_path,
    )

    seq_ids, labels, extra = dataset[0]

    assert torch.equal(seq_ids, torch.tensor([0, 1]))
    assert torch.equal(labels, torch.tensor([1, 0, 1, 0, 0, 1, 0]))
    assert torch.equal(extra["mask"], torch.ones(2, dtype=torch.bool))


def test_gene_labels_follow_sequence_when_padding_is_disabled(tmp_path):
    h5_path = tmp_path / BEND_H5_FILENAME
    frame = pd.DataFrame({
        "sequence": ["AC"],
        "label": [np.array([1, 2])],
        "split": ["train"],
    })
    write_bend_h5(h5_path, {"gene_finding": frame}, mode="w")
    dataset = BendBenchmarkDataset(
        split="train",
        max_length=5,
        dataset_name="gene_finding",
        tokenizer=_tokenizer(),
        tokenizer_name="char",
        use_padding=False,
        return_mask=True,
        use_tokenizer=False,
        h5_path=h5_path,
    )

    seq_ids, labels, extra = dataset[0]

    assert torch.equal(seq_ids, torch.tensor([0, 1]))
    assert torch.equal(labels, torch.tensor([1, 2]))
    assert torch.equal(extra["mask"], torch.ones(2, dtype=torch.bool))


def test_gene_finding_rejects_sequence_label_length_mismatch(tmp_path):
    h5_path = tmp_path / BEND_H5_FILENAME
    frame = pd.DataFrame({
        "sequence": ["AC"],
        "label": [np.array([1])],
        "split": ["train"],
    })
    write_bend_h5(h5_path, {"gene_finding": frame}, mode="w")
    dataset = BendBenchmarkDataset(
        split="train",
        max_length=5,
        dataset_name="gene_finding",
        tokenizer=_tokenizer(),
        tokenizer_name="char",
        use_padding=True,
        use_tokenizer=False,
        h5_path=h5_path,
    )

    with pytest.raises(ValueError, match="sequence and label lengths differ"):
        dataset[0]


@pytest.mark.parametrize(
    "transcript_strand, expected_gene, expected_labels, expected_cpg, expected_control, expected_raw",
    [
        ("negative", "CGTT", [8, 6, 5, 4], "GT", 1, -1),
        ("positive", "AACG", [0, 1, 2, 8], "AC", 0, 1),
    ],
)
def test_canonicalization_updates_shared_h5_and_keeps_controls_separate(
    tmp_path, transcript_strand, expected_gene, expected_labels,
    expected_cpg, expected_control, expected_raw,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    gene_task = DownstreamTask("gene_finding", "hg38", max_length=4)
    cpg_task = DownstreamTask("cpg_methylation", "hg38", max_length=2)
    gene = pd.DataFrame({
        "sequence": ["AACGTT", "AC"],
        "label": [np.array([0, 1, 2, 8, 3, 4]), np.array([1, 2])],
        "chromosome": ["chr1", "chr1"],
        "start": [0, 10],
        "end": [6, 12],
        "split": ["train", "train"],
    })
    cpg = pd.DataFrame({
        "sequence": ["ACAA"],
        "label": [[1, 0, 1, 0, 0, 0, 0]],
        "chromosome": ["chr1"],
        "start": [0],
        "end": [4],
        "split": ["test"],
    })
    write_bend_h5(
        input_dir / BEND_H5_FILENAME,
        {"gene_finding": gene, "cpg_methylation": cpg},
        mode="w",
    )

    canonicalizer = object.__new__(BendBenchmarkCanonicalizer)
    canonicalizer.input_dir = input_dir
    canonicalizer.output_dir = output_dir
    canonicalizer.task_config = SimpleNamespace(skip_tasks=[])
    canonicalizer.config = SimpleNamespace(
        partial_downstream_canonicalization=False,
        transformation="rc",
        should_rc_at_end=False,
    )
    canonicalizer.has_downstream_task_been_processed = {gene_task: False, cpg_task: False}
    reference_genome = SimpleNamespace(
        genome_transcribed_segments={
            "chr1": {
                "positive": IntervalTree(),
                "negative": IntervalTree(),
            }
        }
    )
    # BEND sequences are extracted in plus-reference orientation. Only a negative
    # transcript requires RC, including the matching gene-label transformation.
    reference_genome.genome_transcribed_segments["chr1"][transcript_strand].addi(0, 4)
    reference_genome.align_transcript_position_with_seq = MethodType(
        ParsedGenome.align_transcript_position_with_seq,
        reference_genome,
    )

    canonicalizer.process_task(gene_task, reference_genome)
    canonicalizer.process_task(cpg_task, reference_genome)

    output_path = output_dir / BEND_H5_FILENAME
    canonical_gene = read_bend_split(output_path, "gene_finding", "train")
    assert canonical_gene["sequence"][0] == expected_gene
    assert np.array_equal(canonical_gene["label"][0], np.array(expected_labels))
    assert read_bend_split(output_path, "cpg_methylation", "test")["sequence"][0] == expected_cpg

    control = np.load(output_dir / "control_signals/gene_finding_train.npy", allow_pickle=True)
    raw_control = np.load(output_dir / "raw_control_signals/gene_finding_train.npy", allow_pickle=True)
    assert control.shape == (2,)
    assert np.array_equal(control[0], np.full(4, expected_control))
    assert np.array_equal(control[1], np.zeros(2))
    assert np.array_equal(raw_control[0], np.full(4, expected_raw))
    assert np.array_equal(raw_control[1], np.zeros(2))
