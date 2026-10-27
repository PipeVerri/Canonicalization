from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataloaders import genomics
from src.dataloaders.datasets.bend_benchmark_dataset import BendBenchmarkDataset
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.hg38_dataset import BertHG38Dataset, bert_mask
from src.models.ConvNova.convnova import CNNModel
from src.tasks.decoders import SequenceDecoder
from src.tasks.tasks import MaskedMultiClass
from src.utils.bend_benchmark_h5 import BEND_H5_FILENAME, write_bend_h5


def _tokenizer(padding_side):
    return CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=16,
        add_special_tokens=False,
        padding_side=padding_side,
    )


@pytest.mark.parametrize(
    "values, mask",
    [
        ([3, 4, 5, 99, 99], [1, 1, 1, 0, 0]),
        ([99, 99, 3, 4, 5], [0, 0, 1, 1, 1]),
        ([3, 99, 4, 99, 5], [1, 0, 1, 0, 1]),
    ],
)
def test_masked_pool_ignores_padding_on_either_side(values, mask):
    decoder = SequenceDecoder(d_model=1, l_output=0, mode="pool")
    x = torch.tensor([values], dtype=torch.float32).unsqueeze(-1)
    mask = torch.tensor([mask], dtype=torch.bool)

    output = decoder(x, mask=mask)

    assert output.shape == (1, 1)
    assert output.item() == pytest.approx(4.0)


def test_masked_multiclass_forwards_mask_to_model_and_decoder():
    task = object.__new__(MaskedMultiClass)
    mask = torch.tensor([[True, True, False]])
    seen = {}

    def encoder(x, mask=None):
        return x, {"mask": mask}

    def model(x, mask=None, state=None):
        seen["model_mask"] = mask
        return x, None

    def decoder(x, state=None, mask=None):
        seen["decoder_mask"] = mask
        return x, {"mask": mask}

    batch = (torch.ones(1, 3, 1), torch.zeros(1), {"mask": mask})
    task.forward(batch, encoder, model, decoder, None)

    assert seen["model_mask"] is mask
    assert seen["decoder_mask"] is mask


def test_bert_mask_uses_valid_mask_after_pad_id_is_lost():
    seq = torch.tensor([4, 4, 0, 1, 2])
    valid_mask = torch.tensor([False, False, True, True, True])

    _, mask, labels = bert_mask(
        seq.clone(),
        mask_token_id=4,
        pad_token_id=None,
        vocab_size=5,
        mask_prob=1.0,
        random_token_prob=0.0,
        unchanged_token_prob=0.0,
        special_token_ids=[4],
        valid_mask=valid_mask,
    )

    assert not mask[:2].any()
    assert mask[2:].all()
    assert (labels[:2] == -100).all()
    assert torch.equal(labels[2:], seq[2:])


def test_hg38_returns_real_token_mask_and_never_selects_padding_for_mlm():
    dataset = object.__new__(BertHG38Dataset)
    dataset.max_length = 5
    dataset.pad_max_length = 5
    dataset.tokenizer = _tokenizer("left")
    dataset.tokenizer_name = "char"
    dataset.add_eos = False
    dataset.return_augs = False
    dataset.use_tokenizer = False
    dataset.replace_N_token = False
    dataset.objective = "stdmlm"
    dataset.control_signals = None
    dataset.df = pd.DataFrame([["chr1", 0, 2, "train"]])
    dataset.fasta = lambda *args, **kwargs: "AC"

    masked, _, extra = dataset[0]

    expected_mask = torch.tensor([False, False, False, True, True])
    assert torch.equal(extra["mask"], expected_mask)
    assert not masked[1][~expected_mask].any()


def _bend_dataset(padding_side, tmp_path, with_control=False):
    h5_path = tmp_path / BEND_H5_FILENAME
    frame = pd.DataFrame({
        "sequence": ["AC"],
        "label": [np.array([1, 2])],
        "split": ["train"],
    })
    write_bend_h5(h5_path, {"gene_finding": frame}, mode="w")
    control_signals = {"train": [np.array([9, 8])]} if with_control else None
    return BendBenchmarkDataset(
        split="train",
        max_length=5,
        dataset_name="gene_finding",
        d_output=9,
        tokenizer=_tokenizer(padding_side),
        tokenizer_name="char",
        use_padding=True,
        add_eos=False,
        return_mask=True,
        use_tokenizer=False,
        h5_path=h5_path,
        control_signals=control_signals,
    )


@pytest.mark.parametrize(
    "padding_side, expected_ids, expected_labels, expected_mask",
    [
        ("right", [0, 1, 4, 4, 4], [1, 2, -100, -100, -100], [1, 1, 0, 0, 0]),
        ("left", [4, 4, 4, 0, 1], [-100, -100, -100, 1, 2], [0, 0, 0, 1, 1]),
    ],
)
def test_bend_labels_and_mask_follow_tokenizer_padding_side(
    padding_side, expected_ids, expected_labels, expected_mask, tmp_path
):
    seq_ids, labels, extra = _bend_dataset(padding_side, tmp_path)[0]

    assert torch.equal(seq_ids, torch.tensor(expected_ids))
    assert torch.equal(labels, torch.tensor(expected_labels))
    assert torch.equal(extra["mask"], torch.tensor(expected_mask, dtype=torch.bool))


def test_bend_right_padding_aligns_control_signal(tmp_path):
    seq_ids, _, extra = _bend_dataset("right", tmp_path, with_control=True)[0]

    assert torch.equal(seq_ids[0], torch.tensor([0, 1, 4, 4, 4]))
    assert torch.equal(seq_ids[1], torch.tensor([9, 8, 0, 0, 0]))
    assert torch.equal(extra["mask"], torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool))


@pytest.mark.parametrize(
    "datamodule_cls, dataset_cls_name",
    [
        (genomics.BendBenchmark, "BendBenchmarkDataset"),
        (genomics.NucleotideTransformer, "NucleotideTransformerDataset"),
    ],
)
def test_datamodules_pass_padding_side_to_character_tokenizer(
    monkeypatch, datamodule_cls, dataset_cls_name
):
    constructed = []

    class DatasetStub:
        def __init__(self, **kwargs):
            constructed.append(kwargs["tokenizer"])

    monkeypatch.setattr(genomics, dataset_cls_name, DatasetStub)
    datamodule = object.__new__(datamodule_cls)
    datamodule.tokenizer_name = "char"
    datamodule.padding_side = "right"
    datamodule.max_length = 5
    datamodule.max_length_val = 5
    datamodule.dataset_name = "gene_finding"
    datamodule.use_padding = True
    datamodule.d_output = 9
    datamodule.add_eos = False
    datamodule.rc_aug = False
    datamodule.return_mask = True
    datamodule.use_tokenizer = False
    datamodule.use_control_signal = False
    datamodule.control_signals = None
    datamodule.control_signals_dir = None
    datamodule.dest_path = None
    datamodule.h5_path = "unused.h5"
    datamodule.bed_path = None
    datamodule.labels_path = None
    datamodule.fasta_path = None

    datamodule.setup()

    assert constructed
    assert all(tokenizer.padding_side == "right" for tokenizer in constructed)


def test_convnova_real_outputs_are_invariant_to_added_padding():
    torch.manual_seed(0)
    model = CNNModel(
        SimpleNamespace(hidden_dim=4, num_cnn_stacks=1, dropout=0.0),
        alphabet_size=5,
        for_representation=True,
        pretrain=False,
        dilation=2,
        kernel_size=3,
        num_conv1d=2,
        d_inner=1,
        final_conv=False,
        ffn=True,
    ).eval()
    short = torch.tensor([[0, 1, 2]])
    padded = torch.tensor([[0, 1, 2, 4, 4]])
    short_mask = torch.ones(1, 3, dtype=torch.bool)
    padded_mask = torch.tensor([[True, True, True, False, False]])

    with torch.no_grad():
        short_output = model(short, mask=short_mask)[0]
        padded_output = model(padded, mask=padded_mask)[0]

    torch.testing.assert_close(padded_output[:, :3], short_output, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(padded_output[:, 3:]) == 0
