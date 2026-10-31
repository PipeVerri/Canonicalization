import os
import random

import pytest
import torch

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.hg38_dataset import BertHG38Dataset
from src.dataloaders.datasets.hg38_fixed_dataset import BertHG38FixedDataset
from src.dataloaders.genomics import BertHG38


class TrackingBertHG38Dataset(BertHG38Dataset):
    """Expose the process that last opened this dataset's FASTA handle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fasta_owner = os.getpid()

    def reopen_fasta(self):
        super().reopen_fasta()
        self.fasta_owner = os.getpid()

    def __getitem__(self, idx):
        masked, target, _ = super().__getitem__(idx)
        control = masked[0][1] if self.control_signals is not None else torch.zeros_like(target)
        return target, control, masked[1], os.getpid(), self.fasta_owner


def _tokenizer(length):
    return CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=length + 2,
        add_special_tokens=False,
    )


@pytest.mark.parametrize("with_control", [False, True])
def test_pretraining_workers_reopen_fasta_and_read_correct_intervals(tmp_path, with_control):
    length = 16
    rng = random.Random(42)
    sequence = "".join(rng.choice("ACGT") for _ in range(length * 48))
    fasta_path = tmp_path / "genome.fa"
    fasta_path.write_text(f">chr1\n{sequence}\n")
    bed_path = tmp_path / "intervals.bed"
    bed_path.write_text("".join(
        f"chr1\t{start}\t{start + length}\ttrain\n"
        for start in range(0, len(sequence), length)
    ))
    controls = {"chr1": torch.arange(1, len(sequence) + 1)} if with_control else None
    dataset = TrackingBertHG38Dataset(
        split="train",
        bed_file=bed_path,
        fasta_file=fasta_path,
        max_length=length,
        tokenizer=_tokenizer(length),
        tokenizer_name="char",
        use_tokenizer=False,
        control_signals=controls,
    )
    datamodule = object.__new__(BertHG38)
    datamodule.dataset_train = dataset
    datamodule.dataset_val = dataset
    datamodule.dataset_test = dataset
    datamodule.batch_size = 4
    datamodule.batch_size_eval = 4
    datamodule.shuffle = False
    datamodule.fault_tolerant = False
    datamodule.drop_last = False
    datamodule.pin_memory = False

    try:
        for num_workers in (0, 2):
            datamodule.num_workers = num_workers
            for loader in (datamodule.train_dataloader(), datamodule.val_dataloader(),
                           datamodule.test_dataloader()):
                targets, observed_controls, masks, pids, owners = zip(*list(loader))
                targets = torch.cat(targets)
                observed_controls = torch.cat(observed_controls)
                masks = torch.cat(masks)
                pids = torch.cat(pids)
                owners = torch.cat(owners)
                expected = torch.tensor(["ACGT".index(base) for base in sequence]).view(-1, length)
                assert torch.equal(targets, expected)
                assert torch.equal(pids, owners)
                if num_workers:
                    assert (pids != os.getpid()).all()
                    assert len(pids.unique()) == num_workers
                else:
                    assert (pids == os.getpid()).all()
                if with_control:
                    expected_controls = torch.arange(1, len(sequence) + 1).view(-1, length).float()
                    expected_controls[masks] = 0
                    assert torch.equal(observed_controls, expected_controls)

        # Closing inherited handles in workers must not close the parent's handle.
        assert str(dataset.fasta.seqs["chr1"][:length]) == sequence[:length]
    finally:
        dataset.fasta.seqs.close()


def test_fixed_pretraining_dataset_reopens_with_uppercase_preserved(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    fasta_path.write_text(">chr1\nacgttgca\n")
    dataset = BertHG38FixedDataset(
        fasta_file=fasta_path,
        chr_ranges={"chr1": (0, 8)},
        max_length=8,
        tokenizer=_tokenizer(8),
    )
    try:
        original = dataset.seqs
        dataset.reopen_fasta()
        assert dataset.seqs is not original
        assert str(dataset.seqs["chr1"][:]) == "ACGTTGCA"
    finally:
        dataset.seqs.close()
