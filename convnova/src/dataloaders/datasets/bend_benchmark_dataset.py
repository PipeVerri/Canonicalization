from collections import namedtuple
from tkinter import NO
import token
from pyfaidx import Fasta
import torch
import torch.nn as nn
# from random import random
import random
from pathlib import Path
import numpy as np
import h5py
import pandas as pd
from src.dataloaders.utils.control_signal import trim_and_pad_to_torch

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.utils.bend_benchmark_h5 import read_bend_split
import src.tasks.decoders as _decoders

_BendOutput = namedtuple("BendOutput", ["logits"])


# The LM task expects a NamedTUple with x.logits, which is added using this wrapper
class BendLinearDecoder(nn.Module):
    """Linear decoder that wraps its output in a namedtuple so LMTask's
    ``x.logits`` attribute access works for fine-tuning (non-pretrain) models."""

    def __init__(self, d_model, d_output):
        super().__init__()
        self.linear = nn.Linear(d_model, d_output)

    def forward(self, x, **kwargs):
        return _BendOutput(logits=self.linear(x))


_decoders.registry["bend_linear"] = BendLinearDecoder
_decoders.dataset_attrs["bend_linear"] = ["d_output"]
_decoders.model_attrs["bend_linear"] = ["d_output"]

class BendBenchmarkDataset(torch.utils.data.Dataset):

    '''
    Loop thru fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name=None,
        d_output=2,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        use_tokenizer=True,
        h5_path: Path | None = None,
        control_signals=None,
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask
        self.use_tokenizer = use_tokenizer
        self.dataset_name = dataset_name
        self.control_signals = control_signals

        if split == "val":
            split = "valid"
        self.split = split

        records = read_bend_split(h5_path, dataset_name, split)
        self.sequences = records["sequence"]
        self.labels = records["label"]
        if len(self.sequences) != len(self.labels):
            raise ValueError("BEND sequences and labels must have equal lengths")

        if add_eos:
            raise RuntimeError("EOS not implemented")
        if tokenizer_name != "char":
            raise RuntimeError(f"{tokenizer_name} tokenizer not implemented")
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else 'do_not_pad',
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"] # get input_ids

        assert type(seq_ids)==list
        seq_ids = torch.LongTensor(seq_ids)
        if not self.use_tokenizer:
            seq_ids = seq_ids-7 # Each character id starts at 7(CharacterTokenizer). The rest are special tokens
            mask = (seq_ids >= 4) | (seq_ids < 0)
            seq_ids[mask] = 4

        if self.control_signals is not None:
            control_signal = self.control_signals[self.split][idx]
            control_signal = trim_and_pad_to_torch(control_signal, self.max_length, self.tokenizer, 0, use_padding=self.use_padding)
            seq_ids = torch.stack([seq_ids, control_signal], dim=0)

        # Get the target
        if self.dataset_name not in ("cpg_methylation", "gene_finding"):
            raise RuntimeError("Dataset not supported")
        y = self.labels[idx]
        if self.dataset_name == "gene_finding":
            if len(sequence) != len(y):
                raise ValueError(
                    f"Gene-finding sequence and label lengths differ at index {idx}: "
                    f"{len(sequence)} != {len(y)}"
                )

            target = trim_and_pad_to_torch(y, self.max_length, self.tokenizer, -100, use_padding=self.use_padding)
        else:
            target = torch.LongTensor(y)

        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        return seq_ids, target

# Old variant effects logic
# elif self.dataset_name == "variant_effects_disease" or self.dataset_name == "variant_effects_expression":
# # Start = end, indicates the position of the SNP
# seq_start = df_seq["start"] - (self.max_length // 2)
# seq_end = df_seq["start"] + (self.max_length - (self.max_length // 2))
# seq = self.seqs[df_seq["chromosome"]][seq_start:seq_end].seq

# if seq_start < 0 or len(seq) != (seq_end - seq_start): # The len diff is used to check that seq_end hasn't gone overboard
#     raise ValueError("SNP seq escapes chromosome bounds")

# seq_snp = seq[:] # Copy the string
# seq_snp[len(seq_snp) // 2] = df_seq["alt"] # Apply the snp, the centered nucleotide will always be on the lower bound because of the way that seq is built

# return seq, seq_snp, {"label": df_seq["label"]}
# else:
# raise RuntimeError(f"{self.dataset_name} not implemented in bend benchmark")
