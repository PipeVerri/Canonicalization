from collections import namedtuple
from pyfaidx import Fasta
import torch
import torch.nn as nn
# from random import random
import random
from pathlib import Path
import numpy as np
import h5py
import pandas as pd
from src.dataloaders.utils.control_signal import trim_and_right_pad_to_torch

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
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
        bed_path=None,
        labels_path=None,
        fasta_path=None,
        control_signals=None
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
        self.labels_h5 = None
        self.control_signals = control_signals

        if split == "val":
            split = "valid"

        # use Path object
        bed_path = Path(bed_path)
        fasta_path = Path(fasta_path)
        assert bed_path.exists()
        assert fasta_path.exists()

        self.seqs = Fasta(str(fasta_path))
        
        if dataset_name == "gene_finding":
            bed_columns = ("chromosome", "start", "end", "transcript_id", "strand", "flank_left", "flank_right", "length", "split")
            to_convert_to_int = ("start", "end", "length")
        elif dataset_name == "cpg_methylation":
            bed_columns = ("chromosome", "start", "end", "strand", "label", "split")
            to_convert_to_int = ("start", "end")
        elif dataset_name == "variant_effects_disease":
            bed_columns = ("chromosome", "start", "end", "ref", "alt", "split", "label", "full_label", "Consequence")
            to_convert_to_int = ("start", "end", "label")
        elif dataset_name == "variant_effects_expression":
            bed_columns = ("chromosome", "start", "end", "ref", "alt", "split", "label", "Consequence", "actual")
            to_convert_to_int = ("start", "end", "label")
        else:
            raise RuntimeError(f"Unrecognized dataset {dataset_name}")
        
        df = pd.read_csv(str(bed_path), sep="\t", header=None, names=bed_columns)
        df = df.iloc[1:].reset_index(drop=True) # The first row has the column names
        df = df[df["split"] == split] # train, test, valid

        for col in to_convert_to_int:
            df[col] = df[col].astype(int)
        self.df = df
        
        if labels_path is not None:
            labels_path = Path(labels_path)
            assert labels_path.exists()
            with h5py.File(labels_path, "r") as f:
                dataset = f["labels"]
                self.labels = [np.array(dataset[i]) for i in range(len(dataset))]
        
        if add_eos:
            raise RuntimeError("EOS not implemented")
        if tokenizer_name != "char":
            raise RuntimeError(f"{tokenizer_name} tokenizer not implemented")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_seq = self.df.iloc[idx]
        seq = self.seqs[df_seq["chromosome"]][df_seq["start"]:df_seq["end"]][:].seq # Replaced on the variant_effects tasks, [:] copies the string just in case

        seq = self.tokenizer(seq,
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
            control_signal = self.control_signals[df_seq["chromosome"]][df_seq["start"]:df_seq["end"]][:]
            control_signal = trim_and_right_pad_to_torch(control_signal, self.max_length)
            seq_ids = torch.stack([seq_ids, control_signal], dim=0)

        # Get the target
        if self.dataset_name == "cpg_methylation":
            y = [0] * 7
            if not pd.isna(df_seq["label"]):
                for label_idx in df_seq["label"].split(","):
                    y[int(label_idx)] = 1
        elif self.dataset_name == "gene_finding":
            # The cross_entropy task expects a tensor with the labels themselves, not the one-hot encoded labels
            y = self.labels[df_seq.name]
            # Padding and trimming
            seq_len = df_seq["end"] - df_seq["start"]
            if seq_len > self.max_length:
                y = y[:self.max_length]
            elif seq_len < self.max_length:
                pad_length = self.max_length - seq_len
                pad_width = (0, pad_length) if self.tokenizer.padding_side == "right" else (pad_length, 0)
                y = np.pad(y, pad_width, mode="constant", constant_values=-100)
        elif self.dataset_name == "variant_effects_disease" or self.dataset_name == "variant_effects_expression":
            # Start = end, indicates the position of the SNP
            seq_start = df_seq["start"] - (self.max_length // 2)
            seq_end = df_seq["start"] + (self.max_length - (self.max_length // 2))
            seq = self.seqs[df_seq["chromosome"]][seq_start:seq_end].seq
            
            if seq_start < 0 or len(seq) != (seq_end - seq_start): # The len diff is used to check that seq_end hasn't gone overboard
                raise ValueError("SNP seq escapes chromosome bounds")
            
            seq_snp = seq[:] # Copy the string
            seq_snp[len(seq_snp) // 2] = df_seq["alt"] # Apply the snp, the centered nucleotide will always be on the lower bound because of the way that seq is built
            
            return seq, seq_snp, {"label": df_seq["label"]}
        else:
            raise RuntimeError(f"{self.dataset_name} not implemented in bend benchmark")
        target = torch.LongTensor(y)

        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        return seq_ids, target
