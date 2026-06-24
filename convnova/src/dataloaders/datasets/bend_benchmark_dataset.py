from pyfaidx import Fasta
import torch
# from random import random 
import random
from pathlib import Path
import numpy as np
import h5py
import pandas as pd

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

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
        fasta_path=None
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

        if split == "val":
            split = "valid"

        # use Path object
        bed_path = Path(bed_path)
        fasta_path = Path(fasta_path)
        assert bed_path.exists()
        assert fasta_path.exists()

        self.seqs = Fasta(str(fasta_path))
        
        if dataset_name == "gene_finding":
            bed_columns = ["chromosome", "start", "end", "transcript_id", "strand", "flank_left", "flank_right", "length", "split"]
            to_convert_to_int = ["start", "end", "length"]
        elif dataset_name == "cpg_methylation":
            bed_columns = ["chromosome", "start", "end", "strand", "label", "split"]
            to_convert_to_int = ["start", "end", "length"]
        else:
            raise RuntimeError(f"Unrecognized dataset {dataset_name}")
        
        df = pd.read_csv(str(bed_path), sep="\t", header=None, names=bed_columns)
        df = df.iloc[1:].reset_index(drop=True)
        df = df[df["split"] == split] # train, test, valid split
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
        seq = self.seqs[df_seq["chromosome"]][df_seq["start"]:df_seq["end"]].seq

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

        # Get the target
        if self.dataset_name == "cpg_methylation":
            y = [0] * 7
            if not pd.isna(df_seq["Score"]):
                for label_idx in df_seq["Score"].split(","):
                    y[int(label_idx)] = 1
        elif self.dataset_name == "gene_finding":
            # The cross_entropy task expects a tensor with the labels themselves, not the one-hot encoded labels
            y = self.labels[df_seq.name]
            # Padding and trimming
            seq_len = df_seq["end"] - df_seq["start"]
            if seq_len > self.max_length:
                y = y[:self.max_length]
            elif seq_len < self.max_length:
                #print(f"idx={idx}, seq_len={seq_len}, y.shape={y.shape}, true_idx={df_seq.name}")
                y = np.pad(y, (self.max_length - seq_len, 0), mode="constant", constant_values=-100)
        else:
            raise RuntimeError(f"{self.dataset_name} not implemented in bend benchmark")
        target = torch.LongTensor(y)

        return seq_ids, target
