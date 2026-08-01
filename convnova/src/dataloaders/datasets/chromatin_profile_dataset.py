from pyfaidx import Fasta
import torch
# from random import random 
import random
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
class ChromatinProfileDataset(torch.utils.data.Dataset):

    '''
    Loop thru fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        seqs,
        labels,
        max_length,
        split,
        d_output=919,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        use_tokenizer=True,
    ):
        self.seqs = seqs
        self.labels = labels
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.use_tokenizer = use_tokenizer
        self.split = split

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self, idx):
        x = torch.LongTensor(self.seqs[idx][:])
        y = torch.LongTensor(self.labels[idx][:])
        #print(f"X shape is {x.shape} | y shape is {y.shape}")
        return x, y