from pyfaidx import Fasta
import torch
# from random import random 
import random
from pathlib import Path
import numpy as np
from src.dataloaders.utils.control_signal import trim_and_right_pad_to_torch

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp



class NucleotideTransformerDataset(torch.utils.data.Dataset):

    '''
    Loop thru fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name=None,
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        use_tokenizer=True,
        use_control_signal=False
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
        self.seqs = None
        self.control_signal = None
        self.dataset_name = dataset_name

        # change "val" split to "test".  No val available, just test
        if split == "val":
            split = "test"

        # use Path object
        base_path = Path(dest_path) / dataset_name 
        #print(f"Using {base_path}")
        assert base_path.exists(), 'path to fasta file must exist'

        for file in (base_path.iterdir()):
            if str(file).endswith('.fasta') and split in str(file):
                if self.seqs is not None:
                    raise ValueError("Overwitten self.seqs")
                self.seqs = Fasta(str(file), read_long_names=True)
                if use_control_signal != False:
                    print("**USING CONTROL SIGNAL**")
                    self.control_signal = {}
                    # I can add memmap_mode="r" to check that I'm not writing into it
                    control_signals_npz = np.load(base_path / (f"{file.stem}_raw_signals.npz" if use_control_signal == "raw" else f"{file.stem}.npz"))
                    for seq_key in control_signals_npz:
                        self.control_signal[seq_key] = torch.from_numpy(control_signals_npz[seq_key])

        self.label_mapper = {}
        for i, key in enumerate(self.seqs.keys()):
            self.label_mapper[i] = (key, int(key.rstrip()[-1])) # seq_key, label(0 or 1 for binary classification)

    def __len__(self):
        return len(self.seqs.keys())

    def __getitem__(self, idx):
        seq_id = self.label_mapper[idx][0]
        x = self.seqs[seq_id][:].seq # only one sequence
        y = self.label_mapper[idx][1] # 0 or 1 for binary classification

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else 'do_not_pad',
            max_length=self.max_length,
            truncation=True,
        )
        if self.control_signal is not None:
            assert len(seq) == len(x)

        seq_ids = seq["input_ids"] # get input_ids
        assert type(seq_ids)==list
        seq_ids = torch.LongTensor(seq_ids)
        if not self.use_tokenizer:
            seq_ids = seq_ids-7 # Each character id starts at 7(CharacterTokenizer). The rest are special tokens
            mask = (seq_ids >= 4) | (seq_ids < 0)
            seq_ids[mask] = 4

        # convert to tensor
        # seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        # need to wrap in list
        target = torch.LongTensor([y])  # offset by 1, includes eos

        # Add the control_signal to the seq_ids tensor
        if self.control_signal is not None:
            seq_control_signal = self.control_signal[seq_id]
            if self.use_padding:
                seq_control_signal = trim_and_right_pad_to_torch(seq_control_signal, self.max_length)
            seq_ids = torch.stack([seq_ids, seq_control_signal], dim=0)

        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target
