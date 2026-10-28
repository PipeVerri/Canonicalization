import numpy as np
import torch

def trim_and_pad_to_torch(_sequence, max_length, tokenizer, constant_values, use_padding=True):
    padding_side = tokenizer.padding_side
    truncation_side = tokenizer.truncation_side
    if not padding_side in ("left", "right"):
        raise ValueError(f"Padding side {padding_side} not recognized")
    if not truncation_side in ("left", "right"):
        raise ValueError(f"Truncation {truncation_side} not recognized")

    sequence = np.array(_sequence[:])
    sequence = sequence[:max_length] if truncation_side == "right" else sequence[-max_length:]
    if use_padding:
        padding = (0, max_length - len(sequence)) if padding_side == "right" else (max_length - len(sequence), 0)
        sequence = np.pad(sequence, padding, mode="constant", constant_values=constant_values)
    return torch.LongTensor(sequence)
