import numpy as np
import torch

def trim_and_right_pad_to_torch(control_signal, max_length):
    control_signal = control_signal[:]
    control_signal = control_signal[:max_length]
    control_signal = np.pad(control_signal, (0, max_length - len(control_signal)), mode="constant")
    return torch.LongTensor(control_signal)