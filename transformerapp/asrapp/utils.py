import torch
import numpy as np
#from otrans.data import PAD
PAD=1

def get_enc_padding_mask(tensor, tensor_length):
    return torch.sum(tensor, dim=-1).ne(0).unsqueeze(-2)


def get_seq_mask(targets):
    batch_size, steps = targets.size()
    seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    return seq_mask


def get_dec_seq_mask(targets, targets_length=None):
    steps = targets.size(-1)
    padding_mask = targets.ne(PAD).unsqueeze(-2).bool()
    seq_mask = torch.ones([steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    seq_mask = seq_mask.unsqueeze(0)

    return seq_mask & padding_mask


def get_length_mask(tensor, tensor_length):
    b, t, _ = tensor.size()  
    mask = tensor.new_ones([b, t], dtype=torch.uint8)
    for i, length in enumerate(tensor_length):
        length = length.item()
        mask[i].narrow(0, 0, length).fill_(0)
    return mask.bool()


