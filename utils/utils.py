import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def batch_to_padded_sequence(batch, lengths):
    """

    :param batch:
    :param lengths:
    :return:
    """
    return pack_padded_sequence(batch, lengths=lengths, batch_first=True)


def padded_sequence_to_batch(padded_sequence):
    """

    :param padded_sequence:
    :return:
    """
    return pad_packed_sequence(padded_sequence, batch_first=True)


def apply_noise(input_batch):
    return input_batch


def create_mask():
    pass
