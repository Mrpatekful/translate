import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def embedding_to_padded_sequence(embedding, lengths):
    """

    :param embedding:
    :param lengths:
    :return:
    """
    return pack_padded_sequence(embedding, lengths=lengths)


def padded_sequence_to_embedding(padded_sequence):
    """

    :param padded_sequence:
    :return:
    """
    return pad_packed_sequence()
