import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """

    """

    def __init__(self):
        super().__init__()

    def forward(self, encoder_outputs, decoder_hidden, batch_size, sequence_length):
        """

        """
        return NotImplementedError

    def _score(self, encoder_outputs, decoder_hidden):
        """

        """
        return NotImplementedError


class BahdanauAttention(Attention):
    """

    """

    def __init__(self,
                 hidden_dim,
                 use_cuda):
        """

        :param hidden_dim:
        :param use_cuda:
        """
        super().__init__()
        self.__hidden_dim = hidden_dim
        self.__use_cuda = use_cuda
        self.__attention = nn.Linear(self.__hidden_dim * 2, self.__hidden_dim)

        tr = torch.rand(self.__hidden_dim, 1)

        if self.__use_cuda:
            self.__attention = self.__attention.cuda()
            tr = tr.cuda()

        self.__transformer = nn.Parameter(tr)

    def forward(self,
                hidden_state,
                encoder_outputs,
                batch_size,
                sequence_length):
        """

        :param hidden_state:
        :param encoder_outputs:
        :param batch_size:
        :param sequence_length:
        :return:
        """
        attn_energies = Variable(torch.zeros([batch_size, sequence_length]))

        if self.__use_cuda:
            attn_energies = attn_energies.cuda()

        for step in range(sequence_length):
            attn_energies[:, step] = self._score(encoder_outputs[:, step], hidden_state.squeeze(0))

        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1).unsqueeze(0)
        return context, attn_weights

    def _score(self,
               encoder_output,
               hidden_state):
        """

        :param encoder_output:
        :param hidden_state:
        :return:
        """
        energy = self.__attention(torch.cat((hidden_state, encoder_output), 1))
        energy = torch.mm(energy, self.__transformer)
        return energy


class LuongAttention(Attention):
    """

    """

    def __init__(self):
        super(LuongAttention, self).__init__()

    def forward(self,
                encoder_outputs,
                decoder_hidden,
                batch_size,
                sequence_length):
        """

        :param encoder_outputs:
        :param decoder_hidden:
        :param batch_size:
        :param sequence_length:
        :return:
        """
        return 0

    def _score(self,
               encoder_outputs,
               decoder_hidden):
        """

        :param encoder_outputs:
        :param decoder_hidden:
        :return:
        """
        return NotImplementedError

