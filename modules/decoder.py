import torch
from torch import nn

from . import attention


class Decoder(nn.Module):
    """
    Decoder module of the sequence to sequence model.
    """

    def __init__(self, hidden_dim, embedding_dim, use_cuda, learning_rate):
        super(Decoder, self).__init__()
        self._use_cuda = use_cuda
        self._hidden_dim = hidden_dim

        self._attention = attention.Attention()

        self._embedding = None

        self.gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim)
        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs, hidden):
        """

        :param inputs:
        :param hidden:
        :return:
        """
        embedded = self.embedding(inputs)
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    @property
    def optimizer(self):
        """

        :return:
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        """

        :param value:
        :return:
        """
        self._optimizer = value
