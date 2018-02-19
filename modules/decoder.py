import torch
from torch import nn
from torch.nn import functional

import numpy as np

from . import attention


class Decoder(nn.Module):
    """
    Decoder module of the sequence to sequence model.
    """

    def __init__(self, hidden_dim, embedding_dim, output_dim, use_cuda, learning_rate):
        super(Decoder, self).__init__()
        self._use_cuda = use_cuda
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        self._attention = attention.Attention()

        self._embedding = None

        self._gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim,
                                 num_layers=1, batch_first=True).cuda()

        self._out = nn.Linear(self._hidden_dim, output_dim).cuda()

        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _forward_step(self, step_input, hidden, batch_size, activation):
        sequence_length = step_input.size(1)
        embedded = self.embedding(step_input)

        output, hidden = self._gru(embedded, hidden)

        predicted_softmax = activation(self._out(output.contiguous().view(-1, self._hidden_dim)),
                                       dim=1).view(batch_size, sequence_length,
                                                   self._output_dim)

        return predicted_softmax, hidden

    def forward(self, inputs, lengths, hidden):
        """

        :param inputs:
        :param lengths:
        :param hidden:
        :return:
        """
        batch_size = inputs.size(0)
        symbols = np.zeros((batch_size, int(lengths[0])), dtype='int')

        decoder_output, decoder_hidden = self._forward_step(inputs, hidden, batch_size,
                                                            activation=functional.log_softmax)

        for step in range(int(lengths[0])):
            step_output = decoder_output[:, step, :]
            symbols[:, step] = step_output.topk(1)[1].data.squeeze(-1).cpu().numpy()

        return decoder_output, symbols, hidden

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

    @property
    def output_dim(self):
        """

        :return:
        """
        return self._output_dim

    @property
    def embedding(self):
        """

        :return:
        """
        return self._embedding

    @embedding.setter
    def embedding(self, embedding):
        """

        :param embedding:
        :return:
        """
        self._embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self._embedding.weight = nn.Parameter(embedding)
        self._embedding.weight.requires_grad = False


class BeamDecoder:

    def __init__(self):
        pass
