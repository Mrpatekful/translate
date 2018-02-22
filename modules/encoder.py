import torch
from torch import nn

import torch.optim
from torch.autograd import Variable

from utils import utils


class RNNEncoder(nn.Module):
    """
    Encoder module of the sequence to sequence model.
    """

    def __init__(self, hidden_size, recurrent_layer,
                 embedding_dim, learning_rate,
                 use_cuda):
        super(RNNEncoder, self).__init__()

        self._use_cuda = use_cuda
        self._hidden_dim = hidden_size

        self._embedding = None

        if recurrent_layer == 'LSTM':
            unit_type = torch.nn.LSTM
        else:
            unit_type = torch.nn.GRU

        self._recurrent_layer = unit_type(input_size=embedding_dim, hidden_size=hidden_size,
                                          num_layers=1, bidirectional=False, batch_first=True)

        if use_cuda:
            self._recurrent_layer = self._recurrent_layer.cuda()

        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs, lengths, hidden_state):
        """

        :param inputs:
        :param lengths:
        :param hidden_state:
        :return:
        """
        embedded_inputs = self._embedding(inputs)
        padded_sequence = utils.batch_to_padded_sequence(embedded_inputs, lengths)
        self._recurrent_layer.flatten_parameters()
        outputs, final_hidden_state = self._recurrent_layer(padded_sequence, hidden_state)
        outputs, _ = utils.padded_sequence_to_batch(outputs)

        return outputs, final_hidden_state

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state of the encoder module.
        :return: Variable, (1, 1, hidden_dim) with zeros as initial values.
        """

        result = Variable(torch.zeros(1, batch_size, self._hidden_dim))

        if self._use_cuda:
            result = result.cuda()

        if isinstance(self._recurrent_layer, torch.nn.LSTM):
            return result, result
        else:
            return result

    @property
    def optimizer(self):
        """

        :return:
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """

        :param optimizer:
        :return:
        """
        self._optimizer = optimizer

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

    @property
    def hidden_size(self):
        """
        Property for the hidden size of the recurrent layer.
        :return: int, size of the hidden layer.
        """
        return self._hidden_size


class ConvEncoder(nn.Module):

    def __init__(self, ):
        super(ConvEncoder, self).__init__()

        self._embedding = None
        self._optimizer = None

    def forward(self, inputs, lengths, hidden):
        """
        :param inputs:
        :param lengths:
        :param hidden:
        :return:
        """
        return NotImplementedError

    @property
    def optimizer(self):
        """

        :return:
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """

        :param optimizer:
        :return:
        """
        self._optimizer = optimizer

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

    @property
    def hidden_size(self):
        """
        Property for the hidden size of the recurrent layer.
        :return: int, size of the hidden layer.
        """
        return self._hidden_dim
