import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable


class Encoder(nn.Module):
    """
    Encoder module of the sequence to sequence model.
    """

    def __init__(self, hidden_dim, embedding_dim, learning_rate, use_cuda):
        super(Encoder, self).__init__()
        self._use_cuda = use_cuda
        self._hidden_dim = hidden_dim

        self._embedding = None

        self._gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim,
                           num_layers=1, bidirectional=False)

        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs, hidden):
        """
        :param inputs:
        :param hidden:
        :return:
        """
        print(inputs)
        embedded = self._embedding(inputs.view(1, 1, -1))
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def init_hidden(self):
        """
        Initializes the hidden state of the encoder module.
        :return: Variable, (1, 1, hidden_dim) with zeros as initial values.
        """
        result = Variable(torch.zeros(1, 1, self._hidden_dim))
        if self._use_cuda:
            return result.cuda()
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
        return self._hidden_dim


class Decoder(nn.Module):
    """
    Decoder module of the sequence to sequence model.
    """

    def __init__(self, hidden_dim, embedding_dim, use_cuda, learning_rate):
        super(Decoder, self).__init__()
        self._use_cuda = use_cuda
        self._hidden_dim = hidden_dim

        self._attention = Attention()

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


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        return 0


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self):
        return 0