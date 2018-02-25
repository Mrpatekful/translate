import torch
from torch import nn

import torch.optim
from torch.autograd import Variable

from utils import utils


class RNNEncoder(nn.Module):
    """
    Recurrent encoder module of the sequence to sequence model.
    """

    def __init__(self,
                 hidden_size,
                 recurrent_layer,
                 num_layers,
                 embedding_size,
                 learning_rate,
                 use_cuda):
        """
        A recurrent encoder module for the sequence to sequence model.
        :param hidden_size: int, size of recurrent layer of the LSTM/GRU.
        :param recurrent_layer: str, name of the recurrent layer ('GRU', 'LSTM').
        :param embedding_size: int, dimension of the word embeddings.
        :param learning_rate: float, learning rate.
        :param use_cuda: bool, True if the device has cuda support.
        """
        super(RNNEncoder, self).__init__()

        self.__use_cuda = use_cuda
        self.__hidden_dim = hidden_size
        self.__num_layers = num_layers

        self._embedding = None

        if recurrent_layer == 'LSTM':
            unit_type = torch.nn.LSTM
        else:
            unit_type = torch.nn.GRU

        self.__recurrent_layer = unit_type(input_size=embedding_size,
                                           hidden_size=hidden_size,
                                           num_layers=num_layers,
                                           bidirectional=False,
                                           batch_first=True)

        if use_cuda:
            self.__recurrent_layer = self.__recurrent_layer.cuda()

        self.__optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self,
                inputs,
                lengths,
                hidden_state):
        """
        A forward step of the encoder. The batch of sequences with word ids are
        packed into padded_sequence object, which are processed by the recurrent layer.
        :param inputs: Variable, (batch_size, sequence_length) containing the ids of the words.
        :param lengths: Ndarray, containing the real lengths of the sequences in the batch (prior to padding).
        :param hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.
        :return outputs: Variable, (batch_size, sequence_length, vocab_size) the output at each time
                         step of the encoder.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final hidden state.
        """
        embedded_inputs = self._embedding(inputs)
        padded_sequence = utils.batch_to_padded_sequence(embedded_inputs, lengths)
        self.__recurrent_layer.flatten_parameters()
        outputs, final_hidden_state = self.__recurrent_layer(padded_sequence, hidden_state)
        outputs, _ = utils.padded_sequence_to_batch(outputs)

        return outputs, final_hidden_state

    def init_hidden(self,
                    batch_size):
        """
        Initializes the hidden state of the encoder module.
        :return: Variable, (num_layers*directions, batch_size, hidden_dim) with zeros as initial values.
        """
        result = Variable(torch.zeros(self.__num_layers, batch_size, self.__hidden_dim))

        if self.__use_cuda:
            result = result.cuda()

        if isinstance(self.__recurrent_layer, torch.nn.LSTM):
            return result, result
        else:
            return result

    @property
    def optimizer(self):
        """
        Property for the optimizer of the encoder.
        :return self.__optimizer: Optimizer, the currently used optimizer of the encoder.
        """
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Setter for the optimizer of the encoder.
        :param optimizer: Optimizer, instance to be set as the new optimizer for the encoder.
        """
        self.__optimizer = optimizer

    @property
    def embedding(self):
        """
        Property for the encoder's embedding layer.
        :return: The currently used embeddings of the encoder.
        """
        return self._embedding

    @embedding.setter
    def embedding(self, embedding):
        """
        Setter for the encoder's embedding layer.
        :param embedding: Embedding, to be set as the embedding layer of the encoder.
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


class ConvEncoder(nn.Module):  # TODO
    """
    Convolutional encoder module of the sequence to sequence model.
    """

    def __init__(self, ):
        super(ConvEncoder, self).__init__()

        self._embedding = None
        self._optimizer = None

    def forward(self,
                inputs,
                lengths,
                hidden):
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
