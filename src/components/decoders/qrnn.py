import torch.nn as nn

from src.components.decoders.base import Decoder


class QRNNDecoder(Decoder):  # TODO
    """
    Quasi-recurrent decoder module of the sequence to sequence model.
    """

    def __init__(self):
        super().__init__()

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

    @classmethod
    def abstract(cls):
        return False

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
