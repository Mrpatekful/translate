import torch.nn as nn

from src.components.base import Decoder


class QRNNDecoder(Decoder):  # TODO
    """
    Quasi-recurrent decoder module of the sequence to sequence model.
    """

    @property
    def optimizers(self):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError

    def __init__(self):
        super().__init__()

        self._embedding = None
        self._optimizer = None

    def forward(self, inputs, lengths, hidden):
        return NotImplementedError

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, embedding):
        self._embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self._embedding.weight = nn.Parameter(embedding)
        self._embedding.weight.requires_grad = False

    @property
    def hidden_size(self):
        return self._hidden_dim
