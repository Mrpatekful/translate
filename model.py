import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


EMBEDDING_DIM = 300


class Encoder(nn.Module):
    """
    Encoder for the AutoEncoder and Translation module.

    """

    def __init__(self, vocab_size, hidden_dim=32):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.bi_layer = nn.LSTM(EMBEDDING_DIM, hidden_dim,
                                num_layers=1, bidirectional=False)

    def forward(self, inputs, hidden):
        """
        Encoder for the AutoEncoder and Translation module.
        Args:
            inputs: int
            hidden: boolean
        Yields:
            A pair of Tensors, each shaped [batch_size, num_steps]. The second element
            of the tuple is the same data time-shifted to the right by one.
        """
        embedded = self.embedding(inputs).view(1, 1, -1)
        output = self.bi_layer()
        _, hidden = self.uni_layers(output, hidden)
        return


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
        return 0


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self):
        return 0


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        return 0

