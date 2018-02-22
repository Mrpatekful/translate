import torch
from torch.autograd import Variable
from torch import nn


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        return NotImplementedError

    def score(self, encoder_outputs, decoder_hidden):
        return NotImplementedError


class BahdanauAttention(Attention):

    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self._hidden_dim = hidden_dim
        self._attention = nn.Linear(self._hidden_dim, self._hidden_dim)

    def forward(self, encoder_outputs, h_n):
        seq_len = len(encoder_outputs)
        attn_energies = Variable(torch.zeros(seq_len))

        for step in range(seq_len):
            attn_energies[step] = self._score(encoder_outputs[step], h_n)

        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = 0
        return context

    def _score(self, encoder_output, h_n):
        energy = self.attn(torch.cat((h_n, encoder_output), 1))
        energy = self.other.dot(energy)
        return energy


class LuongAttention(Attention):

    def __init__(self):
        super(LuongAttention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        return 0

    def score(self, encoder_outputs, decoder_hidden):
        return NotImplementedError

