import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable

import random

import numpy as np


class RNNDecoder(nn.Module):
    """
    Decoder module of the sequence to sequence model.
    """

    def __init__(self, hidden_size, embedding_dim, recurrent_layer,
                 output_dim, learning_rate, use_cuda):
        super(RNNDecoder, self).__init__()

        self._use_cuda = use_cuda
        self._hidden_dim = hidden_size
        self._output_dim = output_dim

        self._attention = None
        self._embedding_layer = None

        if recurrent_layer == 'LSTM':
            unit_type = torch.nn.LSTM
        else:
            unit_type = torch.nn.GRU

        self._recurrent_layer = unit_type(input_size=embedding_dim, hidden_size=hidden_size,
                                          num_layers=1, bidirectional=False, batch_first=True)
        self._output_layer = nn.Linear(self._hidden_dim, output_dim)

        if use_cuda:
            self._recurrent_layer = self._recurrent_layer.cuda()
            self._output_layer = self._output_layer.cuda()

        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    # def _forward_step(self, step_input, encoder_outputs, prev_hidden, batch_size, activation):
    #
    #     embedded = self.embedding(step_input)
    #     output, hidden = self._recurrent_layer(embedded, prev_hidden)
    #
    #     predicted_softmax = activation(self._output_layer(output.contiguous().view(-1, self._hidden_dim)),
    #                                    dim=1).view(batch_size, -1,
    #                                                self._output_dim)
    #
    #     return predicted_softmax, hidden
    #
    # def forward(self, inputs, encoder_outputs, lengths, hidden):
    #     batch_size = inputs.size(0)
    #     sequence_length = inputs.size(1)
    #     symbols = np.zeros((batch_size, int(lengths[0])), dtype='int')
    #
    #     decoder_hidden = hidden
    #
    #     decoder_output, decoder_hidden = self._forward_step(step_input=inputs, prev_hidden=decoder_hidden,
    #                                                         encoder_outputs=encoder_outputs, batch_size=batch_size,
    #                                                         activation=functional.log_softmax)
    #
    #     for step in range(int(lengths[0])):
    #         step_output = decoder_output[:, step, :]
    #         symbols[:, step] = step_output.topk(1)[1].data.squeeze(-1).cpu().numpy()
    #
    #     return decoder_output, symbols

    def _forward_step(self, step_input, hidden_state, batch_size, activation):
        embedded_input = self.embedding(step_input)
        output, hidden_state = self._recurrent_layer(embedded_input, hidden_state)

        output = activation(self._output_layer(output.contiguous().view(-1, self._hidden_dim)),
                            dim=1).view(batch_size, -1,
                                        self._output_dim)

        return output, hidden_state

    def forward(self, inputs,
                encoder_outputs, lengths,
                hidden_state, loss_function,
                teacher_forcing_ratio=0):

        batch_size = inputs.size(0)
        sequence_length = inputs.size(1)
        symbols = np.zeros((batch_size, int(lengths[0])), dtype='int')

        use_teacher_forcing = True
        loss = 0

        if use_teacher_forcing:

            for step in range(sequence_length):

                step_output, hidden_state = self._forward_step(step_input=inputs[:, step].unsqueeze(-1),
                                                               hidden_state=hidden_state,
                                                               batch_size=batch_size,
                                                               activation=functional.log_softmax)

                if self._attention is not None:
                    if isinstance(hidden_state, tuple):  # LSTM
                        hidden_state = (self._attention(hidden_state[0], encoder_outputs), hidden_state[1])
                    else:                                # GRU
                        hidden_state = self._attention(hidden_state, encoder_outputs)

                loss += loss_function(step_output.squeeze(1), inputs[:, step])
                symbols[:, step] = step_output.topk(1)[1].data.squeeze(-1).squeeze(-1).cpu().numpy()

        return loss, symbols

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
    def attention(self):
        """

        :return:
        """
        return self._attention

    @attention.setter
    def attention(self, attention):
        """

        :param attention:
        :return:
        """
        self._attention = attention

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
        return self._embedding_layer

    @embedding.setter
    def embedding(self, embedding):
        """

        :param embedding:
        :return:
        """
        self._embedding_layer = nn.Embedding(embedding.size(0), embedding.size(1))
        self._embedding_layer.weight = nn.Parameter(embedding)
        self._embedding_layer.weight.requires_grad = False


class BeamDecoder:

    def __init__(self):
        pass


class ConvDecoder:

    def __init__(self):
        pass

