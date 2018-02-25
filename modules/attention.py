import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional


class RNNAttention(nn.Module):
    """

    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """

        :param hidden_size:
        :param use_cuda:
        """
        super().__init__()
        self._recurrent_layer = None
        self._use_cuda = use_cuda
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self._input_size = hidden_size

    def forward(self,
                step_input,
                hidden_state,
                encoder_outputs,
                batch_size,
                sequence_length):

        return NotImplementedError

    def _calculate_context(self,
                           previous_state,
                           encoder_outputs,
                           batch_size,
                           sequence_length):
        """

        :param previous_state:
        :param encoder_outputs:
        :param batch_size:
        :param sequence_length:
        :return:
        """
        attn_energies = Variable(torch.zeros([batch_size, sequence_length]))

        if self._use_cuda:
            attn_energies = attn_energies.cuda()

        squeezed_output = previous_state.squeeze(1)
        for step in range(sequence_length):
            attn_energies[:, step] = self._score(encoder_outputs[:, step], squeezed_output)

        attn_weights = functional.softmax(attn_energies, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        return context, attn_weights

    def _score(self,
               encoder_outputs,
               decoder_state):

        return NotImplementedError

    @property
    def recurrent_layer(self):
        """

        :return:
        """
        if self._recurrent_layer is None:
            raise ValueError('Recurrent layer must be set.')
        return self._recurrent_layer

    @property
    def input_size(self):
        """

        :return:
        """
        return self._input_size


class BahdanauAttention(RNNAttention):
    """

    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """

        :param hidden_size:
        :param embedding_size:
        :param use_cuda:
        """
        super().__init__(hidden_size=hidden_size,
                         embedding_size=embedding_size,
                         use_cuda=use_cuda)

        self.__attention_layer = nn.Linear(self._hidden_size * 2, self._hidden_size)
        self.__projection_layer = nn.Linear(self._hidden_size + self._embedding_size, self._hidden_size)

        tr = torch.rand(self._hidden_size, 1)

        if self._use_cuda:
            self.__attention_layer = self.__attention_layer.cuda()
            self.__projection_layer = self.__projection_layer.cuda()

            tr = tr.cuda()

        self.__transformer = nn.Parameter(tr)

    def forward(self,
                step_input,
                hidden_state,
                encoder_outputs,
                batch_size,
                sequence_length):
        """

        :param step_input:
        :param hidden_state:
        :param encoder_outputs:
        :param batch_size:
        :param sequence_length:
        :return:
        """
        previous_state = hidden_state[0][-1] if isinstance(hidden_state, tuple) else hidden_state[-1]

        context, attn_weights = self._calculate_context(previous_state=previous_state,
                                                        encoder_outputs=encoder_outputs,
                                                        batch_size=batch_size,
                                                        sequence_length=sequence_length)

        concat_input = self.__projection_layer(torch.cat((step_input, context), dim=2))

        output, hidden_state = self.recurrent_layer(concat_input, hidden_state)

        return output, hidden_state, attn_weights

    def _score(self,
               encoder_output,
               decoder_state):
        """

        :param encoder_output:
        :param decoder_state:
        :return:
        """
        energy = functional.tanh(self.__attention_layer(torch.cat((decoder_state, encoder_output), 1)))
        energy = torch.mm(energy, self.__transformer)
        return energy


class LuongAttention(RNNAttention):
    """

    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """

        :param hidden_size:
        :param use_cuda:
        """
        super().__init__(hidden_size=hidden_size,
                         embedding_size=embedding_size,
                         use_cuda=use_cuda)

        self._projection_layer = nn.Linear(self._hidden_size * 2, self._hidden_size)

        if self._use_cuda:
            self._projection_layer.cuda()

        self._input_size = embedding_size

    def forward(self,
                step_input,
                hidden_state,
                encoder_outputs,
                batch_size,
                sequence_length):
        """

        :param step_input:
        :param hidden_state:
        :param encoder_outputs:
        :param batch_size:
        :param sequence_length:
        :return:
        """
        output, hidden_state = self.recurrent_layer(step_input, hidden_state)

        previous_state = hidden_state[0][-1] if isinstance(hidden_state, tuple) else hidden_state[-1]

        context, attn_weights = self._calculate_context(previous_state=previous_state,
                                                        encoder_outputs=encoder_outputs,
                                                        batch_size=batch_size,
                                                        sequence_length=sequence_length)

        output = self._projection_layer(torch.cat((output, context), dim=2))

        return output, hidden_state, attn_weights


class GeneralAttention(LuongAttention):
    """

    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """

        :param hidden_size:
        :param embedding_size:
        :param use_cuda:
        """
        super().__init__(hidden_size=hidden_size,
                         embedding_size=embedding_size,
                         use_cuda=use_cuda)

        self.__attention_layer = nn.Linear(self._hidden_size, self._hidden_size)

        if use_cuda:
            self.__attention_layer = self.__attention_layer.cuda()

    def _score(self,
               encoder_output,
               decoder_state):
        """

        :param encoder_output:
        :param decoder_state:
        :return:
        """
        energy = self.__attention_layer(encoder_output)
        energy = torch.bmm(decoder_state.unsqueeze(1), energy.unsqueeze(1).transpose(1, 2)).squeeze(-1)
        return energy


class DotAttention(LuongAttention):
    """

    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """

        :param hidden_size:
        :param embedding_size:
        :param use_cuda:
        """
        super().__init__(hidden_size=hidden_size,
                         embedding_size=embedding_size,
                         use_cuda=use_cuda)

    def _score(self,
               encoder_output,
               decoder_state):
        """

        :param encoder_output:
        :param decoder_state:
        :return:
        """
        return torch.bmm(decoder_state.unsqueeze(1), encoder_output.unsqueeze(1).transpose(1, 2)).squeeze(-1)


class ConcatAttention(LuongAttention):
    """

    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """

        :param hidden_size:
        :param embedding_size:
        :param use_cuda:
        """
        super().__init__(hidden_size=hidden_size,
                         embedding_size=embedding_size,
                         use_cuda=use_cuda)

        self.__embedding_size = embedding_size

        self._input_size = embedding_size

        self.__attention_layer = nn.Linear(self._hidden_size * 2, self._hidden_size)
        self.__projection_layer = nn.Linear(self._hidden_size + self.__embedding_size, self._hidden_size)

        tr = torch.rand(hidden_size, 1)

        if use_cuda:
            self.__attention_layer = self.__attention_layer.cuda()
            self.__projection_layer = self.__projection_layer.cuda()

            tr = tr.cuda()

        self.__transformer = nn.Parameter(tr)

    def _score(self,
               encoder_output,
               decoder_state):
        """

        :param encoder_output:
        :param decoder_state:
        :return:
        """
        energy = functional.tanh(self.__attention_layer(torch.cat((decoder_state, encoder_output), 1)))
        energy = torch.mm(energy, self.__transformer)
        return energy
