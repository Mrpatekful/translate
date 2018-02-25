import torch
from .decoder import *
from torch.autograd import Variable
from torch import nn
from torch.nn import functional


class RNNAttention(nn.Module):
    """
    Abstract base class for the attention mechanisms of the recurrent decoder unit.
    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """
        Base class, should not be instantiated.
        :param hidden_size: int, size of the hidden layer of the recurrent decoder.
        :param embedding_size: int, size of the word embedding vectors.
        :param use_cuda: bool, True if cuda is enabled on the device.
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
                           decoder_state,
                           encoder_outputs,
                           batch_size,
                           sequence_length):
        """
        Calculates the context for the decoder, given the encoder outputs and a decoder hidden state.
        The algorithm iterates through the encoder outputs and scores each output based on the similarity
        with the decoder state. The scoring functions are implemented in the child nodes of this class.
        :param decoder_state: Variable, (batch_size, 1, hidden_size) the state of the decoder.
        :param encoder_outputs: Variable, (batch_size, sequence_length, hidden_size) the output of
               each time step of the encoder.
        :param batch_size: int, size of the input batch.
        :param sequence_length: int, size of the sequence.
        :return context: Variable, the weighted sum of the encoder outputs.
        :return attn_weights: Variable, weights used in the calculation of the context.
        """
        attn_energies = Variable(torch.zeros([batch_size, sequence_length]))

        if self._use_cuda:
            attn_energies = attn_energies.cuda()

        squeezed_output = decoder_state.squeeze(1)
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
        Property for the recurrent layer of the decoder.
        :return self._recurrent_layer:
        """
        if self._recurrent_layer is None:
            raise ValueError('Recurrent layer must be set.')
        return self._recurrent_layer

    @property
    def input_size(self):
        """
        Property for the size of the recurrent unit's input layer, which will be
        determined by the currently used attention implementation.
        :return self._input_size: int, size of the required input size for the RNN.
        """
        return self._input_size


class BahdanauAttention(RNNAttention):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is based on:

        https://arxiv.org/pdf/1409.0473.pdf

    The computational path of the method differs from the Luong style, since
    here the context vector contributes to the calculation of the hidden state, by
    concatenating the context with the input of the recurrent unit.

        h(t-1) -> a(t) -> c(t) -> h(t)
    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """
        An implementation of the attention mechanism.
        :param hidden_size: int, size of the hidden layer of the recurrent decoder.
        :param embedding_size: int, size of the word embedding vectors.
        :param use_cuda: bool, True if cuda is enabled on the device.
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
        A forward step for the attention module. The recurrent layer of the decoder also does
        it's computation at the invocation of this method. This is required, because the computation
        path of the multiple types of attentions differ, and in this case the context will be added
        to the input of the recurrent layer.
        :param step_input: Variable, with size of (batch_size, 1), containing word ids for step t.
        :param hidden_state: Variable, with size of (num_layers * directions, batch_size, hidden_size).
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param batch_size: int, size of the input batches.
        :param sequence_length: int, size of the sequence of the input batch.
        :return output: Variable, (batch_size, 1, vocab_size) distribution of probabilities over the words.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final state at time t.
        :return attn_weights: Variable, (batch_size, 1, sequence_length) attention weights for visualization.
        """
        previous_state = hidden_state[0][-1] if isinstance(hidden_state, tuple) else hidden_state[-1]

        context, attn_weights = self._calculate_context(decoder_state=previous_state,
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
        Scoring function of the Bahdanau style attention. The states are concatenated and fed through
        a non-linear activation layer, and the multiplied by a vector to project the attention energies
        to the correct size.
        :param encoder_output: Variable, (batch_size, 1, hidden_layer) output of the encoder at time step t.
        :param decoder_state: Variable, (batch_size, 1, hidden_layer) hidden state of the decoder at time step t.
        :return energy: Variable, similarity between the decoder and encoder state.
        """
        energy = functional.tanh(self.__attention_layer(torch.cat((decoder_state, encoder_output), 1)))
        energy = torch.mm(energy, self.__transformer)
        return energy


class LuongAttention(RNNAttention):
    """
    Attention mechanism for the recurrent decoder module. The algorithm is based on:

        https://arxiv.org/pdf/1508.04025.pdf

    The computational path of the method differs from the Bahdanau style, since
    here the context vector contributes to the calculation of the hidden state, after the
    computations of the recurrent layer.

        h(t) -> a(t) -> c(t) -> h*(t)
    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """
        Abstract base class for the Luong style attention mechanisms. The different types of this attention
        essentially have the same computational path, but they differ in the scoring mechanism of the
        similarity between the encoder and decoder states.
        :param hidden_size: int, size of the hidden layer of the recurrent decoder.
        :param use_cuda: bool, True if cuda is enabled on the device.
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
        Forward step for the Luong style attentions. The context is calculated from the hidden state
        of the decoder at time step t, and is merged into the final output layer through a projection
        layer with linear activation.
        :param step_input: Variable, with size of (batch_size, 1), containing word ids for step t.
        :param hidden_state: Variable, with size of (num_layers * directions, batch_size, hidden_size).
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param batch_size: int, size of the input batches.
        :param sequence_length: int, size of the sequence of the input batch.
        :return output: Variable, (batch_size, 1, vocab_size) distribution of probabilities over the words.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final state at time t.
        :return attn_weights: Variable, (batch_size, 1, sequence_length) attention weights for visualization.
        """
        output, hidden_state = self.recurrent_layer(step_input, hidden_state)

        previous_state = hidden_state[0][-1] if isinstance(hidden_state, tuple) else hidden_state[-1]

        context, attn_weights = self._calculate_context(decoder_state=previous_state,
                                                        encoder_outputs=encoder_outputs,
                                                        batch_size=batch_size,
                                                        sequence_length=sequence_length)

        output = self._projection_layer(torch.cat((output, context), dim=2))

        return output, hidden_state, attn_weights


class GeneralAttention(LuongAttention):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is a specific case
    of Luong style attention, where the scoring is based off of the linear activation
    of the encoder output, and the dot product of the decoder hidden state with the result of the
    activation from the linear layer.
    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """
        A specific case of Luong style attention.
        :param hidden_size: int, size of the hidden layer of the recurrent decoder.
        :param embedding_size: int, size of the word embedding vectors.
        :param use_cuda: bool, True if cuda is enabled on the device.
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
        The score computation is as follows:
            h_d * (W_a * h_eT)
        where h_d is the decoder hidden state, W_a is a linear layer and h_eT is
        the transpose of encoder output at time step t.
        :param encoder_output: Variable, (batch_size, 1, hidden_layer) output of the encoder at time step t.
        :param decoder_state: Variable, (batch_size, 1, hidden_layer) hidden state of the decoder at time step t.
        :return energy: Variable, similarity between the decoder and encoder state.
        """
        energy = self.__attention_layer(encoder_output)
        energy = torch.bmm(decoder_state.unsqueeze(1), energy.unsqueeze(1).transpose(1, 2)).squeeze(-1)
        return energy


class DotAttention(LuongAttention):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is a specific case
    of Luong style attention, where the scoring is based off of only the dot product of the
    encoder and decoder states.
    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """
        A specific case of Luong style attention.
        :param hidden_size: int, size of the hidden layer of the recurrent decoder.
        :param embedding_size: int, size of the word embedding vectors.
        :param use_cuda: bool, True if cuda is enabled on the device.
        """
        super().__init__(hidden_size=hidden_size,
                         embedding_size=embedding_size,
                         use_cuda=use_cuda)

    def _score(self,
               encoder_output,
               decoder_state):
        """
        The score computation is as follows:
            h_d * h_eT
        where h_d is the decoder hidden state, and h_eT is the transpose of encoder output at time step t.
        :param encoder_output: Variable, (batch_size, 1, hidden_layer) output of the encoder at time step t.
        :param decoder_state: Variable, (batch_size, 1, hidden_layer) hidden state of the decoder at time step t.
        :return energy: Variable, similarity between the decoder and encoder state.
        """
        return torch.bmm(decoder_state.unsqueeze(1), encoder_output.unsqueeze(1).transpose(1, 2)).squeeze(-1)


class ConcatAttention(LuongAttention):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is a specific case
    of Luong style attention, where the scoring is based off of the concatenation of the encoder
    and decoder states, which is then passed through a non-linear layer with tanh activation.
    The result is then multiplied by a vector to transform the final result to the correct size.
    The scoring of similarity between encoder and decoder states is essentially the same as Bahdanau's
    method, however the computation path follows the Luong style.
    """
    def __init__(self,
                 hidden_size,
                 embedding_size,
                 use_cuda):
        """
        A specific case of Luong style attention.
        :param hidden_size: int, size of the hidden layer of the recurrent decoder.
        :param embedding_size: int, size of the word embedding vectors.
        :param use_cuda: bool, True if cuda is enabled on the device.
        """
        super().__init__(hidden_size=hidden_size,
                         embedding_size=embedding_size,
                         use_cuda=use_cuda)

        self._input_size = embedding_size

        self.__attention_layer = nn.Linear(self._hidden_size * 2, self._hidden_size)

        tr = torch.rand(hidden_size, 1)

        if use_cuda:
            self.__attention_layer = self.__attention_layer.cuda()

            tr = tr.cuda()

        self.__transformer = nn.Parameter(tr)

    def _score(self,
               encoder_output,
               decoder_state):
        """
        The score computation is as follows:
            v_t * tanh(W_a * [h_d ; h_e])
        where v_t is a vector, that transform the output to the correct size, h_d is the decoder hidden state,
        W_a is a weight matrix and h_e is the encoder output at time step t.
        :param encoder_output: Variable, (batch_size, 1, hidden_layer) output of the encoder at time step t.
        :param decoder_state: Variable, (batch_size, 1, hidden_layer) hidden state of the decoder at time step t.
        :return energy: Variable, similarity between the decoder and encoder state.
        """
        energy = functional.tanh(self.__attention_layer(torch.cat((decoder_state, encoder_output), 1)))
        energy = torch.mm(energy, self.__transformer)
        return energy
