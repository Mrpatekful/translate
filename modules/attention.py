from .decoder import *
from torch.autograd import Variable


class AttentionRNNDecoder(RNNDecoder):
    """
    Abstract base class for the attentional variation of recurrent decoder unit.
    """

    def __init__(self, parameter_setter):
        """
        An abstract base of a recurrent decoder with attention.
        """
        super().__init__(parameter_setter=parameter_setter)

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

        if self._use_cuda.value:
            attn_energies = attn_energies.cuda()

        squeezed_output = decoder_state.squeeze(1)
        for step in range(sequence_length):
            attn_energies[:, step] = self._score(encoder_outputs[:, step], squeezed_output)

        attn_weights = functional.softmax(attn_energies, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        return context, attn_weights

    def forward(self,
                inputs,
                encoder_outputs,
                lengths,
                hidden_state,
                loss_function,
                tf_ratio):
        """
        An attentional forward step. The calculations can be done with or without teacher fording.
        :param inputs: Variable, (batch_size, sequence_length) a batch of word ids.
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param lengths: Ndarray, an array for storing the real lengths of the sequences in the batch.
        :param hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.
        :param loss_function: loss function of the decoder.
        :param tf_ratio: int, if 1, teacher forcing is always used. Set to 0 to disable teacher forcing.
        :return loss: int, loss of the decoding
        :return symbols: Ndarray, the decoded word ids.
        """
        batch_size = inputs.size(0)
        sequence_length = inputs.size(1)

        symbols = np.zeros((batch_size, sequence_length), dtype='int')

        use_teacher_forcing = True
        loss = 0
        if use_teacher_forcing:
            for step in range(sequence_length):
                step_input = inputs[:, step].unsqueeze(-1)
                step_output, hidden_state, attn_weights = self._decode(decoder_input=step_input,
                                                                       hidden_state=hidden_state,
                                                                       encoder_outputs=encoder_outputs,
                                                                       batch_size=batch_size,
                                                                       sequence_length=sequence_length)

                loss += loss_function(step_output.squeeze(1), inputs[:, step])
                symbols[:, step] = step_output.topk(1)[1].data.squeeze(-1).squeeze(-1).cpu().numpy()

        else:
            for step in range(sequence_length):
                step_input = inputs[:, step].unsqueeze(-1)
                step_output, hidden_state, attn_weights = self._decode(decoder_input=step_input,
                                                                       hidden_state=hidden_state,
                                                                       encoder_outputs=encoder_outputs,
                                                                       batch_size=batch_size,
                                                                       sequence_length=sequence_length)

                loss += loss_function(step_output.squeeze(1), inputs[:, step])
                symbols[:, step] = step_output.topk(1)[1].data.squeeze(-1).squeeze(-1).cpu().numpy()

        return loss, symbols

    def _score(self,
               encoder_outputs,
               decoder_state):

        return NotImplementedError


class BahdanauAttentionRNNDecoder(AttentionRNNDecoder):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is based on:

        https://arxiv.org/pdf/1409.0473.pdf

    The computational path of the method differs from the Luong style, since
    here the context vector contributes to the calculation of the hidden state, by
    concatenating the context with the input of the recurrent unit.

        h(t-1) -> a(t) -> c(t) -> h(t)
    """

    def __init__(self, parameter_setter):
        """
        An instance of a decoder, using Bahdanau-style attention mechanism.
        :param parameter_setter: required parameters for the setter object.
            :parameter hidden_size: int, size of recurrent layer of the LSTM/GRU.
            :parameter embedding_size: int, dimension of the word embeddings.
            :parameter output_size: int, size of the (vocabulary) output layer of the decoder.
            :parameter recurrent_layer: str, name of the recurrent layer ('GRU', 'LSTM').
            :parameter num_layers: int, number of stacked RNN layers.
            :parameter learning_rate: float, learning rate.
            :parameter max_length: int, maximum length of the sequence decoding.
            :parameter use_cuda: bool, True if the device has cuda support.
            :parameter tf_ratio: float, teacher forcing ratio.
        """
        super().__init__(parameter_setter=parameter_setter+{'_input_size': '_hidden_size'})

        self._attention_layer = None
        self._projection_layer = None
        self._transformer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        :return: Decoder, with initialized parameters.
        """
        super().init_parameters()

        self._attention_layer = nn.Linear(self._hidden_size.value * 2, self._hidden_size.value)
        self._projection_layer = nn.Linear(self._hidden_size.value + self._embedding_size.value,
                                           self._hidden_size.value)

        tr = torch.rand(self._hidden_size.value, 1)

        if self._use_cuda:
            self._attention_layer = self._attention_layer.cuda()
            self._projection_layer = self._projection_layer.cuda()

            tr = tr.cuda()

        self._transformer = nn.Parameter(tr)
        return self

    def _decode(self,
                decoder_input,
                hidden_state,
                encoder_outputs,
                batch_size,
                sequence_length):
        """
        A decode step for the attentional decoder module. The recurrent layer of the decoder also does
        it's computation at the invocation of this method.
        :param decoder_input: Variable, with size of (batch_size, 1), containing word ids for step t.
        :param hidden_state: Variable, with size of (num_layers * directions, batch_size, hidden_size).
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param batch_size: int, size of the input batches.
        :param sequence_length: int, size of the sequence of the input batch.
        :return output: Variable, (batch_size, 1, vocab_size) distribution of probabilities over the words.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final state at time t.
        :return attn_weights: Variable, (batch_size, 1, sequence_length) attention weights for visualization.
        """
        embedded_input = self.embedding(decoder_input)
        previous_state = hidden_state[0][-1] if isinstance(hidden_state, tuple) else hidden_state[-1]

        context, attn_weights = self._calculate_context(decoder_state=previous_state,
                                                        encoder_outputs=encoder_outputs,
                                                        batch_size=batch_size,
                                                        sequence_length=sequence_length)

        concat_input = self._projection_layer(torch.cat((embedded_input, context), dim=2))
        output, hidden_state = self._recurrent_layer(concat_input, hidden_state)

        output = functional.log_softmax(self._output_layer(output.contiguous().view(-1, self._hidden_size.value)),
                                        dim=1).view(batch_size, -1, self._output_size.value)

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
        energy = functional.tanh(self._attention_layer(torch.cat((decoder_state, encoder_output), 1)))
        energy = torch.mm(energy, self._transformer)
        return energy


class LuongAttentionRNNDecoder(AttentionRNNDecoder):
    """
    Attention mechanism for the recurrent decoder module. The algorithm is based on:

        https://arxiv.org/pdf/1508.04025.pdf

    The computational path of the method differs from the Bahdanau style, since
    here the context vector contributes to the calculation of the hidden state, after the
    computations of the recurrent layer.

        h(t) -> a(t) -> c(t) -> h*(t)
    """

    def __init__(self, parameter_setter):
        """
        Abstract base class for the Luong style attention mechanisms. The different types of this attention
        essentially have the same computational path, but they differ in the scoring mechanism of the
        similarity between the encoder and decoder states.
        """
        super().__init__(parameter_setter=parameter_setter+{'_input_size': '_embedding_size'})

        self._projection_layer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        :return: Decoder, with initialized parameters.
        """
        super().init_parameters()

        self._projection_layer = nn.Linear(self._hidden_size.value * 2, self._hidden_size.value)

        if self._use_cuda.value:
            self._projection_layer.cuda()

        return self

    def _decode(self,
                decoder_input,
                hidden_state,
                encoder_outputs,
                batch_size,
                sequence_length):
        """
        Decoding for the Luong style attentions. The context is calculated from the hidden state
        of the decoder at time step t, and is merged into the final output layer through a projection
        layer with linear activation.
        :param decoder_input: Variable, with size of (batch_size, 1), containing word ids for step t.
        :param hidden_state: Variable, with size of (num_layers * directions, batch_size, hidden_size).
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param batch_size: int, size of the input batches.
        :param sequence_length: int, size of the sequence of the input batch.
        :return output: Variable, (batch_size, 1, vocab_size) distribution of probabilities over the words.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final state at time t.
        :return attn_weights: Variable, (batch_size, 1, sequence_length) attention weights for visualization.
        """
        embedded_input = self.embedding(decoder_input)
        output, hidden_state = self._recurrent_layer(embedded_input, hidden_state)

        previous_state = hidden_state[0][-1] if isinstance(hidden_state, tuple) else hidden_state[-1]

        context, attn_weights = self._calculate_context(decoder_state=previous_state,
                                                        encoder_outputs=encoder_outputs,
                                                        batch_size=batch_size,
                                                        sequence_length=sequence_length)

        output = self._projection_layer(torch.cat((output, context), dim=2))

        output = functional.log_softmax(self._output_layer(output.contiguous().view(-1, self._hidden_size.value)),
                                        dim=1).view(batch_size, -1, self._output_size.value)

        return output, hidden_state, attn_weights


class GeneralAttentionRNNDecoder(LuongAttentionRNNDecoder):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is a specific case
    of Luong style attention, where the scoring is based off of the linear activation
    of the encoder output, and the dot product of the decoder hidden state with the result of the
    activation from the linear layer.
    """

    def __init__(self, parameter_setter):
        """
        A specific case of Luong style attention.
        :param parameter_setter: required parameters for the setter object.
            :parameter hidden_size: int, size of recurrent layer of the LSTM/GRU.
            :parameter embedding_size: int, dimension of the word embeddings.
            :parameter output_size: int, size of the (vocabulary) output layer of the decoder.
            :parameter recurrent_layer: str, name of the recurrent layer ('GRU', 'LSTM').
            :parameter num_layers: int, number of stacked RNN layers.
            :parameter learning_rate: float, learning rate.
            :parameter max_length: int, maximum length of the sequence decoding.
            :parameter use_cuda: bool, True if the device has cuda support.
            :parameter tf_ratio: float, teacher forcing ratio.
        """
        super().__init__(parameter_setter=parameter_setter)

        self.__attention_layer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        :return: Decoder, with initialized parameters.
        """
        super().init_parameters()

        self.__attention_layer = nn.Linear(self._hidden_size.value, self._hidden_size.value)

        if self._use_cuda.value:
            self.__attention_layer = self.__attention_layer.cuda()

        return self

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


class DotAttentionRNNDecoder(LuongAttentionRNNDecoder):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is a specific case
    of Luong style attention, where the scoring is based off of only the dot product of the
    encoder and decoder states.
    """

    def __init__(self, parameter_setter):
        """
        A specific case of Luong style attention.
        :param parameter_setter: required parameters for the setter object.
            :parameter hidden_size: int, size of recurrent layer of the LSTM/GRU.
            :parameter embedding_size: int, dimension of the word embeddings.
            :parameter output_size: int, size of the (vocabulary) output layer of the decoder.
            :parameter recurrent_layer: str, name of the recurrent layer ('GRU', 'LSTM').
            :parameter num_layers: int, number of stacked RNN layers.
            :parameter learning_rate: float, learning rate.
            :parameter max_length: int, maximum length of the sequence decoding.
            :parameter use_cuda: bool, True if the device has cuda support.
            :parameter tf_ratio: float, teacher forcing ratio.
        """
        super().__init__(parameter_setter=parameter_setter)

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        :return: Decoder, with initialized parameters.
        """
        super().init_parameters()

        return self

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


class ConcatAttentionRNNDecoder(LuongAttentionRNNDecoder):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is a specific case
    of Luong style attention, where the scoring is based off of the concatenation of the encoder
    and decoder states, which is then passed through a non-linear layer with tanh activation.
    The result is then multiplied by a vector to transform the final result to the correct size.
    The scoring of similarity between encoder and decoder states is essentially the same as Bahdanau's
    method, however the computation path follows the Luong style.
    """
    def __init__(self, parameter_setter):
        """
        A specific case of Luong style attention.
        :param parameter_setter: required parameters for the setter object.
            :parameter hidden_size: int, size of recurrent layer of the LSTM/GRU.
            :parameter embedding_size: int, dimension of the word embeddings.
            :parameter output_size: int, size of the (vocabulary) output layer of the decoder.
            :parameter recurrent_layer: str, name of the recurrent layer ('GRU', 'LSTM').
            :parameter num_layers: int, number of stacked RNN layers.
            :parameter learning_rate: float, learning rate.
            :parameter max_length: int, maximum length of the sequence decoding.
            :parameter use_cuda: bool, True if the device has cuda support.
            :parameter tf_ratio: float, teacher forcing ratio.
        """
        super().__init__(parameter_setter=parameter_setter)

        self.__attention_layer = None
        self.__transformer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        :return: Decoder, with initialized parameters.
        """
        super().init_parameters()

        self.__attention_layer = nn.Linear(self._hidden_size.value * 2, self._hidden_size.value)

        tr = torch.rand(self._hidden_size.value, 1)

        if self._use_cuda.value:
            self.__attention_layer = self.__attention_layer.cuda()

            tr = tr.cuda()

        self.__transformer = nn.Parameter(tr)

        return self

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
