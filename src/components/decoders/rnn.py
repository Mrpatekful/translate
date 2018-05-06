"""

"""

import numpy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional

from src.components.base import Decoder
from src.components.utils.utils import Optimizer

from src.utils.analysis import AttentionData

from src.utils.utils import ParameterSetter, subtract_dict, Interface


class RNNDecoder(Decoder):
    """
    An implementation of recurrent decoder unit for the sequence to sequence model.
    """

    interface = Interface(**{
        'hidden_size':      (0, None),
        'recurrent_type':   (1, None),
        'num_layers':       (2, None),
        'optimizer_type':   (3, None),
        'learning_rate':    (4, None),
        'max_length':       (5, None),
        'cuda':             (6, 'Experiment:Policy:cuda$'),
        'embedding_size':   (7, 'embedding_size$'),
        'input_size':       (8, 'embedding_size$')
    })

    abstract = False

    @ParameterSetter.pack(interface)
    def __init__(self, parameter_setter):
        """
        A recurrent decoder module for the sequence to sequence model.
        :param parameter_setter: required parameters for the setter object.
            -:parameter hidden_size: int, size of recurrent layer of the LSTM/GRU.
            -:parameter embedding_size: int, dimension of the word embeddings.
            -:parameter recurrent_layer: str, name of the recurrent layer ('GRU', 'LSTM').
            -:parameter num_layers: int, number of stacked RNN layers.
            -:parameter learning_rate: float, learning rate.
            -:parameter max_length: int, maximum length of the sequence decoding.
            -:parameter use_cuda: bool, True if the device has cuda support.
            -:parameter tf_ratio: float, teacher forcing ratio.
        """
        super().__init__()
        self._parameter_setter = parameter_setter

        self._recurrent_layer = None
        self._optimizer = None
        self._tokens = None

        self._outputs = {
            'symbols':    None,
        }

        self.embedding = None
        self.output_layer = None

        self._recurrent_parameters = None

    def init_parameters(self):
        """
        Calls the parameter setter, which initializes the Parameter type attributes.
        After initialization, the main components of the decoder, which require the previously
        initialized parameter values, are created as well.
        """
        self._parameter_setter.initialize(self)

        if self._recurrent_type == 'LSTM':
            unit_type = torch.nn.LSTM
        elif self._recurrent_type == 'GRU':
            unit_type = torch.nn.GRU
        else:
            raise ValueError('Invalid recurrent unit type.')

        self._output_size = self._hidden_size

        self._recurrent_layer = unit_type(input_size=self._input_size,
                                          hidden_size=self._hidden_size,
                                          num_layers=self._num_layers,
                                          dropout=0.5,
                                          bidirectional=False,
                                          batch_first=True)

        if self._cuda:
            self._recurrent_layer = self._recurrent_layer.cuda()

        self._recurrent_parameters = [name for name, _ in self.named_parameters()]

        return self

    def init_optimizer(self):
        """
        Initializes the optimizer for the decoder.
        """
        parameters = [param for name, param in self.named_parameters() if name in self._recurrent_parameters]
        self._optimizer = Optimizer(parameters=parameters,
                                    optimizer_type=self._optimizer_type,
                                    scheduler_type='ReduceLROnPlateau',
                                    learning_rate=self._learning_rate)

        return self

    def _decode(self,
                inputs,
                hidden_state,
                encoder_outputs,
                batch_size,
                sequence_length):
        """
        Decoding of a given input. It can be a single time step or a full sequence as well.
        :param inputs: Variable, with size of (batch_size, X), containing word ids for time step t.
        :param hidden_state: Variable, with size of (num_layers * directions, batch_size, hidden_size).
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size). This
                                parameter is redundant for the standard type of decoding.
        :param batch_size: int, size of the input batches.
        :param sequence_length: int, length of the sequence of the input batch.
        :return output: Variable, (batch_size, 1, vocab_size) result of the decoding, which is a vector, that
                        provides a probability distribution over the words of the vocabulary.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final state at time t.
        """
        output, hidden_state = self._recurrent_layer(inputs, hidden_state)
        output = functional.log_softmax(self.output_layer(output.contiguous().view(-1, self._hidden_size)),
                                        dim=1).view(batch_size, -1, self.output_layer.size)

        return output, hidden_state, None

    def forward(self,
                encoder_outputs,
                hidden_state,
                targets=None,
                max_length=None):
        """
        Forward step of the decoder unit. If the targets parameter is not None, then teacher forcing is used,
        so during decoding, the previous output word will be provided at time step t. If targets is None, decoding
        follows the general method, when the input word for the recurrent unit at time step t, is the output word at
        time step t-1.
        :param targets: Variable, (batch_size, sequence_length) a batch of word ids. If None, then normal teacher
                        forcing is not applied.
        :param max_length: int, maximum length of the decoded sequence. If None, the maximum length parameter from
                           the configuration file will be used as maximum length. This parameter has no effect, if
                           targets parameter is provided, because in that case, the length of the target sequence
                           will be decoding length.
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size). This parameter
                                is redundant for the standard decoder unit.
        :param hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.
        :return decoder_outputs: dict, containing two string keys, symbols: Ndarray, the decoded word ids.
        """
        batch_size = encoder_outputs.size(0)

        if targets is not None:

            predictions = self._forced_decode(targets=targets,
                                              batch_size=batch_size,
                                              hidden_state=hidden_state,
                                              encoder_outputs=encoder_outputs,
                                              input_sequence_length=None)

        else:

            predictions = self._predictive_decode(max_length=max_length,
                                                  batch_size=batch_size,
                                                  hidden_state=hidden_state,
                                                  encoder_outputs=encoder_outputs,
                                                  input_sequence_length=None)

        return self._outputs, predictions

    def _forced_decode(self,
                       targets,
                       batch_size,
                       hidden_state,
                       encoder_outputs,
                       input_sequence_length):
        """
        This method is primarily used during training, when target outputs are provided to the decoder.
        These target sequences start with an <SOS> token, which will serve as the first input to the _decode
        function. During the decoding iterations the decoder's predictions will only be used as final outputs to
        measure the loss, so the input for the (t)-th time step will be the (t-1)-th element of the provided
        targets.
        :param targets:  Variable, (batch_size, sequence_length) a batch of word ids.
        :param batch_size: int, size of the currently processed batch.
        :param hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param input_sequence_length:This parameter is required only by the attentional version of this method.
        """
        output_sequence_length = targets.size(1) - 1

        self._outputs['symbols'] = numpy.zeros((batch_size, output_sequence_length), dtype=numpy.int32)

        predictions = []

        inputs = targets[:, :-1].contiguous()
        embedded_inputs = self.embedding(inputs)

        outputs, hidden_state, _ = self._decode(inputs=embedded_inputs,
                                                hidden_state=hidden_state,
                                                encoder_outputs=None,
                                                batch_size=batch_size,
                                                sequence_length=None)

        for step in range(output_sequence_length):
            self._outputs['symbols'][:, step] = outputs[:, step, :].topk(1)[1].squeeze(-1).data.cpu().numpy()
            predictions.append(outputs[:, step, :])

        return predictions

    def _predictive_decode(self,
                           max_length,
                           batch_size,
                           hidden_state,
                           encoder_outputs,
                           input_sequence_length):
        """
        This method is used during unforced training and evaluation. Targets are not provided, so the decoding
        will go on, until the length of the output reaches the given max_length, or the _max_length attribute.
        During the decoding iteration of the sequence, the input word of the decoder at time step (t) is the
        output word of the decoder from the time step (t-1).
        :param max_length: int, maximum length for the decoded sequence. If None the max_length parameter's
                           value will be used.
        :param batch_size: int, size of the currently processed batch.
        :param hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param input_sequence_length: This parameter is required only by the attentional version of this method.
        """
        output_sequence_length = max_length if max_length is not None else self._max_length
        self._outputs['symbols'] = numpy.zeros((batch_size, output_sequence_length), dtype='int')

        predictions = []

        sos_tokens = torch.from_numpy(numpy.array([self.tokens['<SOS>']] * batch_size)).unsqueeze(-1)
        if self._cuda:
            sos_tokens = sos_tokens.cuda()

        step_input = autograd.Variable(sos_tokens)

        for step in range(output_sequence_length):
            embedded_input = self.embedding(step_input)

            step_output, hidden_state, _ = self._decode(inputs=embedded_input,
                                                        hidden_state=hidden_state,
                                                        encoder_outputs=None,
                                                        batch_size=batch_size,
                                                        sequence_length=None)

            symbol_output = step_output.topk(1)[1].data.squeeze(-1)

            self._outputs['symbols'][:, step] = symbol_output.squeeze(-1).cpu().numpy()
            predictions.append(step_output.squeeze(1))

            step_input = symbol_output

        return predictions

    @property
    def tokens(self):
        """
        Property for the tokens of the decoder.
        """
        return self._tokens

    @tokens.setter
    def tokens(self, tokens):
        """
        Setter function for the tokens of the decoder.
        :param tokens: dict, containing the keys <UNK>, <EOS> and <SOS> tokens with their current ids as values.
        :raises ValueError: the dict doesn't contain the expected tokens.
        """
        try:

            self._tokens = {
                '<UNK>': tokens['<UNK>'],
                '<EOS>': tokens['<EOS>'],
                '<SOS>': tokens['<SOS>']
            }

        except KeyError as error:
            raise ValueError(f'{error} was not provided for the decoder.')

    @property
    def output_types(self):
        return {}

    @property
    def optimizers(self):
        """
        Property for the optimizers of the decoder.
        :return: list, optimizers used by the decoder.
        """
        return [self._optimizer]

    @property
    def state(self):
        """
        Property for the optimizers of the decoder.
        :return: list, optimizers used by the decoder.
        """
        return {
            'weights':      {k: v for k, v in self.state_dict().items() if k in self._recurrent_parameters},
            'optimizer':    self._optimizer.state
        }

    @state.setter
    def state(self, state):
        """
        Property for the state of the decoder.
        :return: dict, containing the state of the weights, and the optimizer.
        """
        parameters = {k: v for k, v in state['weights'].items() if k in self._recurrent_parameters}
        self.load_state_dict(parameters, strict=False)
        self._optimizer.state = state['optimizer']


class AttentionRNNDecoder(RNNDecoder):
    """
    Abstract base class for the attentional variation of recurrent decoder unit.
    """

    abstract = True

    def __init__(self, parameter_setter):
        """
        An abstract base of a recurrent decoder with attention.
        """
        super().__init__(parameter_setter=parameter_setter)

        self._outputs = {
            **self._outputs,
            'alignment_weights':   None,
        }

        self._attention_parameters = None
        self._attention_optimizer = None

    def init_optimizer(self):
        """

        """
        super().init_optimizer()

        parameters = [param for name, param in self.named_parameters() if name in self._attention_parameters]
        self._attention_optimizer = Optimizer(parameters=parameters,
                                              optimizer_type=self._optimizer_type,
                                              scheduler_type='ReduceLROnPlateau',
                                              learning_rate=self._learning_rate)

        return self

    def _calculate_context(self,
                           decoder_state,
                           encoder_outputs,
                           batch_size,
                           sequence_length):
        """
        Calculates the context for the decoder, given the encoder outputs and a decoder hidden state.
        The algorithm iterates through the encoder outputs and scores each output based on the similarity
        with the decoder state. The scoring functions are implemented in the child nodes of this class.

        Args:
            decoder_state:
                Variable, (batch_size, 1, hidden_size) the state of the decoder.

            encoder_outputs:
                Variable, (batch_size, sequence_length, hidden_size) the output of
                each time step of the encoder.

            batch_size:
                int, size of the input batch.

            sequence_length:
                int, size of the sequence.

        Returns:
            context:
                Variable, the weighted sum of the encoder outputs.

            attn_weights:
                Variable, weights used in the calculation of the context.
        """
        attn_energies = torch.zeros([batch_size, sequence_length])

        if self._cuda:
            attn_energies = attn_energies.cuda()

        attn_energies = autograd.Variable(attn_energies)

        squeezed_output = decoder_state.squeeze(1)
        for step in range(sequence_length):
            attn_energies[:, step] = self._score(encoder_outputs[:, step], squeezed_output)

        attn_weights = functional.softmax(attn_energies, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        return context, attn_weights

    def forward(self,
                encoder_outputs,
                hidden_state,
                targets=None,
                max_length=None):
        """
        Forward step of the attentional decoder unit. If the targets parameter is not None, then teacher forcing is
        used, so during decoding, the previous output word will be provided at time step t. If targets is None, decoding
        follows the general method, when the input word for the recurrent unit at time step t, is the output word at
        time step t-1.

        Args:
            targets:
                Variable, (batch_size, sequence_length) a batch of word ids.

            max_length:
                int, maximum length for the decoded sequence. If None the max_length parameter's
                value will be used.

            encoder_outputs:
                Variable, with size of (batch_size, sequence_length, hidden_size).

            hidden_state:
                Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.

        Returns:
            outputs: dict, containing three string keys, symbols: Ndarray, the decoded word ids,
                     alignment_weights:
        """
        batch_size = encoder_outputs.size(0)
        input_sequence_length = encoder_outputs.size(1)

        if targets is not None:

            predictions = self._forced_decode(targets=targets,
                                              batch_size=batch_size,
                                              hidden_state=hidden_state,
                                              encoder_outputs=encoder_outputs,
                                              input_sequence_length=input_sequence_length)

        else:

            predictions = self._predictive_decode(max_length=max_length,
                                                  batch_size=batch_size,
                                                  hidden_state=hidden_state,
                                                  encoder_outputs=encoder_outputs,
                                                  input_sequence_length=input_sequence_length)

        return self._outputs, predictions

    def _forced_decode(self,
                       targets,
                       batch_size,
                       hidden_state,
                       encoder_outputs,
                       input_sequence_length):
        """
        This method is primarily used during training, when target outputs are provided to the decoder.
        These target sequences start with an <SOS> token, which will serve as the first input to the _decode
        function. During the decoding iterations the decoder's predictions will only be used as final outputs to
        measure the loss, so the input for the (t)-th time step will be the (t-1)-th element of the provided
        targets.

        Args:
            targets:
                Variable, (batch_size, sequence_length) a batch of word ids.

            batch_size:
                int, size of the currently processed batch.

            hidden_state:
                Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.

            encoder_outputs:
                Variable, with size of (batch_size, sequence_length, hidden_size).

            input_sequence_length:
                int, length of the input (for the encoder) sequence.
        """
        output_sequence_length = targets.size(1) - 1

        inputs = targets[:, :-1].contiguous()
        embedded_inputs = self.embedding(inputs)

        predictions = []

        self._outputs['symbols'] = numpy.zeros((batch_size, output_sequence_length), dtype='int')
        self._outputs['alignment_weights'] = numpy.zeros((batch_size, output_sequence_length, input_sequence_length))

        for step in range(output_sequence_length):
            step_input = embedded_inputs[:, step, :]
            step_input = step_input.unsqueeze(1)
            step_output, hidden_state, attn_weights = self._decode(inputs=step_input,
                                                                   hidden_state=hidden_state,
                                                                   encoder_outputs=encoder_outputs,
                                                                   batch_size=batch_size,
                                                                   sequence_length=input_sequence_length)

            predictions.append(step_output.squeeze(1))
            self._outputs['alignment_weights'][:, step, :] = attn_weights.data.squeeze(1).cpu().numpy()
            self._outputs['symbols'][:, step] = step_output.topk(1)[1].data.squeeze(-1).squeeze(-1).cpu().numpy()

        return predictions

    def _predictive_decode(self,
                           max_length,
                           batch_size,
                           hidden_state,
                           encoder_outputs,
                           input_sequence_length):
        """
        This method is used during unforced training and evaluation. Targets are not provided, so the decoding
        will go on, until the length of the output reaches the given max_length, or the _max_length attribute.
        During the decoding iteration of the sequence, the input word of the decoder at time step (t) is the
        output word of the decoder from the time step (t-1).

        Args:
            max_length:
                int, maximum length for the decoded sequence. If None the max_length parameter's
                value will be used.

            batch_size:
                int, size of the currently processed batch.

            hidden_state:
                Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.

            encoder_outputs:
                Variable, with size of (batch_size, sequence_length, hidden_size).

            input_sequence_length:
                int, length of the input (for the encoder) sequence.
        """
        output_sequence_length = max_length if max_length is not None else self._max_length

        predictions = []

        self._outputs['symbols'] = numpy.zeros((batch_size, output_sequence_length), dtype='int')
        self._outputs['alignment_weights'] = numpy.zeros((batch_size, output_sequence_length, input_sequence_length))

        sos_tokens = torch.from_numpy(numpy.array([self.tokens['<SOS>']] * batch_size)).unsqueeze(-1)

        if self._cuda:
            sos_tokens = sos_tokens.cuda()

        step_input = autograd.Variable(sos_tokens)

        for step in range(output_sequence_length):
            embedded_input = self.embedding(step_input)

            step_output, hidden_state, attn_weights = self._decode(inputs=embedded_input,
                                                                   hidden_state=hidden_state,
                                                                   encoder_outputs=encoder_outputs,
                                                                   batch_size=batch_size,
                                                                   sequence_length=input_sequence_length)

            symbol_output = step_output.topk(1)[1].data.squeeze(-1)

            predictions.append(step_output.squeeze(1))
            self._outputs['alignment_weights'][:, step, :] = attn_weights.data.squeeze(1).cpu().numpy()
            self._outputs['symbols'][:, step] = symbol_output.squeeze(-1).cpu().numpy()

            step_input = symbol_output

        return predictions

    def _score(self, encoder_outputs, decoder_state):
        raise NotImplementedError

    @property
    def output_types(self):
        return {
            'attention': AttentionData
        }

    @property
    def optimizers(self):
        return [
            self._optimizer,
            self._attention_optimizer
        ]

    @property
    def state(self):
        """
        Property for the optimizers of the decoder.
        :return: list, optimizers used by the decoder.
        """
        return {
            'weights':              {k: v for k, v in self.state_dict().items()
                                     if k in self._recurrent_parameters or k in self._attention_parameters},
            'optimizer':            self._optimizer.state,
            'attention_optimizer':  self._attention_optimizer.state
        }

    @state.setter
    def state(self, state):
        """
        Property for the state of the decoder.
        """
        parameters = {k: v for k, v in state['weights'].items()
                      if k in self._recurrent_parameters or k in self._attention_parameters}
        self.load_state_dict(parameters, strict=False)
        self._optimizer.state = state['optimizer']
        self._attention_optimizer.state = state['attention_optimizer']


class BahdanauAttentionRNNDecoder(AttentionRNNDecoder):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is based on:

        https://arxiv.org/pdf/1409.0473.pdf

    The computational path of the method differs from the Luong style, since
    the context vector contributes to the calculation of the hidden state as well, created
    by the recurrent unit at time step t.

        h(t-1) -> a(t) -> c(t) -> h(t)

    The attention weights are derived from the similarity scores of the previous recurrent hidden
    state (from time step t-1) and the encoder outputs. The created context vector is then merged with
    the output of the recurrent unit as well, to get the final output of a softmax layer, providing the
    probability distribution over the word ids.
    """

    interface = Interface(**{
        **subtract_dict(RNNDecoder.interface.dictionary, {'input_size': None}),
        'input_size': (Interface.last_key(subtract_dict(RNNDecoder.interface.dictionary, {'input_size': None})) + 1,
                       'Decoder:embedding_size$ + Decoder:hidden_size$')
    })

    abstract = False

    @ParameterSetter.pack(interface)
    def __init__(self, parameter_setter):
        """
        An a bahdanau-type attentional RNN decoder instance.
        """
        super().__init__(parameter_setter=parameter_setter)

        self._projection_layer = None
        self._attention_layer = None
        self._transformer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        """
        super().init_parameters()

        self._attention_layer = nn.Linear(self._hidden_size * 2, self._hidden_size)
        self._projection_layer = nn.Linear(self._hidden_size + self._embedding_size, self._hidden_size)

        tr = torch.rand(self._hidden_size, 1)

        if self._cuda:
            self._attention_layer = self._attention_layer.cuda()
            self._projection_layer = self._projection_layer.cuda()
            tr = tr.cuda()

        self._transformer = nn.Parameter(tr)

        self._attention_parameters = [name for name, _ in self.named_parameters()
                                      if name not in self._recurrent_parameters]

        return self

    def _decode(self,
                inputs,
                hidden_state,
                encoder_outputs,
                batch_size,
                sequence_length):
        """
        A decode step for the bahdanau variation of attentional decoder module. The computations are described
        in header docstring of the class.
        :param inputs: Variable, with size of (batch_size, 1), containing word ids for step t.
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

        concat_input = torch.cat((inputs, context), dim=2)
        output, hidden_state = self._recurrent_layer(concat_input, hidden_state)

        output = self._projection_layer(torch.cat((inputs, output), dim=2))

        output = functional.log_softmax(self.output_layer(output.contiguous().view(-1, self._hidden_size)),
                                        dim=1).view(batch_size, -1, self.output_layer.size)

        return output, hidden_state, attn_weights

    def _score(self, encoder_output, decoder_state):
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
        super().__init__(parameter_setter=parameter_setter)

        self._projection_layer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        """
        super().init_parameters()

        self._projection_layer = nn.Linear(self._hidden_size * 2, self._hidden_size)

        if self._cuda:
            self._projection_layer = self._projection_layer.cuda()

        return self

    def _decode(self,
                inputs,
                hidden_state,
                encoder_outputs,
                batch_size,
                sequence_length):
        """
        Decoding for the Luong style attentions. The context is calculated from the hidden state
        of the decoder at time step t, and is merged into the final output layer through a projection
        layer with linear activation.
        :param inputs: Variable, with size of (batch_size, 1), containing word ids for step t.
        :param hidden_state: Variable, with size of (num_layers * directions, batch_size, hidden_size).
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param batch_size: int, size of the input batches.
        :param sequence_length: int, size of the sequence of the input batch.
        :return output: Variable, (batch_size, 1, vocab_size) distribution of probabilities over the words.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final state at time t.
        :return attn_weights: Variable, (batch_size, 1, sequence_length) attention weights for visualization.
        """
        self._recurrent_layer.flatten_parameters()

        output, hidden_state = self._recurrent_layer(inputs, hidden_state)

        previous_state = hidden_state[0][-1] if isinstance(hidden_state, tuple) else hidden_state[-1]

        context, attn_weights = self._calculate_context(decoder_state=previous_state,
                                                        encoder_outputs=encoder_outputs,
                                                        batch_size=batch_size,
                                                        sequence_length=sequence_length)

        output = self._projection_layer(torch.cat((output, context), dim=2))

        output = functional.log_softmax(self.output_layer(output.contiguous().view(-1, self._hidden_size)),
                                        dim=1).view(batch_size, -1, self.output_layer.size)

        return output, hidden_state, attn_weights

    def _score(self, encoder_outputs, decoder_state):
        raise NotImplementedError


class GeneralAttentionRNNDecoder(LuongAttentionRNNDecoder):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is a specific case
    of Luong style attention, where the scoring is based off of the linear activation
    of the encoder output, and the dot product of the decoder hidden state with the result of the
    activation from the linear layer.
    """

    interface = RNNDecoder.interface

    abstract = False

    @ParameterSetter.pack(interface)
    def __init__(self, parameter_setter):
        super().__init__(parameter_setter=parameter_setter)

        self._attention_layer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        """
        super().init_parameters()

        self._attention_layer = nn.Linear(self._hidden_size, self._hidden_size)

        if self._cuda:
            self._attention_layer = self._attention_layer.cuda()

        self._attention_parameters = [name for name, _ in self.named_parameters()
                                      if name not in self._recurrent_parameters]

        return self

    def _score(self, encoder_output, decoder_state):
        """
        The score computation is as follows:

            h_d * (W_a * h_eT)

        where h_d is the decoder hidden state, W_a is a linear layer and h_eT is
        the transpose of encoder output at time step t.
        :param encoder_output: Variable, (batch_size, 1, hidden_layer) output of the encoder at time step t.
        :param decoder_state: Variable, (batch_size, 1, hidden_layer) hidden state of the decoder at time step t.
        :return energy: Variable, similarity between the decoder and encoder state.
        """
        energy = self._attention_layer(encoder_output)
        energy = torch.bmm(decoder_state.unsqueeze(1), energy.unsqueeze(1).transpose(1, 2)).squeeze(-1)

        return energy


class DotAttentionRNNDecoder(LuongAttentionRNNDecoder):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is a specific case
    of Luong style attention, where the scoring is based off of only the dot product of the
    encoder and decoder states.
    """

    interface = RNNDecoder.interface

    abstract = False

    @ParameterSetter.pack(interface)
    def __init__(self, parameter_setter):
        super().__init__(parameter_setter=parameter_setter)

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        """
        super().init_parameters()

        self._attention_parameters = [name for name, _ in self.named_parameters()
                                      if name not in self._recurrent_parameters]

        return self

    def _score(self, encoder_output, decoder_state):
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

    interface = RNNDecoder.interface

    abstract = False

    @ParameterSetter.pack(interface)
    def __init__(self, parameter_setter):
        super().__init__(parameter_setter=parameter_setter)

        self._attention_layer = None
        self._transformer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        """
        super().init_parameters()

        self._attention_layer = nn.Linear(self._hidden_size * 2, self._hidden_size)

        tr = torch.rand(self._hidden_size, 1)

        if self._cuda:
            self._attention_layer = self._attention_layer.cuda()

            tr = tr.cuda()

        self._transformer = nn.Parameter(tr)

        self._attention_parameters = [name for name, _ in self.named_parameters()
                                      if name not in self._recurrent_parameters]

        return self

    def _score(self, encoder_output, decoder_state):
        """
        The score computation is as follows:

            v_t * tanh(W_a * [h_d ; h_e])

        where v_t is a vector, that transform the output to the correct size, h_d is the decoder hidden state,
        W_a is a weight matrix and h_e is the encoder output at time step t.
        :param encoder_output: Variable, (batch_size, 1, hidden_layer) output of the encoder at time step t.
        :param decoder_state: Variable, (batch_size, 1, hidden_layer) hidden state of the decoder at time step t.
        :return energy: Variable, similarity between the decoder and encoder state.
        """
        energy = functional.tanh(self._attention_layer(torch.cat((decoder_state, encoder_output), 1)))
        energy = torch.mm(energy, self._transformer)

        return energy
