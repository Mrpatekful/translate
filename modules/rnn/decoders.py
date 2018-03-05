from modules.base.decoder import Decoder

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.autograd as autograd

from utils.utils import Parameter

import numpy as np


class RNNDecoder(Decoder):
    """
    An implementation of recurrent decoder unit for the sequence to sequence model.
    """

    _param_dict = {
        'hidden_size': Parameter(name='_hidden_size',       doc='int, size of recurrent layer of the LSTM/GRU.'),
        'embedding_size': Parameter(name='_embedding_size', doc='int, dimension of the word embeddings.'),
        'output_size': Parameter(name='_output_size',       doc='int, size of the output layer of the decoder.'),
        'input_size': Parameter(name='_input_size',         doc='int, size of the input layer of the RNN.'),
        'recurrent_type': Parameter(name='_recurrent_type', doc='str, name of the recurrent layer (GRU, LSTM).'),
        'num_layers': Parameter(name='_num_layers',         doc='int, number of stacked RNN layers.'),
        'learning_rate': Parameter(name='_learning_rate',   doc='float, learning rate.'),
        'max_length': Parameter(name='_max_length',         doc='int, maximum length of the sequence decoding.'),
        'use_cuda': Parameter(name='_use_cuda',             doc='bool, True if the device has cuda support.'),
        'tf_ratio': Parameter(name='_tf_ratio',             doc='float, teacher forcing ratio.')
    }

    def __init__(self, parameter_setter):
        """
        A recurrent decoder module for the sequence to sequence model.
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
        super().__init__()
        self._parameter_setter = parameter_setter

        self._recurrent_layer = None
        self._embedding_layer = None
        self._output_layer = None
        self._optimizer = None

        self._decoder_outputs = {
            'symbols': None,
            'attention': None,
            'loss': None
        }

    def init_parameters(self):
        """
        Calls the parameter setter, which initializes the Parameter type attributes.
        After initialization, the main components of the decoder, which require the previously
        initialized parameter values, are created as well.
        """
        for parameter in self._param_dict:
            self.__dict__[self._param_dict[parameter].name] = self._param_dict[parameter]

        self._parameter_setter(self.__dict__)

        if self._recurrent_type.value == 'LSTM':
            unit_type = torch.nn.LSTM
        else:
            unit_type = torch.nn.GRU

        self._recurrent_layer = unit_type(input_size=self._input_size.value,
                                          hidden_size=self._hidden_size.value,
                                          num_layers=self._num_layers.value,
                                          bidirectional=False,
                                          batch_first=True)

        self._output_layer = nn.Linear(self._hidden_size.value, self._output_size.value)

        if self._use_cuda:
            self._recurrent_layer = self._recurrent_layer.cuda()
            self._output_layer = self._output_layer.cuda()

        return self

    def init_optimizer(self):
        """
        Initializes the optimizer for the decoder.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate.value)

        return self

    def _decode(self,
                decoder_input,
                hidden_state,
                encoder_outputs,
                batch_size,
                sequence_length):
        """
        Decoding of a given input. It can be a single time step or a full sequence as well.
        :param decoder_input: Variable, with size of (batch_size, X), containing word ids for step t.
        :param hidden_state: Variable, with size of (num_layers * directions, batch_size, hidden_size).
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param batch_size: int, size of the input batches.
        :param sequence_length: int, size of the sequence of the input batch.
        :return output: Variable, (batch_size, 1, vocab_size) distribution of probabilities over the words.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final state at time t.
        """
        embedded_input = self.embedding(decoder_input)
        output, hidden_state = self._recurrent_layer(embedded_input, hidden_state)
        output = functional.log_softmax(self._output_layer(output.contiguous().view(-1, self._hidden_size.value)),
                                        dim=1).view(batch_size, -1, self._output_size.value)

        return output, hidden_state, None

    def forward(self,
                targets,
                outputs,
                lengths,
                hidden_state,
                loss_function):
        """
        A forward step of the decoder. Processing can be done with different methods, with or
        without attention mechanism and teacher forcing.
        :param targets: Variable, (batch_size, sequence_length) a batch of word ids.
        :param outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param lengths: Ndarray, an array for storing the real lengths of the sequences in the batch.
        :param hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.
        :param loss_function: loss function of the decoder.
        :return loss: int, loss of the decoding
        :return symbols: Ndarray, the decoded word ids.
        """
        batch_size = targets.size(0)
        sequence_length = targets.size(1)

        self._decoder_outputs['symbols'] = np.zeros((batch_size, sequence_length), dtype='int')
        self._decoder_outputs['loss'] = 0

        use_teacher_forcing = True

        if use_teacher_forcing:
            outputs, hidden_state, _ = self._decode(decoder_input=targets,
                                                    hidden_state=hidden_state,
                                                    encoder_outputs=None,
                                                    batch_size=batch_size,
                                                    sequence_length=None)

            for step in range(sequence_length):
                self._decoder_outputs['symbols'][:, step] = outputs[:, step, :].topk(1)[1]\
                    .squeeze(-1).data.cpu().numpy()

            self._decoder_outputs['loss'] = loss_function(outputs.view(-1, self._output_size.value), targets.view(-1))

        else:
            for step in range(sequence_length):
                step_input = targets[:, step].unsqueeze(-1)
                step_output, hidden_state, _ = self._decode(decoder_input=step_input,
                                                            hidden_state=hidden_state,
                                                            encoder_outputs=None,
                                                            batch_size=batch_size,
                                                            sequence_length=sequence_length)

                self._decoder_outputs['loss'] += loss_function(step_output.squeeze(1), targets[:, step])
                self._decoder_outputs['symbols'][:, step] = step_output.topk(1)[1].data.squeeze(-1).\
                    squeeze(-1).cpu().numpy()

        return self._decoder_outputs

    @classmethod
    def assemble(cls, params):
        """
        Creates the required dictionary of parameters required for the initialization of a decoder object.
        :param params: dict, parameters that describe the decoder object.
        :return: dict, processed and well-formatted parameters for an RNNDecoder object.
        """
        return {
            **{cls._param_dict[param].name: params[param] for param in params if param in cls._param_dict},
            cls._param_dict['embedding_size'].name: params['target_language'].embedding_size,
            cls._param_dict['input_size'].name: params['target_language'].embedding_size,
            cls._param_dict['output_size'].name: params['target_language'].vocab_size
        }

    @classmethod
    def abstract(cls):
        return False

    @property
    def optimizer(self):
        """
        Property for the optimizer of the decoder.
        :return self.__optimizer: Optimizer, the currently used optimizer of the decoder.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Setter for the optimizer of the decoder.
        :param optimizer: Optimizer, instance to be set as the new optimizer for the decoder.
        """
        self._optimizer = optimizer

    @property
    def output_dim(self):
        """
        Property for the output size of the decoder. This is also the size of the vocab.
        :return: int, size of the output at the decoder's final layer.
        """
        return self._output_size.value

    @property
    def embedding(self):
        """
        Property for the decoder's embedding layer.
        :return: The currently used embeddings of the decoder.
        """
        return self._embedding_layer

    @embedding.setter
    def embedding(self, embedding):
        """
        Setter for the decoder's embedding layer.
        :param embedding: Embedding, to be set as the embedding layer of the decoder.
        """
        self._embedding_layer = nn.Embedding(embedding.size(0), embedding.size(1))
        self._embedding_layer.weight = nn.Parameter(embedding)
        self._embedding_layer.weight.requires_grad = False


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
        attn_energies = autograd.Variable(torch.zeros([batch_size, sequence_length]))

        if self._use_cuda.value:
            attn_energies = attn_energies.cuda()

        squeezed_output = decoder_state.squeeze(1)
        for step in range(sequence_length):
            attn_energies[:, step] = self._score(encoder_outputs[:, step], squeezed_output)

        attn_weights = functional.softmax(attn_energies, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        return context, attn_weights

    def forward(self,
                targets,
                outputs,
                lengths,
                hidden_state,
                loss_function):
        """
        An attentional forward step. The calculations can be done with or without teacher fording.
        :param targets: Variable, (batch_size, sequence_length) a batch of word ids.
        :param outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param lengths: Ndarray, an array for storing the real lengths of the sequences in the batch.
        :param hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.
        :param loss_function: loss function of the decoder.
        :return loss: int, loss of the decoding
        :return symbols: Ndarray, the decoded word ids.
        """
        batch_size = targets.size(0)
        sequence_length = targets.size(1)

        self._decoder_outputs['symbols'] = np.zeros((batch_size, sequence_length), dtype='int')
        self._decoder_outputs['attention'] = np.zeros((batch_size, sequence_length, sequence_length))
        self._decoder_outputs['loss'] = 0

        use_teacher_forcing = True

        if use_teacher_forcing:
            for step in range(sequence_length):
                step_input = targets[:, step].unsqueeze(-1)
                step_output, hidden_state, attn_weights = self._decode(decoder_input=step_input,
                                                                       hidden_state=hidden_state,
                                                                       encoder_outputs=outputs,
                                                                       batch_size=batch_size,
                                                                       sequence_length=sequence_length)

                self._decoder_outputs['loss'] += loss_function(step_output.squeeze(1), targets[:, step])
                self._decoder_outputs['attention'][:, step, :] = attn_weights.data.squeeze(1).cpu().numpy()
                self._decoder_outputs['symbols'][:, step] = step_output.topk(1)[1].data.squeeze(-1)\
                    .squeeze(-1).cpu().numpy()

        else:
            for step in range(sequence_length):
                step_input = targets[:, step].unsqueeze(-1)
                step_output, hidden_state, attn_weights = self._decode(decoder_input=step_input,
                                                                       hidden_state=hidden_state,
                                                                       encoder_outputs=outputs,
                                                                       batch_size=batch_size,
                                                                       sequence_length=sequence_length)

                self._decoder_outputs['loss'] += loss_function(step_output.squeeze(1), targets[:, step])
                self._decoder_outputs['attention'][:, step, :] = attn_weights.data.squeeze(1).cpu().numpy()
                self._decoder_outputs['symbols'][:, step] = step_output.topk(1)[1].data.squeeze(-1)\
                    .squeeze(-1).cpu().numpy()

        return self._decoder_outputs

    def _score(self,
               encoder_outputs,
               decoder_state):

        return NotImplementedError

    @classmethod
    def abstract(cls):
        return True


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
        An attentional rnn decoder object.
        """
        super().__init__(parameter_setter=parameter_setter)

        self._attention_layer = None
        self._projection_layer = None
        self._transformer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        """
        super().init_parameters()

        self._attention_layer = nn.Linear(self._hidden_size.value * 2, self._hidden_size.value)
        tr = torch.rand(self._hidden_size.value, 1)

        if self._use_cuda:
            self._attention_layer = self._attention_layer.cuda()
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

        concat_input = torch.cat((embedded_input, context), dim=2)
        output, hidden_state = self._recurrent_layer(concat_input, hidden_state)

        output = functional.log_softmax(self._output_layer(output.contiguous().view(-1, self._hidden_size.value)),
                                        dim=1).view(batch_size, -1, self._output_size.value)

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

    @classmethod
    def assemble(cls, params):
        """
        Formats the provided parameters for a BahdanauAttentionRNNDecoder object.
        :param params: dict, parameters that describe the decoder object.
        :return: dict, processed and well-formatted parameters for an BahdanauAttentionRNNDecoder object.
        """
        return {
            **{cls._param_dict[param].name: params[param] for param in params if param in cls._param_dict},
            cls._param_dict['embedding_size'].name: params['target_language'].embedding_size,
            cls._param_dict['input_size'].name: params['target_language'].embedding_size + params['hidden_size'],
            cls._param_dict['output_size'].name: params['target_language'].vocab_size
        }

    @classmethod
    def abstract(cls):
        return False


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
        super().__init__(parameter_setter=parameter_setter)

        self._attention_layer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        """
        super().init_parameters()

        self._attention_layer = nn.Linear(self._hidden_size.value, self._hidden_size.value)

        if self._use_cuda.value:
            self._attention_layer = self._attention_layer.cuda()

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

    @classmethod
    def abstract(cls):
        return False


class DotAttentionRNNDecoder(LuongAttentionRNNDecoder):
    """
    Global attention mechanism for the recurrent decoder module. The algorithm is a specific case
    of Luong style attention, where the scoring is based off of only the dot product of the
    encoder and decoder states.
    """

    def __init__(self, parameter_setter):
        super().__init__(parameter_setter=parameter_setter)

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        """
        super().init_parameters()

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

    @classmethod
    def abstract(cls):
        return False


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
        super().__init__(parameter_setter=parameter_setter)

        self._attention_layer = None
        self._transformer = None

    def init_parameters(self):
        """
        Initializes the parameters for the decoder.
        """
        super().init_parameters()

        self._attention_layer = nn.Linear(self._hidden_size.value * 2, self._hidden_size.value)

        tr = torch.rand(self._hidden_size.value, 1)

        if self._use_cuda.value:
            self._attention_layer = self._attention_layer.cuda()

            tr = tr.cuda()

        self._transformer = nn.Parameter(tr)

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

    @classmethod
    def abstract(cls):
        return False
