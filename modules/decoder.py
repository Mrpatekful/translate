import torch
from torch import nn
from torch.nn import functional
from utils.utils import Parameter


import numpy as np


class RNNDecoder(nn.Module):
    """
    Decoder module of the sequence to sequence model.
    """

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
        super(RNNDecoder, self).__init__()
        self._parameter_setter = parameter_setter

        self._hidden_size = Parameter(name='_hidden_size',       doc='int, size of recurrent layer of the LSTM/GRU.')
        self._embedding_size = Parameter(name='_embedding_size', doc='int, dimension of the word embeddings.')
        self._output_size = Parameter(name='_output_size',       doc='int, size of the output layer of the decoder.')
        self._input_size = Parameter(name='_input_size',         doc='int, size of the input layer of the RNN.')
        self._recurrent_type = Parameter(name='_recurrent_type', doc='str, name of the recurrent layer (GRU, LSTM).')
        self._num_layers = Parameter(name='_num_layers',         doc='int, number of stacked RNN layers.')
        self._learning_rate = Parameter(name='_learning_rate',   doc='float, learning rate.')
        self._max_length = Parameter(name='_max_length',         doc='int, maximum length of the sequence decoding.')
        self._use_cuda = Parameter(name='_use_cuda',             doc='bool, True if the device has cuda support.')
        self._tf_ratio = Parameter(name='_tf_ratio',             doc='float, teacher forcing ratio.')

        self._recurrent_layer = None
        self._embedding_layer = None
        self._output_layer = None
        self._optimizer = None

    def init_parameters(self):
        """
        Calls the parameter setter, which initializes the Parameter type attributes.
        After initialization, the main components of the decoder, which require the previously
        initialized parameter values, are created as well.
        """
        self._parameter_setter(self.__dict__)

        if self._recurrent_type.value == 'LSTM':
            unit_type = torch.nn.LSTM
        else:
            unit_type = torch.nn.GRU

        try:
            input_size = self._input_size.value
        except ValueError:
            input_size = self._embedding_size.value

        self._recurrent_layer = unit_type(input_size=input_size,
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
        :return attn_weights: Variable, (batch_size, 1, sequence_length) attention weights for visualization.
        """
        embedded_input = self.embedding(decoder_input)
        output, hidden_state = self._recurrent_layer(embedded_input, hidden_state)
        output = functional.log_softmax(self._output_layer(output.contiguous().view(-1, self._hidden_size.value)),
                                        dim=1).view(batch_size, -1, self._output_size.value)

        return output, hidden_state

    def forward(self,
                inputs,
                encoder_outputs,
                lengths,
                hidden_state,
                loss_function,
                tf_ratio):
        """
        A forward step of the decoder. Processing can be done with different methods, with or
        without attention mechanism and teacher forcing.
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
            outputs, hidden_state = self._decode(decoder_input=inputs,
                                                 hidden_state=hidden_state,
                                                 encoder_outputs=None,
                                                 batch_size=batch_size,
                                                 sequence_length=None)

            for step in range(sequence_length):
                symbols[:, step] = outputs[:, step, :].topk(1)[1].squeeze(-1).data.cpu().numpy()

            loss = loss_function(outputs.view(-1, self._output_size.value), inputs.view(-1))

        else:
            for step in range(sequence_length):
                step_input = inputs[:, step].unsqueeze(-1)
                step_output, hidden_state = self._decode(decoder_input=step_input,
                                                         hidden_state=hidden_state,
                                                         encoder_outputs=None,
                                                         batch_size=batch_size,
                                                         sequence_length=sequence_length)

                loss += loss_function(step_output.squeeze(1), inputs[:, step])
                symbols[:, step] = step_output.topk(1)[1].data.squeeze(-1).squeeze(-1).cpu().numpy()

        return loss, symbols

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


class BeamDecoder:

    def __init__(self):
        pass


class ConvDecoder:

    def __init__(self):
        pass

