import torch
from torch import nn
from torch.nn import functional
from utils.utils import Parameter
from modules import attention


import numpy as np


class RNNDecoder(nn.Module):
    """
    Decoder module of the sequence to sequence model.
    """

    def __init__(self,
                 parameter_setter):
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
        self._recurrent_type = Parameter(name='_recurrent_type', doc='str, name of the recurrent layer (GRU, LSTM).')
        self._num_layers = Parameter(name='_num_layers',         doc='int, number of stacked RNN layers.')
        self._learning_rate = Parameter(name='_learning_rate',   doc='float, learning rate.')
        self._max_length = Parameter(name='_max_length',         doc='int, maximum length of the sequence decoding.')
        self._use_cuda = Parameter(name='_use_cuda',             doc='bool, True if the device has cuda support.')
        self._tf_ratio = Parameter(name='_tf_ratio',             doc='float, teacher forcing ratio.')

        self.__attention = None

        self.__embedding_layer = None
        self.__recurrent_layer = None
        self.__optimizer = None

        self._init_parameters()
        self._init_optimizer()

    def _init_parameters(self):
        self._parameter_setter(self.__dict__)

        self.__attention = attention.GeneralAttention(hidden_size=50,
                                                      embedding_size=self._embedding_size.value,
                                                      use_cuda=self._use_cuda.value)

        if self._recurrent_type.value == 'LSTM':
            unit_type = torch.nn.LSTM
        else:
            unit_type = torch.nn.GRU

        if self.__attention is not None:
            input_size = self.__attention.input_size
        else:
            input_size = self._embedding_size.value

        self.__recurrent_layer = unit_type(input_size=input_size,
                                           hidden_size=self._hidden_size.value,
                                           num_layers=self._num_layers.value,
                                           bidirectional=False,
                                           batch_first=True)

        self.__output_layer = nn.Linear(self._hidden_size.value, self._output_size.value)

        if self._use_cuda:
            self.__recurrent_layer = self.__recurrent_layer.cuda()
            self.__output_layer = self.__output_layer.cuda()

    def _init_optimizer(self):
        params = self.parameters()
        if attention is not None:
            self.__attention._recurrent_layer = self.__recurrent_layer
            params = list(list(params) + list(self.__attention.parameters()))

        self.__optimizer = torch.optim.Adam(params, lr=self._learning_rate.value)

    def _decode_step(self,
                     step_input,
                     hidden_state,
                     encoder_outputs,
                     batch_size,
                     sequence_length):
        """
        A single step decoder function. Attention is applied to the batches. The recurrent unit
        calculates the t-th hidden state inside the attention mechanism's forward call.
        :param step_input: Variable, with size of (batch_size, 1), containing word ids for step t.
        :param hidden_state: Variable, with size of (num_layers * directions, batch_size, hidden_size).
        :param encoder_outputs: Variable, with size of (batch_size, sequence_length, hidden_size).
        :param batch_size: int, size of the input batches.
        :param sequence_length: int, size of the sequence of the input batch.
        :return output: Variable, (batch_size, 1, vocab_size) distribution of probabilities over the words.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final state at time t.
        :return attn_weights: Variable, (batch_size, 1, sequence_length) attention weights for visualization.
        """
        embedded_input = self.embedding(step_input)

        output, hidden_state, attn_weights = self.__attention.forward(step_input=embedded_input,
                                                                      hidden_state=hidden_state,
                                                                      encoder_outputs=encoder_outputs,
                                                                      batch_size=batch_size,
                                                                      sequence_length=sequence_length)

        output = functional.log_softmax(self.__output_layer(output.contiguous().view(-1, self._hidden_size.value)),
                                        dim=1).view(batch_size, -1, self._output_size.value)

        return output, hidden_state, attn_weights

    def _decode(self,
                inputs,
                hidden_state,
                batch_size):
        """
        A full 'roll-out' of the RNN. The sequence is fully decoded, and the outputs are returned
        for each time step. Attention is not used, but the calculations are must faster.
        :param inputs: Variable, (batch_size, sequence_length) containing the ids of the words.
        :param hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) initial hidden state.
        :param batch_size: int, size of the batch.
        :return outputs: Variable, (batch_size, sequence_length, vocab_size)the output at each time step of the decoder.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final hidden state.
        """
        embedded_inputs = self.embedding(inputs)
        outputs, hidden_state = self.__recurrent_layer(embedded_inputs, hidden_state)
        outputs = functional.log_softmax(self.__output_layer(outputs.contiguous().view(-1, self._hidden_size.value)),
                                         dim=1).view(batch_size, -1, self._output_size.value)

        return outputs, hidden_state

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
        decoded_lengths = np.zeros(batch_size)

        symbols = np.zeros((batch_size, sequence_length), dtype='int')

        use_teacher_forcing = True
        loss = 0
        if use_teacher_forcing:
            if self.__attention is not None:
                for step in range(sequence_length):
                    step_input = inputs[:, step].unsqueeze(-1)
                    step_output, hidden_state, attn_weights = self._decode_step(step_input=step_input,
                                                                                hidden_state=hidden_state,
                                                                                encoder_outputs=encoder_outputs,
                                                                                batch_size=batch_size,
                                                                                sequence_length=sequence_length)

                    loss += loss_function(step_output.squeeze(1), inputs[:, step])
                    symbols[:, step] = step_output.topk(1)[1].data.squeeze(-1).squeeze(-1).cpu().numpy()

            else:
                outputs, hidden_state = self._decode(inputs=inputs,
                                                     hidden_state=hidden_state,
                                                     batch_size=batch_size)

                for step in range(sequence_length):
                    symbols[:, step] = outputs[:, step, :].topk(1)[1].squeeze(-1).data.cpu().numpy()

                loss = loss_function(outputs.view(-1, self._output_size.value), inputs.view(-1))

        return loss, symbols

    @property
    def optimizer(self):
        """
        Property for the optimizer of the decoder.
        :return self.__optimizer: Optimizer, the currently used optimizer of the decoder.
        """
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Setter for the optimizer of the decoder.
        :param optimizer: Optimizer, instance to be set as the new optimizer for the decoder.
        """
        self.__optimizer = optimizer

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
        return self.__embedding_layer

    @embedding.setter
    def embedding(self, embedding):
        """
        Setter for the decoder's embedding layer.
        :param embedding: Embedding, to be set as the embedding layer of the decoder.
        """
        self.__embedding_layer = nn.Embedding(embedding.size(0), embedding.size(1))
        self.__embedding_layer.weight = nn.Parameter(embedding)
        self.__embedding_layer.weight.requires_grad = False


class BeamDecoder:

    def __init__(self):
        pass


class ConvDecoder:

    def __init__(self):
        pass

