import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim

from modules.encoder import Encoder
from utils.utils import Parameter
from utils.utils import batch_to_padded_sequence
from utils.utils import padded_sequence_to_batch


class RNNEncoder(Encoder):
    """
    Recurrent encoder module of the sequence to sequence model.
    """

    _param_dict = {
        'hidden_size': Parameter(name='_hidden_size',        doc='int, size of recurrent layer of the LSTM/GRU.'),
        'embedding_size': Parameter(name='_embedding_size',  doc='int, dimension of the word embeddings.'),
        'recurrent_type':  Parameter(name='_recurrent_type', doc='str, name of the recurrent layer (GRU, LSTM).'),
        'num_layers': Parameter(name='_num_layers',          doc='int, number of stacked RNN layers.'),
        'optimizer_type': Parameter(name='_optimizer_type',  doc='Optimizer, for parameter optimalization.'),
        'learning_rate': Parameter(name='_learning_rate',    doc='float, learning rate.'),
        'use_cuda':  Parameter(name='_use_cuda',             doc='bool, True if the device has cuda support.')
    }

    def __init__(self, parameter_setter):
        """
        A recurrent encoder module for the sequence to sequence model.
        :param parameter_setter: ParameterSetter object, that requires the following parameters.
            -:parameter hidden_size: int, size of recurrent layer of the LSTM/GRU.
            -:parameter recurrent_layer: str, name of the recurrent layer ('GRU', 'LSTM').
            -:parameter embedding_size: int, dimension of the word embeddings.
            -:parameter optimizer: Optimizer, for parameter optimalization.
            -:parameter learning_rate: float, learning rate.
            -:parameter use_cuda: bool, True if the device has cuda support.
        """
        super().__init__()
        self._parameter_setter = parameter_setter

        self._recurrent_layer = None
        self._embedding_layer = None
        self._optimizer = None

        self._outputs = {
            'hidden_state': None,
            'encoder_outputs': None
        }

    def init_parameters(self):
        """
        Calls the parameter setter, which initializes the Parameter type attributes.
        After initialization, the main components of the encoder, which require the previously
        initialized parameter values, are created as well.
        """
        for parameter in self._param_dict:
            self.__dict__[self._param_dict[parameter].name] = self._param_dict[parameter]

        self._parameter_setter(self.__dict__)

        if self._recurrent_type.value == 'LSTM':
            unit_type = torch.nn.LSTM
        elif self._recurrent_type.value == 'GRU':
            unit_type = torch.nn.GRU
        else:
            raise ValueError('Invalid recurrent unit type.')

        self._recurrent_layer = unit_type(input_size=self._embedding_size.value,
                                          hidden_size=self._hidden_size.value,
                                          num_layers=self._num_layers.value,
                                          bidirectional=False,
                                          batch_first=True)

        if self._use_cuda.value:
            self._recurrent_layer = self._recurrent_layer.cuda()

        return self

    def init_optimizer(self):
        """
        Initializes the optimizer for the encoder.
        """
        optimizers = {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
            'RMSProp': torch.optim.RMSprop,
        }
        self._optimizer = optimizers[self._optimizer_type.value](self.parameters(), lr=self._learning_rate.value)

        return self

    def forward(self,
                inputs,
                lengths):
        """
        A forward step of the encoder. The batch of sequences with word ids are
        packed into padded_sequence object, which are processed by the recurrent layer.
        :param inputs: Variable, (batch_size, sequence_length) containing the ids of the words.
        :param lengths: Ndarray, containing the real lengths of the sequences in the batch (prior to padding).
        :return outputs: Variable, (batch_size, sequence_length, vocab_size) the output at each time
                         step of the encoder.
        :return hidden_state: Variable, (num_layers * directions, batch_size, hidden_size) the final hidden state.
        """
        initial_state = self._init_hidden(inputs.size(0))

        embedded_inputs = self._embedding_layer(inputs)
        padded_sequence = batch_to_padded_sequence(embedded_inputs, lengths)

        self._recurrent_layer.flatten_parameters()

        outputs, self._outputs['hidden_state'] = self._recurrent_layer(padded_sequence, initial_state)
        self._outputs['encoder_outputs'], _ = padded_sequence_to_batch(outputs)

        return self._outputs

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the encoder module.
        :return: Variable, (num_layers*directions, batch_size, hidden_dim) with zeros as initial values.
        """
        result = autograd.Variable(torch.zeros(self._num_layers.value, batch_size, self._hidden_size.value))

        if self._use_cuda.value:
            result = result.cuda()

        if isinstance(self._recurrent_layer, torch.nn.LSTM):
            return result, result
        else:
            return result

    @classmethod
    def abstract(cls):
        return False

    @classmethod
    def assemble(cls, params):
        return {
            **{cls._param_dict[param].name: params[param] for param in params if param in cls._param_dict},
            cls._param_dict['embedding_size'].name: params['language'].embedding_size,
        }

    @property
    def optimizer(self):
        """
        Property for the optimizer of the encoder.
        :return self.__optimizer: Optimizer, the currently used optimizer of the encoder.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Setter for the optimizer of the encoder.
        :param optimizer: Optimizer, instance to be set as the new optimizer for the encoder.
        """
        self._optimizer.value = optimizer

    @property
    def embedding(self):
        """
        Property for the encoder's embedding layer.
        :return: The currently used embeddings of the encoder.
        """
        return self._embedding_layer

    @embedding.setter
    def embedding(self, embedding):
        """
        Setter for the encoder's embedding layer.
        :param embedding: Embedding, to be set as the embedding layer of the encoder.
        """
        self._embedding_layer = nn.Embedding(embedding['weights'].size(0), embedding['weights'].size(1))
        self._embedding_layer.weight = nn.Parameter(embedding['weights'])
        self._embedding_layer.weight.requires_grad = embedding['requires_grad']

    @property
    def hidden_size(self):
        """
        Property for the hidden size of the recurrent layer.
        :return: int, size of the hidden layer.
        """
        return self._hidden_size.value
