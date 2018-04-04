import torch
import torch.autograd as autograd
import torch.optim

from modules.encoder import Encoder
from modules.utils.utils import Optimizer

from utils.utils import ParameterSetter
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from collections import OrderedDict


class RNNEncoder(Encoder):
    """
    Recurrent encoder module of the sequence to sequence model.
    """

    interface = OrderedDict(**{
        'hidden_size':      None,
        'recurrent_type':   None,
        'num_layers':       None,
        'optimizer_type':   None,
        'learning_rate':    None,
        'use_cuda':        'Task:use_cuda$',
        'embedding_size':  'embedding_size$'
    })

    abstract = False

    @ParameterSetter.pack(interface)
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
        self._optimizer = None

        self._outputs = {
            'hidden_state':     None,
            'encoder_outputs':  None
        }

        self.embedding = None

    def init_parameters(self):
        """
        Calls the parameter setter, which initializes the Parameter type attributes.
        After initialization, the main components of the encoder, which require the previously
        initialized parameter values, are created as well.
        """
        self._parameter_setter.initialize(self)

        if self._recurrent_type == 'LSTM':
            unit_type = torch.nn.LSTM
        elif self._recurrent_type == 'GRU':
            unit_type = torch.nn.GRU
        else:
            raise ValueError('Invalid recurrent unit type.')

        self._recurrent_layer = unit_type(input_size=self._embedding_size,
                                          hidden_size=self._hidden_size,
                                          num_layers=self._num_layers,
                                          bidirectional=False,
                                          batch_first=True)

        if self._use_cuda:
            self._recurrent_layer = self._recurrent_layer.cuda()

        return self

    def init_optimizer(self):
        """
        Initializes the optimizer for the encoder.
        """
        self._optimizer = Optimizer(parameters=self.parameters(),
                                    optimizer_type=self._optimizer_type,
                                    scheduler_type='ReduceLROnPlateau',
                                    learning_rate=self._learning_rate)

        return self

    def forward(self, inputs, lengths):
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
        embedded_inputs = self.embedding(inputs)
        padded_sequence = pack_padded_sequence(embedded_inputs, lengths=lengths, batch_first=True)

        self._recurrent_layer.flatten_parameters()

        outputs, self._outputs['hidden_state'] = self._recurrent_layer(padded_sequence, initial_state)
        self._outputs['encoder_outputs'], _ = pad_packed_sequence(outputs, batch_first=True)

        return self._outputs

    def _init_hidden(self, batch_size):  # TODO double state
        """
        Initializes the hidden state of the encoder module.
        :return: Variable, (num_layers*directions, batch_size, hidden_dim) with zeros as initial values.
        """
        state = autograd.Variable(torch.zeros(self._num_layers, batch_size, self._hidden_size))

        if self._use_cuda:
            state = state.cuda()

        if isinstance(self._recurrent_layer, torch.nn.LSTM):
            return state, state
        else:
            return state

    @property
    def optimizers(self):
        """
        Property for the optimizers of the encoder.
        :return: list, optimizers used by the encoder.
        """
        return [
            self._optimizer,
            *self.embedding.optimizer,
        ]

    @property
    def state(self):
        """
        Property for the state of the encoder.
        :return: dict, containing the state of the weights, and the optimizer.
        """
        return {
            'weights':      self.state_dict(),
            'optimizer':    self._optimizer.state
        }

    # noinspection PyMethodOverriding
    @state.setter
    def state(self, state):
        """
        Setter method for the weights of the encoder, and the optimizer.
        :param state: dict, containing the states.
        """
        self.load_state_dict({k: v for k, v in state['weights'].items() if k in self.state_dict()})
        self._optimizer.state = state['optimizer']
