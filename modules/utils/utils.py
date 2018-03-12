import torch

from torch.nn import Module
from torch.nn import Linear
from torch.nn import LeakyReLU

from utils.utils import Component

from collections import OrderedDict


class Discriminator(Module, Component):
    """
    Abstract base class for the discriminator modules, mainly used for the unsupervised
    translation task. Any newly added discriminator type module must inherit from this
    super class, otherwise it won't be discoverable by the hierarchy builder utility.
    """

    def __init__(self, learning_rate, use_cuda):
        super().__init__()

        self._learning_rate = learning_rate
        self._use_cuda = use_cuda

    def forward(self, *args, **kwargs):
        return NotImplementedError

    def get_optimizer_states(self):
        return NotImplementedError

    def set_optimizer_states(self, *args, **kwargs):
        return NotImplementedError


class FFDiscriminator(Discriminator):
    """
    Feed-forward discriminator module for the unsupervised neural translation task.
    """

    @staticmethod
    def interface():
        return OrderedDict(**{
            'hidden_size': None,
            'learning_rate': None,
            'optimizer_type': None,
            'use_cuda': 'Task:use_cuda$',
            'input_size': 'Encoder:hidden_size$'
        })

    @classmethod
    def abstract(cls):
        return False

    def __init__(self,
                 input_size,
                 hidden_size,
                 learning_rate,
                 optimizer_type,
                 use_cuda):
        """
        An instance of a feed-forward discriminator.
        :param input_size: int, input size of the feed-forward network.
        :param hidden_size: int, hidden size of the feed forward neural network.
        :param learning_rate: float, learning rate of the optimizer.
        :param use_cuda: bool, true if cuda support is enabled.
        """
        super().__init__(learning_rate=learning_rate, use_cuda=use_cuda)

        self._input_layer = Linear(input_size, hidden_size)
        self._hidden_layer = Linear(hidden_size, hidden_size)
        self._output_layer = Linear(hidden_size, 1)
        self._activation = LeakyReLU()

        if self._use_cuda:
            self._input_layer = self._input_layer.cuda()
            self._hidden_layer = self._hidden_layer.cuda()
            self._output_layer = self._output_layer.cuda()
            self._activation = self._activation.cuda()

        optimizers = {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
            'RMSProp': torch.optim.RMSprop,
        }

        self._optimizer = optimizers[optimizer_type](self.parameters(), lr=self._learning_rate)

    def forward(self, inputs):
        """
        Forward step for the discriminator.
        :param inputs: Variable, (batch_size, input_size), where input_size is equal to the encoder's
                       hidden_size.
        :return: Variable, (batch_size, 1).
        """
        output = self._activation(self._input_layer(inputs))
        output = self._activation(self._hidden_layer(output))
        output = torch.sigmoid(self._output_layer(output))

        return output

    def get_optimizer_states(self):
        return NotImplementedError

    def set_optimizer_states(self, *args, **kwargs):
        return NotImplementedError

    @property
    def optimizer(self):
        """
        Property for the optimizer of the discriminator.
        :return: Optimizer, the currently used optimizer of the discriminator.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Setter for the optimizer of the discriminator.
        :param optimizer: Optimizer, instance to be set as the new optimizer for the discriminator.
        """
        self._optimizer = optimizer


class RNNDiscriminator(Discriminator):
    """
    Recurrent discriminator module for the unsupervised neural translation task.
    """

    @staticmethod
    def interface():
        return OrderedDict(**{
            'hidden_size': None,
            'learning_rate': None,
            'optimizer_type': None,
            'use_cuda': 'Task:use_cuda$',
            'input_size': 'Encoder:hidden_size$'
        })

    @classmethod
    def abstract(cls):
        return False

    def __init__(self,
                 input_size,
                 hidden_size,
                 learning_rate,
                 optimizer_type,
                 use_cuda):
        """
        An instance of a recurrent discriminator.
        :param input_size: int, input size of the feed-forward network.
        :param hidden_size: int, hidden size of the feed forward neural network.
        :param learning_rate: float, learning rate of the optimizer.
        :param use_cuda: bool, true if cuda support is enabled.
        """
        super().__init__(learning_rate=learning_rate, use_cuda=use_cuda)

        self._recurrent_layer = torch.nn.GRU(input_size=input_size,
                                             hidden_size=hidden_size,
                                             batch_first=True)

        self._output_layer = Linear(self._hidden_size, 1)

        if self._use_cuda:
            self._recurrent_layer = self._recurrent_layer.cuda()
            self._output_layer = self._output_layer.cuda()

        optimizers = {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
            'RMSProp': torch.optim.RMSprop,
        }

        self._optimizer = optimizers[optimizer_type](self.parameters(), lr=self._learning_rate)

    def forward(self, inputs):
        """
        Forward step for the discriminator.
        :param inputs: Variable, (batch_size, input_size), where input_size is equal to the encoder's
                       hidden_size.
        :return: Variable, (batch_size, 1).
        """
        outputs = self._recurrent_layer.forward(inputs)
        final_output = torch.sigmoid(self._output_layer(outputs[-1]))

        return final_output

    def get_optimizer_states(self):
        """
        Setter for the state dicts of the discriminator's optimizer(s).
        :return: dict, containing the state of the optimizer(s).
        """
        return {
            'discriminator_optimizer': self.optimizer.state_dict()
        }

    def set_optimizer_states(self, discriminator_optimizer):
        """
        Setter for the state dicts of the discriminator's optimizer(s).
        :param discriminator_optimizer: dict, state of the discriminator's optimizer.
        """
        self.optimizer.load_state_dict(discriminator_optimizer)

    @property
    def optimizer(self):
        """
        Property for the optimizer of the decoder.
        :return: Optimizer, the currently used optimizer of the discriminator.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Setter for the optimizer of the discriminator.
        :param optimizer: Optimizer, instance to be set as the new optimizer of the discriminator.
        """
        self._optimizer = optimizer


class Noise:

    def __init__(self):
        pass

    def __call__(self, input_batch):
        return input_batch

