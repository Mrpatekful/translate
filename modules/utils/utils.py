import torch

from torch.nn import Module
from torch.nn import Linear
from torch.nn import LeakyReLU

from utils.utils import Component

from collections import OrderedDict


class Discriminator(Module, Component):  # TODO states
    """
    Abstract base class for the discriminator modules, mainly used for the unsupervised
    translation task. Any newly added discriminator type module must inherit from this
    super class, otherwise it won't be discoverable by the hierarchy builder utility.
    """

    interface = OrderedDict(**{
            'hidden_size':      None,
            'learning_rate':    None,
            'optimizer_type':   None,
            'use_cuda':        'Task:use_cuda$',
            'input_size':      'Encoder:hidden_size$'
        })

    def __init__(self, learning_rate, use_cuda):
        super().__init__()

        self._learning_rate = learning_rate
        self._use_cuda = use_cuda

    def forward(self, *args, **kwargs):
        return NotImplementedError

    @property
    def optimizers(self):
        return NotImplementedError


class FFDiscriminator(Discriminator):
    """
    Feed-forward discriminator module for the unsupervised neural translation task.
    """

    abstract = False

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

        self._optimizers = [
            Optimizer(parameters=self.parameters(),
                      optimizer_type=optimizer_type,
                      scheduler_type='ReduceLROnPlateau',
                      learning_rate=self._learning_rate)
        ]

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

    @property
    def optimizers(self):
        return self._optimizers


class RNNDiscriminator(Discriminator):
    """
    Recurrent discriminator module for the unsupervised neural translation task.
    """

    abstract = False

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

        self._optimizers = [
            Optimizer(parameters=self.parameters(),
                      optimizer_type=optimizer_type,
                      scheduler_type='ReduceLROnPlateau',
                      learning_rate=self._learning_rate)
        ]

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

    @property
    def optimizers(self):
        return self._optimizers


class Embedding(Module):
    """
    Wrapper class for the embedding layers of the models. The optional training of the embeddings
    is done by a built-in optimizer.
    """

    def __init__(self,
                 embedding_size,
                 vocab_size,
                 use_cuda,
                 weights=None,
                 requires_grad=True):
        """
        An embedding instance.
        :param embedding_size: int, size of the word vectors.
        :param vocab_size: int, number of words in the vocab.
        :param use_cuda: bool, true if cuda support is enabled.
        :param weights: Tensor, if weights are provided, these will be used
        :param requires_grad:
        """
        super().__init__()

        self._layer = torch.nn.Embedding(vocab_size, embedding_size)

        if weights is not None:
            self._layer.weight = torch.nn.Parameter(weights)

        if use_cuda:
            self._layer = self._layer.cuda()

        self._optimizer = []

        if requires_grad:
            self._optimizer = [
                Optimizer(parameters=self.parameters(),
                          optimizer_type='SGD',
                          scheduler_type='ReduceLROnPlateau',
                          learning_rate=0.01)
            ]

    def forward(self, inputs):
        """
        Propagates the inputs through the embedding layer.
        :param inputs: Variable, word id-s, that will be translated to word vector representations.
        :return: Variable, word vectors of the given input.
        """
        return self._layer(inputs)

    @property
    def optimizer(self):
        """
        Property for the optimizer of the embedding.
        :return: tuple, that only yields an optimizer, if it was defined as required.
        """
        return self._optimizer

    @property
    def state(self):
        """
        Property for the state of the embedding.
        :return: dict, containing the state of the weights and the optimizer.
        """
        return {
            'weight': self.state_dict(),
            'optimizer': None if len(self._optimizer) == 0 else self._optimizer[0].state
        }

    @state.setter
    def state(self, states):
        """
        Setter method for the state of the embedding.
        :param states: dict, containing the state of the weights and optimizer.
        """
        self.load_state_dict(states['weight'])
        if len(self._optimizer) > 0:
            self._optimizer[0].state = states['optimizer']


class Optimizer:
    """
    Wrapper class for the optimizers. Additionally to the optimizers provided by torch,
    this type has built-in learning rate scheduling.
    """

    _algorithms = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'RMSProp': torch.optim.RMSprop,
    }

    _schedulers = {
        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau
    }

    def __init__(self,
                 parameters,
                 optimizer_type,
                 scheduler_type,
                 learning_rate):
        """
        Optimizer type object.
        :param parameters: iterable, containing the parameters, that will be optimized by the provided
                           optimalization algorithm.
        :param optimizer_type: str, type of the algorithm to be used for optimalization.
        :param scheduler_type: str, type of the scheduler to be used for learning rate adjustments.
        :param learning_rate: float, the initial learning rate.
        """
        try:

            self._algorithm = self._algorithms[optimizer_type](params=parameters, lr=learning_rate)
            self._scheduler = self._schedulers[scheduler_type](self._algorithm)

        except KeyError as error:
            raise RuntimeError('Invalid optimizer/scheduler type: %s' % error)

    def step(self):
        """
        Executes the optimalization step on the parameters, that have benn provided to the optimizer.
        """
        self._algorithm.step()

    def clear(self):
        """
        Clears the gradients of the parameters, which are being optimized by the algorithm.
        """
        self._algorithm.zero_grad()

    def adjust(self, metric):
        """
        Adjust the learning rate, given a metric.
        :param metric:
        """
        self._scheduler.step(metric)

    @property
    def state(self):
        """
        Property for the state of the optimizer.
        :return: dict, state of the optimizer.
        """
        return self._algorithm.state_dict()

    @state.setter
    def state(self, state):
        """
        Setter method for the state of the optimizer.
        :param state: dict, state of the optimizer.
        """
        self._algorithm.load_state_dict(state)


class Noise:

    def __init__(self):
        pass

    def __call__(self, input_batch):
        return input_batch
