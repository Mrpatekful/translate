"""

"""

import torch
import torch.autograd

from torch.nn import LeakyReLU
from torch.nn import Linear
from torch.nn import Module

from torch.nn.functional import softmax

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils.utils import Component, Interface


class Classifier(Module, Component):
    """
    Abstract base class for the discriminator modules, mainly used for the unsupervised
    translation task. Any newly added discriminator type module must inherit from this
    super class, otherwise it won't be discoverable by the hierarchy builder utility.
    """

    interface = Interface(**{
        'hidden_size':      (0, None),
        'learning_rate':    (1, None),
        'optimizer_type':   (2, None),
        'output_size':      (3, None),
        'cuda':             (4, 'Experiment:Policy:cuda$'),
        'input_size':       (5, 'Encoder:hidden_size$')
    })

    def __init__(self,
                 hidden_size:       int,
                 input_size:        int,
                 output_size:       int,
                 learning_rate:     float,
                 optimizer_type:    str,
                 cuda:              bool):
        """
        An instance of a feed-forward discriminator.

        Arguments:
            input_size:
                int, input size of the feed-forward network.

            hidden_size:
                int, hidden size of the feed forward neural network.

            learning_rate:
                float, learning rate of the optimizer.

            optimizer_type:
                str, type of the optimizer.

            cuda:
                bool, true if cuda support is enabled.
        """
        super().__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._output_size = output_size

        self._optimizer_type = optimizer_type
        self._learning_rate = learning_rate
        self._cuda = cuda

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def optimizer(self):
        raise NotImplementedError

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class FFClassifier(Classifier):
    """
    Feed-forward classifier module for the unsupervised neural translation task.
    """

    abstract = False

    def __init__(self,
                 input_size:        int,
                 hidden_size:       int,
                 output_size:       int,
                 learning_rate:     float,
                 optimizer_type:    str,
                 cuda:              bool):
        """
        An instance of a feed-forward discriminator.

        Arguments:
            input_size:
                int, input size of the feed-forward network.

            hidden_size:
                int, hidden size of the feed forward neural network.

            learning_rate:
                float, learning rate of the optimizer.

            cuda:
                bool, true if cuda support is enabled.
        """
        super().__init__(hidden_size=hidden_size,
                         output_size=output_size,
                         input_size=input_size,
                         learning_rate=learning_rate,
                         optimizer_type=optimizer_type,
                         cuda=cuda)

        self._input_layer = Linear(input_size, hidden_size)
        self._hidden_layer = Linear(hidden_size, hidden_size)
        self._output_layer = Linear(hidden_size, self._output_size)
        self._activation = LeakyReLU()

        if self._cuda:
            self._input_layer = self._input_layer.cuda()
            self._hidden_layer = self._hidden_layer.cuda()
            self._output_layer = self._output_layer.cuda()
            self._activation = self._activation.cuda()

        self._optimizer = Optimizer(parameters=self.parameters(),
                                    optimizer_type=optimizer_type,
                                    scheduler_type='ReduceLROnPlateau',
                                    learning_rate=self._learning_rate)


    def forward(self, *args, inputs, **kwargs):
        """
        Forward step for the classifier.

        Args:
            inputs:
                Variable, (batch_size, input_size), where input_size is equal to the encoder's
                hidden_size.

        Returns:
            output:
                Variable, (batch_size, 1).
        """
        output = self._activation(self._input_layer(inputs))
        output = self._activation(self._hidden_layer(output))
        output = self._output_layer(output)

        softmax_output = softmax(output, dim=1)

        return output, softmax_output

    @property
    def optimizer(self):
        return self._optimizer


class RNNClassifier(Classifier):
    """
    Recurrent discriminator module for the unsupervised neural translation task.
    """

    abstract = False

    interface = Interface(**{
        **Classifier.interface.dictionary,
        'num_layers': (Interface.last_key(Classifier.interface.dictionary) + 1, None)
    })

    def __init__(self,
                 input_size:        int,
                 hidden_size:       int,
                 output_size:       int,
                 learning_rate:     float,
                 num_layers:        int,
                 optimizer_type:    str,
                 cuda:              bool):
        """
        An instance of a recurrent discriminator.

        Args:
            input_size:
                int, input size of the feed-forward network.

            hidden_size:
                int, hidden size of the feed forward neural network.

            learning_rate:
                float, learning rate of the optimizer.

            cuda:
                bool, true if cuda support is enabled.
        """
        super().__init__(hidden_size=hidden_size,
                         input_size=input_size,
                         output_size=output_size,
                         learning_rate=learning_rate,
                         optimizer_type=optimizer_type,
                         cuda=cuda)

        self._num_layers = num_layers

        self._recurrent_layer = torch.nn.GRU(input_size=input_size,
                                             num_layers=num_layers,
                                             hidden_size=hidden_size,
                                             batch_first=True)

        self._output_layer = Linear(self._hidden_size, self._output_size)

        if self._cuda:
            self._recurrent_layer = self._recurrent_layer.cuda()
            self._output_layer = self._output_layer.cuda()

        self._optimizer = Optimizer(parameters=self.parameters(),
                                    optimizer_type=optimizer_type,
                                    scheduler_type='ReduceLROnPlateau',
                                    learning_rate=self._learning_rate)

    def forward(self, *args, inputs, lengths, **kwargs):
        """
        Forward step for the discriminator.

        Args:
            inputs:
                Variable, (batch_size, input_size), where input_size is equal to the encoder's
                hidden_size.

            lengths:

        Returns:
            final_output:
                Variable, (batch_size, 1).
        """
        initial_state = self._init_hidden(inputs.size(0))
        padded_sequence = pack_padded_sequence(inputs, lengths=lengths, batch_first=True)

        self._recurrent_layer.flatten_parameters()

        outputs, _ = self._recurrent_layer(padded_sequence, initial_state)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        outputs = self._output_layer(outputs[:, -1, :])

        softmax_outputs = softmax(outputs, dim=1)

        return outputs, softmax_outputs

    def _init_hidden(self, batch_size):
        """
        Initializes the hidden state of the encoder module.
        :return: Variable, (num_layers*directions, batch_size, hidden_dim) with zeros as initial values.
        """
        state = torch.autograd.Variable(torch.randn(self._num_layers, batch_size, self._hidden_size))

        if self._use_cuda:
            state = state.cuda()

        return state

    @property
    def optimizer(self):
        return self._optimizer


class Embedding(Module):
    """
    Wrapper class for the embedding layers of the models. The optional training of the embeddings
    is done by a built-in optimizer.
    """

    def __init__(self,
                 embedding_size:    int,
                 vocab_size:        int,
                 cuda:              bool,
                 weights:           torch.FloatTensor=None,
                 requires_grad:     bool=True):
        """
        An embedding instance.

        Args:
            embedding_size:
                Int, size of the word vectors.

            vocab_size:
                Int, number of words in the vocab.

            cuda:
                Bool, true if cuda support is enabled.

            weights:
                Tensor, if weights are provided, these will be used

            requires_grad:

        """
        super().__init__()

        self._layer = torch.nn.Embedding(vocab_size, embedding_size)
        self._requires_grad = requires_grad

        if weights is not None:
            self._layer.weight = torch.nn.Parameter(weights, requires_grad=self._requires_grad)

        if cuda:
            self._layer = self._layer.cuda()

        self._optimizer = None

        if requires_grad:
            self._optimizer = Optimizer(parameters=self.parameters(),
                                        optimizer_type='SGD',
                                        scheduler_type='ReduceLROnPlateau',
                                        learning_rate=0.01)

    def forward(self, inputs):
        """
        Propagates the inputs through the embedding layer.

        Args:
            inputs:
                Variable, word id-s, that will be translated to word vector representations.

        Returns:
            outputs:
                Variable, word vectors of the given input.
        """
        return self._layer(inputs)

    def freeze(self):
        """
        Freezes the parameters of the embedding layer. While frozen, the parameters can't be
        modified by the optimizer.
        """
        if self._requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """
       Unfreezes the parameters of the optimizers. By calling this method, the optimizer will
       be able to modify the weights of the embedding layer.
       """
        if self._requires_grad:
            for param in self.parameters():
                param.requires_grad = True

    @property
    def optimizer(self):
        """
        Property for the optimizer of the embedding.
        """
        return self._optimizer

    @property
    def state(self):
        """
        Property for the state of the embedding.
        """
        return {
            'weight':       self.state_dict(),
            'optimizer':    None if self._optimizer is None else self._optimizer.state
        }

    @state.setter
    def state(self, states):
        """
        Setter method for the state of the embedding.
        """
        self.load_state_dict(states['weight'])
        if self._optimizer is not None:
            self._optimizer.state = states['optimizer']


class Layer(Module):

    def __init__(self, input_size, output_size, use_cuda):
        """


        Args:
            input_size:

            output_size:

            use_cuda:

        """
        super().__init__()

        self._weights = Linear(input_size, output_size)
        self.size = output_size

        if use_cuda:
            self._weights = self._weights.cuda()

        self._optimizer = Optimizer(parameters=self.parameters(),
                                    optimizer_type='Adam',
                                    scheduler_type='ReduceLROnPlateau',
                                    learning_rate=0.001)

    def forward(self, inputs):
        """


        Args:
            inputs:

        Returns:
            outputs:

        """
        return self._weights(inputs)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def optimizer(self):
        """
        Property for the optimizer of the layer.
        """
        return self._optimizer

    @property
    def state(self):
        """
        Property for the state of the embedding.
        """
        return {
            'weight':       self.state_dict(),
            'optimizer':    self._optimizer.state
        }

    @state.setter
    def state(self, states):
        """
        Setter method for the state of the embedding.
        :param states: dict, containing the state of the weights and optimizer.
        """
        self.load_state_dict(states['weight'])
        self._optimizer.state = states['optimizer']


class Optimizer:
    """
    Wrapper class for the optimizers. Additionally to the optimizers provided by torch,
    this type has built-in learning rate scheduling.
    """

    _algorithms = {
        'Adam':     torch.optim.Adam,
        'SGD':      torch.optim.SGD,
        'RMSProp':  torch.optim.RMSprop,
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

        Args:
            parameters:
                Iterable, containing the parameters, that will be optimized by the provided
                optimalization algorithm.

            optimizer_type:
                Str, type of the algorithm to be used for optimalization.

            scheduler_type:
                Str, type of the scheduler to be used for learning rate adjustments.

            learning_rate:
                Float, the initial learning rate.
        """
        try:

            self._algorithm = self._algorithms[optimizer_type](params=parameters, lr=learning_rate)
            self._scheduler = self._schedulers[scheduler_type](self._algorithm)

        except KeyError as error:
            raise RuntimeError(f'Invalid optimizer/scheduler type: {error}')

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
        """
        self._scheduler.step(metric)

    @property
    def state(self):
        """
        Property for the state of the optimizer.
        """
        return self._algorithm.state_dict()

    @state.setter
    def state(self, state):
        """
        Setter method for the state of the optimizer.
        """
        self._algorithm.load_state_dict(state)
