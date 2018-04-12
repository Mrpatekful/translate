import torch
import numpy

import torch.autograd

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
        'tokens':          'Task:tokens$',
        'use_cuda':        'Task:Policy:use_cuda$',
        'input_size':      'Encoder:hidden_size$'
    })

    def __init__(self,
                 hidden_size,
                 learning_rate,
                 optimizer_type,
                 tokens,
                 use_cuda,
                 input_size):
        """


        Args:
            hidden_size:

            learning_rate:

            optimizer_type:

            use_cuda:

            input_size:

        """
        super().__init__()

        self._output_size = len(tokens)

        self._hidden_size = hidden_size
        self._optimizer_type = optimizer_type
        self._input_size = input_size
        self._learning_rate = learning_rate
        self._use_cuda = use_cuda

    def forward(self, *args, **kwargs):
        return NotImplementedError

    @property
    def optimizers(self):
        return NotImplementedError

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


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
                 tokens,
                 use_cuda):
        """
        An instance of a feed-forward discriminator.

        Args:

            input_size:
                int, input size of the feed-forward network.

            hidden_size:
                int, hidden size of the feed forward neural network.

            learning_rate:
                float, learning rate of the optimizer.

            use_cuda:
                bool, true if cuda support is enabled.
        """
        super().__init__(hidden_size=hidden_size,
                         optimizer_type=optimizer_type,
                         input_size=input_size,
                         learning_rate=learning_rate,
                         tokens=tokens,
                         use_cuda=use_cuda)

        self._input_layer = Linear(input_size, hidden_size)
        self._hidden_layer = Linear(hidden_size, hidden_size)
        self._output_layer = Linear(hidden_size, self._output_size)
        self._activation = LeakyReLU()

        if self._use_cuda:
            self._input_layer = self._input_layer.cuda()
            self._hidden_layer = self._hidden_layer.cuda()
            self._output_layer = self._output_layer.cuda()
            self._activation = self._activation.cuda()

        self._optimizer = Optimizer(parameters=self.parameters(),
                                    optimizer_type=optimizer_type,
                                    scheduler_type='ReduceLROnPlateau',
                                    learning_rate=self._learning_rate)


    def forward(self, inputs):
        """
        Forward step for the discriminator.

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
        output = torch.sigmoid(self._output_layer(output))

        return output

    @property
    def optimizer(self):
        return self._optimizer


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
                 tokens,
                 use_cuda):
        """
        An instance of a recurrent discriminator.

        Args:
            input_size:
                int, input size of the feed-forward network.

            hidden_size:
                int, hidden size of the feed forward neural network.

            learning_rate:
                float, learning rate of the optimizer.

            use_cuda:
                bool, true if cuda support is enabled.
        """
        super().__init__(hidden_size=hidden_size,
                         optimizer_type=optimizer_type,
                         input_size=input_size,
                         learning_rate=learning_rate,
                         tokens=tokens,
                         use_cuda=use_cuda)

        self._recurrent_layer = torch.nn.GRU(input_size=input_size,
                                             hidden_size=hidden_size,
                                             batch_first=True)

        self._output_layer = Linear(self._hidden_size, self._output_size)

        if self._use_cuda:
            self._recurrent_layer = self._recurrent_layer.cuda()
            self._output_layer = self._output_layer.cuda()

        self._optimizer = Optimizer(parameters=self.parameters(),
                                    optimizer_type=optimizer_type,
                                    scheduler_type='ReduceLROnPlateau',
                                    learning_rate=self._learning_rate)

    def forward(self, inputs):
        """
        Forward step for the discriminator.

        Args:
            inputs:
                Variable, (batch_size, input_size), where input_size is equal to the encoder's
                hidden_size.

        Returns:
            final_output:
                Variable, (batch_size, 1).
        """
        outputs, _ = self._recurrent_layer.forward(inputs)
        final_output = torch.sigmoid(self._output_layer(outputs[:, -1, :]))

        return final_output

    @property
    def optimizer(self):
        return self._optimizer


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

        Args:
            embedding_size:
                Int, size of the word vectors.

            vocab_size:
                Int, number of words in the vocab.

            use_cuda:
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

        if use_cuda:
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


class Noise:
    """

    """

    def __init__(self, use_cuda, p=0.1, k=3):
        """


        Args:
            p:

            k:

        """
        self._use_cuda = use_cuda
        self._p = p
        self._k = k

    def __call__(self, inputs, padding):
        """


        Args:
            inputs:

        Returns:
            noisy_inputs:

        """

        return self._drop_out(inputs, padding)

    def _drop_out(self, inputs, padding_value):
        """


        Args:
            inputs:

            padding_value:

        Returns:
            outputs:

            lengths:

        """
        inputs = inputs.cpu().numpy()
        noisy_inputs = numpy.zeros((inputs.shape[0], inputs.shape[1] + 1))
        mask = numpy.array(numpy.random.rand(inputs.shape[0], inputs.shape[1] - 1) > self._p, dtype=numpy.int32)
        noisy_inputs[:, 1:-1] = mask * inputs[:, 1:]
        noisy_inputs[:, 0] = inputs[:, 0]
        for index in range(inputs.shape[0]):
            remaining = noisy_inputs[index][noisy_inputs[index] != 0]
            padding = numpy.array([padding_value]*len(noisy_inputs[index][noisy_inputs[index] == 0]))
            padding[-1] = remaining.shape[0]
            noisy_inputs[index, :] = numpy.concatenate((remaining, padding))

        noisy_inputs = noisy_inputs[numpy.argsort(-noisy_inputs[:, -1])]

        return torch.from_numpy(noisy_inputs[:, :-1]).long(), numpy.array(noisy_inputs[:, -1], dtype=int)


class Translator(Component):

    abstract = True

    interface = OrderedDict({
        'vocabs':   None
    })

    def __init__(self, vocabs):
        self._vocabs = vocabs

    def translate(self, inputs,  source_lang_index, target_lang_index):
        pass


class NaiveTranslator(Translator):

    abstract = False

    def __init__(self, vocabs):
        super().__init__(vocabs)

    def translate(self, inputs,  source_lang_index, target_lang_index):
        pass

