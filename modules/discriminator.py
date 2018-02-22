import torch
from torch import nn


class Discriminator(nn.Module):

    def __init__(self, input_dim, hidden_dim,
                 learning_rate, use_cuda):
        super(Discriminator, self).__init__()

        self._input_layer = nn.Linear(input_dim, hidden_dim)
        self._hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self._hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self._output_layer = nn.Linear(hidden_dim, 1)
        self._activation = nn.LeakyReLU()

        if use_cuda:
            self._input_layer = self._input_layer.cuda()
            self._hidden_layer_1 = self._hidden_layer_1.cuda()
            self._hidden_layer_2 = self._hidden_layer_2.cuda()
            self._output_layer = self._output_layer.cuda()
            self._activation = self._activation.cuda()

        self._optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        output = self._activation(self._input_layer(inputs))
        output = self._activation(self._hidden_layer_1(output))
        output = self._activation(self._hidden_layer_2(output))
        output = self._activation(self._output_layer(output))
        # TODO smoothing coefficient 0.1
        return output

    @property
    def optimizer(self):
        """

        :return:
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """

        :param optimizer:
        :return:
        """
        self._optimizer = optimizer
