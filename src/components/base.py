from torch.nn import Module

from src.utils.utils import Component


class Encoder(Module, Component):
    """
    Abstract base class for the encoder modules of the application. An encoder must
    inherit from this class, otherwise it won't be discoverable by the hierarchy
    builder utility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def optimizers(self):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError

    @state.setter
    def state(self, value):
        raise NotImplementedError


class Decoder(Module, Component):
    """
    Abstract base class for the decoder modules of the application. A decoder must
    inherit from this class, otherwise it won't be discoverable by the hierarchy
    builder utility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._output_size = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def optimizers(self):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError

    @state.setter
    def state(self, value):
        raise NotImplementedError

    @property
    def output_size(self):
        return self._output_size
