from torch.nn import Module

from utils.utils import Component


class Encoder(Module, Component):
    """
    Abstract base class for the encoder modules of the application. An encoder must
    inherit from this class, otherwise it won't be discoverable by the hierarchy
    builder utility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError

    @property
    def optimizers(self):
        return NotImplementedError

    @property
    def state(self):
        return NotImplementedError
