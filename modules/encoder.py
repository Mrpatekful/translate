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

    def get_optimizer_states(self):
        return NotImplementedError

    def set_optimizer_states(self, *args, **kwargs):
        return NotImplementedError
