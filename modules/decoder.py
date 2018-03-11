from torch.nn import Module

from utils.utils import Component


class Decoder(Module, Component):
    """
    Abstract base class for the decoder modules of the application. A decoder must
    inherit from this class, otherwise it won't be discoverable by the hierarchy
    builder utility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError
