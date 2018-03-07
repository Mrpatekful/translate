import torch.nn as nn


class Encoder(nn.Module):
    """
    Abstract base class for the encoder modules of the application. An encoder must
    inherit from this class, otherwise it won't be discoverable by the hierarchy
    builder utility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError

    @classmethod
    def assemble(cls, params):
        return NotImplementedError

    @classmethod
    def abstract(cls):
        return True
