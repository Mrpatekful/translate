import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError

    @classmethod
    def assemble(cls, params):
        return NotImplementedError

    @classmethod
    def abstract(cls):
        return True
