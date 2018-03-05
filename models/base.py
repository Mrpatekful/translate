import torch.nn as nn


class Model(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError

    def step(self):
        return NotImplementedError

    def zero_grad(self):
        return NotImplementedError

    @classmethod
    def assemble(cls, params):
        return NotImplementedError

    @classmethod
    def abstract(cls):
        return True
