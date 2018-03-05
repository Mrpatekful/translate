import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError
