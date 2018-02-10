import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import reader


def main():
    rd = reader.Lang()
    voc = rd.load_vocab('/home/patrik/GitHub/NMT/eng_vec')


if __name__ == '__main__':
    main()
