import numpy
import torch
import sys
import argparse

from utils.config import Config
from utils.session import Session

numpy.random.seed(2)
torch.manual_seed(2)

TASK_CONFIG = 'configs/tasks/unmt.json'


def main():
    task = Config(TASK_CONFIG).assemble()

    session = Session(task)
    session.start()

    # if sys.argv[1] == 'clear':
    #     session.start()
    # elif sys.argv[1] == 'load':
    #     session.load(sys.argv[2])
    #     session.start()
    # elif sys.argv[1] == 'eval':
    #     session.load(sys.argv[2])
    #     session.eval()


if __name__ == '__main__':
    main()
