import numpy
import torch
import argparse

from utils.config import Config
from utils.session import Session


numpy.random.seed(2)
torch.manual_seed(2)

TASK_CONFIG = 'configs/tasks/unmt.json'


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', action='store', dest='config', default=TASK_CONFIG)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--train', action='store_true', dest='train')
    group.add_argument('-e', '--eval', action='store_true', dest='eval')

    arguments = parser.parse_args(['-e'])

    task, log_dir = Config(arguments.config).assemble()

    session = Session(task, log_dir)

    if arguments.train:
        session.train()
    else:
        session.evaluate()


if __name__ == '__main__':
    main()
