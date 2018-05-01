import argparse

import numpy
import torch
import logging

from src.utils.config import Config
from src.utils.session import Session

numpy.random.seed(2)
torch.manual_seed(2)

EXPERIMENT_CONFIG = 'configs/experiments/unmt_1.json'


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiment', action='store', dest='config', default=EXPERIMENT_CONFIG)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--train', action='store_true', dest='train')
    group.add_argument('-i', '--inference', action='store_true', dest='inference')
    parser.add_argument('-c', '--clear', action='store_true', dest='clear')
    parser.add_argument('-v', '--verbose', action='store', dest='log_level', default=logging.INFO)

    arguments = parser.parse_args(['-t', '-c'])

    clear_logs = arguments.clear if arguments.train else False

    session = Session(*Config(arguments.config, arguments.log_level).assemble(), clear=clear_logs)

    if arguments.train:
        session.train()
    else:
        session.evaluate()


if __name__ == '__main__':
    main()
