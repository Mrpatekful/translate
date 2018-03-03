import numpy
import torch

from utils import parser
from utils import trainer
from utils import reader

from models.seqtoseq import SeqToSeq

numpy.random.seed(2)
torch.manual_seed(2)

CONFIG_PATH = 'configs/rnn.json'


def main():
    parameters = parser.parse_params(CONFIG_PATH)
    seq2seq_trainer = trainer.UnsupervisedTrainer(**parameters)
    seq2seq_trainer.fit(epochs=10)


if __name__ == '__main__':
    main()
