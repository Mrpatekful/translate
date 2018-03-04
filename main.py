import numpy
import torch

from utils import parser
from utils import tasks


numpy.random.seed(2)
torch.manual_seed(2)

CONFIG_PATH = 'configs/rnn.json'


def main():
    parameters = parser.parse_params(CONFIG_PATH)
    seq2seq_trainer = tasks.UnsupervisedTranslation(**parameters)
    seq2seq_trainer.fit_model(epochs=10)


if __name__ == '__main__':
    main()
