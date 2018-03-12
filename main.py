import numpy
import torch

from utils.config import Config

numpy.random.seed(2)
torch.manual_seed(2)

TASK_CONFIG = 'configs/tasks/unmt.json'


def main():
    task = Config(TASK_CONFIG).assemble()
    # task.load_checkpoint()
    task.fit_model(epochs=1)


if __name__ == '__main__':
    main()
