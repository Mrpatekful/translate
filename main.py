import numpy
import torch

from utils.config import Config

numpy.random.seed(2)
torch.manual_seed(2)

TASK_CONFIG = 'configs/tasks/nmt.json'
MODEL_CONFIG = 'configs/models/sts.json'


def main():
    task = Config(task_config=TASK_CONFIG, model_config=MODEL_CONFIG).assemble()
    task.fit_model(epochs=10)


if __name__ == '__main__':
    main()
