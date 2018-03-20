import torch

import os
import re

from os.path import join

from utils.utils import execute


class Session:

    def __init__(self, task, log_dir):
        self._task = task
        self._log_dir = log_dir
        self._state = self._load()

        if self._state is not None:
            self._task.state = self._state['task']

    def _load(self):
        checkpoints = sorted(os.listdir(self._log_dir), key=lambda x: x.split('_')[1])
        if len(checkpoints) == 0:
            self._last_log = 0
            return None

        else:
            self._last_log = int(re.search('\d', checkpoints[-1]).group(0))
            return torch.load(join(self._log_dir, checkpoints[-1]))

    def _save(self, state):
        checkpoint = join(self._log_dir, 'checkpoint_%d.pt' % (self._last_log + 1))
        self._last_log += 1
        torch.save(state, checkpoint)

    def evaluate(self):
        if self._state is None:
            raise RuntimeError('There is no available model for evaluation.')
        with EvaluationContext(session=self) as ec:
            ec.evaluate()

    def train(self):
        with TrainingContext(session=self) as tc:
            for epoch in tc.epochs:
                tc.train()

                with EvaluationContext(session=self) as ec:
                    ec.evaluate()

                tc.epoch = epoch
                self._save({
                    'task': self._task.state,
                    'epoch': epoch
                })

    @property
    def task(self):
        return self._task

    @property
    def state(self):
        return self._state


class TrainingContext:

    EPOCHS = 10000

    def __init__(self, session):
        self._session = session
        self.epoch = 0
        if self._session.state is not None:
            self.epoch = self._session.state['epoch']

    def __enter__(self):
        execute('train', self._session.task.readers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        execute('train', self._session.task.readers, {'boolean': False})

    def train(self):
        print('Epoch: %d' % self.epoch)
        outputs = self._session.task.train()
        print(outputs)

    @property
    def epochs(self):
        return range(self.epoch, self.EPOCHS, 1)


class EvaluationContext:

    def __init__(self, session):
        self._session = session

    def __enter__(self):
        execute('eval', self._session.task.readers)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        execute('eval', self._session.task.readers, {'boolean': False})

    def evaluate(self):
        outputs = self._session.task.evaluate()
        for output in outputs:
            print(output['symbols'])
