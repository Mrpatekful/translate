import torch

import os
import re

from os.path import join

from utils.utils import call


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

    def _save_state(self, state):
        checkpoint = join(self._log_dir, 'checkpoint_%d.pt' % (self._last_log + 1))
        self._last_log += 1
        torch.save(state, checkpoint)

    def _save_outputs(self, outputs):
        pass

    def evaluate(self):
        if self._state is None:
            raise RuntimeError('There is no available model for evaluation.')
        with EvaluationContext(session=self) as ec:
            ec.evaluate()

    def train(self):
        with TrainingContext(session=self) as tc:
            for epoch in tc.epochs:

                tc.epoch = epoch

                tc.train()

                with EvaluationContext(session=self) as ec:
                    ec.evaluate()

                self._save_state({
                    'task':   self._task.state,
                    'epoch':  epoch
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
        self._start_epoch = self.epoch

    def __enter__(self):
        call('train', self._session.task.input_pipelines)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        call('train', self._session.task.input_pipelines, {'boolean': False})

    def train(self):
        print(f'\nEpoch: {self.epoch}\n')
        outputs = self._session.task.train()
        print(f'\nAverage loss: {outputs[0]:.5}\n')

    @property
    def epochs(self):
        return range(self._start_epoch, self.EPOCHS, 1)


class EvaluationContext:

    def __init__(self, session):
        self._session = session

    def __enter__(self):
        call('eval', self._session.task.input_pipelines)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        call('eval', self._session.task.input_pipelines, {'boolean': False})

    def evaluate(self):
        outputs = self._session.task.evaluate()
        print('\nOutputs:\n')
        for output in outputs:
            print(output['symbols'])
