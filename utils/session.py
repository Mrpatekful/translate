import torch

import os
import re
import numpy

from os.path import join

from utils.utils import call

numpy.set_printoptions(precision=4)


class Session:

    def __init__(self, task, checkpoint_dir, output_dir):
        self._task = task
        self._checkpoint_dir = checkpoint_dir
        self._output_dir = output_dir
        self._state = self._load()

        if self._state is not None:
            self._task.state = self._state['task']

    def _load(self):
        def find_int(x):
            result = re.search('^\d+', x.split('_')[1])
            if result is None:
                return 0
            else:
                return int(result.group(0))

        checkpoints = sorted(
            os.listdir(self._checkpoint_dir),
            key=lambda x: find_int(x)
        )
        if len(checkpoints) == 0:
            self._last_log = 0
            return None

        else:
            self._last_log = find_int(checkpoints[-1])
            return torch.load(join(self._checkpoint_dir, checkpoints[-1]))

    def _save_state(self, state):
        checkpoint = join(self._checkpoint_dir, f'checkpoint_{self._last_log + 1}.pt')
        self._last_log += 1
        torch.save(state, checkpoint)

    def evaluate(self):
        if self._state is None:
            raise RuntimeError('There is no available model for evaluation.')
        with EvaluationContext(session=self) as ec:
            ec.evaluate('Test')

    def train(self):
        with TrainingContext(session=self) as tc:
            for epoch in tc.epochs:

                tc.epoch = epoch

                train_log = tc.train()

                with EvaluationContext(session=self) as ec:
                    ec.evaluate('Validation')
                    ec.save(tc.epoch, train_log)

                self._save_state({
                    'task':     self._task.state,
                    'epoch':    epoch
                })

    @property
    def task(self):
        return self._task

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def state(self):
        return self._state


class TrainingContext:

    EPOCHS = 10000

    def __init__(self, session):
        self._session = session
        self.epoch = 0
        if self._session.state is not None:
            self.epoch = self._session.state['epoch'] + 1
        self._start_epoch = self.epoch

    def __enter__(self):
        call('train', self._session.task.input_pipelines)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        call('train', self._session.task.input_pipelines, {'boolean': False})

    def train(self):
        def average_as_list(key):
            return numpy.array([float(outputs[index].data[key].average()) for index in range(len(outputs))])

        print(f'\nEpoch: {self.epoch}\n')
        outputs = self._session.task.train()
        print('\nTrain outputs:\n')
        print(f'''Total Loss::          {average_as_list('total_loss')}\n''')
        print(f'''Translation Loss:     {average_as_list('translation_loss')}\n''')
        print(f'''Auto-Encoding Loss:   {average_as_list('auto_encoding_loss')}\n''')
        print(f'''Reguralization Loss:  {average_as_list('reguralization_loss')}\n''')
        print(f'''Discriminator Loss:   {average_as_list('discriminator_loss')}\n''')

        return outputs

    @property
    def epochs(self):
        return range(self._start_epoch, self.EPOCHS, 1)


class EvaluationContext:

    def __init__(self, session):
        self._session = session
        self._output_dir = self._session.output_dir
        self._outputs = None
        self._get_last_log()

    def __enter__(self):
        call('eval', self._session.task.input_pipelines)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        call('eval', self._session.task.input_pipelines, {'boolean': False})

    def evaluate(self, mode):
        def average_as_list(key):
            return numpy.array([float(self._outputs[index].data[key].average()) for index in range(len(self._outputs))])

        def data_as_list(key):
            return [self._outputs[index].data[key] for index in range(len(self._outputs))]

        self._outputs = self._session.task.evaluate()

        print(f'\n{mode} outputs:\n')
        print(f'''Total Loss:           {average_as_list('total_loss')}\n''')
        print(f'''Translation Loss:     {average_as_list('translation_loss')}\n''')
        print(f'''Auto-Encoding Loss:   {average_as_list('auto_encoding_loss')}\n''')
        print(f'''Reguralization Loss:  {average_as_list('reguralization_loss')}\n''')
        print(f'''Discriminator Loss:   {average_as_list('discriminator_loss')}\n''')
        print(f'''Texts:                {data_as_list('auto_encoding_text')}\n''')

    def save(self, epoch, train_log):
        log = {
            'epoch':            epoch,
            'validation_log':   self._outputs,
            'training_log':     train_log
        }
        output_file = join(self._output_dir, f'outputs_{self._last_log + 1}.pt')
        self._last_log += 1
        torch.save(log, output_file)

    def _get_last_log(self):
        def find_int(x):
            result = re.search('^\d+', x.split('_')[1])
            if result is None:
                return 0
            else:
                return int(result.group(0))

        analysis_files = sorted(
            os.listdir(self._output_dir),
            key=lambda x: find_int(x)
        )
        if len(analysis_files) == 0:
            self._last_log = 0
        else:
            self._last_log = find_int(analysis_files[-1])
            print(self._last_log)
