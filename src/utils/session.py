"""

"""

import logging
import os
import re
from os.path import join

import numpy
import torch
import pickle
import sys

from src.utils.analysis import DataLog, ScalarData


class Session:

    CHECKPOINT_DIR = 'checkpoints'
    OUTPUT_DIR = 'outputs'
    LOG_DIR = 'info.log'

    INTERRUPT = 'interrupt'

    def __init__(self, experiment, model_dir, clear=False):
        numpy.set_printoptions(precision=4)
        self.experiment = experiment
        self._model_dir = model_dir
        self._checkpoint_dir = os.path.join(model_dir, Session.CHECKPOINT_DIR)
        self._output_dir = os.path.join(model_dir, Session.OUTPUT_DIR)
        self._info_dir = os.path.join(model_dir, Session.LOG_DIR)
        self.interrupted = False

        if not os.path.exists(self._checkpoint_dir):
            os.mkdir(self._checkpoint_dir)
        if not os.path.exists(self._output_dir):
            os.mkdir(self._output_dir)
        if not os.path.exists(self._info_dir):
            os.mkdir(self._output_dir)

        if clear:
            self._clear_logs()

        self._state = self._load()
        if self._state is not None:
            self.experiment.state = self._state['task']

    def _clear_logs(self):
        try:
            for file in os.listdir(self._checkpoint_dir):
                os.remove(join(self._checkpoint_dir, file))
            for file in os.listdir(self._output_dir):
                os.remove(join(self._output_dir, file))
        except FileNotFoundError as error:
            logging.error(f'Directory does not exists {error}')
            sys.exit()

    def _load(self):
        def find_int(x):
            try:
                result = re.search('^\d+', x.split('_')[1])
            except IndexError:
                return 0
            if result is None:
                return 0
            else:
                return int(result.group(0))

        checkpoints = sorted(
            os.listdir(self._checkpoint_dir),
            key=find_int
        )

        if len(checkpoints) == 0:
            self._last_log = 0
            return None
        else:
            self._last_log = find_int(checkpoints[-1])
            candidates = [file for file in checkpoints if int(find_int(file)) == self._last_log]
            if len(candidates) == 2:
                checkpoint = sorted(candidates)[-1]
            else:
                checkpoint = checkpoints[-1]
            if checkpoint.split('_')[0] == Session.INTERRUPT:
                self.interrupted = True
                logging.info(f'Found an interrupted experiment. Loading the latest state.')
            else:
                self.interrupted = False
            return torch.load(join(self._checkpoint_dir, checkpoint))

    def save_state(self, state, name='checkpoint'):
        if name == 'checkpoint':
            self._last_log += 1
        checkpoint = join(self._checkpoint_dir, f'{name}_{self._last_log}.pt')
        torch.save(state, checkpoint)

    def test(self):
        if self._state is None:
            raise RuntimeError('There is no available model for testing.')
        with TestContext(session=self) as tc:
            tc.test()
            tc.save()

    def evaluate(self):
        if self._state is None:
            raise RuntimeError('There is no available model for evaluation.')
        with EvaluationContext(session=self) as ec:
            ec.evaluate()
            ec.save()

    def train(self):
        with TrainingContext(session=self) as tc:
            for epoch in tc.epochs:

                tc.epoch = epoch

                train_log = tc.train()

                with ValidationContext(session=self) as vc:
                    vc.validate()
                    vc.save(tc.epoch, train_log)

                self.save_state({
                    'task':     self.experiment.state,
                    'epoch':    epoch
                })
                self.interrupted = False

    @property
    def task(self):
        return self.experiment

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def model_dir(self):
        return self._model_dir

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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.save_state({
            'task':     self._session.experiment.state,
            'epoch':    self.epoch - 1
        }, Session.INTERRUPT)
        logging.info(f'Training interrupted {exc_type}')

    def train(self):
        logging.info(f'Processing Epoch {self.epoch} ...')

        outputs = self._session.task.train(self.epoch)

        keys = []
        for data_id in outputs:
            keys = [*keys, *list(outputs[data_id].data.keys())]

        for key in set(keys):
            data = average_as_list(outputs=outputs, key=key)
            if data is not None:
                logging.info(f'''Train | {key}:     {data}''')

        return outputs

    @property
    def epochs(self):
        return range(self._start_epoch, self.EPOCHS, 1)


class ValidationContext:

    def __init__(self, session):
        self._session = session
        self._output_dir = self._session.output_dir
        self._outputs = None
        self._get_last_log()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def validate(self):
        self._outputs = self._session.task.validate()

        keys = []
        for data_id in self._outputs:
            keys = [*keys, *list(self._outputs[data_id].data.keys())]

        for key in set(keys):
            data = average_as_list(outputs=self._outputs, key=key)
            if data is not None:
                logging.info(f'''Validation | {key}:     {data}''')

    def save(self, epoch, train_log):
        log = {
            'epoch':            epoch,
            'validation_log':   self._outputs,
            'training_log':     train_log
        }
        output_file = join(self._output_dir, f'outputs_{self._last_log + 1}.pt')
        if self._last_log == 0:
            language_token = [key for key in self._outputs if key != DataLog.MUTUAL_TOKEN_ID][0]
            meta = {
                'epoch': 0,
                'validation_log': [
                    *self._outputs[DataLog.MUTUAL_TOKEN_ID].data.keys(),
                    *self._outputs[language_token].data.keys()
                ],
                'training_log': [
                    *train_log[DataLog.MUTUAL_TOKEN_ID].data.keys(),
                    *train_log[language_token].data.keys()
                ]
            }
            pickle.dump(meta, open(join(self._output_dir, '.meta'), 'wb'))
        self._last_log += 1
        pickle.dump(log, open(output_file, 'wb'))

    def _get_last_log(self):
        def find_int(x):
            try:
                result = re.search('^\d+', x.split('_')[1])
            except IndexError:
                return 0
            if result is None:
                return 0
            else:
                return int(result.group(0))

        analysis_files = sorted(
            os.listdir(self._output_dir),
            key=find_int
        )

        if len(analysis_files) == 0:
            self._last_log = 0
        else:
            self._last_log = find_int(analysis_files[-1])


class TestContext:

    TEST_FILE = 'test.pt'

    def __init__(self, session):
        self._session = session
        self._output_dir = os.path.join(self._session.model_dir, TestContext.TEST_FILE)
        self._outputs = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def test(self):
        self._outputs = self._session.task.test()

    def save(self):
        pickle.dump(self._outputs, open(self._output_dir, 'wb'))


class EvaluationContext:

    EVAL_FILE = 'eval.pt'

    def __init__(self, session):
        self._session = session
        self._output_dir = os.path.join(self._session.model_dir, EvaluationContext.EVAL_FILE)
        self._outputs = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def evaluate(self):
        self._outputs = self._session.task.evaluate()

    def save(self):
        pickle.dump(self._outputs, open(self._output_dir, 'wb'))


def average_as_list(outputs, key):
    if isinstance(outputs[DataLog.MUTUAL_TOKEN_ID].data.get(key, None), ScalarData):
        return numpy.array([outputs[DataLog.MUTUAL_TOKEN_ID].data[key].average()])

    elif all([isinstance(outputs[data_id].data.get(key, None), ScalarData) for data_id in outputs
              if data_id != DataLog.MUTUAL_TOKEN_ID]):
        return numpy.array([float(outputs[data_id].data[key].average()) for data_id in outputs if
                            data_id != DataLog.MUTUAL_TOKEN_ID])
