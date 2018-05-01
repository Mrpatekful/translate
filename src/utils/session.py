import logging
import os
import re
from os.path import join

import numpy
import torch
import pickle

from src.utils.analysis import DataLog

from src.utils.utils import call


class Session:


    def __init__(self, task, checkpoint_dir, output_dir, info_dir, clear=False):
        numpy.set_printoptions(precision=4)
        self._task = task
        self._checkpoint_dir = checkpoint_dir
        self._output_dir = output_dir
        self._info_dir = info_dir

        if clear:
            self._clear_logs()

        self._state = self._load()
        if self._state is not None:
            self._task.state = self._state['task']

    def _clear_logs(self):
        for file in os.listdir(self._checkpoint_dir):
            os.remove(join(self._checkpoint_dir, file))
        for file in os.listdir(self._output_dir):
            os.remove(join(self._output_dir, file))

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
            return torch.load(join(self._checkpoint_dir, checkpoints[-1]))

    def _save_state(self, state):
        checkpoint = join(self._checkpoint_dir, f'checkpoint_{self._last_log + 1}.pt')
        self._last_log += 1
        torch.save(state, checkpoint)

    def evaluate(self):
        if self._state is None:
            raise RuntimeError('There is no available model for evaluation.')
        with ValidationContext(session=self) as ec:
            ec.evaluate('Test')

    def train(self):
        with TrainingContext(session=self) as tc:
            for epoch in tc.epochs:

                tc.epoch = epoch

                train_log = tc.train()

                with ValidationContext(session=self) as ec:
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def train(self):
        def average_as_list(key):
            if key in outputs[DataLog.MUTUAL_TOKEN_ID].data:
                return numpy.array([outputs[DataLog.MUTUAL_TOKEN_ID].data[key].average()])
            else:
                return numpy.array([float(outputs[data_id].data[key].average()) for data_id in outputs if
                                    data_id != DataLog.MUTUAL_TOKEN_ID])

        logging.info(f'Processing Epoch {self.epoch} ...')

        outputs = self._session.task.train(self.epoch)

        logging.info(f'''Train | Total Model Loss:     {average_as_list('total_loss')}''')
        logging.info(f'''Train | Translation Loss:     {average_as_list('translation_loss')}''')
        logging.info(f'''Train | Auto-Encoding Loss:   {average_as_list('auto_encoding_loss')}''')
        logging.info(f'''Train | Reguralization Loss:  {average_as_list('reguralization_loss')}''')
        logging.info(f'''Train | Discriminator Loss:   {average_as_list('discriminator_loss')}''')

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

    def evaluate(self, mode):
        def average_as_list(key):
            if key in self._outputs[DataLog.MUTUAL_TOKEN_ID].data:
                return numpy.array([self._outputs[DataLog.MUTUAL_TOKEN_ID].data[key].average()])

            else:
                return numpy.array([float(self._outputs[data_id].data[key].average())
                                    for data_id in self._outputs if data_id != DataLog.MUTUAL_TOKEN_ID])

        def data_as_list(key):
            return [self._outputs[index].data[key] for index in range(len(self._outputs))]

        self._outputs = self._session.task.evaluate()

        logging.info(f'''{mode} | Total Model Loss:     {average_as_list('total_loss')}''')
        logging.info(f'''{mode} | Translation Loss:     {average_as_list('translation_loss')}''')
        logging.info(f'''{mode} | Auto-Encoding Loss:   {average_as_list('auto_encoding_loss')}''')
        logging.info(f'''{mode} | Reguralization Loss:  {average_as_list('reguralization_loss')}''')
        logging.info(f'''{mode} | Discriminator Loss:   {average_as_list('discriminator_loss')}''')
        # logging.info(f'''Texts:                {data_as_list('auto_encoding_text')}''')

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


class InferenceContext:
    pass
