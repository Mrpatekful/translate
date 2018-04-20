import matplotlib.pyplot as plt

import pickle
import logging
import sys
import os
import re

from os.path import join


class Plot:

    @staticmethod
    def plot(*args, **kwargs):
        return NotImplementedError


class Data:

    def __init__(self):
        pass

    def add(self, identifier, value):
        return NotImplementedError


class TextData(Data):

    def __init__(self):
        super().__init__()
        self._values = {}

    def __repr__(self):
        text = ''
        for key in self._values:
            text += f'\n{key}:'
            for log in self._values[key]:
                for line in log:
                    text += line
                text += '\n'
        return text

    def add(self, identifier, value):
        if self._values.get(identifier, None) is None:
            self._values[identifier] = []
        self._values[identifier].append(value)


class ScalarData(Data, Plot):

    @staticmethod
    def summed_average(scalar_iterable, identifiers=None):
        return (sum([scalar.average(identifiers) for scalar in scalar_iterable])
                / len(scalar_iterable))

    # noinspection PyMethodOverriding
    @staticmethod
    def plot(iterable, epoch_range, identifiers=None, plot_size=5):
        plt.figure(1, figsize=(plot_size, plot_size))

        if epoch_range is not None:
            assert isinstance(epoch_range, tuple), 'Epoch range must be a tuple.'
            assert epoch_range[0] < epoch_range[1], 'Epoch range must be a valid interval.'
            epoch_range = list(range(*epoch_range, 1))
        else:
            epoch_range = list(range(len(scalar_iterable)))

        data = []

        for epoch, scalar in enumerate(scalar_iterable):
            if epoch in epoch_range:
                data.append(scalar.average())

    def __init__(self):
        super().__init__()
        self._values = {}

    def add(self, identifier, value):
        if self._values.get(identifier, None) is None:
            self._values[identifier] = [0, 0]
        self._values[identifier][0] += value
        self._values[identifier][1] += 1

    def average(self, identifiers=None):
        if identifiers is None:
            identifiers = self._values.keys()
        if len(self._values) == 0:
            return 0
        return (sum([self._values[key][0]/self._values[key][1] for key in self._values if key in identifiers])
                / len(self._values))


class LatentStateData(Data, Plot):

    def __init__(self):
        super().__init__()
        self._values = {}

    def add(self, identifier, value):
        return NotImplementedError

    # noinspection PyMethodOverriding
    @staticmethod
    def plot(x, y, plot_size):
        pass


class AttentionData(Data, Plot):

    def __init__(self):
        super().__init__()

    # noinspection PyMethodOverriding
    @staticmethod
    def plot(x, y, plot_size):
        plt.plot()


class DataLog:

    TRAIN_DATA_ID = 'train_id'
    MUTUAL_TOKEN_ID = 'mutual_token'

    def __init__(self, data_interface):
        self._data = dict(zip(list(data_interface.keys()), [data_type() for data_type in data_interface.values()]))
        self._data_interface = data_interface
        self._identifiers = set()

    # noinspection PyUnresolvedReferences
    def add(self, identifier, key, value):
        self._data[key].add(identifier, value)
        self._identifiers.add(identifier)

    @property
    def data(self):
        return self._data

    @property
    def identifiers(self):
        return self._identifiers


class DataLogContainer:

    def __init__(self, log_dict, mutual_log_iterable, language_log_iterable_dict):
        self._mutual_data_logs = {}
        self._language_data_logs = {}
        self._log_dict = log_dict

        # for data_log in mutual_log_iterable:
        #     for key in data_log.data:
        #         if key not in self._mutual_data_logs:
        #             self._mutual_data_logs[key] = []
        #         self._mutual_data_logs[key].append(data_log[key])
        #
        # language_tokens = list(language_log_iterable_dict.keys())
        # for index in range(len(language_log_iterable_dict[language_tokens[0]])):
        #     for key in language_log_iterable_dict[language_tokens[0]][0].data:
        #         if key not in self._language_data_logs:
        #             self._language_data_logs[key] = dict(zip(language_tokens,
        #                                                      [[] for _ in range(len(language_tokens))]))
        #         for language in language_tokens:
        #             self._language_data_logs[key][language].append(language_log_iterable_dict[language][index])

    def _plot_mutual_data(self, keys):
        self._mutual_data_logs[key][0].plot(iterable=[self._mutual_data_logs[key]], keys=key)

    def _plot_language_specific_data(self, keys, identifier):
        pass

    def _plot(self, keys, tokens, identifiers=None):
        if identifiers is None:
            identifiers = [DataLog.TRAIN_DATA_ID]

        assert DataLog.MUTUAL_TOKEN_ID not in tokens or (tokens[0] == DataLog.MUTUAL_TOKEN_ID
                                                         and tokens[-1] == DataLog.MUTUAL_TOKEN_ID), ''

        for key in keys:
            type(self._log_dict[token][0]).plot(iterable=[self._log_dict[token]])


class Analyzer:

    _meta_file = '.meta'

    def __init__(self, directory):
        self._directory = directory
        self._train_meta, self._validation_meta = self._load_meta_data()
        self._train_ids = None
        self._validation_ids = None

    def _load_meta_data(self):
        try:
            _meta = pickle.load(open(join(self._directory, self._meta_file), 'rb'))

        except FileNotFoundError:
            logging.error(''' ' .meta ' file was not found.''')
            sys.exit()

        return _meta['training_log'], _meta['validation_log']

    def show_available_metrics(self):
        def print_format(element):
            return f'< {str(element)} > ,'

        print('> [Train metrics]:\t%s\n\n> [Validation metrics]:\t%s' %
              ('\n\t\t\t'.join(map(print_format, self._train_meta)),
               '\n\t\t\t'.join(map(print_format, self._validation_meta))))

    def show_validation_identifiers(self):
        def find_int(x):
            try:
                result = re.search('^\d+', x.split('_')[1])
            except IndexError:
                return 0
            if result is None:
                return 0
            else:
                return int(result.group(0))

        output_files = sorted(os.listdir(self._directory), key=find_int)

        last_log = pickle.load(open(join(self._directory, output_files[-1]), 'rb'))

        self._validation_ids = {index: data_log.identifiers for index, data_log
                                in enumerate(last_log['validation_log'])}

        self._train_ids = {index: data_log.identifiers for index, data_log
                           in enumerate(last_log['training_log'])}

        print('Identifiers: {\n%s\n}\n' % (', '.join(map(str, self._validation_ids[0]))))

    # noinspection PyTypeChecker
    def plot_train_logs(self, epoch_range=None):
        def add_to_language(l_id, ep, dat):
            if l_id not in logs:
                logs[l_id] = {}
            logs[l_id][ep] = dat

        def add_to_mutual(ep, dat):
            if 'mutual' not in logs:
                logs['mutual'] = {}
            logs['mutual'][ep] = dat

        if epoch_range is not None:
            assert isinstance(epoch_range, tuple), 'Epoch range must be a tuple.'
            assert epoch_range[0] < epoch_range[1], 'Epoch range element 0 must be less than element 1.'

        output_files = os.listdir(self._directory)

        if len(output_files) < 1:
            print('No output files have been found.')
            return

        logs = {}
        for file in output_files:
            if re.match('^outputs_[0-9]+.pt$', file):
                log = pickle.load(open(join(self._directory, file), 'rb'))
                for language_id in range(len(log['training_log'])-1):
                    add_to_language(language_id, log['epoch'], log['training_log'][language_id].data)
                add_to_mutual(log['epoch'], log['training_log'][-1].data)

        num_languages = len(logs)-1

        try:

            data = {index: dict(zip(list(logs[0][0].keys()), [[] for _ in range(len(logs[0][0].keys()))]))
                    for index in range(num_languages)}

            data = {**data, 'mutual': dict(zip(list(logs['mutual'][0].keys()),
                    [[] for _ in range(len(list(logs['mutual'][0].keys())))]))}

            epochs = range(epoch_range[0], epoch_range[1]) \
                if epoch_range is not None else range(len(logs[0]))

            for epoch in epochs:
                for language in logs:
                    for key in logs[language][epoch]:
                        data[language][key].append(float(logs[language][epoch][key].average()))

        except KeyError as error:
            raise ValueError(f'{error} has not been found in the logs.')

        plt.figure(1, figsize=(5 * (num_languages + 1), 5))

        for index, key in enumerate(data):

            plt.subplot(int(f'1{num_languages+1}{index+1}'))
            epochs = list(range(epoch_range[0], epoch_range[1]) if epoch_range is not None else range(len(logs[0])))

            for data_label in data[key]:
                plt.plot(epochs, data[key][data_label], '--', label=data_label)
                plt.xticks(epochs)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')

            plt.legend(loc='upper right', fontsize='small')

        plt.show()

    def plot_validation_logs(self, language_id, identifiers, metrics=None):
        pass
