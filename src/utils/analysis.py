import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import MaxNLocator


from collections import OrderedDict

import pickle
import numpy
import logging
import sys
import os
import re
import faiss
import tqdm

from os.path import join


class Plot:
    """

    """

    PLOT_SIZE = 8

    @staticmethod
    def plot(data, plot_size, epochs, epoch_range, identifiers=None, **params):
        return NotImplementedError


class Data:
    """

    """

    def __init__(self):
        self._values = {}

    def add(self, identifier, value):
        return NotImplementedError

    def get_required_keys(self):
        return ()

    @property
    def data(self):
        return self._values

    @property
    def identifiers(self):
        return list(self._values.keys())


class TextData(Data):
    """

    """

    def __init__(self):
        super().__init__()

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
    """

    """

    @staticmethod
    def summed_average(scalar_iterable, identifiers=None):
        return (sum([scalar.average(identifiers) for scalar in scalar_iterable])
                / len(scalar_iterable))

    # noinspection PyMethodOverriding
    @staticmethod
    def plot(data, plot_size, epochs, epoch_range, identifiers=None):

        if identifiers is None:
            identifiers = [None]

        plt.figure(1, figsize=(plot_size * 1.5, plot_size * len(identifiers)))

        for index, identifier in enumerate(identifiers):
            plt.subplot(int(f'{len(identifiers)}1{index+1}'))
            if identifier is None:
                _identifier = None
                plt.title('Mutual')
            else:
                _identifier = [identifier]
                plt.title(identifier)
            for token_id in data:
                plt.plot(epochs, list(map(lambda x: x.average(identifiers=_identifier),
                                          data[token_id][epoch_range])), '--', label=token_id)
                plt.xticks(epochs)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            if len(data.keys()) > 1:
                plt.legend(loc='upper right', fontsize='medium')

        plt.tight_layout()
        plt.show()

    def __init__(self):
        super().__init__()

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
    """

    """

    @staticmethod
    def plot(data, plot_size, epochs, epoch_range, identifiers=None, **params):

        assert len(data.keys()) > 2, 'At least two languages must be given'

        attn_data = data[list(data.keys())[0]][epoch_range][-1]

        input_words = attn_data.data[identifiers[0]]['input_text']

        weights = numpy.around(numpy.transpose(attn_data.data[identifiers[0]]['hidden_state'][0]), decimals=2)

        fig, ax = plt.subplots(figsize=(plot_size, plot_size * len(identifiers)))

        ax.grid(color='r', linestyle='--', linewidth=2)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fig.tight_layout()
        plt.show()

    def __init__(self):
        super().__init__()

    def add(self, identifier, value):
        self._values[identifier] = value

    def get_required_keys(self):
        return 'input_text', 'hidden_state'


class AttentionData(Data, Plot):
    """

    """

    @staticmethod
    def plot(data, plot_size, epochs, epoch_range, identifiers=None, **params):

        assert len(data.keys()) == 1, 'Only a single language can be plotted'
        assert identifiers is not None, 'At least 1 identifier must be given'
        assert isinstance(identifiers, int) or isinstance(identifiers, str), \
            'Invalid identifier type, must be a string or int'

        print('Showing the last element of the epoch range.')

        attn_data = data[list(data.keys())[0]][epoch_range][-1]

        input_words = attn_data.data[identifiers]['input_text']
        output_words = attn_data.data[identifiers]['output_text']

        weights = numpy.around(numpy.transpose(attn_data.data[identifiers]['alignment_weights'][0]), decimals=2)

        fig, ax = plt.subplots(figsize=(plot_size, plot_size))
        im = ax.imshow(weights, cmap=cm.Greys)

        ax.set_xticks(numpy.arange(len(output_words)))
        ax.set_yticks(numpy.arange(len(input_words)))

        ax.set_xlabel('Output words')
        ax.set_ylabel('Input words')

        ax.set_xticklabels(output_words)
        ax.set_yticklabels(input_words)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(output_words)):
            for j in range(len(input_words)):
                text = ax.text(j, i, weights[i, j], ha="center", va="center", color="tab:orange")

        fig.tight_layout()
        plt.show()

    def __init__(self):
        super().__init__()

    def get_required_keys(self):
        return 'input_text', 'output_text', 'alignment_weights'

    def add(self, identifier, value):
        self._values[identifier] = value


class DataLog:
    """

    """

    TRAIN_DATA_ID: str = '<TRAIN>'
    MUTUAL_TOKEN_ID: str = '<MUTUAL>'

    def __init__(self, data_interface):
        self._data = dict(zip(list(data_interface.keys()),
                              [data_type() for data_type in data_interface.values()]))
        self._data_interface = data_interface
        self._identifiers = set()

    # noinspection PyUnresolvedReferences
    def add(self, identifier, key, value):
        self._data[key].add(identifier, value)
        self._identifiers.add(identifier)

    def get_required_keys(self, key):
        return self._data[key].get_required_keys()

    @property
    def data(self):
        return self._data

    @property
    def identifiers(self):
        return self._identifiers


class DataLogContainer:
    """

    """

    def __init__(self):
        self._log_dict = {}

    def add(self, data_logs):
        for log_key in data_logs:
            for metric in data_logs[log_key].data:
                if metric not in self._log_dict:
                    self._log_dict[metric] = {}
                if log_key not in self._log_dict[metric]:
                    self._log_dict[metric][log_key] = []
                self._log_dict[metric][log_key].append(data_logs[log_key].data[metric])

    def plot(self, metric, tokens, identifiers, plot_size, epoch_range=(None, ), **kwargs):
        if tokens is None:
            tokens = list(self._log_dict[metric].keys())

        assert isinstance(epoch_range, tuple), 'Epoch range must be a tuple.'

        # Checking epoch range

        if epoch_range[0] is not None and len(epoch_range) > 1 and epoch_range[1] is not None:
            assert epoch_range[0] < epoch_range[1], 'Epoch range must be a valid interval.'

        if epoch_range[0] is not None and len(epoch_range) == 1:
            epoch_range = (epoch_range[0], None)

        assert DataLog.MUTUAL_TOKEN_ID not in \
            tokens or (tokens[0] == DataLog.MUTUAL_TOKEN_ID and tokens[-1] == DataLog.MUTUAL_TOKEN_ID), \
            f'Provided tokens must not contain {DataLog.MUTUAL_TOKEN_ID}'

        data = dict(zip(tokens, [self._log_dict[metric][token] for token in tokens]))
        token_ids = list(data.keys())

        # Creating a list of numbers, marking the logged epochs of the data

        if epoch_range[0] is not None and epoch_range[1] is not None:
            epochs = list(range(*epoch_range))
        elif epoch_range[0] is not None:
            epochs = list(range(epoch_range[0], len(data[token_ids[0]])))
        else:
            epochs = list(range(len(data[token_ids[0]])))

        epoch_range = slice(*epoch_range)

        type(self._log_dict[metric][tokens[0]][0]).plot(
            data=data,
            plot_size=plot_size,
            epochs=epochs,
            epoch_range=epoch_range,
            identifiers=identifiers,
            **kwargs)

    @property
    def metrics(self):
        return list(self._log_dict.keys())


class Analyzer:
    """

    """

    _meta_file = '.meta'

    TRAIN_MODE = 'train'
    VALIDATION_MODE = 'validation'

    def __init__(self, directory):
        self._directory = directory
        self._train_meta, self._validation_meta = self._load_meta_data()
        self._validation_ids = None
        self._train_ids = None
        self._train_log_container = None
        self._validation_log_container = None

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

    def _load_logs(self):
        output_files = os.listdir(self._directory)

        if len(output_files) < 1:
            print('No output files have been found.')
            return

        self._train_log_container = DataLogContainer()
        self._validation_log_container = DataLogContainer()

        logs = []

        for file in output_files:
            if re.match('^outputs_[0-9]+.pt$', file):
                logs.append(pickle.load(open(join(self._directory, file), 'rb')))

        for log in sorted(logs, key=lambda x: x['epoch']):
            self._train_log_container.add(log['training_log'])
            self._validation_log_container.add(log['validation_log'])

    def plot(self, metric, mode, plot_size=Plot.PLOT_SIZE, tokens=None, epoch_range=(None, ), identifiers=None):
        self._load_logs()
        if mode == self.TRAIN_MODE:
            assert metric in self._train_log_container.metrics, f'{metric} is not a train metric.'
            self._train_log_container.plot(metric=metric,
                                           epoch_range=epoch_range,
                                           tokens=tokens,
                                           identifiers=identifiers,
                                           plot_size=plot_size)
        elif mode == self.VALIDATION_MODE:
            assert metric in self._validation_log_container.metrics, f'{metric} is not a validation metric.'
            self._validation_log_container.plot(metric=metric,
                                                epoch_range=epoch_range,
                                                tokens=tokens,
                                                identifiers=identifiers,
                                                plot_size=plot_size)
        else:
            raise ValueError(f'Mode must one of the following: {self.TRAIN_MODE}, {self.VALIDATION_MODE}')


class TextAnalyzer:
    """

    """
    @staticmethod
    def create_report(corpora, vocab):
        """

        Args:

        """
        vocab_set = set()

        with open(vocab, 'r') as f:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Collecting vocab data')
                for line in f:
                    p_bar.update()
                    vocab_set.add(line.strip().split()[0])

        vocab_size = len(vocab_set)
        
        words = {}
        line_lengths = {}
        unknown_lines = {}
        word_frequency = OrderedDict()
        lines = 0

        with open(corpora, 'r') as f:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Collecting corpora data')
                for line in f:
                    p_bar.update()
                    lines += 1
                    line_as_list = line.strip().split()
                    line_lengths[len(line_as_list)] = line_lengths.get(len(line_as_list), 0) + 1
                    unknown_word_count = 0
                    for word in line_as_list:
                        if word not in vocab_set:
                            unknown_word_count += 1
                        if words.get(word, None) is not None:
                            word_frequency[words[word]] -= 1
                            if word_frequency[words[word]] == 0:
                                del word_frequency[words[word]]
                        words[word] = words.get(word, 0) + 1
                        word_frequency[words[word]] = word_frequency.get(words[word], 0) + 1
                    unknown_lines[unknown_word_count] = unknown_lines.get(unknown_word_count, 0) + 1

        known_words = 0

        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Checking vocab coverage')
            for word in words:
                p_bar.update()
                if word in vocab_set:
                    known_words += 1
        
        del words

        if len(unknown_lines) > 1:
            fig, (ax_lines, ax_words) = plt.subplots(2, 1, figsize=(15, 10))

            ax_words.bar(list(unknown_lines.keys()), list(unknown_lines.values()))

            ax_words.set_title('Lines with unknown words')

            ax_words.set_xticks(list(unknown_lines.keys()))
            ax_words.get_xaxis().set_major_locator(MaxNLocator(integer=True))

            ax_words.set_xlabel('Unknown word count')
            ax_words.set_ylabel('Number of lines')

        else:
            fig, ax_lines = plt.subplots(figsize=(15, 5))
        
        ax_lines.set_title('Sequence lengths')

        ax_lines.bar(list(line_lengths.keys()), list(line_lengths.values()))
        ax_lines.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        ax_lines.set_xlabel('Length')
        ax_lines.set_ylabel('Number of lines')
        
        sparse = 0
        for num_word in word_frequency:
            if num_word < 5:
                sparse += word_frequency[num_word]

        all_words = sum(word_frequency.values())

        print(f'Number of lines:                                    {lines}')
        print(f'Average line length:                                '
              f'{sum([key*line_lengths[key] for key in line_lengths])/lines:.4}')
        print(f'Number of unique words in the corpora:              {all_words}')
        print(f'Number of words, with an occurrence of less than 5: '
              f'{sparse} ({float(sparse/all_words)*100:.4}% of unique words)')
        print(f'Number of singleton words:                          '
              f'{word_frequency.get(1, 0)} ({float(word_frequency.get(1, 0)/all_words*100):.4}% of unique words)')
        print(f'Vocab coverage:                                     {float(known_words/all_words)*100:.4}%')
        print(f'Vocab usage:                                        {float(known_words/vocab_size)*100:.4}%')
        if len(unknown_lines) == 1:
            print(f'Number of unknown words:                            {list(unknown_lines.keys())[0]}/line')

        fig.tight_layout()
        plt.show()
