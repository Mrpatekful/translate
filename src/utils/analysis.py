"""

"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator


from collections import OrderedDict

import pickle
import numpy
import logging
import sys
import os
import re
import tqdm

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.scores import precision, accuracy, recall, f_measure

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from os.path import join


class Plot:
    """

    """

    PLOT_SIZE = 8

    @staticmethod
    def display(data, plot_size, epochs, epoch_range, identifiers=None, **params):
        raise NotImplementedError


class Data:
    """

    """

    def __init__(self):
        self._values = {}

    def add(self, identifier, value):
        raise NotImplementedError

    def get_required_keys(self):
        raise NotImplementedError

    @property
    def data(self):
        return self._values

    @property
    def identifiers(self):
        return list(self._values.keys())


class TextData(Data, Plot):
    """

    """

    @staticmethod
    def display(data, plot_size, epochs, epoch_range, identifiers=None, **params):
        print('Showing the last state of the epoch range.\n')
        languages = list(data.keys()) if identifiers is None else list(identifiers.keys())
        for language in languages:
            metrics, entries = data[language][epoch_range][-1].calculate_metrics()
            print('Averaged metrics of %s data with %d entries:\n' % (language, entries))
            for metric in metrics:
                print('> {}:\t{:.3}'.format(metric, metrics[metric]))
            print('\n')
            for identifier in data[language][epoch_range][-1].data:
                if identifiers is None or identifier in identifiers[language]:
                    print(data[language][epoch_range][-1].as_str(identifier))

    def __init__(self):
        super().__init__()

    def as_str(self, identifier):
        input_text = self._values[identifier]['input_text']
        target_text = self._values[identifier].get('target_text', None)
        output_text = self._values[identifier]['output_text']

        if target_text is None:
            return'[%d]\n>Input: %s\n>Output: %s\n\n' % (identifier, ' '.join(input_text), ' '.join(output_text))
        else:
            cc = SmoothingFunction()
            bleu_score = sentence_bleu([target_text], output_text, smoothing_function=cc.method4)
            return'[{}]\n> Input: {}\n> Output: {}\n> Target: {}\nBLEU: {:.2}\n\n'.format(
                identifier, ' '.join(input_text), ' '.join(output_text), ' '.join(target_text), float(bleu_score))

    def add(self, identifier, value):
        self._values[identifier] = value

    def get_required_keys(self):
        return ()

    def calculate_metrics(self):
        included_logs = 0
        metrics = {}
        cc = SmoothingFunction()
        for identifier in self._values:
            if self._values[identifier].get('target_text', None) is not None:
                included_logs += 1
                target_text = self._values[identifier]['target_text']
                output_text = self._values[identifier]['output_text']
                metrics['BLEU'] = metrics.get('BLEU', 0) + sentence_bleu([target_text], output_text,
                                                                         smoothing_function=cc.method4)
                metrics['accuracy'] = metrics.get('accuracy', 0) + accuracy(target_text, output_text)
                target_text = set(target_text)
                output_text = set(output_text)
                metrics['precision'] = metrics.get('precision', 0) + precision(target_text, output_text)
                metrics['recall'] = metrics.get('recall', 0) + recall(target_text, output_text)
                metrics['f_measure'] = metrics.get('f_measure', 0) + f_measure(target_text, output_text)

        if included_logs != 0:
            for metric in metrics:
                metrics[metric] /= included_logs

        return metrics, included_logs


class ScalarData(Data, Plot):
    """

    """

    @staticmethod
    def summed_average(scalar_iterable, identifiers=None):
        return (sum([scalar.average(identifiers) for scalar in scalar_iterable])
                / len(scalar_iterable))

    # noinspection PyMethodOverriding
    @staticmethod
    def display(data, plot_size, epochs, epoch_range, identifiers=None):
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

    def get_required_keys(self):
        return ()


class LatentStateData(Data, Plot):
    """

    """

    @staticmethod
    def display(data, plot_size, epochs, epoch_range, identifiers=None, **params):
        def draw_line(p1, p2, c='blue'):
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color=c, linewidth=0.5)

        print('Showing the last state of the epoch range.')

        dimension = params.get('dimension', 2)
        analyzer_type = params.get('analyzer_type', 'PCA')
        with_progression = params.get('with_progression', False)
        distinct = params.get('distinct', False)

        analyzer = LatentStateData._create_analyzer(data, epochs[-1], dimension, analyzer_type)

        plt.figure(figsize=(15, 15), dpi=80)

        colors = dict(zip(list(data.keys()), ['black', 'tab:orange', 'blue', 'green']))

        plot_data = {}

        if identifiers is None:
            languages = list(data.keys())
        else:
            languages = list(identifiers.keys())

        for language in languages:
            plot_data[language] = {}
            for identifier in data[language][epoch_range][-1].data:
                if identifiers is None or identifier in identifiers[language]:
                    data_log = data[language][epoch_range][-1].data[identifier]

                    hidden_state = data_log['hidden_state']
                    hidden_state = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state

                    plot_data[language][' '.join(data_log['input_text'])] = \
                        analyzer.transform(hidden_state.data.cpu().numpy()[-1, 0, :].reshape(1, -1)).reshape(-1)

        if dimension == 2:
            for language in plot_data:
                _data = numpy.array(list(plot_data[language].values()))
                x = _data[:, 0]
                y = _data[:, 1]
                plt.scatter(x, y, marker='x', label=language, color=colors[language])
                plt.legend(loc='upper right', fontsize='x-large')
                if identifiers is not None:
                    for sentence in plot_data[language]:
                        color = colors[language]
                        x_coord, y_coord = plot_data[language][sentence][0], plot_data[language][sentence][1]
                        plt.annotate(sentence, xy=(x_coord, y_coord), xytext=(0, 0), textcoords='offset points',
                                     fontsize=19, color=color, weight='bold')

            if with_progression:
                analyzers = []
                for epoch in epochs:
                    analyzers.append(LatentStateData._create_analyzer(data, epoch, dimension, analyzer_type))

                progression_data = {}
                for language in languages:
                    progression_data[language] = {}
                    for index, epoch_log in enumerate(data[language][epoch_range]):
                        for identifier in epoch_log.data:
                            if identifiers is None or identifier in identifiers[language]:
                                data_log = epoch_log.data[identifier]

                                hidden_state = data_log['hidden_state']
                                hidden_state = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state

                                if progression_data[language].get(identifier, None) is None:
                                    progression_data[language][identifier] = []

                                progression_data[language][identifier].append(
                                    analyzers[index].transform(hidden_state.data.cpu().numpy()[-1, 0, :]
                                                                       .reshape(1, -1)).reshape(-1))

                for language in progression_data:
                    for identifier in progression_data[language]:
                        for index in range(len(progression_data[language][identifier])-1):
                            if distinct:
                                draw_line(progression_data[language][identifier][index],
                                          progression_data[language][identifier][index+1], colors[language])
                            else:
                                draw_line(progression_data[language][identifier][index],
                                          progression_data[language][identifier][index + 1])


        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.title('Visualization of the multilingual encoder latent space')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _create_analyzer(data, epoch, dimension, analyzer_type):

        assert dimension == 2 or dimension == 3, 'dim must be 2 or 3'
        assert analyzer_type == 'PCA' or analyzer_type == 'TSNE', 'analyzer must either be PCA or TSNE'

        train_data = []
        for key in data:
            for identifier in data[key][epoch].data:
                hidden_state = data[key][epoch].data[identifier]['hidden_state']
                hidden_state = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
                train_data = [
                    *train_data,
                    hidden_state.data.cpu().numpy()[-1, 0, :].reshape(-1)
                    ]

        if analyzer_type == 'PCA':
            analyzer = PCA(n_components=dimension, whiten=True)
            analyzer.fit(numpy.array(train_data))
        else:
            analyzer = TSNE(n_components=dimension, n_iter=3000, verbose=2)
            analyzer.fit(numpy.array(train_data))

        return analyzer

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
    def display(data, plot_size, epochs, epoch_range, identifiers=None, **params):

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
        ax.imshow(weights, cmap=cm.Greys)

        ax.set_xticks(numpy.arange(len(output_words)))
        ax.set_yticks(numpy.arange(len(input_words)))

        ax.set_xlabel('Output words')
        ax.set_ylabel('Input words')

        ax.set_xticklabels(output_words)
        ax.set_yticklabels(input_words)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(input_words)):
            for j in range(len(output_words)):
                ax.text(j, i, weights[i, j], ha="center", va="center", color="tab:orange")

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

    def display(self, metric, tokens, identifiers, plot_size, epoch_range=(None, ), **kwargs):
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

        type(self._log_dict[metric][tokens[0]][0]).display(
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

    TEST_FILE = 'test.pt'
    EVAL_FILE = 'eval.pt'
    OUTPUT_DIR = 'outputs'

    _meta_file = '.meta'

    TRAIN_MODE = 'train'
    VALIDATION_MODE = 'validation'
    TEST_MODE = 'test'
    EVALUATION_MODE = 'evaluation'

    def __init__(self, directory):
        self._directory = directory
        self._train_meta, self._validation_meta = self._load_train_meta_data()

        self._validation_ids = None
        self._train_ids = None

        self._train_log_container = None
        self._validation_log_container = None

        self._test_data = None
        self._test_metrics = None
        if os.path.isfile(os.path.join(self._directory, self.TEST_FILE)):
            self._test_data = pickle.load(open(os.path.join(self._directory, self.TEST_FILE), 'rb'))

        self._evaluation_data = None
        self._evaluation_metrics = None
        if os.path.isfile(os.path.join(self._directory, self.EVAL_FILE)):
            self._evaluation_data = pickle.load(open(os.path.join(self._directory, self.EVAL_FILE), 'rb'))

    def _load_train_meta_data(self):
        try:
            _meta = pickle.load(open(join(self._directory, self.OUTPUT_DIR, self._meta_file), 'rb'))

        except FileNotFoundError:
            logging.error(''' ' .meta ' file was not found.''')
            sys.exit()

        return _meta['training_log'], _meta['validation_log']

    def show_available_metrics(self):
        def print_format(element):
            return f'< {element} > ,'

        print('> [Train metrics]:\t%s\n\n> [Validation metrics]:\t%s\n' %
              ('\n\t\t\t'.join(map(print_format, self._train_meta)),
               '\n\t\t\t'.join(map(print_format, self._validation_meta))))

        if self._test_data is not None:
            keys = []
            for data_id in self._test_data:
                keys = [*keys, *list(self._test_data[data_id].data.keys())]
            self._test_metrics = set(keys)
            print('> [Test metrics]:\t%s\n' % '\n\t\t\t'.join(map(print_format, self._test_metrics)))

        if self._evaluation_data is not None:
            keys = []
            for data_id in self._test_data:
                keys = [*keys, *list(self._test_data[data_id].data.keys())]
            self._evaluation_metrics = set(keys)
            print('> [Evaluation metrics]:\t%s' % '\n\t\t\t'.join(map(print_format, self._evaluation_metrics)))

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

        output_files = sorted(os.listdir(join(self._directory, self.OUTPUT_DIR)), key=find_int)

        last_log = pickle.load(open(join(self._directory, self.OUTPUT_DIR, output_files[-1]), 'rb'))

        self._validation_ids = {index: data_log.identifiers for index, data_log
                                in enumerate(last_log['validation_log'])}

        self._train_ids = {index: data_log.identifiers for index, data_log
                           in enumerate(last_log['training_log'])}

        print('Identifiers: {\n%s\n}\n' % (', '.join(map(str, self._validation_ids[0]))))

    def _load_logs(self):
        output_files = os.listdir(join(self._directory, self.OUTPUT_DIR))

        if len(output_files) < 1:
            print('No output files have been found.')
            return

        self._train_log_container = DataLogContainer()
        self._validation_log_container = DataLogContainer()

        logs = []

        for file in output_files:
            if re.match('^outputs_[0-9]+.pt$', file):
                logs.append(pickle.load(open(join(self._directory, self.OUTPUT_DIR, file), 'rb')))

        for log in sorted(logs, key=lambda x: x['epoch']):
            self._train_log_container.add(log['training_log'])
            self._validation_log_container.add(log['validation_log'])

    def display(self, metric, mode, plot_size=Plot.PLOT_SIZE, tokens=None, epoch_range=(None,),
                identifiers=None, **kwargs):

        self._load_logs()
        if mode == self.TRAIN_MODE:
            assert metric in self._train_log_container.metrics, f'{metric} is not a train metric.'
            self._train_log_container.display(metric=metric,
                                              epoch_range=epoch_range,
                                              tokens=tokens,
                                              identifiers=identifiers,
                                              plot_size=plot_size,
                                              **kwargs)
        elif mode == self.VALIDATION_MODE:
            assert metric in self._validation_log_container.metrics, f'{metric} is not a validation metric.'
            self._validation_log_container.display(metric=metric,
                                                   epoch_range=epoch_range,
                                                   tokens=tokens,
                                                   identifiers=identifiers,
                                                   plot_size=plot_size,
                                                   **kwargs)
        elif mode == self.TEST_MODE:
            assert metric in self._validation_log_container.metrics, f'{metric} is not a validation metric.'
            pass

        elif mode == self.EVALUATION_MODE:
            assert metric in self._validation_log_container.metrics, f'{metric} is not a validation metric.'
            pass

        else:
            raise ValueError(f'Mode must one of the following: {self.TRAIN_MODE}, '
                             f'{self.VALIDATION_MODE}, {self.TEST_MODE}, {self.EVALUATION_MODE}')


def create_report(corpora, vocab):
    """


    Arguments:
        corpora:

        vocab:

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


def create_embedding_analyzer(vocab_paths, save_path, dimension=2, analyzer_type='PCA'):
    """


    Arguments:
        vocab_paths:

        dimension:

        save_path:

        analyzer_type:

    """
    train_data = []

    try:

        assert dimension == 2 or dimension == 3, 'dim must be 2 or 3'
        assert analyzer_type == 'PCA' or analyzer_type == 'TSNE', 'analyzer must either be PCA or TSNE'

        for path in vocab_paths:
            with open(path, 'r') as f:
                with tqdm.tqdm() as p_bar:
                    p_bar.set_description('Collecting vocab data')
                    for line in f:
                        p_bar.update()
                        line_as_list = line.strip().split()
                        train_data.append(list(map(float, line_as_list[1:])))

        if analyzer_type == 'PCA':
            transformer = PCA(n_components=dimension, whiten=True)
            transformer.fit(numpy.array(train_data))
        else:
            transformer = TSNE(n_components=dimension, n_iter=3000, verbose=2)
            transformer.fit(numpy.array(train_data))

        analyzer_path = os.path.join(save_path, f'{analyzer_type.lower()}-analyzer')
        pickle.dump(transformer, open(analyzer_path, 'wb'))
        logging.info(f'{analyzer_type} analyzer has been created and dumped to {analyzer_path}')

    except AssertionError as error:
        del train_data
        logging.error(error)

    except MemoryError:
        del train_data
        logging.error('Out of memory')


def analyze_embeddings(vocab_paths, words, analyzer_path, dim=2):
    """


    Arguments:
        vocab_paths:

        words:

        analyzer_path:

        dim:

    """
    try:

        analyzer = pickle.load(open(analyzer_path, 'rb'))

        annotations = {}
        points = []

        for index, language in enumerate(vocab_paths):
            annotations[language] = {}
            with open(vocab_paths[language], 'r') as f:
                with tqdm.tqdm() as p_bar:
                    p_bar.set_description(f'Searching for words in {language} vocab')
                    for c, line in enumerate(f):
                        p_bar.update()
                        line_as_list = line.strip().split()
                        if line_as_list[0] in words[language]:
                            annotations[language][line_as_list[0]] = analyzer.transform(numpy.array(
                                line_as_list[1:], dtype=float).reshape(1, -1)).reshape(-1)
                        if c % 100 == 0:
                            points.append([*analyzer.transform(numpy.array(
                                line_as_list[1:], dtype=float).reshape(1, -1)).reshape(-1), index])

        for language in words:
            for word in words[language]:
                assert word in annotations[language], f'{word} is not in the vocabulary'

    except AssertionError as error:
        logging.error(error)
        return

    plt.figure(figsize=(15, 15), dpi=80)

    colors = dict(zip(list(vocab_paths.keys()), ['black', 'tab:orange', 'blue', 'green']))

    if dim == 2:
        for language in annotations:
            data = numpy.array(list(annotations[language].values()))
            x = data[:, 0]
            y = data[:, 1]
            plt.scatter(x, y, marker='x', label=language, color=colors[language])
            plt.legend(loc='upper right', fontsize='x-large')
            for word in annotations[language]:
                color = colors[language]
                x_coord, y_coord = annotations[language][word][0], annotations[language][word][1]
                plt.annotate(word, xy=(x_coord, y_coord), xytext=(0, 0), textcoords='offset points', fontsize=19,
                             color=color, weight='bold')

        points = numpy.array(points)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['grey', 'tab:orange'])
        # Plot the training points
        plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap=cm_bright, alpha=0.1)


    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.title('Visualization of the multilingual word embedding space')
    plt.tight_layout()
    plt.show()
