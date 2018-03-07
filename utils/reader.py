import numpy as np

import torch
from torch.autograd import Variable

from utils import utils
from tqdm import trange

import sklearn.utils
import copy


def parallel_data_loader(data_path, separator_token='<SEP>'):
    """
    Loader function for parallel data.
    :return: list, the data stored as a list of strings.
    """
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            data.append(line[:-1].split(separator_token))
    if len(data) == 0:
        raise ValueError('The given file is empty.')
    return data


def mono_data_loader(data_path):
    """
    Loader function for parallel data.
    :return: list, the data stored as a list of strings.
    """
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            data.append(line)
    if len(data) == 0:
        raise ValueError('The given file is empty.')
    return data


def ids_from_sentence(language, sentence):
    """
    Convenience method, for converting a sequence of words to ids.
    :param language: Language, object of the language to use the look up of.
    :param sentence: string, a tokenized sequence of words.
    :return: list, containing the ids (int) of the sentence in the same order.
    """
    return [language.word_to_id[word] for word in sentence.split(' ')[:-1]]


def sentence_from_ids(language, ids):
    """
    Convenience method, for converting a sequence of ids to words.
    :param language:
    :param ids:
    :return:
    """
    return [language.id_to_word[word_id] for word_id in ids]


class DataQueue:
    """
    A queue object for the data feed. This can be later configured to load the data to
    memory asynchronously.
    """

    def __init__(self, data_path, max_segment_size):
        """
        :param data_path: str, location of the data.
        """
        self._MAX_LEN = 10

        self._data_path = data_path
        self._max_segment_size = max_segment_size
        self._data_segment_queue = []

    def data_generator(self):
        """
        Data is retrieved directly from a file, and loaded into data chunks of size MAX_CHUNK_SIZE.
        :return: list of strings, a data chunk.
        """
        with open(self._data_path, 'r') as file:
            data_segment = []
            for line in file:
                if len(data_segment) < self._max_segment_size:
                    data_segment.append(line[:-1])
                else:
                    temp_data_segment = copy.deepcopy(data_segment)
                    del data_segment[:]
                    yield temp_data_segment
            yield data_segment  # the segment is not filled, returning with the remaining values


class Reader:
    """
    Derived classes should implement the reading logic for the seq2seq model. Readers divide the
    data into segments. The purpose of this behaviour, is to keep the sentences with similar lengths
    in segments, so they can be freely shuffled without mixing them together with larger sentences.
    """

    def batch_generator(self):
        """
        The role of this function is to generate batches for the seq2seq model. The batch generation
        should include the logic of shuffling the samples. A full iteration should include
        all of the data samples.
        :return: PyTorch padded-sequence object.
        """
        return NotImplementedError

    def print_validation_format(self, *args, **kwargs):
        return NotImplementedError

    def assemble(self, params):
        return NotImplementedError

    @classmethod
    def abstract(cls):
        return True


class FileReader(Reader):  # TODO
    """
    An implementation of the reader class. Batches are read from the source in file real-time.
    This version of the reader should only be used if the source file is too large to be stored
    in memory.
    """

    def __init__(self,
                 language,
                 data_path,
                 batch_size,
                 use_cuda,
                 max_segment_size):
        """
        An instance of a file reader.
        :param language: Language, instance of the used language object.
        :param data_path: str, absolute path of the data location.
        :param batch_size: int, size of the input batches.
        :param use_cuda: bool, True if the device has cuda support.
        """
        self._language = language
        self._use_cuda = use_cuda
        self._batch_size = batch_size
        self._data_queue = DataQueue(data_path, max_segment_size)
        self._max_segment_size = max_segment_size

    def batch_generator(self):
        """
        Generator for mini-batches. Data is read indirectly from a file through a DataQueue object.
        :return: Torch padded-sequence object
        """
        for data_segment in self._data_queue.data_generator():
            shuffled_data_segment = sklearn.utils.shuffle(data_segment)
            for index in range(0, len(shuffled_data_segment)-self._batch_size, self._batch_size):
                batch = sorted(shuffled_data_segment[index:index + self._batch_size],
                               key=lambda x: len(x), reverse=True)

                yield self._variable_from_sentences(batch)

    def _variable_from_sentences(self, sentences):
        """
        Creates PyTorch Variable object from a tokenized sequence.
        :param sentences: string, a tokenized sequence of words.
        :return: Variable, containing the ids of the sentence.
        """
        ids = np.empty((len(sentences), len(sentences[0].split(' '))-2))
        for index in range(len(sentences)):
            ids[index, :len(sentences[index])] = np.array(ids_from_sentence(self._language, sentences[index]))

        wrapped_ids = Variable(torch.from_numpy(ids))
        if self._use_cuda:
            return wrapped_ids.cuda()
        else:
            return wrapped_ids

    def print_validation_format(self, *args, **kwargs):
        return NotImplementedError

    @classmethod
    def assemble(cls, params):
        return NotImplementedError

    @classmethod
    def abstract(cls):
        return False

    @property
    def language(self):
        """
        Property for the reader object's source language.
        :return: Language object, the source language.
        """
        return self._language


class FastReader(Reader):
    """
    A faster implementation of reader class than FileReader. The source data is fully loaded into
    the memory.
    """

    def __init__(self,
                 language,
                 data_path,
                 batch_size,
                 use_cuda,
                 corpus_type,
                 padding_type,
                 format_batch,
                 max_segment_size):
        """
        An instance of a fast reader.
        :param language: Language, instance of the used language object.
        :param data_path: str, absolute path of the data location.
        :param batch_size: int, size of the input batches.
        :param use_cuda: bool, True if the device has cuda support.
        :param corpus_type: str, type of the corpus, that could either be monolingual, or parallel.
        :param padding_type: str, type of padding that will be used during training. The sequences in
                             the mini-batches may vary in length, so padding must be applied to convert
                             them to equal lengths.
        :param format_batch: function, that defines the form of batches, that will be generated by
                             the batch generator function. This method must be provided by the task.
        :param max_segment_size: int, the size of each segment, that will contain the similar length data.
        """
        self._data_path = data_path
        self._language = language
        self._use_cuda = use_cuda
        self._batch_size = batch_size
        self._max_segment_size = max_segment_size
        self._format_batch = format_batch

        padding = utils.subclasses(Padding)

        self._data_processor = padding[padding_type](language, max_segment_size)

        data_loaders = {
            'Monolingual': mono_data_loader,
            'Parallel': parallel_data_loader
        }

        self._data = self._data_processor(data_loaders[corpus_type](data_path))

    def batch_generator(self):
        """
        Generator for mini-batches. Data is read from memory. The _format_batch function comes from the
        definition of the task. It is a wrapper function that transform the generated batch of data into a form,
        that is convenient for the current task.
        :return: tuple, a PyTorch Variable of dimension (Batch_size, Sequence_length), containing
                 the ids of words, sorted by their length in descending order. Each sample is
                 padded to the length of the longest sequence in the batch/segment.
                 The latter behaviour may vary. Second element of the tuple is a numpy array
                 of the lengths of the original sequences (without padding).
        """
        for data_segment in self._segment_generator():
            shuffled_data_segment = sklearn.utils.shuffle(data_segment)
            for index in range(0, len(shuffled_data_segment)-self._batch_size, self._batch_size):
                batch = self._data_processor.create_batch(shuffled_data_segment[index:index + self._batch_size])
                yield self._format_batch(batch, self._use_cuda)

    def _segment_generator(self):
        """
        Divides the data to segments of size MAX_SEGMENT_SIZE.
        """
        t = trange(0, len(self._data), self._max_segment_size)
        for index in t:
            yield copy.deepcopy(self._data[index:index + self._max_segment_size])

    def print_validation_format(self, **kwargs):
        """
        Convenience function for printing the parameters of the function, to the standard output.
        The parameters must be provided as keyword arguments. Each argument must contain a 2D
        array containing word ids, which will be converted to the represented words from the
        dictionary of the language, used by the reader instance.
        """
        id_batches = np.array(list(kwargs.values()))
        expression = ''
        for index, ids in enumerate(zip(*id_batches)):
            expression += '{%d}:\n' % index
            for param in zip(kwargs, ids):
                expression += ('> [%s]:\t%s\n' % (param[0], '\t'.join(sentence_from_ids(self._language, param[1]))))
            expression += '\n'
        print(expression)

    @classmethod
    def abstract(cls):
        return False

    @classmethod
    def assemble(cls, params):
        return {
            'use_cuda':         params['use_cuda'],
            'data_path':        params['corpus']['params']['path'],
            'batch_size':       params['batch_size'],
            'max_segment_size': params['max_segment_size'],
            'corpus_type':      params['corpus']['type'],
            'padding_type':     params['padding_type'],
            'language':         utils.Language(**{
                **params['corpus']['params']['embedding'],
                'use_cuda': params['use_cuda'],
            })
        }

    @property
    def language(self):
        """
        Property for the reader object's source language.
        :return: Language object, the source language.
        """
        return self._language


class Padding:

    def create_batch(self, ):
        return NotImplementedError

    @classmethod
    def abstract(cls):
        return True


class PostPadding(Padding):
    """
    Data is padded during the training iterations. Padding is determined by the longest
    sequence in the batch.
    """

    def __init__(self, language, max_segment_size):
        """
        An instance of a post-padder object.
        :param language: Language, instance of the used language object.
        """
        super().__init__()
        self._language = language
        self._max_segment_size = max_segment_size

    def __call__(self, data):
        """
        Converts the data of (string) sentences to ids of the words.
        :param data: list, strings of the sentences.
        :return: list, list of (int) ids of the words in the sentences.
        """
        data_to_ids = []
        for index in range(0, len(data), self._max_segment_size):
            for line in data[index:index + self._max_segment_size]:
                ids = ids_from_sentence(self._language, line)
                ids.append(len(ids))
                data_to_ids.append(ids)
        return data_to_ids

    @staticmethod
    def create_batch(data):
        """
        Creates a sorted batch from the data. Each line of the data is padded to the
        length of the longest sequence in the batch.
        :param data:
        :return:
        """
        sorted_data = sorted(data, key=lambda x: x[-1], reverse=True)
        batch_length = sorted_data[0][-1]
        for index in range(len(sorted_data)):
            while len(sorted_data[index])-1 < batch_length:
                sorted_data[index].insert(-1, 0)

        return np.array(sorted_data, dtype='int')

    @classmethod
    def abstract(cls):
        return False


class PrePadding(Padding):
    """
    Data is padded previously to the training iterations. The padding is determined
    by the longest sequence in the data segment.
    """

    def __init__(self, language, max_segment_size):
        """
        An instance of a pre-padder object.
        :param language: Language, instance of the used language object.
        """
        super().__init__()
        self._language = language
        self._max_segment_size = max_segment_size

    def __call__(self, data):
        """
        Converts the data of (string) sentences to ids of the words.
        Sentences are padded to the length of the longest sentence in the data segment.
        Length of a segment is determined by MAX_SEGMENT_SIZE.
        :param data: list, strings of the sentences.
        :return: list, list of (int) ids of the words in the sentences.
        """
        data_to_ids = []
        for index in range(0, len(data), self._max_segment_size):
            segment_length = len(ids_from_sentence(self._language, data[index:index + self._max_segment_size][0]))
            for line in data[index:index + self._max_segment_size]:
                ids = ids_from_sentence(self._language, line)
                ids_len = len(ids)
                while len(ids) < segment_length:
                    ids.append(0)
                data_line = np.zeros((segment_length + 1), dtype='int')
                data_line[:-1] = ids
                data_line[-1] = ids_len
                data_to_ids.append(data_line)

        return data_to_ids

    @staticmethod
    def create_batch(data):
        """
        Creates the batch, by sorting the elements in descending order with respect to the
        lengths of the sequences.
        :param data: list, containing lists of the ids.
        :return: Numpy Array, sorted batch.
        """
        return np.array(sorted(data, key=lambda x: x[-1], reverse=True))

    @classmethod
    def abstract(cls):
        return False
