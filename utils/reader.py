import numpy as np

import torch
from torch.autograd import Variable

from tqdm import tqdm

import sklearn.utils
import copy


def data_loader(data_path):
    """
    Loader function for the data.
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
    # TODO [1:-1] for \n and <ENG> is out of place
    return [language.word_to_id[word] for word in sentence.split(' ')[1:-1]]


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
                    data_segment.append(line)
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

    @classmethod
    def abstract(cls):
        return True


class FileReader(Reader):
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
        # TODO length for the sentences and padding
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
        result = np.empty((len(sentences), len(sentences[0].split(' '))-2))
        for idx in range(len(sentences)):
            result[idx, :len(sentences[idx])] = np.array(ids_from_sentence(self._language, sentences[idx]))

        result = Variable(torch.from_numpy(result))
        if self._use_cuda:
            return result.cuda()
        else:
            return result

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
                 max_segment_size):
        """
        An instance of a fast reader.
        :param language: Language, instance of the used language object.
        :param data_path: str, absolute path of the data location.
        :param batch_size: int, size of the input batches.
        :param use_cuda: bool, True if the device has cuda support.
        """
        self._data_path = data_path
        self._language = language
        self._use_cuda = use_cuda
        self._batch_size = batch_size
        self._max_segment_size = max_segment_size
        self._data_processor = self.PostPadding(language,
                                                max_segment_size)
        self._data = self._data_processor(data_loader(data_path))

    def batch_generator(self):
        """
        Generator for mini-batches. Data is read from memory.
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
                ids = torch.from_numpy(batch[:, :-1])
                lengths = batch[:, -1]
                if self._use_cuda:
                    ids = ids.cuda()
                yield Variable(ids), lengths

    def _segment_generator(self):
        """
        Divides the data to segments of size MAX_SEGMENT_SIZE.
        """
        for index in tqdm(range(0, len(self._data), self._max_segment_size)):
            yield copy.deepcopy(self._data[index:index + self._max_segment_size])

    @classmethod
    def abstract(cls):
        return False

    class PrePadding:
        """
        Data is padded previously to the training iterations. The padding is determined
        by the longest sequence in the data segment.
        """

        def __init__(self, language, max_segment_size):
            """
            An instance of a pre-padder object.
            :param language: Language, instance of the used language object.
            """
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
                segment_length = len(ids_from_sentence(self._language,
                                                       data[index:index + self._max_segment_size][0]))
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

    class PostPadding:
        """
        Data is padded during the training iterations. Padding is determined by the longest
        sequence in the batch.
        """

        def __init__(self, language, max_segment_size):
            """
            An instance of a post-padder object.
            :param language: Language, instance of the used language object.
            """
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

    @property
    def language(self):
        """
        Property for the reader object's source language.
        :return: Language object, the source language.
        """
        return self._language
