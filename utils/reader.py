import numpy as np

import torch
from torch.autograd import Variable

import sklearn.utils
import abc
import re
import copy


EMBEDDING_DIM = 3
VOCAB_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_voc'

LANG_TGT_VOC = None
LANG_SRC_VOC = None
LANG_TGT_TOK = None
LANG_SRC_TOK = None

MAX_SEGMENT_SIZE = 1000  # this constant is used by DataQueue and FastReader to divide the data into smaller segments


USE_CUDA = torch.cuda.is_available()


def vocab_creator(path):
    """
    Temporary function for testing purposes. Creates a vocab file from a text, with random
    embedding weights.
    :param path: string, the absolute path of the text to create a vocab of.
    """
    def add_word(w, voc):
        if w not in voc:
            voc[w] = [n for n in np.random.rand(EMBEDDING_DIM)]

    vocab = {}
    with open(path, 'r') as file:
        for line in file:
            line_as_list = re.split(r"[\s|\n]+", line)
            for token in line_as_list:
                add_word(token, vocab)

    with open(VOCAB_PATH, 'w') as file:
        file.write('{0} {1}\n'.format(len(vocab), EMBEDDING_DIM))
        for word in vocab.keys():
            line = str(word)
            for elem in vocab[word]:
                line += (' ' + str(elem))
            line += '\n'
            file.write(line)


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


class Language:
    """
    Wrapper class for the lookup tables of the languages.
    """

    def __init__(self):
        self._word_to_id = {}
        self._id_to_word = {}
        self._word_to_count = {}

        self._embedding = None
        self._train_data = None

    def load_vocab(self, path):
        """
        Loads the vocabulary from a file. Path is assumed to be a text
        file, where each line contains a word and its corresponding embedding weights, separated by spaces.
        :param path: string, the absolute path of the vocab.
        """
        with open(path, 'r') as file:
            first_line = file.readline().split(' ')
            num_of_words = int(first_line[0])
            embedding_dim = int(first_line[1])
            self._embedding = np.empty((num_of_words + 4, embedding_dim), dtype='float')

            for index, line in enumerate(file):
                line_as_list = list(line.split(' '))
                self._word_to_id[line_as_list[0]] = index + 1  # all values are incremented by 1 because 0 is <PAD>
                self._embedding[index + 1, :] = np.array([float(element) for element in line_as_list[1:]], dtype=float)

            self._word_to_id['<PAD>'] = 0
            self._word_to_id['<SOS>'] = len(self._word_to_id)
            self._word_to_id['<EOS>'] = len(self._word_to_id)
            self._word_to_id['<UNK>'] = len(self._word_to_id)

            self._id_to_word = dict(zip(self._word_to_id.values(),
                                        self._word_to_id.keys()))

            self._embedding[0, :] = np.zeros(embedding_dim)
            self._embedding[-1, :] = np.zeros(embedding_dim)
            self._embedding[-2, :] = np.zeros(embedding_dim)
            self._embedding[-3, :] = np.zeros(embedding_dim)

            self._embedding = torch.from_numpy(self._embedding).float()

            if USE_CUDA:
                self._embedding = self._embedding.cuda()

    @property
    def embedding(self):
        """
        Property for the embedding matrix.
        :return: A PyTorch Variable object, that contains the embedding matrix
                for the language.
        """
        if self._embedding is None:
            raise ValueError('The vocabulary has not been initialized for the language.')
        return self._embedding

    @property
    def embedding_size(self):
        """
        Property for the dimension of the embeddings.
        :return: int, length of the embedding vectors (dim 1 of the embedding matrix).
        """
        if self._embedding is None:
            raise ValueError('The vocabulary has not been initialized for the language.')
        return self._embedding.shape[1]

    @property
    def vocab_size(self):
        """
        Property for the dimension of the embeddings.
        :return: int, length of the vocabulary (dim 1 of the embedding matrix).
        """
        if self._embedding is None:
            raise ValueError('The vocabulary has not been initialized for the language.')
        return self._embedding.shape[0]

    @property
    def word_to_id(self):
        """
        Property for the word to id dictionary.
        :return: dict, containing the word-id pairs.
        """
        return self._word_to_id

    @property
    def id_to_word(self):
        """
        Property for the word to id dictionary.
        :return: dict, containing the word-id pairs.
        """
        return self._id_to_word


class DataQueue:
    """
    A queue object for the data feed. This can be later configured to load the data to
    memory asynchronously.
    """

    def __init__(self, data_path):
        """
        :param data_path: str, location of the data.
        """
        self._MAX_LEN = 10

        self._data_path = data_path
        self._data_segment_queue = []

    def data_generator(self):
        """
        Data is retrieved directly from a file, and loaded into data chunks of size MAX_CHUNK_SIZE.
        :return: list of strings, a data chunk.
        """
        with open(self._data_path, 'r') as file:
            data_segment = []
            for line in file:
                if len(data_segment) < MAX_SEGMENT_SIZE:
                    data_segment.append(line)
                else:
                    temp_data_segment = copy.deepcopy(data_segment)
                    del data_segment[:]
                    yield temp_data_segment
            yield data_segment  # the segment is not filled, returning with the remaining values


class Reader(metaclass=abc.ABCMeta):
    """
    Derived classes should implement the reading logic for the seq2seq model. Readers divide the
    data into segments. The purpose of this behaviour, is to keep the sentences with similar lengths
    in segments, so they can be freely shuffled without mixing them together with larger sentences.
    """

    @abc.abstractmethod
    def batch_generator(self):
        """
        The role of this function is to generate batches for the seq2seq model. The batch generation
        should include the logic of shuffling the samples. A full iteration should include
        all of the data samples.
        :return: PyTorch padded-sequence object.
        """


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
                 use_cuda):
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
        self._data_queue = DataQueue(data_path)

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
                 use_cuda):
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
        self._data_processor = self.PostPadding(language)  # PrePadding <-> PostPadding
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
            # batches must always be the same size so len(..) - batch_size is the termination index
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
        for index in range(0, len(self._data), MAX_SEGMENT_SIZE):
            yield copy.deepcopy(self._data[index:index + MAX_SEGMENT_SIZE])

    # =============================================================================== #
    # Two versions of FastReader:                                                     #
    #       1. PrePadding:                                                            #
    #          Padding is located in the process __call__ function, so _create_batch  #
    #          only has to sort the batch. The draw back is there might be batches,   #
    #          where even the longest sequence is padded.                             #
    #       2. PostPadding:                                                           #
    #          Padding is located in _create_batch function, this way the sentences   #
    #          are padded to the length of the longest sentence in the batch, but     #
    #          input feeding is slower.                                               #
    # =============================================================================== #

    # =============================================================================== #
    # ----------------------------------Version 1.----------------------------------- #
    # =============================================================================== #

    class PrePadding:
        """
        Data is padded previously to the training iterations. The padding is determined
        by the longest sequence in the data segment.
        """

        def __init__(self, language):
            """
            An instance of a pre-padder object.
            :param language: Language, instance of the used language object.
            """
            self._language = language

        def __call__(self, data):
            """
            Converts the data of (string) sentences to ids of the words.
            Sentences are padded to the length of the longest sentence in the data segment.
            Length of a segment is determined by MAX_SEGMENT_SIZE.
            :param data: list, strings of the sentences.
            :return: list, list of (int) ids of the words in the sentences.
            """
            data_to_ids = []
            for index in range(0, len(data), MAX_SEGMENT_SIZE):
                # length of the longest line in the segment
                segment_length = len(ids_from_sentence(self._language,
                                                       data[index:index + MAX_SEGMENT_SIZE][0]))
                for line in data[index:index + MAX_SEGMENT_SIZE]:
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

    # =============================================================================== #
    # ----------------------------------Version 2.----------------------------------- #
    # =============================================================================== #

    class PostPadding:
        """
        Data is padded during the training iterations. Padding is determined by the longest
        sequence in the batch.
        """

        def __init__(self, language):
            """
            An instance of a post-padder object.
            :param language: Language, instance of the used language object.
            """
            self._language = language

        def __call__(self, data):
            """
            Converts the data of (string) sentences to ids of the words.
            :param data: list, strings of the sentences.
            :return: list, list of (int) ids of the words in the sentences.
            """
            data_to_ids = []
            for index in range(0, len(data), MAX_SEGMENT_SIZE):
                for line in data[index:index + MAX_SEGMENT_SIZE]:
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
            batch_length = sorted_data[0][-1]  # length of the longest sequence in the batch
            for index in range(len(sorted_data)):
                while len(sorted_data[index])-1 < batch_length:  # subtracting the length [-1] element
                    sorted_data[index].insert(-1, 0)

            return np.array(sorted_data, dtype='int')

    @property
    def language(self):
        """
        Property for the reader object's source language.
        :return: Language object, the source language.
        """
        return self._language
