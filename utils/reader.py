import torch
from torch.autograd import Variable

from utils import utils
from tqdm import trange

import sklearn.utils
import copy
import numpy
import inspect

from utils.utils import Component

from collections import OrderedDict


class Reader(Component):
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


class FastReader(Reader):
    """
    A faster implementation of reader class than FileReader. The source data is fully loaded into
    the memory.
    """

    def __init__(self,
                 batch_size,
                 use_cuda,
                 corpus,
                 padding_type,
                 max_segment_size):
        """
        An instance of a fast reader.
        :param batch_size: int, size of the input batches.
        :param use_cuda: bool, True if the device has cuda support.
        :param padding_type: str, type of padding that will be used during training. The sequences in
                             the mini-batches may vary in length, so padding must be applied to convert
                             them to equal lengths.
        :param max_segment_size: int, the size of each segment, that will contain the similar length data.
        """
        self._corpus = corpus
        self._use_cuda = use_cuda
        self._batch_size = batch_size
        self._max_segment_size = max_segment_size
        self._batch_format = None

        padding = utils.subclasses(Padding)

        self._data_processor = padding[padding_type](self._corpus.source_language, max_segment_size)

        self._train_data = self._data_processor(self._corpus.train)
        self._dev_data = self._data_processor(self._corpus.dev)
        self._test_data = self._data_processor(self._corpus.test)

        self._modes = {
            'train': self._train_data,
            'dev': self._dev_data,
            'test': self._test_data
        }

        self._mode = 'train'

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
                yield self.batch_format(batch, self._use_cuda)

    def _segment_generator(self):
        """
        Divides the data to segments of size MAX_SEGMENT_SIZE.
        """
        data = self._modes[self._mode]
        t = trange(0, len(data), self._max_segment_size)
        for index in t:
            yield copy.deepcopy(data[index:index + self._max_segment_size])

    def print_validation_format(self, **kwargs):
        """
        Convenience function for printing the parameters of the function, to the standard output.
        The parameters must be provided as keyword arguments. Each argument must contain a 2D
        array containing word ids, which will be converted to the represented words from the
        dictionary of the language, used by the reader instance.
        """
        id_batches = numpy.array(list(kwargs.values()))
        expression = ''
        for index, ids in enumerate(zip(*id_batches)):
            expression += '{%d}:\n' % index
            for param in zip(kwargs, ids):
                expression += ('> [%s]:\t%s\n' % (param[0], '\t'.join(sentence_from_ids(self._corpus.source_language,
                                                                                        param[1]))))
            expression += '\n'
        print(expression)

    @classmethod
    def abstract(cls):
        return False

    @classmethod
    def interface(cls):
        return OrderedDict(
            max_segment_size=None,
            batch_size=None,
            padding_type=None,
            use_cuda='Task:use_cuda$',
            corpus=Corpus
        )

    @property
    def batch_format(self):
        """
        Property for the currently used batch production format for the batch generation.
        This value must be provided by the task, to define a convenient way of reading the
        data from segments.
        :return: function, currently used batch formatter function.
        """
        return self._batch_format

    @batch_format.setter
    def batch_format(self, batch_format):
        """
        Setter for the batch format of the reader.
        :param batch_format: function, that defines the way of creating batches.
                             The function receives a batch, that has dimensions (batch_size, seq_length),
                             and a bool, that indicates whether cuda is enabled.
        """
        self._batch_format = batch_format

    @property
    def mode(self):
        """
        Property for the mode of the reader.
        :return: str, mode of the reader.
        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        """
        Setter for the mode of the reader.
        :param mode: str, the value may be 'train', 'dev' or 'test'
        :raises ValueError: if the value is not a valid mode.
        """
        if mode not in self._modes.keys():
            raise ValueError('Incorrect mode.')
        self._mode = mode

    @property
    def source_language(self):
        """
        Property for the reader object's source language.
        :return: Language object, the source language.
        """
        return self._corpus.source_language

    @property
    def target_language(self):
        """
        Property for the reader object's target language.
        :return: Language object, the source language.
        """
        return self._corpus.target_language


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
        ids = numpy.empty((len(sentences), len(sentences[0].split(' '))-2))
        for index in range(len(sentences)):
            ids[index, :len(sentences[index])] = numpy.array(ids_from_sentence(self._language, sentences[index]))

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
    def interface(cls):
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


class Corpus(Component):
    """
    Wrapper class for the corpus of the task. Stores information about the corpus, and
    stores the location of the train, development and test data.
    """

    def __init__(self,
                 train,
                 dev,
                 test,
                 use_cuda):
        """
        An instance of a corpora.
        :param train: str, path of the train data.
        :param dev: str, path of the development data.
        :param test: str, path of the test data.
        """
        self._train = train
        self._dev = dev
        self._test = test
        self._use_cuda = use_cuda

        self._source_language = None
        self._target_language = None

    def _load_data(self, data_path):
        return NotImplementedError

    @property
    def source_language(self):
        """
        Property for the source language of the text corpora.
        :return: Language, instance of the wrapper class for the source language.
        """
        if self._source_language is None:
            raise ValueError('Source language has not been set.')
        return self._source_language

    @property
    def target_language(self):
        """
        Property for the target language of the text corpora.
        :return: Language, instance of the wrapper class for the target language.
        """
        if self._target_language is None:
            raise ValueError('Target language has not been set.')
        return self._target_language

    @property
    def source_embedding_size(self):
        """
        Property for the embedding size of the source language.
        :return: int, size of the source language's word embedding.
        """
        if self._source_language is None:
            raise ValueError('Source language has not been set.')
        return self.source_language.embedding_size

    @property
    def target_embedding_size(self):
        """
        Property for the embedding size of the target language.
        :return: int, size of the target language's word embedding.
        """
        if self._target_language is None:
            raise ValueError('Target language has not been set.')
        return self.target_language.embedding_size

    @property
    def source_vocab_size(self):
        """
        Property for the vocab size of the source language.
        :return: int, number of words in the source language.
        """
        if self._source_language is None:
            raise ValueError('Source language has not been set.')
        return self.source_language.vocab_size

    @property
    def target_vocab_size(self):
        """
        Property for the vocab size of the target language.
        :return: int, number of words in the target language.
        """
        if self._target_language is None:
            raise ValueError('Target language has not been set.')
        return self.target_language.vocab_size

    def properties(self):
        return {
            name: getattr(self, name) for (name, _) in
            inspect.getmembers(type(self), lambda x: isinstance(x, property)) if name not in ['train', 'dev', 'test']
        }

    @property
    def train(self):
        """
        Property for the train data segment of the corpora.
        :return: str, the whole train data.
        """
        if self._train is None:
            raise ValueError('Train data path has not been set.')
        return self._train

    @property
    def dev(self):
        """
        Property for the development data segment of the corpora.
        :return: str, the whole development data.
        """
        if self._train is None:
            raise ValueError('Development data path has not been set.')
        return self._dev

    @property
    def test(self):
        """
        Property for the test data segment of the corpora.
        :return: str, the whole test data.
        """
        if self._train is None:
            raise ValueError('Test data path has not been set.')
        return self._test

    @classmethod
    def interface(cls):
        return OrderedDict(
            train=None,
            dev=None,
            test=None,
            use_cuda='Task:use_cuda$'
        )

    @classmethod
    def abstract(cls):
        return True


class Monolingual(Corpus):
    """
    Special case of Corpus class, where the data read from the files only have a single language.
    """

    def __init__(self,
                 train,
                 dev,
                 test,
                 trained,
                 vocab,
                 token,
                 use_cuda):
        """
        Instance of a wrapper for Monolingual text corpora.
        :param train: str, path of the train data.
        :param dev: str, path of the development data.
        :param test: str, path of the test data.
        :param trained: bool, indicates, whether the embeddings, used by the language have been pre-trained.
        :param vocab: str, path of the vocabulary, containing the embeddings.
        :param token: str, token for the language. E.g: <ESP>, <ENG>, <GER>
        :param use_cuda: bool, true, if cuda support is enabled.
        """

        super().__init__(train,
                         dev,
                         test,
                         use_cuda)

        self._train = self._load_data(self.train)
        self._dev = self._load_data(self.dev)
        self._test = self._load_data(self.test)

        language = Language(vocab, token, trained, use_cuda)

        self._source_language = language
        self._target_language = language

    def _load_data(self, data_path):
        """
        Loader function for the monolingual corpora, where there is only a single language.
        :return: list, the data stored as a list of strings.
        """
        data = []
        with open(data_path, 'r') as file:
            for line in file:
                data.append(line)
        if len(data) == 0:
            raise ValueError('The given file is empty.')
        return data

    @classmethod
    def interface(cls):
        return OrderedDict(
            **super().interface(),
            vocab=None,
            trained=None,
            token=None
        )

    @classmethod
    def abstract(cls):
        return False


class Parallel(Corpus):
    """
    Wrapper class for the corpora, that yields two languages. The languages are paired, and
    are separated by a special separator token.
    """

    def __init__(self,
                 train,
                 dev,
                 test,
                 source_trained,
                 source_vocab,
                 source_token,
                 target_trained,
                 target_vocab,
                 target_token,
                 separator_token,
                 use_cuda):
        """
        An instance of a wrapper for parallel corpora.
        :param train: str, path of the train data.
        :param dev: str, path of the development data.
        :param test: str, path of the test data.
        :param source_trained: bool, true, if the embeddings, used by the source language have been pre-trained.
        :param source_vocab: str, path of the vocabulary, containing the embeddings for the source language.
        :param source_token: str, token for the source language. E.g: <ESP>, <ENG>, <GER>
        :param target_trained: bool, true, if the embeddings, used by the target language have been pre-trained.
        :param target_vocab: str, path of the vocabulary, containing the embeddings for the target language.
        :param target_token: str, token for the target language. E.g: <ESP>, <ENG>, <GER>
        :param separator_token: str, a special separator token, that divides the paired corpus.
        :param use_cuda: bool, true, if cuda support is enabled.
        """

        super().__init__(train,
                         dev,
                         test,
                         use_cuda)

        self._separator_token = separator_token

        self._source_language = Language(source_vocab, source_token, source_trained, use_cuda)
        self._target_language = Language(target_vocab, target_token, target_trained, use_cuda)

    def _load_data(self, data_path):
        """
        Loader function for parallel data. Data is read from the provided path, and separated
        by the separator token.
        :return: list, the data stored as a list of strings.
        """
        data = []
        with open(data_path, 'r') as file:
            for line in file:
                data.append(line[:-1].split(self._separator_token))
        if len(data) == 0:
            raise ValueError('The given file is empty.')
        return data

    @classmethod
    def interface(cls):
        return OrderedDict(
            **super().interface(),
            source_trained=None,
            source_vocab=None,
            source_token=None,
            target_trained=None,
            target_vocab=None,
            target_token=None,
            separator_token=None
        )

    @classmethod
    def abstract(cls):
        return False


class Language:
    """
    Wrapper class for the lookup tables of the languages.
    """

    def __init__(self,
                 path,
                 token,
                 trained,
                 use_cuda):
        """
        A language instance for storing the embedding and vocabulary.
        :param path: str, path of the embedding/vocabulary for the language.
        :param token: str, token, that identifies the language.
        :param trained: bool, true, if the embeddings have been pre-trained.
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._word_to_count = {}
        self._language_token = token
        self._use_cuda = use_cuda

        self.requires_grad = not trained

        self._embedding = None
        self._load_vocab(path)

    def _load_vocab(self, path):
        """
        Loads the vocabulary from a file. Path is assumed to be a text
        file, where each line contains a word and its corresponding embedding weights, separated by spaces.
        :param path: string, the absolute path of the vocab.
        """
        with open(path, 'r') as file:
            first_line = file.readline().split(' ')
            num_of_words = int(first_line[0])
            embedding_dim = int(first_line[1])
            self._embedding = numpy.empty((num_of_words + 5, embedding_dim), dtype='float')

            for index, line in enumerate(file):
                line_as_list = list(line.split(' '))
                self._word_to_id[line_as_list[0]] = index + 1  # all values are incremented by 1 because 0 is <PAD>
                self._embedding[index + 1, :] = numpy.array([float(element) for element in line_as_list[1:]],
                                                            dtype=float)

            self._word_to_id['<PAD>'] = 0
            self._word_to_id[self._language_token] = len(self._word_to_id)
            self._word_to_id['<SOS>'] = len(self._word_to_id)
            self._word_to_id['<EOS>'] = len(self._word_to_id)
            self._word_to_id['<UNK>'] = len(self._word_to_id)

            self._id_to_word = dict(zip(self._word_to_id.values(),
                                        self._word_to_id.keys()))

            self._embedding[0, :] = numpy.zeros(embedding_dim)
            self._embedding[-1, :] = numpy.zeros(embedding_dim)
            self._embedding[-2, :] = numpy.zeros(embedding_dim)
            self._embedding[-3, :] = numpy.zeros(embedding_dim)
            self._embedding[-4, :] = numpy.random.rand(embedding_dim)

            self._embedding = torch.from_numpy(self._embedding).float()

            if self._use_cuda:
                self._embedding = self._embedding.cuda()

    @property
    def tokens(self):
        """
        Property for the tokens of the language.
        :return: dict, <UNK>, <EOS> and <SOS> tokens with their ids.
        """
        return {
            '<UNK>': self._word_to_id['<UNK>'],
            '<EOS>': self._word_to_id['<EOS>'],
            '<SOS>': self._word_to_id['<SOS>']
        }

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


def ids_from_sentence(language, sentence):
    """
    Convenience method, for converting a sequence of words to ids.
    :param language: Language, object of the language to use the look up of.
    :param sentence: string, a tokenized sequence of words.
    :return: list, containing the ids (int) of the sentence in the same order.
    """
    return [language.word_to_id[word.rstrip()] for word in sentence.split(' ') if word.rstrip() != '']


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


class Padding:
    """
    Base class for the padding types.
    """

    def __init__(self, language, max_segment_size):
        self._language = language
        self._max_segment_size = max_segment_size

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
        super().__init__(language, max_segment_size)

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

        return numpy.array(sorted_data, dtype='int')

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
        super().__init__(language, max_segment_size)

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
                data_line = numpy.zeros((segment_length + 1), dtype='int')
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
        return numpy.array(sorted(data, key=lambda x: x[-1], reverse=True))

    @classmethod
    def abstract(cls):
        return False
