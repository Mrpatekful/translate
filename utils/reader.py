import torch
import torch.nn

from torch.autograd import Variable

import sklearn.utils
import copy
import numpy
import inspect

from utils.utils import Component
from utils.utils import ParameterSetter
from utils.utils import subtract_dict
from utils.utils import ids_from_sentence
from utils.utils import subclasses

from modules.utils.utils import Embedding

from collections import OrderedDict


class Corpora(Component):
    """
    Wrapper class for the corpus of the task. Stores information about the corpus, and
    stores the location of the train, development and test data.
    """

    interface = OrderedDict(**{
            'train':     None,
            'dev':       None,
            'test':      None,
            'use_cuda': 'Task:use_cuda$'
        })

    abstract = True

    @ParameterSetter.pack(interface)
    def __init__(self, parameter_setter):
        """
        An instance of a corpora.
        :param parameter_setter: ParameterSetter object, that requires the following parameters.
            -:parameter train: str, path of the train data.
            -:parameter dev: str, path of the development data.
            -:parameter test: str, path of the test data.
        """
        parameter_setter.initialize(self)

        self._source_vocabulary = None
        self._target_vocabulary = None

    def _load_data(self, data_path):
        return NotImplementedError

    # The following 'size' properties are required for the parameter value resolver mechanism.

    @property
    def source_vocabulary(self):
        """
        Property for the source language of the text corpora.
        :return: Language, instance of the wrapper class for the source language.
        """
        if self._source_vocabulary is None:
            raise ValueError('Source vocabulary has not been set.')
        return self._source_vocabulary

    @property
    def target_vocabulary(self):
        """
        Property for the target language of the text corpora.
        :return: Language, instance of the wrapper class for the target language.
        """
        if self._target_vocabulary is None:
            raise ValueError('Target vocabulary has not been set.')
        return self._target_vocabulary

    @property
    def source_embedding_size(self):
        """
        Property for the embedding size of the source language.
        :return: int, size of the source language's word embedding.
        """
        if self._source_vocabulary is None:
            raise ValueError('Source vocabulary has not been set.')
        return self.source_vocabulary.embedding_size

    @property
    def target_embedding_size(self):
        """
        Property for the embedding size of the target language.
        :return: int, size of the target language's word embedding.
        """
        if self._target_vocabulary is None:
            raise ValueError('Target vocabulary has not been set.')
        return self.target_vocabulary.embedding_size

    @property
    def source_vocab_size(self):
        """
        Property for the vocab size of the source language.
        :return: int, number of words in the source language.
        """
        if self._source_vocabulary is None:
            raise ValueError('Source vocabulary has not been set.')
        return self.source_vocabulary.vocab_size

    @property
    def target_vocab_size(self):
        """
        Property for the vocab size of the target language.
        :return: int, number of words in the target language.
        """
        if self._target_vocabulary is None:
            raise ValueError('Target vocabulary has not been set.')
        return self.target_vocabulary.vocab_size

    def properties(self):
        return {
            name: getattr(self, name) for (name, _) in
            inspect.getmembers(type(self), lambda x: isinstance(x, property))
            if name not in ['train', 'dev', 'test']
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
        if self._dev is None:
            raise ValueError('Development data path has not been set.')
        return self._dev

    @property
    def test(self):
        """
        Property for the test data segment of the corpora.
        :return: str, the whole test data.
        """
        if self._test is None:
            raise ValueError('Test data path has not been set.')
        return self._test


class Monolingual(Corpora):
    """
    Special case of Corpora class, where the data read from the files only have a single language.
    """

    interface = OrderedDict(**{
            **Corpora.interface,
            'vocab':    None,
            'provided': None,
            'fixed':    None,
            'token':    None
        })

    abstract = False

    @ParameterSetter.pack(interface)
    def __init__(self, parameter_setter):
        """
        Instance of a wrapper for Monolingual text corpora.
        :param parameter_setter:ParameterSetter object, that requires the following parameters.
            -:parameter train: str, path of the train data.
            -:parameter dev: str, path of the development data.
            -:parameter test: str, path of the test data.
            -:parameter vocab: str, path of the vocabulary, containing the embeddings.
            -:parameter trained: bool, indicates, whether the embeddings, used by the language have been pre-trained.
            -:parameter provided bool, indicates, whether the embeddings are provided to the model.
            -:parameter token: str, token for the language. E.g: <ESP>, <ENG>, <GER>
            -:parameter use_cuda: bool, true, if cuda support is enabled.
        """
        super().__init__(parameter_setter=parameter_setter)

        vocabulary = Vocabulary(**parameter_setter.extract(Vocabulary.interface))

        parameter_setter.initialize(self, subtract_dict(self.interface, Corpora.interface))

        self._train = self._load_data(self.train)
        self._dev = self._load_data(self.dev)
        self._test = self._load_data(self.test)

        self._source_vocabulary = vocabulary
        self._target_vocabulary = vocabulary

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


class Parallel(Corpora):  # TODO vocabulary parameter extraction
    """
    Wrapper class for the corpora, that yields two languages. The languages are paired, and
    are separated by a special separator token.
    """

    interface = OrderedDict(**{
            **Corpora.interface,
            'source_trained':   None,
            'source_vocab':     None,
            'source_token':     None,
            'target_trained':   None,
            'target_vocab':     None,
            'target_token':     None,
            'separator_token':  None
        })

    abstract = False

    @ParameterSetter.pack(interface)
    def __init__(self, parameter_setter):
        """
        An instance of a wrapper for parallel corpora.
        :param parameter_setter: ParameterSetter object, that requires the following parameters.
            -:parameter train: str, path of the train data.
            -:parameter dev: str, path of the development data.
            -:parameter test: str, path of the test data.
            -:parameter source_trained: bool, true, if the embeddings of the source language have been pre-trained.
            -:parameter source_vocab: str, path of the vocabulary, containing the embeddings for the source language.
            -:parameter source_token: str, token for the source language. E.g: <ESP>, <ENG>, <GER>
            -:parameter target_trained: bool, true, if the embeddings, if the target language have been pre-trained.
            -:parameter target_vocab: str, path of the vocabulary, containing the embeddings for the target language.
            -:parameter target_token: str, token for the target language. E.g: <ESP>, <ENG>, <GER>
            -:parameter separator_token: str, a special separator token, that divides the paired corpus.
            -:parameter use_cuda: bool, true, if cuda support is enabled.
        """
        super().__init__(parameter_setter=parameter_setter)

        self._source_language = Vocabulary(**parameter_setter.extract(Vocabulary.interface))
        self._target_language = Vocabulary(**parameter_setter.extract(Vocabulary.interface))

        parameter_setter.initialize(self, subtract_dict(self.interface, super().interface))

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


class Reader(Component):
    """
    Derived classes should implement the reading logic for the seq2seq model. Readers divide the
    data into segments. The purpose of this behaviour, is to keep the sentences with similar lengths
    in segments, so they can be freely shuffled without mixing them together with larger sentences.
    """
    def __init__(self):
        self._train = None
        self._eval = None

    def batch_generator(self):
        """
        The role of this function is to generate batches for the seq2seq model. The batch generation
        should include the logic of shuffling the samples. A full iteration should include
        all of the data samples.
        :return: PyTorch padded-sequence object.
        """
        return NotImplementedError

    def train(self, boolean):
        self._train = boolean

    def eval(self, boolean):
        self._eval = boolean


class FastReader(Reader):
    """
    A faster implementation of reader class than FileReader. The source data is fully loaded into
    the memory.
    """

    interface = OrderedDict(**{
            'max_segment_size':  None,
            'batch_size':        None,
            'padding_type':      None,
            'use_cuda':         'Task:use_cuda$',
            'corpora':           Corpora
        })

    abstract = False

    def __init__(self,
                 batch_size,
                 use_cuda,
                 corpora,
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
        super().__init__()

        self._corpora = corpora
        self._use_cuda = use_cuda
        self._max_segment_size = max_segment_size

        self._batch_format = None

        padding = subclasses(Padding)

        self._data_processor = padding[padding_type](self._corpora.source_vocabulary, max_segment_size)

        self._modes = {
            'train': {
                'batch_size': batch_size,
                'data': self._data_processor(self._corpora.train)
            },
            'dev': {
                'batch_size': 1,
                'data': self._data_processor(self._corpora.dev)
            },
            'test': {
                'batch_size': 1,
                'data': self._data_processor(self._corpora.test)
            }
        }

        self._mode = None

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
            for index in range(0, len(shuffled_data_segment)-self._modes[self._mode]['batch_size'],
                               self._modes[self._mode]['batch_size']):
                batch = self._data_processor.create_batch(
                    shuffled_data_segment[index:index + self._modes[self._mode]['batch_size']])
                yield self.batch_format(batch, self._use_cuda)

    def _segment_generator(self):
        """
        Divides the data to segments of size MAX_SEGMENT_SIZE.
        """
        t = range(0, len(self._modes[self._mode]['data']), self._max_segment_size)
        for index in t:
            yield copy.deepcopy(self._modes[self._mode]['data'][index:index + self._max_segment_size])

    def train(self, boolean=True):
        super().train(boolean)
        if self._train:
            self._mode = 'train'
        else:
            self._mode = None

    def eval(self, boolean=True):
        super().eval(boolean)
        if self._train and self._eval:
            self._mode = 'dev'
        elif self._train and not self._eval:
            self._mode = 'train'
        elif not self._train and self._eval:
            self._mode = 'test'
        else:
            self._mode = None

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
    def corpora(self):
        """
        Property for the corpora of the reader.
        """
        return self._corpora

    @property
    def source_vocabulary(self):
        """
        Property for the reader object's source language.
        :return: Language object, the source language.
        """
        return self._corpora.source_vocabulary

    @property
    def target_vocabulary(self):
        """
        Property for the reader object's target language.
        :return: Language object, the source language.
        """
        return self._corpora.target_vocabulary


class FileReader(Reader):  # TODO
    """
    An implementation of the reader class. Batches are read from the source in file real-time.
    This version of the reader should only be used if the source file is too large to be stored
    in memory.
    """

    interface = None

    abstract = False

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
        super().__init__()

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

    @property
    def language(self):
        """
        Property for the reader object's source language.
        :return: Language object, the source language.
        """
        return self._language


class Vocabulary:
    """
    Wrapper class for the lookup tables of the languages.
    """
    PAD = -1

    interface = OrderedDict(**{
            'vocab':      None,
            'token':      None,
            'provided':   None,
            'fixed':      None,
            'use_cuda':   None
        })

    def __init__(self, vocab, token, provided, fixed, use_cuda):
        """
        A language instance for storing the embedding and vocabulary.
        :param vocab: str, path of the embedding/vocabulary for the language.
        :param token: str, token, that identifies the language.
        :param provided: bool, indicates, whether the embeddings are provided to the model.
        :param fixed: bool, true, if the embeddings are fixed, and do not require updates.
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._word_to_count = {}

        self._language_token = token
        self._use_cuda = use_cuda
        self._provided = provided

        self.requires_grad = not fixed

        self._vocab_size = None
        self._embedding_size = None
        self._embedding_weights = None

        self._load_data(vocab)

        self._embedding = Embedding(embedding_size=self._embedding_size,
                                    vocab_size=self._vocab_size,
                                    use_cuda=self._use_cuda,
                                    weights=self._embedding_weights,
                                    requires_grad=self.requires_grad)

    def _load_data(self, path):
        """
        Loads the vocabulary from a file. Path is assumed to be a text
        file, where each line contains a word and its corresponding embedding weights, separated by spaces.
        :param path: string, the absolute path of the vocab.
        """
        with open(path, 'r') as file:
            first_line = file.readline().split(' ')
            self._vocab_size = int(first_line[0]) + 4
            self._embedding_size = int(first_line[1])

            if self._provided:
                self._embedding_weights = numpy.empty((self._vocab_size, self._embedding_size), dtype='float')

            for index, line in enumerate(file):
                line_as_list = list(line.split(' '))
                self._word_to_id[line_as_list[0]] = index
                if self._provided:
                    self._embedding_weights[index + 1, :] = numpy.array([float(element) for element
                                                                         in line_as_list[1:]], dtype=float)

            self._word_to_id[self._language_token] = len(self._word_to_id)

            self._word_to_id['<SOS>'] = len(self._word_to_id)
            self._word_to_id['<EOS>'] = len(self._word_to_id)
            self._word_to_id['<UNK>'] = len(self._word_to_id)

            self._vocab_size = len(self._word_to_id)

            self._word_to_id['<PAD>'] = self.PAD

            self._id_to_word = dict(zip(self._word_to_id.values(), self._word_to_id.keys()))

            if self._provided:
                self._embedding_weights[-1, :] = numpy.zeros(self._embedding_size)
                self._embedding_weights[-2, :] = numpy.zeros(self._embedding_size)
                self._embedding_weights[-3, :] = numpy.zeros(self._embedding_size)
                self._embedding_weights[-4, :] = numpy.random.rand(self._embedding_size)

                self._embedding_weights = torch.from_numpy(self._embedding_weights).float()

                if self._use_cuda:
                    self._embedding_weights = self._embedding_weights.cuda()

    def __call__(self, expression):
        """
        Translates the given expression to it's corresponding word or id.
        :param expression: str or int, if str (word) is provided, then the id will be returned, and
                           the behaviour is the same for the other case.
        :return: int or str, (id or word) of the provided expression.
        """
        if isinstance(expression, str):
            return self._word_to_id[expression]

        elif isinstance(expression, int):
            return self._id_to_word[expression]

        else:
            raise ValueError('Expression must either be a string or an int.')

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
        Property for the embedding layer.
        :return: A PyTorch Embedding type object.
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
        return self._embedding_size

    @property
    def vocab_size(self):
        """
        Property for the dimension of the embeddings.
        :return: int, length of the vocabulary (dim 1 of the embedding matrix).
        """
        return self._vocab_size


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
            yield data_segment


class Padding:
    """
    Base class for the padding types.
    """

    abstract = True

    def __init__(self, language, max_segment_size):
        self._language = language
        self._max_segment_size = max_segment_size

    def create_batch(self, ):
        return NotImplementedError


class PostPadding(Padding):
    """
    Data is padded during the training iterations. Padding is determined by the longest
    sequence in the batch.
    """

    abstract = False

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
            while len(sorted_data[index]) - 1 < batch_length:
                sorted_data[index].insert(-1, Vocabulary.PAD)

        return numpy.array(sorted_data, dtype='int')

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


class PrePadding(Padding):
    """
    Data is padded previously to the training iterations. The padding is determined
    by the longest sequence in the data segment.
    """

    abstract = False

    @staticmethod
    def create_batch(data):
        """
        Creates the batch, by sorting the elements in descending order with respect to the
        lengths of the sequences.
        :param data: list, containing lists of the ids.
        :return: Numpy Array, sorted batch.
        """
        return numpy.array(sorted(data, key=lambda x: x[-1], reverse=True))

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
                    ids.append(Vocabulary.PAD)
                data_line = numpy.zeros((segment_length + 1), dtype='int')
                data_line[:-1] = ids
                data_line[-1] = ids_len
                data_to_ids.append(data_line)

        return data_to_ids
