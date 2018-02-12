import numpy as np
import torch
from torch.autograd import Variable
import sklearn.utils
import re
import copy


EMBEDDING_DIM = 3
VOCAB_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_voc'

LANG_TGT_VOC = None
LANG_SRC_VOC = None
LANG_TGT_TOK = None
LANG_SRC_TOK = None


def vocab_creator(path):
    """
    Temporary function for testing purposes. Creates a vocab file from a text, with random
    embedding weights.
    :param path: string, the absolute path of the text to create a vocab of.
    """
    def _add_word(w, voc):
        if w not in voc:
            voc[w] = [n for n in np.random.rand(EMBEDDING_DIM)]

    vocab = {}
    with open(path, 'r') as file:
        for line in file:
            line_as_list = re.split(r"[\s|\n]+", line)
            for token in line_as_list:
                _add_word(token, vocab)

    with open(VOCAB_PATH, 'w') as file:
        file.write('{0} {1}\n'.format(len(vocab), EMBEDDING_DIM))
        for word in vocab.keys():
            line = str(word)
            for elem in vocab[word]:
                line += (' ' + str(elem))
            line += '\n'
            file.write(line)


class Language:
    """
    Wrapper class for the lookup tables of the languages.
    """
    def __init__(self):
        self._word_to_id = {}
        self._word_to_count = {}

        self._embedding = None
        self._train_data = None

    def load_vocab(self, path):
        """
        Loads the vocabulary from a file. Path is assumed to be a text
        file, where each line contains a word and its corresponding embedding weights, separated by spaces.
        :param path: string, the absolute path of the vocab.
        """
        # TODO add tokens to embedding
        with open(path, 'r') as file:
            first_line = file.readline().split(' ')
            self._embedding = np.empty((int(first_line[0]), int(first_line[1])), dtype=float)

            for index, line in enumerate(file):
                line_as_list = list(line.split(' '))
                self._word_to_id[line_as_list[0]] = index
                self._embedding[index, :] = np.array([float(element) for element in line_as_list[1:]], dtype=float)

            self._word_to_id['<SOS>'] = len(self._word_to_id)
            self._word_to_id['<EOS>'] = len(self._word_to_id)
            self._word_to_id['<UNK>'] = len(self._word_to_id)

            self._embedding = torch.FloatTensor(self._embedding)

    @property
    def embedding(self):
        """
        Property for the embedding matrix.
        :return: A PyTorch Variable object, with disabled gradient, that contains the embedding matrix
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
    def word_to_id(self):
        """
        Property for the word to id dictionary.
        :return: dict, containing the word-id pairs.
        """
        return self._word_to_id


class DataQueue:
    """
    A queue object for the data feed. This can be later configured to load the data to
    memory asynchronously.
    """
    def __init__(self, data_path, full_load):
        """
        :param data_path: str, location of the data.
        :param full_load: boolean, if true, the data will be fully loaded to the memory else,
                            it will be loaded into smaller data chunks.
        """
        self._MAX_LEN = 3
        self._MAX_CHUNK_SIZE = 5000

        self._data_path = data_path
        self._data_chunk_queue = []

    def data_generator(self):
        """
        Data is retrieved directly from a file, and loaded into data chunks of size MAX_CHUNK_SIZE.
        :return: list of strings, a data chunk.
        """
        with open(self._data_path, 'r') as file:
            data_chunk = []
            for line in file:
                if len(data_chunk) < self._MAX_CHUNK_SIZE:
                    data_chunk.append(line)
                else:
                    temp_data_chunk = copy.deepcopy(data_chunk)
                    del data_chunk[:]
                    yield temp_data_chunk
            yield data_chunk  # the chunk is not filled, returning with the remaining values


class Reader:
    """
    Input stream generator class for the seq2seq model.
    """
    def __init__(self, language, data_path, batch_size, full_load, use_cuda):
        self._language = language
        self._use_cuda = use_cuda
        self._batch_size = batch_size
        self._data_queue = DataQueue(data_path, full_load)
        self._data_chunk_generator = self._data_queue.data_generator

    def batch_generator(self):
        """
        Generator for mini-batches. Data is read indirectly from a file through a DataQueue object,
        or if the full_load option is active, it is read from a list.
        :return: Torch padded-sequence object
        """
        for data_chunk in self._data_chunk_generator():
            shuffled_data_chunk = sklearn.utils.shuffle(data_chunk)
            for index in range(0, len(shuffled_data_chunk), self._batch_size):
                batch = sorted(shuffled_data_chunk[index:index + self._batch_size],
                               key=lambda x: len(x), reverse=True)
                yield self._variable_from_sentences(batch)

    def _indexes_from_sentence(self, sentence):
        """
        Convenience method, for converting a sequence of words to ids.
        :param sentence: string, a tokenized sequence of words.
        :return: list, containing the ids (int) of the sentence in the same order.
        """
        # TODO [1:-1] for \n and <ENG> is out of place
        return [self._language.word_to_id[word] for word in sentence.split(' ')[1:-1]]

    def _variable_from_sentences(self, sentences):
        """
        Creates PyTorch Variable object from a tokenized sequence.
        :param sentences: string, a tokenized sequence of words.
        :return: Variable, containing the ids of the sentence.
        """
        result = np.empty((len(sentences), len(sentences[0].split(' '))-2))
        for idx in range(len(sentences)):
            result[idx, :len(sentences[idx])] = np.array(self._indexes_from_sentence(sentences[idx]))

        result = Variable(torch.LongTensor(result))
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

