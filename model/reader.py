import numpy as np
import torch
from torch.autograd import Variable
import re


EMBEDDING_DIM = 3
VOCAB_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/text_voc'

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

    def load_vocab(self, path):
        """
        Loads the vocabulary from a file. Path is assumed to be a text
        file, where each line contains a word and its corresponding embedding weights, separated by spaces.
        :param path: string, the absolute path of the vocab.
        """
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

            self._embedding = Variable(torch.FloatTensor(self._embedding), requires_grad=False)

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


class Reader:
    """
    Input stream generator class for the training.
    """
    def __init__(self, lang_src, lang_tgt, use_cuda):
        self._lang_src = lang_src
        self._lang_tgt = lang_tgt
        self._use_cuda = use_cuda

    @staticmethod
    def _indexes_from_sentence(lang, sentence):
        """
        Convenience method, for converting a sequence of words to ids.
        :param lang: The language of the sequence. This parameter must be set with the object's
                    reference of the target or source language.
        :param sentence: string, a tokenized sequence of words.
        :return: list, containing the ids (int) of the sentence in the same order.
        """
        return [lang.word_to_index[word] for word in sentence.split(' ')]

    def variable_from_sentence(self, lang, sentence):
        """
        Creates PyTorch Variable object from a tokenized sequence.
        :param lang: The language of the sequence. This parameter must be set with the object's
                    reference of the target or source language.
        :param sentence: string, a tokenized sequence of words.
        :return: Variable, containing the ids of the sentence.
        """
        indexes = self._indexes_from_sentence(lang, sentence)
        result = Variable(torch.LongTensor(indexes))
        if self._use_cuda:
            return result.cuda()
        else:
            return result

    def variables_from_pair(self, lang_src, lang_tgt, pair):
        """
        Creates a pair of variables from a pair of sentences.
        :param lang_src: The source language of the translation, this parameter must be a reference,
                        to the corresponding Language object of this class.
        :param lang_tgt: The target language of the translation, the requirement is the same as the previous param's.
        :param pair: tuple, a pair of sentences, the first element is the source, and the second element is
                    the target sentence.
        :return: tuple, containing the Variable objects of the source and target sentence.
        """
        src_variable = self.variable_from_sentence(lang_src, pair[0])
        tgt_variable = self.variable_from_sentence(lang_tgt, pair[1])
        return src_variable, tgt_variable

    @property
    def source_language(self):
        """
        Property for the reader object's source language.
        :return: Language object, the source language.
        """
        return self._lang_src

    @property
    def target_language(self):
        """
        Property for the reader object's target language.
        :return: Language object, the target language.
        """
        return self._lang_tgt

