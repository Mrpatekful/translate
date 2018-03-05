import re
import numpy

EMBEDDING_DIM = 3
VOCAB_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_voc'


def vocab_creator(path):
    def add_word(w, voc):
        if w not in voc:
            voc[w] = [n for n in numpy.random.rand(EMBEDDING_DIM)]

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


def main():
    pass


if __name__ == '__main__':
    main()