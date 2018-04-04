import re
import numpy

EMBEDDING_DIM = 25

ENG = {
    'vocab':    '../data/eng/eng_voc',
    'corpora':  '../data/eng/eng_tok'
}

FRA = {
    'vocab':    '../data/fra/fra_voc',
    'corpora':  '../data/fra/fra_tok'
}


def vocab_creator(lang):
    def add_word(w, voc):
        if w not in voc:
            voc[w] = numpy.array([n for n in numpy.random.rand(EMBEDDING_DIM)], dtype=numpy.float32)

    vocab = {}
    with open(lang['corpora'], 'r') as file:
        for line in file:
            line_as_list = re.split(r"[\s|\n]+", line)
            for token in line_as_list:
                add_word(token, vocab)

    with open(lang['vocab'], 'w') as file:
        file.write('{0} {1}\n'.format(len(vocab), EMBEDDING_DIM))
        for word in vocab.keys():
            line = str(word)
            for elem in vocab[word]:
                line += (' ' + str(elem))
            line += '\n'
            file.write(line)


def main():
    vocab_creator(ENG)
    vocab_creator(FRA)


if __name__ == '__main__':
    main()
