"""

"""

import numpy
import tqdm

EMBEDDING_DIM = 50

ENG = {
    'vocab':    '/media/patrik/1EDB65B8599DD93E/data/server/eng/eng_vocab',
    'corpora':  '/media/patrik/1EDB65B8599DD93E/data/server/eng/ENG_DATA_SYNC'
}

FRA = {
    'vocab':    '/media/patrik/1EDB65B8599DD93E/data/server/fra/FRA_VOCAB',
    'corpora':  '/media/patrik/1EDB65B8599DD93E/data/server/fra/FRA_DATA_SYNC'
}


def vocab_creator(lang):
    def add_word(w, voc):
        if w not in voc:
            voc[w] = numpy.random.rand(EMBEDDING_DIM).astype(numpy.float32)

    vocab = {}
    with open(lang['corpora'], 'r') as file:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Creating vocab')
            for line in file:
                p_bar.update()
                line_as_list = line.strip().split()
                for token in line_as_list:
                    add_word(token, vocab)

    with open(lang['vocab'], 'w') as file:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Saving vocab')
            file.write('{0} {1}\n'.format(len(vocab), EMBEDDING_DIM))
            for word in vocab.keys():
                p_bar.update()
                file.write('%s %s\n' % (word, ' '.join(list(map(str, list(vocab[word]))))))


def main():
    vocab_creator(ENG)
    vocab_creator(FRA)


if __name__ == '__main__':
    main()
