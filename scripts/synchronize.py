"""

"""

import tqdm
import argparse


DEFAULT_CORPORA = '/media/patrik/1EDB65B8599DD93E/data/fra/FRA_DATA'
DEFAULT_VOCAB = '/media/patrik/1EDB65B8599DD93E/data/fra/wiki.fr.vec'

DEFAULT_OUTPUT_CORPORA = '/media/patrik/1EDB65B8599DD93E/data/fra/FRA_DATA_SYNC'
DEFAULT_OUTPUT_VOCAB = '/media/patrik/1EDB65B8599DD93E/data/fra/vectors-fr_SYNC'


# The maximum number of out of vocab words in a sentence
DEFAULT_OOV_LIMIT = 3

DEFAULT_MIN_LENGTH = 3


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputC', action='store', dest='old_corpora', type=str, default=DEFAULT_CORPORA,
                        help='path of the old corpora')
    parser.add_argument('--inputV', action='store', dest='old_vocab', type=str, default=DEFAULT_VOCAB,
                        help='path of the old vocab')
    parser.add_argument('--outputC', action='store', dest='new_corpora', type=str, default=DEFAULT_OUTPUT_CORPORA,
                        help='path of the new corpora')
    parser.add_argument('--outputV', action='store', dest='new_vocab', type=str, default=DEFAULT_OUTPUT_VOCAB,
                        help='path of the new vocab')
    parser.add_argument('--limit', action='store', dest='oov_limit', type=int, default=DEFAULT_OOV_LIMIT,
                        help='out of vocab limit per line')
    parser.add_argument('--min', action='store', dest='min_length', type=int, default=DEFAULT_MIN_LENGTH,
                        help='minimum length of a line')

    arguments = parser.parse_args()

    _vocab_path = arguments.old_vocab
    _corpora_path = arguments.old_corpora
    _new_vocab_path = arguments.new_vocab
    _new_corpora_path = arguments.new_corpora
    _oov_limit = arguments.oov_limit
    _min_length = arguments.min_length

    vocab = set()

    with open(_vocab_path, 'r', encoding='utf-8') as _vocab:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Loading vocab')
            for line in _vocab:
                p_bar.update()
                vocab.add(line.strip().split()[0])

    old_vocab_size = len(vocab)
    c_removed_line_count = 0
    c_lines = 0

    # Copying the corpora to the new location, while removing the lines with high oov count.

    with open(_corpora_path, 'r', encoding='utf-8') as corpora:
        with open(_new_corpora_path, 'w', encoding='utf-8') as new_corpora:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Creating new corpora')
                for line in corpora:
                    p_bar.update()
                    c_lines += 1
                    line_as_list = line.strip().split()
                    unknown_word_count = 0
                    new_line = []
                    for word in line_as_list:
                        if word not in vocab:
                            unknown_word_count += 1
                        else:
                            new_line.append(word)
                    if unknown_word_count <= _oov_limit and len(new_line) >= _min_length:
                        new_corpora.write('%s\n' % ' '.join(new_line))
                    else:
                        c_removed_line_count += 1

    del vocab

    new_vocab = set()

    # Loading the set of distinct words from the newly created corpora

    with open(_new_corpora_path, 'r', encoding='utf-8') as new_corpora:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Loading new vocab')
            for line in new_corpora:
                p_bar.update()
                line_as_list = line.strip().split()
                for word in line_as_list:
                    new_vocab.add(word)

    new_vocab_size = len(new_vocab)

    v_removed_line_count = 0

    # Copying the old vocab to the new location, while removing the unnecessary words

    with open(_vocab_path, 'r', encoding='utf-8') as _vocab:
        with open(_new_vocab_path, 'w', encoding='utf-8') as _new_vocab:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Creating new vocab')
                for line in _vocab:
                    p_bar.update()
                    if line.strip().split()[0] in new_vocab:
                        _new_vocab.write(line)
                    else:
                        v_removed_line_count += 1

    print(f'(Corpora) Number of removed lines: '
          f'{c_removed_line_count} ({float(c_removed_line_count/c_lines)*100:.4}%)')
    print(f'(Corpora) New size:                {c_lines-c_removed_line_count}')
    print(f'(Vocab) Number of removed lines:   '
          f'{v_removed_line_count} ({float(v_removed_line_count/old_vocab_size)*100:.4}%)')
    print(f'(Vocab) New size:                  {new_vocab_size}')


if __name__ == '__main__':
    main()
