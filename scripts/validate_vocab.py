"""

"""

import argparse
import tqdm
import sys

DEFAULT_INPUT_VOCAB = '/media/patrik/1EDB65B8599DD93E/data/eng/vectors-en_SYNC'
DEFAULT_OUTPUT_VOCAB = '/media/patrik/1EDB65B8599DD93E/data/eng/wiki.en.vec_synced_validated'

DEFAULT_DIM = 300


def validate(path, dim):
    with open(path, 'r') as vocab:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Validating vocabulary')
            for index, line in enumerate(vocab):
                p_bar.update()
                line_as_list = line.strip().split()

                try:
                    assert len(line_as_list) == dim + 1, f'Invalid line length at index {index}\n{line}'

                    map(float, line_as_list)

                except AssertionError as error:
                    raise RuntimeError(error)

                except ValueError as error:
                    raise RuntimeError(f'Non-numeric value at index {index} {error}\n {line}')


def copy(path_from, path_to, dim):
    with open(path_from, 'r') as old_vocab:
        with open(path_to, 'w') as new_vocab:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Creating new vocabulary')
                for line in old_vocab:
                    p_bar.update()
                    line_as_list = line.strip().split()
                    new_vocab.write('%s %s\n' % (line_as_list[0], ' '.join(line_as_list[-dim:])))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', action='store', dest='input_vocab', type=str, default=DEFAULT_INPUT_VOCAB,
                        help='path of the input vocab')
    parser.add_argument('--output', action='store', dest='output_vocab', type=str, default=DEFAULT_OUTPUT_VOCAB,
                        help='path of the output vocab')
    parser.add_argument('--dim', action='store', dest='dim', type=int, default=DEFAULT_DIM,
                        help='dimension of the word embeddings')

    arguments = parser.parse_args()

    input_vocab = arguments.input_vocab
    output_vocab = arguments.output_vocab
    dim = arguments.dim

    try:
        validate(path=input_vocab, dim=dim)
    except RuntimeError as error:
        tqdm.tqdm.write('\n%s' % str(error))
        tqdm.tqdm.write('Validation failed, creating new vocabulary...')
        copy(path_from=input_vocab, path_to=output_vocab, dim=dim)
        try:
            validate(path=output_vocab, dim=dim)
        except RuntimeError as error:
            tqdm.tqdm.write('\n%s' % str(error))
            tqdm.tqdm.write('Validation failed, could not create new vocabulary...')
            sys.exit()

    tqdm.tqdm.write('Validation success')


if __name__ == '__main__':
    main()
