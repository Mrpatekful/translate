"""

"""

import tqdm
import numpy
import argparse
import faiss
import torch
import pickle

from numpy.linalg import inv


DEFAULT_VOCAB_SRC = '/media/patrik/1EDB65B8599DD93E/data/mapping/vectors-en-sync.txt'
DEFAULT_VOCAB_TGT = '/media/patrik/1EDB65B8599DD93E/data/mapping/vectors-fr-sync.txt'

DEFAULT_ALIGNMENT_PATH_SRC = '/media/patrik/1EDB65B8599DD93E/data/eng/ALIGNMENT_EN_to_FR_2'
DEFAULT_ALIGNMENT_PATH_TGT = '/media/patrik/1EDB65B8599DD93E/data/fra/ALIGNMENT_FR_to_EN_2'

DEFAULT_MAPPING_PATH = '/media/patrik/1EDB65B8599DD93E/data/mapping/best_mapping.pth'

DEFAULT_DIM = 300

DEFAULT_SIZE_SRC = 0
DEFAULT_SIZE_TGT = 0


def measure_length(path):
    lines = 0
    with open(path, 'r') as file:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Measuring length')
            for _ in file:
                p_bar.update()
                lines += 1

    return lines


def load_vocab(path, desc, shape):
    size = 0
    vocab = numpy.empty(shape)
    words = []
    with open(path, 'r', encoding='utf-8') as _vocab:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description(desc)
            for line in _vocab:
                p_bar.update()
                try:
                    line_as_list = line.strip().split()
                    words.append(line_as_list[0])
                    vector = line_as_list[1:]
                    for index, element in enumerate(vector[-shape[1]:]):
                        vocab[size, index] = float(element)

                except ValueError as error:
                    raise RuntimeError(f'Invalid value ({error}) at row {size} (Vector length: {len(vector)})\n{line}')

                except IndexError:
                    raise RuntimeError(f'Invalid vector length at row {size}')
                size += 1

    return vocab.astype('float32'), words


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', action='store', dest='source', type=str, default=DEFAULT_VOCAB_SRC,
                        help='path of a vocabulary')
    parser.add_argument('--target', action='store', dest='target', type=str, default=DEFAULT_VOCAB_TGT,
                        help='path of a vocabulary')
    parser.add_argument('--st_path', action='store', dest='st_path', type=str, default=DEFAULT_ALIGNMENT_PATH_SRC,
                        help='path of the produced alignment A->B')
    parser.add_argument('--ts_path', action='store', dest='ts_path', type=str, default=DEFAULT_ALIGNMENT_PATH_TGT,
                        help='path of the produced alignment B->A')
    parser.add_argument('--mapping_path', action='store', dest='mapping_path', type=str, default=DEFAULT_MAPPING_PATH,
                        help='path of the mapping matrix')
    parser.add_argument('--mapping', action='store_true', dest='mapping_required')
    parser.add_argument('--dim', action='store', dest='dim', type=int, default=DEFAULT_DIM,
                        help='dimension of the vectors')
    parser.add_argument('--size_src', action='store', dest='size_src', type=int, default=DEFAULT_SIZE_SRC,
                        help='path of the new vocab')
    parser.add_argument('--size_tgt', action='store', dest='size_tgt', type=int, default=DEFAULT_SIZE_TGT,
                        help='out of vocab limit per line')

    arguments = parser.parse_args()

    vocab_path_src = arguments.source
    vocab_path_tgt = arguments.target
    alignment_path_st = arguments.st_path
    alignment_path_ts = arguments.ts_path
    mapping_path = arguments.mapping_path
    mapping_required = arguments.mapping_required
    dim = arguments.dim

    size_src = arguments.size_src if arguments.size_src != 0 else measure_length(vocab_path_src)
    size_tgt = arguments.size_tgt if arguments.size_tgt != 0 else measure_length(vocab_path_tgt)

    vocab_src, words_src = load_vocab(path=vocab_path_src, desc='Loading source vocab', shape=(size_src, dim))
    vocab_tgt, words_tgt = load_vocab(path=vocab_path_tgt, desc='Loading target vocab', shape=(size_tgt, dim))

    mapping_matrix = torch.load(mapping_path)

    print('Mapping SRC -> TGT ...')
    if mapping_required:
        projected_vocab_src = numpy.dot(vocab_src, mapping_matrix)
    else:
        projected_vocab_src = vocab_src
    index = faiss.IndexFlatL2(dim)
    index.add(vocab_tgt)
    _, indexes = index.search(projected_vocab_src, 1)

    pickle.dump(obj={words_src[src_index]: words_tgt[tgt_index]
                     for src_index, tgt_index in enumerate(list(indexes.reshape(-1)))},
                file=open(alignment_path_st, 'wb'))

    print('Mapping TGT -> SRC ...')
    if mapping_required:
        inverse_mapping_matrix = inv(mapping_matrix)
        projected_vocab_tgt = numpy.dot(vocab_tgt, inverse_mapping_matrix)
    else:
        projected_vocab_tgt = vocab_tgt
    index = faiss.IndexFlatL2(dim)
    index.add(vocab_src)
    _, indexes = index.search(projected_vocab_tgt, 1)

    pickle.dump(obj={words_tgt[tgt_index]: words_src[src_index]
                     for tgt_index, src_index in enumerate(list(indexes.reshape(-1)))},
                file=open(alignment_path_ts, 'wb'))


if __name__ == '__main__':
    main()
