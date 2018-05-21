"""

"""

import argparse
import tqdm
import numpy
import os
import torch
import pickle
import faiss


def location_scheme(path, corpora_type):
    return os.path.join(os.path.dirname(os.path.realpath(path)), f'{corpora_type}')


def synchronize(vocab_path, new_vocab_path, corpora_path, new_corpora_path, oov_limit, min_length):
    tqdm.tqdm.write('Synchronizing corpora with vocab')
    vocab = set()

    with open(vocab_path, 'r', encoding='utf-8') as _vocab:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Loading vocab')
            for line in _vocab:
                p_bar.update()
                vocab.add(line.strip().split()[0])

    old_vocab_size = len(vocab)
    c_removed_line_count = 0
    c_lines = 0

    # Copying the corpora to the new location, while removing the lines with high oov count.

    with open(corpora_path, 'r', encoding='utf-8') as corpora:
        with open(new_corpora_path, 'w', encoding='utf-8') as new_corpora:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Creating new corpora')
                for line in corpora:
                    p_bar.update()
                    c_lines += 1
                    line_as_list = line.strip().split()
                    line_as_list = list(map(lambda x: str(x).lower(), line_as_list))
                    unknown_word_count = 0
                    new_line = []
                    for word in line_as_list:
                        if word not in vocab:
                            unknown_word_count += 1
                        else:
                            new_line.append(word)
                    if unknown_word_count <= oov_limit and  len(new_line) >= min_length:
                        new_corpora.write('%s\n' % ' '.join(new_line))
                    else:
                        c_removed_line_count += 1

    del vocab

    new_vocab = set()

    # Loading the set of distinct words from the newly created corpora

    with open(new_corpora_path, 'r', encoding='utf-8') as new_corpora:
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

    with open(vocab_path, 'r', encoding='utf-8') as _vocab:
        with open(new_vocab_path, 'w', encoding='utf-8') as _new_vocab:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Creating new vocab')
                for line in _vocab:
                    p_bar.update()
                    if line.strip().split()[0] in new_vocab:
                        _new_vocab.write(line)
                    else:
                        v_removed_line_count += 1

    tqdm.tqdm.write(f'(Corpora) Number of removed lines: '
                    f'{c_removed_line_count} ({float(c_removed_line_count/c_lines)*100:.4}%)')
    tqdm.tqdm.write(f'(Corpora) New size:                {c_lines-c_removed_line_count}')
    tqdm.tqdm.write(f'(Vocab) Number of removed lines:   '
                    f'{v_removed_line_count} ({float(v_removed_line_count/old_vocab_size)*100:.4}%)')
    tqdm.tqdm.write(f'(Vocab) New size:                  {new_vocab_size}')


def create_vocab(vocab_path, corpora_path, embedding_dim):


    words = set()

    with open(corpora_path, 'r', encoding='utf-8') as corpora:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Reading corpora')
            for line in corpora:
                p_bar.update()
                line_as_list = line.strip().split()
                for word in line_as_list:
                    words.add(word)

    tqdm.tqdm.write('Creating vocab with %d words, %d dimension embedding' %
                    (len(words), embedding_dim))

    with open(vocab_path, 'w', encoding='utf-8') as vocab:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Creating vocab')
            vocab.write('%d %d\n' % (len(words), embedding_dim))
            for word in words:
                p_bar.update()
                vocab.write('%s %s\n' % (
                    word,
                    ' '.join(list(map(str, numpy.random.random(embedding_dim).astype(numpy.float32)))))
                )


def add_tokens(corpora_path, new_corpora_path):
    tqdm.tqdm.write('Adding tokens to corpora')
    with open(corpora_path, 'r', encoding='utf-8') as corpora:
        with open(new_corpora_path, 'w', encoding='utf-8') as new_corpora:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Creating new corpora')
                for line in corpora:
                    p_bar.update()
                    line_as_list = line.strip().split()
                    line_as_list.insert(0, '<SOS>')
                    line_as_list.append('<EOS>')
                    new_corpora.write('%s\n' % ' '.join(line_as_list))


def split_corpora(corpora_path, train_split, validation_split):
    lines = measure_length(corpora_path)

    tqdm.tqdm.write('Splitting corpora Train: %d | Validation: %f | Test: %f' %
                    (int(lines*train_split),
                     int(lines*validation_split),
                     int(lines*(1 - train_split - validation_split))))

    with open(corpora_path, 'r') as corpora:
        with open(location_scheme(corpora_path, 'train'), 'w') as train:
            with open(location_scheme(corpora_path, 'validation'), 'w') as dev:
                with open(location_scheme(corpora_path, 'test'), 'w') as test:
                    with tqdm.tqdm() as p_bar:
                        p_bar.set_description('Creating train split')
                        for index, line in enumerate(corpora):
                            p_bar.update()
                            if index < int(train_split*lines):
                                train.write(line)
                            elif index < int(train_split*lines) + int(validation_split*lines):
                                p_bar.set_description('Creating dev split')
                                dev.write(line)
                            else:
                                p_bar.set_description('Creating test split')
                                test.write(line)


def synchronize_vocabs(candidate_vocab_path, reference_vocab_path, new_vocab_path):
    tqdm.tqdm.write('Removing unused words from alignment vocab')

    vocab = set()
    with open(reference_vocab_path, 'r', encoding='utf-8') as reference_vocab:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Loading reference vocab')
            for line in reference_vocab:
                p_bar.update()
                vocab.add(line.strip().split()[0])

    with open(candidate_vocab_path, 'r', encoding='utf-8') as candidate_vocab:
        with open(new_vocab_path, 'w', encoding='utf-8') as new_vocab:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Creating new vocab')
                for line in candidate_vocab:
                    p_bar.update()
                    word = line.strip().split()[0]
                    if word in vocab:
                        new_vocab.write(line)


def measure_length(path):
    lines = 0
    with open(path, 'r') as file:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Measuring length')
            for _ in file:
                p_bar.update()
                lines += 1

    return lines


def load_vocab_for_alignment(path, desc, shape):
    size = 0
    vocab = numpy.empty(shape)
    words = []
    with open(path, 'r', encoding='utf-8') as _vocab:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description(desc)
            for line in _vocab:
                p_bar.update()
                line_as_list = line.strip().split()
                words.append(line_as_list[0])
                vector = line_as_list[1:]
                for index, element in enumerate(vector[-shape[1]:]):
                    vocab[size, index] = float(element)
                size += 1

    return vocab.astype('float32'), words


def align_vocabs(source_vocab_path, target_vocab_path, alignment_embedding_dimension):

    size_src = measure_length(source_vocab_path)
    size_tgt = measure_length(target_vocab_path)

    vocab_src, words_src = load_vocab_for_alignment(
        path=source_vocab_path, desc='Loading source vocab', shape=(size_src, alignment_embedding_dimension))
    vocab_tgt, words_tgt = load_vocab_for_alignment(
        path=target_vocab_path, desc='Loading target vocab', shape=(size_tgt, alignment_embedding_dimension))

    tqdm.tqdm.write('Mapping SRC -> TGT ...')

    index = faiss.IndexFlatL2(alignment_embedding_dimension)
    index.add(vocab_tgt)
    _, indexes = index.search(vocab_src, 1)

    pickle.dump(obj={words_src[src_index]: words_tgt[tgt_index]
                     for src_index, tgt_index in enumerate(list(indexes.reshape(-1)))},
                file=open(location_scheme(source_vocab_path, 'alignment'), 'wb'))


def remove_unique(corpora_path, new_corpora_path, min_count, max_removed, min_length, max_vocab_size, max_corpora_size):
    word_frequency = {}

    with open(corpora_path, 'r', encoding='utf-8') as corpora:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Reading corpora')
            for line in corpora:
                p_bar.update()
                line_as_list = line.strip().split()
                for word in line_as_list:
                    word_frequency[word] = word_frequency.get(word, 0) + 1

    ordered_words = set(sorted([word for word in word_frequency if word_frequency[word] >= min_count],
                               key=lambda x: word_frequency[x])[::-1][:max_vocab_size])

    with open(corpora_path, 'r', encoding='utf-8') as corpora:
        with open(new_corpora_path, 'w', encoding='utf-8') as new_corpora:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Creating new corpora')
                for index, line in enumerate(corpora):
                    p_bar.update()
                    line_as_list = line.strip().split()
                    new_line = []
                    unique_count = 0
                    for word in line_as_list:
                        if word in ordered_words:
                            new_line.append(word)
                        else:
                            unique_count += 1
                    if unique_count < max_removed and len(new_line) > min_length:
                        if index > max_corpora_size:
                            break
                        new_corpora.write('%s\n' % ' '.join(new_line))


def tokenize(word):
    sub_words = []
    sub_word = ''
    for index, char in enumerate(word):
        if not char.isalnum() and char != '\'':
            if sub_word != '':
                sub_words.append(sub_word)
                sub_word = ''
            sub_words.append(char)
        else:
            sub_word += char
    if sub_word != '':
        sub_words.append(sub_word)
    return sub_words


def tokenize_corpora(corpora_path, new_corpora_path, min_length, max_length):
    removed_line_count = 0
    line_count = 0

    with open(corpora_path, 'r', encoding='utf-8') as file_raw:
        with open(new_corpora_path, 'w', encoding='utf-8') as file_tok:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Tokenizing corpora')
                for line in file_raw:
                    p_bar.update()
                    line_count += 1
                    line_as_list = line.strip().split()
                    for word_index in range(len(line_as_list) - 1, -1, -1):
                        if len(line_as_list[word_index]) > 1 and not line_as_list[word_index].isalnum():
                            sub_words = tokenize(line_as_list[word_index])
                            del line_as_list[word_index]

                            for sub_word_index in range(len(sub_words)):
                                line_as_list.insert(word_index + sub_word_index, sub_words[sub_word_index])

                    line_as_list = list(filter(lambda x: x != '' and x != ' ', line_as_list))

                    if max_length > len(line_as_list) > min_length:
                        file_tok.write('%s\n' % ' '.join(list(map(lambda x: x.lower(), line_as_list))))
                    else:
                        removed_line_count += 1


def remove_duplicates(vocab_path, new_vocab_path, dimension):
    vocab_dict = {}
    with tqdm.tqdm() as p_bar:
        with open(vocab_path, 'r', encoding='utf-8') as vocab:
            for line in vocab:
                p_bar.update()
                line_as_list = line.strip().split()
                vocab_dict[line_as_list[0]] = list(map(float, line_as_list[-dimension:]))
    tqdm.tqdm.write('Vocab size: %d' % len(vocab_dict))
    with tqdm.tqdm() as p_bar:
        with open(new_vocab_path, 'w', encoding='utf-8') as new_vocab:
            for word in vocab_dict:
                p_bar.update()
                new_vocab.write('%s %s\n' % (word, ' '.join(list(map(str, vocab_dict[word])))))


def main():

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--input', action='store', dest='input_corpora', type=str, default=DEFAULT_CORPORA,
    #                     help='path of the input corpora')
    # parser.add_argument('--train', action='store', dest='train', type=float, default=DEFAULT_TRAIN_SPLIT,
    #                     help='path of the input corpora')
    # parser.add_argument('--dev', action='store', dest='dev', type=float, default=DEFAULT_DEV_SPLIT,
    #                     help='path of the input corpora')
    #
    # arguments = parser.parse_args()

    embedding_dim = 100
    max_vocab_size = 30000
    max_corpora_size = 3000000

    source_corpora = '/media/patrik/1EDB65B8599DD93E/data/test/eng/eng_raw'
    source_vocab = '/media/patrik/1EDB65B8599DD93E/data/test/eng/vectors-en.txt'

    target_corpora = '/media/patrik/1EDB65B8599DD93E/data/test/fra/fra_raw'
    target_vocab = '/media/patrik/1EDB65B8599DD93E/data/test/fra/vectors-fr.txt'

    synced_source_corpora = location_scheme(source_corpora, 'synced_data')
    synced_source_vocab = location_scheme(source_vocab, 'synced_vocab')

    synced_target_corpora = location_scheme(target_corpora, 'synced_data')
    synced_target_vocab = location_scheme(target_vocab, 'synced_vocab')

    tokenized_source_corpora = location_scheme(source_corpora, 'tokenized_data')

    tokenized_target_corpora = location_scheme(target_corpora, 'tokenized_data')

    tokenize_corpora(corpora_path=source_corpora, new_corpora_path=tokenized_source_corpora,
                     min_length=3, max_length=30)

    tokenize_corpora(corpora_path=target_corpora, new_corpora_path=tokenized_target_corpora,
                     min_length=3, max_length=30)

    synchronize(corpora_path=tokenized_source_corpora, vocab_path=source_vocab,
                new_corpora_path=synced_source_corpora,
                new_vocab_path=synced_source_vocab,
                min_length=3, oov_limit=4)

    synchronize(corpora_path=tokenized_target_corpora, vocab_path=target_vocab,
                new_corpora_path=synced_target_corpora,
                new_vocab_path=synced_target_vocab,
                min_length=3, oov_limit=4)

    reduced_source_corpora = location_scheme(source_corpora, 'reduced_data')

    remove_unique(
        corpora_path=synced_source_corpora,
        new_corpora_path=reduced_source_corpora,
        max_removed=2, min_count=4, min_length=3,
        max_vocab_size=max_vocab_size,
        max_corpora_size=max_corpora_size
    )

    reduced_source_vocab = location_scheme(source_corpora, 'reduced_vocab')

    create_vocab(vocab_path=reduced_source_vocab, corpora_path=reduced_source_corpora,
                 embedding_dim=embedding_dim)

    reduced_target_corpora = location_scheme(target_corpora, 'reduced_data')

    remove_unique(
        corpora_path=synced_target_corpora,
        new_corpora_path=reduced_target_corpora,
        max_removed=2, min_count=4, min_length=3,
        max_vocab_size=max_vocab_size,
        max_corpora_size=max_corpora_size
    )

    reduced_target_vocab = location_scheme(target_corpora, 'reduced_vocab')

    create_vocab(vocab_path=reduced_target_vocab, corpora_path=reduced_target_corpora,
                 embedding_dim=embedding_dim)

    tokenized_source_corpora = location_scheme(source_corpora, 'tokenized_data')
    tokenized_target_corpora = location_scheme(target_corpora, 'tokenized_data')

    add_tokens(corpora_path=reduced_source_corpora, new_corpora_path=tokenized_source_corpora)
    add_tokens(corpora_path=reduced_target_corpora, new_corpora_path=tokenized_target_corpora)

    reduced_source_align_vocab = location_scheme(source_corpora, 'reduced_align_vocab')
    reduced_target_align_vocab = location_scheme(target_corpora, 'reduced_align_vocab')

    synchronize_vocabs(candidate_vocab_path=source_vocab,
                       new_vocab_path=reduced_source_align_vocab,
                       reference_vocab_path=reduced_source_vocab)

    synchronize_vocabs(candidate_vocab_path=target_vocab,
                       new_vocab_path=reduced_target_align_vocab,
                       reference_vocab_path=reduced_target_vocab)

    remove_duplicates(vocab_path=reduced_source_align_vocab,
                      new_vocab_path=reduced_source_align_vocab,
                      dimension=300)

    remove_duplicates(vocab_path=reduced_target_align_vocab,
                      new_vocab_path=reduced_target_align_vocab,
                      dimension=300)

    align_vocabs(alignment_embedding_dimension=300,
                 source_vocab_path=reduced_source_align_vocab,
                 target_vocab_path=reduced_target_align_vocab)

    align_vocabs(alignment_embedding_dimension=300,
                 source_vocab_path=reduced_target_align_vocab,
                 target_vocab_path=reduced_source_align_vocab)

    split_corpora(corpora_path=tokenized_source_corpora,
                  train_split=0.975,
                  validation_split=0.0125)

    split_corpora(corpora_path=tokenized_target_corpora,
                  train_split=0.975,
                  validation_split=0.0125)



if __name__ == '__main__':
    main()
