"""

"""

import tqdm
import argparse


DEFAULT_INPUT = '/media/patrik/1EDB65B8599DD93E/data/eng/test'
DEFAULT_OUTPUT = '/media/patrik/1EDB65B8599DD93E/data/eng/test_tok'

DEFAULT_MIN = 3
DEFAULT_MAX = 60


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--min', action='store', dest='min_length', type=int, default=DEFAULT_MIN,
                        help='minimum length of a line')
    parser.add_argument('--max', action='store', dest='max_length', type=int, default=DEFAULT_MAX,
                        help='maximum length of a line')
    parser.add_argument('-i', '--input', action='store', dest='input_file', type=str, default=DEFAULT_INPUT,
                        help='path of the input file')
    parser.add_argument('-o', '--output', action='store', dest='output_file', type=str, default=DEFAULT_OUTPUT,
                        help='path of the output file')

    arguments = parser.parse_args()

    _min_length = arguments.min_length
    _max_length = arguments.max_length
    _file_input = arguments.input_file
    _file_output = arguments.output_file

    removed_line_count = 0
    line_count = 0

    with open(_file_input, 'r', encoding='utf-8') as file_raw:
        with open(_file_output, 'w', encoding='utf-8') as file_tok:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Tokenizing corpora')
                for line in file_raw:
                    p_bar.update()
                    line_count += 1
                    line_as_list = line.strip().split()
                    for word_index in range(len(line_as_list)-1, -1, -1):
                        if len(line_as_list[word_index]) > 1 and not line_as_list[word_index].isalnum():
                            sub_words = tokenize(line_as_list[word_index])
                            del line_as_list[word_index]

                            for sub_word_index in range(len(sub_words)):
                                line_as_list.insert(word_index + sub_word_index, sub_words[sub_word_index])

                    line_as_list = list(filter(lambda x: x != '' and x != ' ', line_as_list))

                    if _max_length > len(line_as_list) > _min_length:
                        file_tok.write('%s\n' % ' '.join(list(map(lambda x: x.lower(), line_as_list))))
                    else:
                        removed_line_count += 1

    print(f'Number of removed lines: {removed_line_count} ({float(removed_line_count/line_count)*100:.4}%)')


if __name__ == '__main__':
    main()
