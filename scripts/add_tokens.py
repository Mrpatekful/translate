"""

"""

import argparse
import tqdm

DEFAULT_FILE_INPUT = '/media/patrik/1EDB65B8599DD93E/data/eng/ENG_DATA_SYNC'
DEFAULT_FILE_OUTPUT = '/media/patrik/1EDB65B8599DD93E/data/eng/ENG_DATA_SYNC_SEG'


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', action='store', dest='input_corpora', type=str, default=DEFAULT_FILE_INPUT,
                        help='path of the input corpora')
    parser.add_argument('--output', action='store', dest='output_corpora', type=str, default=DEFAULT_FILE_OUTPUT,
                        help='path of the output corpora')

    arguments = parser.parse_args()

    input_corpora = arguments.input_corpora
    output_corpora = arguments.output_corpora

    with open(input_corpora, 'r') as file_tok:
        with open(output_corpora, 'w') as file_seg:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Adding tokens')
                for line in file_tok:
                    p_bar.update()
                    line_as_list = line.strip().split()
                    line_as_list.insert(0, '<SOS>')
                    line_as_list.append('<EOS>')

                    file_seg.write('%s\n' % ' '.join(line_as_list))


if __name__ == '__main__':
    main()
