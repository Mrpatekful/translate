import re

FILE_INPUT = '/media/patrik/1EDB65B8599DD93E/data/eng/eng_tok'
FILE_OUTPUT = '/media/patrik/1EDB65B8599DD93E/data/eng/eng_tok_short'

MIN_LENGTH = 3
MAX_LENGTH = 60

END_OF_LINE_PUNCTUATIONS = ['.', '!', '?']


def main():
    with open(FILE_INPUT, 'r') as file_tok:
        with open(FILE_OUTPUT, 'w') as file_seg:
            for c, line in enumerate(file_tok):
                if c % 1000 == 0:
                    print(c)
                line_as_list = line.strip().split()
                if not (MIN_LENGTH < len(line_as_list) < MAX_LENGTH):
                    continue

                file_seg.write(line + '\n')


if __name__ == '__main__':
    main()
