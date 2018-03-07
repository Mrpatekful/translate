import re

FILE_INPUT = '../data/eng/eng_org'
FILE_OUTPUT = '../data/eng/eng_seg'

MIN_LENGTH = 5
MAX_LENGTH = 10

END_OF_LINE_PUNCTUATIONS = ['.', '!', '?']


def main():
    with open(FILE_INPUT, 'r') as file_tok:
        with open(FILE_OUTPUT, 'w') as file_seg:
            for line in file_tok:
                line_as_list = re.split(r"[\s]+", line)
                if not (MIN_LENGTH < len(line_as_list) < MAX_LENGTH):
                    continue
                line_as_list.insert(-1, '<EOS>')
                line_as_list.insert(0, '<ENG>')
                line_as_list.insert(0, '<SOS>')

                new_line = str(line_as_list[0])
                for word in line_as_list[1:]:
                    new_line += str(word) if str(word) == '<SOS>' else ' ' + str(word)

                file_seg.write(new_line + '\n')


if __name__ == '__main__':
    main()
