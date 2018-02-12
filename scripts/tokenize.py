import re


FILE_INPUT = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_raw'
FILE_OUTPUT = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_tok'

END_OF_LINE_PUNCTUATIONS = ['.', '!', '?']

# true if the raw file is organized, so there is
# only a single sentence or expression in a line
SENTENCES_SEPARATED = True


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
    with open(FILE_INPUT, 'r') as file_raw:
        with open(FILE_OUTPUT, 'w') as file_tok:
            for line in file_raw:
                line_as_list = re.split(r"[\s]+", line)
                for word_index in range(len(line_as_list)-1, -1, -1):
                    if len(line_as_list[word_index]) > 1 and not line_as_list[word_index].isalnum():
                        sub_words = tokenize(line_as_list[word_index])
                        del line_as_list[word_index]

                        for sub_word_index in range(len(sub_words)):
                            line_as_list.insert(word_index + sub_word_index, sub_words[sub_word_index])

                line_as_list = list(filter(lambda x: x != '' and x != ' ', line_as_list))

                if len(line_as_list) == 0:
                    continue

                new_line = str(line_as_list[0]).lower()
                line_break = ''
                for word in line_as_list[1:]:
                    # if sentences are not separated '.' '?' '!' are assumed to be the end of the sentence.
                    if str(word) in END_OF_LINE_PUNCTUATIONS and not SENTENCES_SEPARATED:
                        line_break = '\n'
                    new_line += (' ' + str(word).lower() + line_break)
                line_break = ''
                if SENTENCES_SEPARATED:
                    line_break = '\n'
                file_tok.write(new_line + line_break)


if __name__ == '__main__':
    main()
