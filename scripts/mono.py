
FILE_ENG_FRA = '../data/eng/eng-fra'
FILE_ENG = '../data/eng/eng_raw'


def main():
    eng_list = []
    punctuation_marks = []
    with open(FILE_ENG_FRA, 'r') as eng_fra_file:
        for line in eng_fra_file:
            eng = line.split('\t')[0]
            if eng[:-1] not in eng_list:
                eng_list.append(eng[:-1])
                punctuation_marks.append(eng[-1])
    with open(FILE_ENG, 'w') as eng_file:
        for line, mark in zip(eng_list, punctuation_marks):
            eng_file.write(line + mark + '\n')


if __name__ == '__main__':
    main()
