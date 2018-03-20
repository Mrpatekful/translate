
FILE_ENG_FRA = '../data/eng-fra'
FILE_ENG = '../data/eng/eng_raw'
FILE_FRA = '../data/fra/fra_raw'


def eng():
    split_list = []
    punctuation_marks = []
    with open(FILE_ENG_FRA, 'r') as eng_fra_file:
        for line in eng_fra_file:
            split = line.split('\t')[0]
            if split[:-1] not in split_list:
                split_list.append(split[:-1])
                punctuation_marks.append(split[-1])
    with open(FILE_ENG, 'w') as mono_file:
        for line, mark in zip(split_list, punctuation_marks):
            mono_file.write(line + mark + '\n')


def fra():
    split_list = []
    punctuation_marks = []
    with open(FILE_ENG_FRA, 'r') as eng_fra_file:
        for line in eng_fra_file:
            split = line.split('\t')[1]
            if split[:-1] not in split_list:
                split_list.append(split[:-2])
                punctuation_marks.append(split[-2])
    with open(FILE_FRA, 'w') as mono_file:
        for line, mark in zip(split_list, punctuation_marks):
            mono_file.write(line + mark + '\n')


def main():
    fra()


if __name__ == '__main__':
    main()
