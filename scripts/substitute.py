"""

"""

from nltk.metrics import edit_distance
import tqdm

FILE_INPUT: str = '/media/patrik/1EDB65B8599DD93E/data/fra_sorted.txt'
FILE_OUTPUT: str = '/media/patrik/1EDB65B8599DD93E/data/fra_tokenized.txt'
VOCAB: str = '/media/patrik/1EDB65B8599DD93E/data/vectors-fr.txt'

MATCH_LOG: str = 'matched_words.log'
UNMATCHED_LOG: str = 'unmatched_words.log'

MIN_LENGTH: int = 3
MAX_LENGTH: int = 60

WORD_DIST_TOL: int = 2


def similarity_predicate_long(distance, word_len):
    return distance < WORD_DIST_TOL and word_len > 5


def similarity_predicate_short(distance, word_len):
    return distance < WORD_DIST_TOL - 1 and word_len > 3


def find_best_match(string, vocab):
    best_match = None
    best_distance = None
    str_len = len(string)
    for word in vocab:
        if abs(len(word) - str_len) < 3:
            distance = edit_distance(string, word)
            if similarity_predicate_long(distance, str_len) or similarity_predicate_short(distance, str_len):
                best_match = word
                best_distance = distance
            elif best_distance is None or distance < best_distance:
                best_match = word
    if best_match is not None and best_distance is not None:
        normalized_distance = best_distance / (len(best_match) + str_len) / 2
    else:
        normalized_distance = None

    return normalized_distance, best_match


def main():
    matched_words = {}
    unmatched_words = {}

    vocab = set()

    with open(VOCAB, 'r') as vocab_f:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Reading vocab')
            for line in vocab_f:
                p_bar.update()
                vocab.add(line.strip().split()[0])

    average_levenshtein_distance = 0
    inserted_lines = 0
    original_lines = 0

    average_inserted_length = 0
    average_original_length = 0

    shortened_lines = 0

    with open(FILE_INPUT, 'r') as old_corpora:
        with open(FILE_OUTPUT, 'w') as new_corpora:
            with tqdm.tqdm() as p_bar:
                p_bar.set_description('Creating new corpora')
                for line in old_corpora:
                    p_bar.update()
                    original_lines += 1
                    line_as_list = line.strip().split()
                    new_list = []
                    for word in line_as_list:
                        if word in vocab:
                            new_list.append(word)
                        elif word in matched_words:
                            new_list.append(matched_words)
                        else:
                            levenshtein_distance, best_match = find_best_match(word, vocab)
                            if levenshtein_distance is not None:
                                new_list.append(best_match)
                                matched_words[word] = best_match
                                average_levenshtein_distance += levenshtein_distance
                            else:
                                unmatched_words[word] = best_match
                    if (((len(line_as_list) - len(new_list)) < 3 and len(line_as_list) > 9) or
                                (len(line_as_list) - len(new_list)) < 2):
                        new_corpora.write('%s\n' % ' '.join(new_list))
                        inserted_lines += 1
                        if len(line_as_list) != len(new_list):
                            shortened_lines += 1

    print(f'Average Total Levensthein Distance: {float(average_levenshtein_distance/len(matched_words)):.3}')
    print(f'New file length: {original_lines}')
    print(f'New file length: {inserted_lines}')
    print(f'Average line length of the original corpora: {float(average_original_length)/original_lines:.3}')
    print(f'Average line length of the new corpora:      {float(average_inserted_length)/inserted_lines:.3}')
    print(f'New corpora length: {float(inserted_lines/original_lines)*100:.3}% of the original length')

    with open(MATCH_LOG, 'w') as log:
        for word in matched_words:
            log.write(f'Original word: {word} Best match: {matched_words[word]}\n')

    with open(UNMATCHED_LOG, 'w') as log:
        for word in unmatched_words:
            log.write(f'Original word: {word} Best match: {matched_words[word]}\n')


if __name__ == '__main__':
    main()
