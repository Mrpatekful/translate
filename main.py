from model import reader

FILE_TOK = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/text_tok'  # tokenized text file
VOCAB_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/text_voc'  # vocabulary file


def main():
    eng = reader.Language()
    reader.vocab_creator(FILE_TOK)
    eng.load_vocab(VOCAB_PATH)


if __name__ == '__main__':
    main()
