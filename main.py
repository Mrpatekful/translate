from model import seq2seq

FILE_SEG = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_seg'  # correctly segmented and tokenized file
FILE_TOK = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_tok'  # tokenized text file
VOCAB_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_voc'  # vocabulary file


def main():
    auto_encoder = seq2seq.Model()
    auto_encoder.fit(epochs=10)


if __name__ == '__main__':
    main()
