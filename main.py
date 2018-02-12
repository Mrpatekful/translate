from model import seq2seq
from data import reader

FILE_SEG = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_seg'  # correctly segmented and tokenized file
FILE_TOK = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_tok'  # tokenized text file
VOCAB_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_voc'  # vocabulary file


def main():
    # auto_encoder = seq2seq.Model()
    # auto_encoder.fit(2)
    reader.vocab_creator(FILE_TOK)
    lang = reader.Language()
    lang.load_vocab(VOCAB_PATH)
    r = reader.Reader(data_path=FILE_SEG,
                      batch_size=2, language=lang, full_load=False, use_cuda=True)
    for epoch in r.batch_generator():
        print('asd')


if __name__ == '__main__':
    main()
