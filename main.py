from utils import reader

FILE_SEG = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_seg'  # correctly segmented and tokenized file
FILE_TOK = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_tok'  # tokenized text file
VOCAB_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_voc'  # vocabulary file


def main():
    # auto_encoder = seq2seq.Model()
    # auto_encoder.fit(2)
    reader.vocab_creator(FILE_TOK)
    lang = reader.Language()
    lang.load_vocab(VOCAB_PATH)
    r = reader.FastReader(data_path=FILE_SEG, batch_size=50, language=lang, use_cuda=True)
    for index, batch in enumerate(r.batch_generator()):
        if index == 200:
            print(batch)


if __name__ == '__main__':
    main()
