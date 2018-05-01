"""

"""

import argparse
import tqdm
import os

DEFAULT_CORPORA = '/media/patrik/1EDB65B8599DD93E/data/fra/FRA_DATA_SYNC_SEG'

DEFAULT_TRAIN_SPLIT = 0.975
DEFAULT_DEV_SPLIT = 0.0125


def location_scheme(path, corpora_type):
    return os.path.join(os.path.dirname(os.path.realpath(path)), f'{corpora_type}')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', action='store', dest='input_corpora', type=str, default=DEFAULT_CORPORA,
                        help='path of the input corpora')
    parser.add_argument('--train', action='store', dest='train', type=float, default=DEFAULT_TRAIN_SPLIT,
                        help='path of the input corpora')
    parser.add_argument('--dev', action='store', dest='dev', type=float, default=DEFAULT_DEV_SPLIT,
                        help='path of the input corpora')

    arguments = parser.parse_args()

    input_corpora = arguments.input_corpora

    train_split = arguments.train
    dev_split = arguments.dev

    train_file = location_scheme(input_corpora, 'TRAIN')
    dev_file = location_scheme(input_corpora, 'DEV')
    test_file = location_scheme(input_corpora, 'TEST')

    lines = 0
    with open(input_corpora, 'r') as corpora:
        with tqdm.tqdm() as p_bar:
            p_bar.set_description('Measuring corpora length')
            for _ in corpora:
                p_bar.update()
                lines += 1

    with open(input_corpora, 'r') as corpora:
        with open(train_file, 'w') as train:
            with open(dev_file, 'w') as dev:
                with open(test_file, 'w') as test:
                    with tqdm.tqdm() as p_bar:
                        p_bar.set_description('Creating train split')
                        for index, line in enumerate(corpora):
                            p_bar.update()
                            if index < int(train_split*lines):
                                train.write(line)
                            elif index < int(train_split*lines) + int(dev_split*lines):
                                p_bar.set_description('Creating dev split')
                                dev.write(line)
                            else:
                                p_bar.set_description('Creating test split')
                                test.write(line)


if __name__ == '__main__':
    main()
