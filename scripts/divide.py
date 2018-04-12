
lang = 'fra'

FILE_INPUT = f'../data/{lang}/{lang}_seg'

FILE_OUTPUTS = {
    'train': f'../data/{lang}/{lang}_train',
    'dev': f'../data/{lang}/{lang}_dev',
    'test': f'../data/{lang}/{lang}_test',
}

TRAIN_SIZE = 0.999
DEV_SIZE = 0.0005


def main():
    file_size = 0
    with open(FILE_INPUT, 'r') as file_seg:
        for _ in file_seg:
            file_size += 1

    with open(FILE_INPUT, 'r') as file_seg:
        with open(FILE_OUTPUTS['train'], 'w') as file_train:
            for index, line in enumerate(file_seg):
                if index + 1 < TRAIN_SIZE*file_size:
                    file_train.write(line)

    with open(FILE_INPUT, 'r') as file_seg:
        with open(FILE_OUTPUTS['dev'], 'w') as file_dev:
            for index, line in enumerate(file_seg):
                if index + 1 > TRAIN_SIZE*file_size:
                    file_dev.write(line)

    with open(FILE_INPUT, 'r') as file_seg:
        with open(FILE_OUTPUTS['test'], 'w') as file_test:
            for index, line in enumerate(file_seg):
                if index + 1 > (TRAIN_SIZE + DEV_SIZE)*file_size:
                    file_test.write(line)




if __name__ == '__main__':
    main()
