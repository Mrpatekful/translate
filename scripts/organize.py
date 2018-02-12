import os
from itertools import zip_longest
import copy


FILE_INPUT = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_tok'
FILE_OUTPUT = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_org'


BUFFER_MAX_LENGTH = 5000


def insert_to_buffer(line_buffer, line_dict):
    for index in range(len(line_buffer)):
        if line_buffer[index]['length'] <= line_dict['length']:
            line_buffer.insert(index, line_dict)
            return
    line_buffer.append(line_dict)


def write_longest(line_a, line_b, line_buffer):
    if len(line_buffer) == 0:
        if line_b is None:
            longest = line_a
        elif line_a is None:
            longest = line_b
        else:
            lines = sorted([{'sentence': line_a, 'length': len(line_a.split(' '))},
                            {'sentence': line_b, 'length': len(line_b.split(' '))}],
                           key=lambda x: x['length'])

            longest = lines[-1]['sentence']
            insert_to_buffer(line_buffer, lines[0])

    elif line_b is None or line_a is None:  # line_a and line_b can't be simultaneously None
        line = line_a if line_b is None else line_b
        if len(line.split(' ')) >= line_buffer[0]['length']:
            longest = line
        else:
            longest = copy.deepcopy(line_buffer[0]['sentence'])
            del line_buffer[0]
            insert_to_buffer(line_buffer, {'sentence': line, 'length': len(line.split(' '))})

    else:
        line_a_dict = {'sentence': line_a, 'length': len(line_a.split(' '))}
        line_b_dict = {'sentence': line_b, 'length': len(line_b.split(' '))}

        if line_buffer[0]['length'] >= line_a_dict['length'] and \
           line_buffer[0]['length'] >= line_b_dict['length']:

            longest = copy.deepcopy(line_buffer[0]['sentence'])
            del line_buffer[0]
            insert_to_buffer(line_buffer, line_a_dict)
            insert_to_buffer(line_buffer, line_b_dict)

        else:
            lines = sorted([line_a_dict, line_b_dict], key=lambda x: x['length'])
            longest = lines[1]['sentence']
            insert_to_buffer(line_buffer, lines[0])

    return longest


def write_to_temp_file(data_buffer, temp_files):
    temp_file = FILE_OUTPUT + str(len(temp_files))
    with open(temp_file, 'w') as temp_output_file:
        for line_as_list in sorted(data_buffer, key=lambda x: len(x), reverse=True):
            sentence = line_as_list[0]
            for word in line_as_list[1:]:
                sentence += (' ' + word)
            temp_output_file.write(sentence)
    temp_files.append(temp_file)
    del data_buffer[:]


def main():
    temp_files = []
    with open(FILE_INPUT, 'r') as file_input:
        data_buffer = []
        for line in file_input:
            if len(data_buffer) < BUFFER_MAX_LENGTH:
                data_buffer.append(line.split(' '))
            else:
                write_to_temp_file(data_buffer=data_buffer, temp_files=temp_files)

        # if the chunk is larger than the leftover sentences write the leftover to a file
        if len(data_buffer) > 0:
            write_to_temp_file(data_buffer=data_buffer, temp_files=temp_files)

    file_id = len(temp_files)
    while len(temp_files) > 1:
        with open(temp_files[0], 'r') as temp_file_a:
            with open(temp_files[1], 'r') as temp_file_b:
                temp_output_file = FILE_OUTPUT + str(file_id)
                # temp_file_c is the output of the merge operation of temp_file_a and time_file_b
                with open(temp_output_file, 'w') as temp_file_c:
                    line_buffer = []
                    for line_a, line_b in zip_longest(temp_file_a, temp_file_b, fillvalue=None):
                        temp_file_c.write(write_longest(line_a, line_b, line_buffer))
                    if len(line_buffer) != 0:
                        for line in line_buffer:
                            temp_file_c.write(line['sentence'])

                    del line_buffer

        os.remove(temp_files[0])
        os.remove(temp_files[1])
        del temp_files[0]
        del temp_files[0]
        temp_files.insert(0, FILE_OUTPUT + str(file_id))
        file_id += 1

    # unnecessary, copying the last sorted text file to the original file
    with open(temp_files[0], 'r') as temp_file:
        with open(FILE_OUTPUT, 'w') as file_output:
            for line in temp_file:
                file_output.write(line)
    os.remove(temp_files[0])


if __name__ == '__main__':
    main()

