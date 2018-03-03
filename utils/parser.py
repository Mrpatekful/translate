from utils import reader
from utils import utils

import json


def parse_params(config_path):
    parameters = json.load(open(config_path, 'r'))
    parameters = {
        'model_params': {
            'model_type': 'SeqToSeq',
            'encoder': 'RNNEncoder',
            'decoder': 'RNNDecoder',
            'encoder_params': {
                '_hidden_size': 50,
                '_recurrent_type': 'LSTM',
                '_num_layers': 2,
                '_learning_rate': 0.01,
                '_use_cuda': True
              },
            'decoder_params': {
                '_hidden_size': 50,
                '_recurrent_type': 'LSTM',
                '_num_layers': 2,
                '_learning_rate': 0.01,
                '_max_length': 15,
                '_tf_ratio': 0,
                '_use_cuda': True
             }
            },
        'trainer_params': {
            'source_reader': 'FastReader',
            'target_reader': 'FastReader',
            'source_data_path': '/data/eng_seg',
            'source_vocab_path': 'data/eng_voc',
            'target_data_path': 'data/text_seg',
            'target_vocab_path': 'data/text_voc',
            'batch_size': 32,
          },
        'use_cuda': True
         }
    readers = {cls.__name__: cls for cls in reader.Reader.__subclasses__()}

    source_reader = readers[parameters['trainer_params']['source_readers']]
    target_reader = readers[parameters['trainer_params']['target_readers']]

    source_language = utils.Language(parameters['trainer_params']['source_vocab_path'])
    target_language = utils.Language(parameters['trainer_params']['target_vocab_path'])

    source_reader = source_reader(language=source_language,
                                  data_path=parameters['trainer_params']['source_data_path'],
                                  batch_size=parameters['trainer_params']['batch_size'],
                                  use_cuda=parameters['trainer_params']['use_cuda'])

    target_reader = target_reader(language=target_language,
                                  data_path=parameters['trainer_params']['target_data_path'],
                                  batch_size=parameters['trainer_params']['batch_size'],
                                  use_cuda=parameters['trainer_params']['use_cuda'])

    parameters['model_params']['encoder_params']['_embedding_size'] = source_language.embedding_size
    parameters['model_params']['decoder_params']['_embedding_size'] = source_language.embedding_size
    parameters['model_params']['decoder_params']['_output_size'] = source_language.vocab_size

    parameters = {**parameters['model_params'],
                  'source_reader': source_reader,
                  'target_reader': target_reader,
                  'source_language': source_language,
                  'target_language': target_language}

    return parameters
