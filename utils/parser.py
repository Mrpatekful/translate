from utils import reader
from utils import utils
from models import models

import json


class Config:
    """

    """

    _readers = {cls.__name__: cls for cls in reader.Reader.__subclasses__()}
    _models = utils.subclasses(models.Models)

    def __init__(self, config_path):
        """

        :param config_path:
        """
        self._config = json.load(open(config_path, 'r'))
        self._task_params = self._set_task_params()
        self._model_params = self._set_model_params()

    def _set_model_params(self):
        model_params = self._config['model_params']

        try:
            model = self._models[model_params['model_type']]
            processed_params = model.parameter_builder(components=model_params['components'],
                                                       task_params=self._task_params)

        except KeyError:
            print('Invalid JSON file.')
            return

        return processed_params

    def _set_task_params(self):
        task_params = self._config['task_params']
        path_params = task_params['paths']
        reader_params = task_params['readers']

        try:
            source_language = utils.Language(path_params['source_vocab_path'])
            target_language = utils.Language(path_params['target_vocab_path'])

            source_reader = self._readers[reader_params['source_reader']]
            source_reader = source_reader(source_language=source_language,
                                          max_segment_size=task_params['max_segment_size'],
                                          data_path=path_params['source_data_path'],
                                          batch_size=task_params['batch_size'],
                                          use_cuda=self._config['use_cuda'])

            target_reader = self._readers[reader_params['target_reader']]
            target_reader = target_reader(source_language=target_language,
                                          max_segment_size=task_params['max_segment_size'],
                                          data_path=path_params['target_data_path'],
                                          batch_size=task_params['batch_size'],
                                          use_cuda=self._config['use_cuda'])

            processed_params = {
                'source_language': source_language,
                'target_language': target_language,
                'source_reader': source_reader,
                'target_reader': target_reader
            }

        except KeyError:
            print('Invalid JSON file.')
            return

        return processed_params

    @property
    def parameters(self):
        return {
            **self._task_params,
            **self._model_params
        }
