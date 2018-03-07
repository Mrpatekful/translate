import json

from models import models
from tasks import tasks
from utils import utils

import sys


class Config:
    """
    Class for handling configurations of models and tasks.
    The configs are defined in JSON format files, which are parsed,
    and instantiated by the corresponding modules.
    """

    def __init__(self, task_config, model_config):
        """
        An instance of a configuration parser. The provided files are
        parsed, stored as dictionary objects.
        :param task_config: str, path of the task configuration file.
        :param model_config: str, path of the model configuration file.
        """
        self._task_config = json.load(open(task_config, 'r'))
        self._model_config = json.load(open(model_config, 'r'))

    def _assemble_model(self, task_params):
        """
        Assembles the model, described by the model configuration file.
        :param task_params: dict, the dictionary of the parsed task parameters.
                            These are necessary for the creation of the model, since
                            it yields information about the size of the language vocab,
                            and embedding dimension.
        :except KeyError: Invalid JSON file configuration.
        :return model: Model, the class of the model described by the configuration file.
        :return model_params: dict, the parameters required by the model class.
        """

        try:

            _models = utils.subclasses(models.Model)
            model = _models[self._model_config['type']]
            model_params = model.assemble({
                **task_params,
                **self._model_config['params'],
                'use_cuda': self._task_config['use_cuda']
            })

        except KeyError as error:
            print('Error: Model configuration file does not follow the '
                  'required format. %s' % error)
            sys.exit()

        return model, model_params

    def _assemble_task(self):
        """
        Assembles the task, described by the task configuration file.
        :except KeyError: Invalid JSON file configuration.
        :return task: Task, the class of the task, which was described in task_type tag.
        :return task_params: dict, the parameters of the task, which was created by
                             the assembler function of the specified task.
        """

        try:

            _tasks = utils.subclasses(tasks.Task)
            task = _tasks[self._task_config['type']]
            task_params = task.assemble({
                **self._task_config['params'],
                'use_cuda': self._task_config['use_cuda']
            })

        except KeyError as error:
            print('Error: Task configuration file does not follow the '
                  'required format. %s' % error)
            sys.exit()

        return task, task_params

    def assemble(self):
        """
        Combines the task and the model, with the parsed parameters.
        :return: Task, an instance of the described task.
        """
        task, task_params = self._assemble_task()
        model, model_params = self._assemble_model(task_params)

        return task(model=model, model_params=model_params, **task_params)
