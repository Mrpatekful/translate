from utils import utils
from utils import tasks
from models import models

import json


class Config:
    """

    """
    _models = utils.subclasses(models.Model)
    _tasks = utils.subclasses(tasks.Task)

    def __init__(self, task_config, model_config):
        self._task_config = json.load(open(task_config, 'r'))
        self._model_config = json.load(open(model_config, 'r'))

    def _assemble_model(self, task_params):
        """

        :param task_params:
        :return:
        """
        try:
            model = self._models[self._model_config['model_type']]
            model_params = model.assemble({**task_params,
                                           **self._model_config['components'],
                                           'use_cuda': self._task_config['use_cuda']})

        except KeyError:
            print('Invalid JSON file for the given model.')
            return

        return model, model_params

    def _assemble_task(self):
        """

        :return:
        """
        try:
            task = self._tasks[self._task_config['task_type']]
            task_params = task.assemble({**self._task_config['components'],
                                         'use_cuda': self._task_config['use_cuda']})

        except KeyError:
            print('Invalid JSON file for the given task.')
            return

        return task, task_params

    def assemble(self):
        """

        :return:
        """
        task, task_params = self._assemble_task()
        model, model_params = self._assemble_model(task_params)

        return task(model=model, model_params=model_params, **task_params)
