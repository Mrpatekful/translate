from utils import utils
from models import models

import torch
import abc


class Task(metaclass=abc.ABCMeta):
    """

    """

    @abc.abstractmethod
    def fit_model(self, *args, **kwargs):
        """

        """

    @abc.abstractmethod
    def _train_step(self, *args, **kwargs):
        """

        """
    @abc.abstractmethod
    def _save_model(self):
        """

        """

    @abc.abstractmethod
    def _load_model(self):
        """

        """


class UnsupervisedTranslation(Task):
    """

    """
    _models = utils.subclasses(models.Models)

    def __init__(self,
                 source_reader,
                 target_reader,
                 source_language,
                 target_language,
                 model_type,
                 model_params):
        """

        :param source_reader:
        :param target_reader:
        :param source_language:
        :param target_language:
        :param model_type:
        :param model_params:
        """
        self._source_language = source_language
        self._target_language = target_language

        self._source_reader = source_reader
        self._target_reader = target_reader

        self._model = self._models[model_type](**model_params)

    def fit_model(self, epochs):
        """
        :param epochs:
        :return:
        """
        loss_function = torch.nn.NLLLoss(ignore_index=0)
        noise_function = utils.NoiseModel()

        for epoch in range(epochs):
            self._model.encoder_embedding = self._source_language.embedding
            self._model.decoder_embedding = self._source_language.embedding
            loss = 0

            for batch, lengths in self._source_reader.batch_generator():
                loss += self._train_step(input_batch=batch,
                                         lengths=lengths,
                                         loss_function=loss_function,
                                         noise_function=noise_function)

            print(epoch, loss)

    def _train_step(self,
                    input_batch,
                    lengths,
                    noise_function,
                    loss_function):
        """

        :param input_batch:
        :param lengths:
        :param noise_function:
        :param loss_function:
        :return:
        """
        self._model.zero_grad()

        noisy_input_batch = noise_function(input_batch)

        decoder_outputs, encoder_outputs = self._model.forward(inputs=noisy_input_batch,
                                                               targets=input_batch,
                                                               lengths=lengths,
                                                               loss_function=loss_function)

        decoder_outputs['loss'].backward()

        self._model.step()

        return decoder_outputs['loss']

    def _save_model(self):
        """

        :return:
        """
        pass

    def _load_model(self):
        """

        :return:
        """
        pass
