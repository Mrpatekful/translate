from utils import utils
from utils import reader

import torch


class Task:

    def fit_model(self, *args, **kwargs):
        return NotImplementedError

    def test_model(self, *args, **kwargs):
        return NotImplementedError

    @classmethod
    def assemble(cls, params):
        return NotImplementedError

    @classmethod
    def abstract(cls):
        return True


class UnsupervisedTranslation(Task):
    """

    """

    def __init__(self,
                 source_reader,
                 target_reader,
                 source_language,
                 target_language,
                 model,
                 model_params):
        """

        :param source_reader:
        :param target_reader:
        :param source_language:
        :param target_language:
        :param model:
        :param model_params:
        """
        self._source_language = source_language
        self._target_language = target_language

        self._source_reader = source_reader
        self._target_reader = target_reader

        self._model = model(**model_params)
        print(self._model)

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
                loss += self._step(input_batch=batch,
                                   lengths=lengths,
                                   loss_function=loss_function,
                                   noise_function=noise_function)

            print(epoch, loss)

    def test_model(self):
        """

        :return:
        """
        # TODO

    def _step(self,
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
        # TODO

    def _load_model(self):
        """

        :return:
        """
        # TODO

    @classmethod
    def assemble(cls, params):
        """

        :param params:
        :return:
        """
        readers = utils.subclasses(reader.Reader)

        source_language = utils.Language(params['data']['source_vocab'])
        target_language = utils.Language(params['data']['target_vocab'])

        source_reader = readers[params['reader']['source_reader']]
        source_reader = source_reader(language=source_language,
                                      max_segment_size=params['max_segment_size'],
                                      data_path=params['data']['source_data'],
                                      batch_size=params['batch_size'],
                                      use_cuda=params['use_cuda'])

        target_reader = readers[params['reader']['target_reader']]
        target_reader = target_reader(language=target_language,
                                      max_segment_size=params['max_segment_size'],
                                      data_path=params['data']['target_data'],
                                      batch_size=params['batch_size'],
                                      use_cuda=params['use_cuda'])

        return {
            'source_language': source_language,
            'target_language': target_language,
            'source_reader': source_reader,
            'target_reader': target_reader,
        }

    @classmethod
    def abstract(cls):
        return False
