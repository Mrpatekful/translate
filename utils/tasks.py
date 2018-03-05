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
    Translation task, without parallel corpus. The method follows the main principles described
    in this article:

        https://arxiv.org/abs/1711.00043

    The main goal of this task is to train a denoising auto-encoder, that learns to map
    sentences to sentences in two ways. The first way is to transform a noisy version of
    the source sentence to it's original form, and the second way is to transform a translated
    version of a sentence to it's original form. There is an additional factor during training,
    which is an adversial reguralization, that learns to discriminate the hidden representations
    of the source and target languages.
    """

    def __init__(self,
                 source_reader,
                 target_reader,
                 source_language,
                 target_language,
                 model,
                 model_params):
        """
        An instance of an unsupervised translation task.
        :param source_reader: Reader, an instance of a reader object, that may be a FastReader or FileReader.
        :param target_reader: Reader, that is the same as the source reader, but for the target language.
        :param source_language: Language, the language object, that will be used as the source language.
        :param target_language: Language, that will be used as the target language.
        :param model: Model, the class of the model, that will be used for this task.
        :param model_params: dict, the parameters that are required by the model.
        """
        self._source_language = source_language
        self._target_language = target_language

        self._source_reader = source_reader
        self._target_reader = target_reader

        self._model = model(**model_params)
        print(self._model)

    def fit_model(self, epochs):
        """
        Fits the model to the data. The training session is run until convergence, or
        the given epochs.
        :param epochs: int, the number of maximum epochs.
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
        A single step of the training. A single batch of data is propagated forward the model,
        evaluated, and back-propagated. The parameters are updated by calling the
        :param input_batch: Variable, containing the ids of the words.
        :param lengths: Ndarray, containing the lengths of each sentence in the input batch.
        :param noise_function: The noise model, that will be applied to the input sentences. As
                               written in the task description, this could serve as a dropout like
                               mechanism, or a translation model from the previous iteration.
        :param loss_function: The loss function used for the calculation of the error.
        :return: int, loss at the current time step, produced by this iteration.
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
        Assembler function for the unsupervised translation task. This method
        creates the reader and language objects described by the parameters.
        :param params: dict, containing the parameters for the source and target objects that
                       are required by this task.
        :return: dict, containing the instantiated reader and language objects.
        """
        readers = utils.subclasses(reader.Reader)

        source_language = utils.Language(params['data']['source_vocab'])
        target_language = utils.Language(params['data']['target_vocab'])

        source_reader = readers[params['reader']['source_reader']]
        source_reader = source_reader(language=source_language,
                                      max_segment_size=params['reader']['max_segment_size'],
                                      data_path=params['data']['source_data'],
                                      batch_size=params['reader']['batch_size'],
                                      use_cuda=params['use_cuda'])

        target_reader = readers[params['reader']['target_reader']]
        target_reader = target_reader(language=target_language,
                                      max_segment_size=params['reader']['max_segment_size'],
                                      data_path=params['data']['target_data'],
                                      batch_size=params['reader']['batch_size'],
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
