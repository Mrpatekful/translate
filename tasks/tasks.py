from utils import utils
from utils import reader

from modules.utils.utils import Noise

import numpy
import torch
import torch.autograd


class Task:

    def fit_model(self, *args, **kwargs):
        return NotImplementedError

    @classmethod
    def assemble(cls, params):
        return NotImplementedError

    @staticmethod
    def format_batch(batch, use_cuda):
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
                 model,
                 model_params):
        """
        An instance of an unsupervised translation task.
        :param source_reader: Reader, an instance of a reader object, that may be a FastReader or FileReader.
        :param target_reader: Reader, that is the same as the source reader, but for the target language.
        :param model: Model, the class of the model, that will be used for this task.
        :param model_params: dict, the parameters that are required by the model.
        """
        self._source_reader = source_reader
        self._target_reader = target_reader

        self._model = model(**model_params)

    def fit_model(self, epochs):
        """
        Fits the model to the data. The training session is run until convergence, or
        the given epochs.
        :param epochs: int, the number of maximum epochs.
        """
        def fit_step(**kwargs):
            """
            A single batch of data is propagated forward the model,
            evaluated, and back-propagated. The parameters are updated by calling the step
            function of the components' optimizers.
            :param kwargs: inputs for the _step function.
            :return: dict, containing the outputs of the model.
            """
            nonlocal self

            ous = self._step(**kwargs)
            ous['loss'].backward()

            self._model.step()

            return ous

        loss_function = torch.nn.NLLLoss(ignore_index=0)
        noise_function = Noise()

        for epoch in range(epochs):

            self._model.encoder_embedding = {
                'weights': self._source_reader.language.embedding,
                'requires_grad': self._source_reader.language.requires_grad
            }

            self._model.decoder_embedding = {
                'weights': self._source_reader.language.embedding,
                'requires_grad': self._source_reader.language.requires_grad
            }

            loss = 0

            for inputs, targets, lengths in self._source_reader.batch_generator():
                outputs = fit_step(inputs=inputs,
                                   targets=targets,
                                   lengths=lengths,
                                   loss_function=loss_function,
                                   noise_function=noise_function)

                loss += outputs['loss']

                inp = inputs.cpu().data[:, :].numpy()
                out = outputs['symbols'][:, :]
                tgt = targets.cpu().data[:, 1:].numpy()

                # self._source_reader.print_validation_format(input=inp, output=out, target=tgt)

            print(loss)

    def _step(self,
              inputs,
              targets,
              lengths,
              noise_function,
              loss_function):
        """
        Single forward step of the model. A batch of inputs and targets are provided, from which
        the output is calculated, with the current parameter values of the model.
        :param inputs: Variable, containing the ids of the words.
        :param lengths: Ndarray, containing the lengths of each sentence in the input batch.
        :param noise_function: The noise model, that will be applied to the input sentences. As
                               written in the task description, this could serve as a dropout like
                               mechanism, or a translation model from the previous iteration.
        :param loss_function: The loss function used for the calculation of the error.
        :return: int, loss at the current time step, produced by this iteration.
        """
        self._model.zero_grad()

        noisy_inputs = noise_function(inputs)

        outputs = self._model.forward(inputs=noisy_inputs,
                                      targets=targets,
                                      lengths=lengths,
                                      loss_function=loss_function)

        return outputs

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

    @staticmethod
    def format_batch(batch, use_cuda):
        """
        The special batch format, that is required by the task. This function is passed to the reader,
        and will be used to produce batches and targets, in a way, that is convenient for this particular task.
        :param batch:
        :param use_cuda:
        :return:
        """
        inputs = torch.from_numpy(batch[:, 1:-2])
        targets = torch.from_numpy(numpy.hstack((batch[:, 0].reshape(-1, 1), batch[:, 2:-1])))
        lengths = batch[:, -1]-2
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        return torch.autograd.Variable(inputs), torch.autograd.Variable(targets), lengths

    @classmethod
    def assemble(cls, params):
        """
        Assembler function for the unsupervised translation task. This method
        creates the reader and language objects described by the parameters.
        :param params: dict, containing the parameters for the source and target objects that
                       are required by this task.
        :raises KeyError: invalid identifier in the configuration file.
        :return: dict, containing the instantiated reader and language objects.
        """
        readers = utils.subclasses(reader.Reader)

        try:

            source_reader = readers[params['readers']['source']['type']]
            source_reader_params = source_reader.assemble({
                **params['readers']['source']['params'],
                'use_cuda': params['use_cuda'],
            })

            source_reader = source_reader(**{
                **source_reader_params,
                'format_batch': cls.format_batch
            })

        except KeyError as error:
            raise KeyError('%s is not a valid identifier for source reader.' % error)

        try:

            target_reader = readers[params['readers']['target']['type']]
            target_reader_params = target_reader.assemble({
                **params['readers']['target']['params'],
                'use_cuda': params['use_cuda'],
            })

            target_reader = target_reader(**{
                **target_reader_params,
                'format_batch': cls.format_batch
            })

        except KeyError as error:
            raise KeyError('%s is not a valid identifier for target reader.' % error)

        return {
            'source_reader': source_reader,
            'target_reader': target_reader,
        }

    @classmethod
    def abstract(cls):
        return False
