from utils import utils
from utils import reader
from models import models

from modules.utils import utils
from utils.utils import Component

import numpy
import torch
import torch.autograd

from collections import OrderedDict


class Task(Component):
    """
    Abstract base class for the tasks.
    """

    @staticmethod
    def format_batch(batch, use_cuda):
        return NotImplementedError

    def save_checkpoint(self, state):
        return NotImplementedError

    def load_checkpoint(self):
        return NotImplementedError

    def fit_model(self, *args, **kwargs):
        return NotImplementedError


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
    _checkpoint = 'logs/checkpoints/checkpoint.pt'

    @staticmethod
    def format_batch(batch, use_cuda):
        """
        The special batch format, that is required by the task. This function is passed to the reader,
        and will be used to produce batches and targets, in a way, that is convenient for this particular task.
        :param batch:
        :param use_cuda:
        :return inputs: Variable, the inputs provided to the decoder. The <SOS> and <EOS> tokens are
                        cut from the original input.
        :return targets: Variable, the targets, provided to the decoder. The <ENG> token is removed
                         from the original batch.
        :return lengths: Ndarray, the lengths of the inputs provided to the encoder. These are used
                         for sequence padding.
        """
        inputs = torch.from_numpy(batch[:, 1:-2])
        targets = torch.from_numpy(numpy.hstack((batch[:, 0].reshape(-1, 1), batch[:, 2:-1])))
        lengths = batch[:, -1] - 2

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        return torch.autograd.Variable(inputs), torch.autograd.Variable(targets), lengths

    @classmethod
    def interface(cls):
        return OrderedDict(**{
            'use_cuda': None,
            'readers':  OrderedDict(**{
                'source': reader.Reader,
                'target': reader.Reader
            }),
            'model': models.Model,
            'reguralization': utils.Discriminator
        })

    @classmethod
    def abstract(cls):
        return False

    def __init__(self,
                 source,
                 target,
                 model,
                 use_cuda,
                 reguralization):
        """
        An instance of an unsupervised translation task.
        :param source: Reader, an instance of a reader object, that may be a FastReader or FileReader.
        :param target: Reader, that is the same as the source reader, but for the target language.
        :param model: Model, the class of the model, that will be used for this task.
        """
        self._source_reader = source
        self._target_reader = target

        self._source_reader.batch_format = self.format_batch
        self._target_reader.batch_format = self.format_batch

        self._reguralization = reguralization

        self._use_cuda = use_cuda

        self._model = model

        self._set_embeddings(self._source_reader.source_language, self._source_reader.source_language)

    def fit_model(self, epochs):
        """
        Fits the model to the data. The training session runs until convergence, or
        the given epochs.
        :param epochs: int, the number of maximum epochs.
        """
        loss_function = torch.nn.NLLLoss(ignore_index=0, reduce=False)
        noise_function = utils.Noise()

        for epoch in range(epochs):

            self._set_embeddings(self._source_reader.source_language, self._source_reader.source_language)

            self._train(loss_function, noise_function)

            self._evaluate(noise_function)

        self.save_checkpoint(self._create_checkpoint())

    def _train(self, loss_function, noise_function):
        """

        :param loss_function:
        :param noise_function:
        :return:
        """
        self._source_reader.mode = 'train'
        loss = 0
        steps = 0

        for inputs, targets, lengths in self._source_reader.batch_generator():
            outputs = self._fit_step(inputs=inputs,
                                     targets=targets,
                                     lengths=lengths,
                                     loss_function=loss_function,
                                     noise_function=noise_function)
            steps += 1
            loss += outputs['loss']

        print(loss / steps)

    def _evaluate(self, noise_function):
        """

        :param noise_function:
        :return:
        """
        self._source_reader.mode = 'dev'
        for inputs, targets, lengths in self._source_reader.batch_generator():
            max_length = targets.size(1)
            outputs = self._step(inputs=inputs,
                                 targets=targets,
                                 lengths=lengths,
                                 max_length=max_length,
                                 noise_function=noise_function)

            inputs = inputs.cpu().data[:, :].numpy()
            outputs = outputs['symbols'][:, :]
            targets = targets.cpu().data[:, 1:].numpy()

            self._source_reader.print_validation_format(input=inputs, output=outputs, target=targets)

    def _set_embeddings(self, encoder_language, decoder_language):
        """
        Sets the embeddings for the mode.
        :param encoder_language: Language, encoder's language.
        :param decoder_language: Language, decoder's language.
        """
        self._model.encoder_embedding = {
            'weights': encoder_language.embedding,
            'requires_grad': encoder_language.requires_grad
        }

        self._model.decoder_embedding = {
            'weights': decoder_language.embedding,
            'requires_grad': decoder_language.requires_grad
        }
        self._model.decoder_tokens = decoder_language.tokens

    def _step(self,
              inputs,
              targets,
              lengths,
              max_length,
              noise_function):
        """
        Single forward step of the model. A batch of inputs and targets are provided, from which
        the output is calculated, with the current parameter values of the model.
        :param inputs: Variable, containing the ids of the words.
        :param lengths: Ndarray, containing the lengths of each sentence in the input batch.
        :param noise_function: The noise model, that will be applied to the input sentences. As
                               written in the task description, this could serve as a dropout like
                               mechanism, or a translation model from the previous iteration.
        :return: int, loss at the current time step, produced by this iteration.
        """
        self._model.zero_grad()

        noisy_inputs = noise_function(inputs)

        outputs = self._model.forward(inputs=noisy_inputs, targets=targets, max_length=max_length, lengths=lengths)

        return outputs

    def _fit_step(self,
                  inputs,
                  targets,
                  lengths,
                  loss_function,
                  noise_function):
        """
        A single batch of data is propagated forward the model,
        evaluated, and back-propagated. The parameters are updated by calling the step
        function of the components' optimizers.
        :param inputs: Variable, containing the ids of the words.
        :param lengths: Ndarray, containing the lengths of each sentence in the input batch.
        :param loss_function: Loss function of the model.
        :param noise_function: The noise model, that will be applied to the input sentences. As
                               written in the task description, this could serve as a dropout like
                               mechanism, or a translation model from the previous iteration.
        :return: int, loss at the current time step, produced by this iteration.
        """
        batch_size = targets.size(0)
        max_length = targets.size(1) - 1

        outputs = self._step(inputs=inputs,
                             targets=None,
                             lengths=lengths,
                             max_length=max_length,
                             noise_function=noise_function)

        # ous['loss'] += self._reguralization(ous)
        outputs['loss'] = 0
        for step, step_output in enumerate(outputs['outputs']):
            outputs['loss'] += loss_function(step_output, targets[:, step + 1])

        lengths = torch.from_numpy(lengths).float()

        if self._use_cuda:
            lengths = lengths.cuda()

        outputs['loss'] = outputs['loss'] / torch.autograd.Variable(lengths)
        outputs['loss'] = outputs['loss'].sum() / batch_size
        outputs['loss'].backward()

        self._model.step()

        return outputs

    def _create_checkpoint(self):
        return {
            'model': self._model.state_dict(),
            'optimizers': self._model.get_optimizer_states()
        }

    def save_checkpoint(self, checkpoint):
        torch.save(checkpoint, self._checkpoint)

    def load_checkpoint(self):
        checkpoint = torch.load(self._checkpoint)
        self._model.load_state_dict(checkpoint['model'])
        self._model.set_optimizer_states(**checkpoint['optimizers'])


class SupervisedTranslation(Task):

    @classmethod
    def interface(cls):
        return OrderedDict(**{
            'use_cuda': None,
            'reader': reader.Reader,
            'model': models.Model
        })

    def __init__(self):
        pass
