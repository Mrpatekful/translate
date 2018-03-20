from utils import utils
from utils import reader
from models import models

from modules.utils import utils
from utils.utils import Component
from utils.utils import execute
from utils.utils import format_outputs

from utils.reader import Vocabulary

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

    def train(self):
        return NotImplementedError

    def evaluate(self):
        return NotImplementedError

    @property
    def readers(self):
        return NotImplementedError

    @property
    def state(self):
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
    which is an adversarial reguralization, that learns to discriminate the hidden representations
    of the source and target languages.
    """

    interface = OrderedDict(**{
        'use_cuda': None,
        'readers': OrderedDict(**{
            'reader_fst': reader.Reader,
            'reader_snd': reader.Reader
        }),
        'model': models.Model,
        'reguralization': utils.Discriminator
    })

    abstract = False

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
        formatted_batch = {'inputs': torch.from_numpy(batch[:, 1:-2]),
                           'targets': torch.from_numpy(numpy.hstack((batch[:, 0].reshape(-1, 1), batch[:, 2:-1]))),
                           'lengths': batch[:, -1] - 2}

        if use_cuda:
            formatted_batch['inputs'] = formatted_batch['inputs'].cuda()
            formatted_batch['targets'] = formatted_batch['targets'].cuda()

        formatted_batch['inputs'] = torch.autograd.Variable(formatted_batch['inputs'])
        formatted_batch['targets'] = torch.autograd.Variable(formatted_batch['targets'])

        return formatted_batch

    def __init__(self,
                 model,
                 use_cuda,
                 reader_fst,
                 reader_snd,
                 reguralization):
        """
        An instance of an unsupervised translation task.
        :param model: Model, that will be used to solve the task.
        :param use_cuda: bool, if cuda support in enabled-
        :param reader_fst: Reader, the reader for the first language of the task.
        :param reader_snd: Reader, the reader for the second language.
        :param reguralization: Reguralization, that will be used as an adversarial reguralizer during training.
        """

        self.reader_fst = reader_fst
        self.reader_snd = reader_snd

        self.reader_fst.batch_format = self.format_batch
        self.reader_snd.batch_format = self.format_batch

        self._reguralization = reguralization
        self._use_cuda = use_cuda

        self._model = model

        self.loss_function = torch.nn.NLLLoss(ignore_index=Vocabulary.PAD, reduce=False)
        self.noise_function = utils.Noise()

    def train(self):
        """
        Training logic for the unsupervised translation task. The method iterates through
        the training corpora, updates the parameters of the model, based on the generated loss.
        :return: average loss during training, normalized by the sequence length, batch size and
        number of steps in an epoch.
        """
        loss = 0
        steps = 0

        for batch_fst, batch_snd in zip(self.reader_fst.batch_generator(), self.reader_snd.batch_generator()):

            self._set_lookup(self.reader_fst)
            outputs = self._train_step(inputs=batch_fst['inputs'],
                                       targets=batch_fst['targets'],
                                       lengths=batch_fst['lengths'],
                                       noise_function=self.noise_function)

            steps += 1
            loss += outputs['loss']

        return loss / steps

    def evaluate(self):
        """
        Evaluation of the trained model. The parameters are not updated by this method.
        :return: outputs of the evaluation. The it is a list, that stores the encoder and
        decoder outputs, produced for a given sample.
        """
        outputs = []

        for batch in self.reader_fst.batch_generator():

            self._set_lookup(self.reader_fst)
            max_length = batch['targets'].size(1)
            output = self._step(inputs=batch['inputs'],
                                targets=batch['targets'],
                                lengths=batch['lengths'],
                                max_length=max_length,
                                noise_function=self.noise_function)

            outputs.append({
                'outputs': output,
                'symbols': format_outputs(
                    (self.reader_fst.source_vocabulary, batch['inputs']),
                    (self.reader_fst.source_vocabulary, batch['targets']),
                    (self.reader_fst.source_vocabulary, output['symbols'][0])
                )
            })

        return outputs

    def _set_lookup(self, reader_instance):
        """
        Sets the lookups (embeddings) for the encoder and decoder.
        :param reader_instance: Reader, that yields the new embeddings for the decoder and encoder.
        """
        self._model.set_embeddings(reader_instance.source_vocabulary.embedding,
                                   reader_instance.target_vocabulary.embedding)

        self._model.decoder_tokens = reader_instance.target_vocabulary.tokens

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
        noisy_inputs = noise_function(inputs)

        outputs = self._model.forward(inputs=noisy_inputs,
                                      targets=targets,
                                      max_length=max_length,
                                      lengths=lengths)

        return outputs

    def _train_step(self,
                    inputs,
                    targets,
                    lengths,
                    noise_function):
        """
        A single batch of data is propagated forward the model,
        evaluated, and back-propagated. The parameters are updated by calling the step
        function of the components' optimizers.
        :param inputs: Variable, containing the ids of the words.
        :param lengths: Ndarray, containing the lengths of each sentence in the input batch.
        :param noise_function: The noise model, that will be applied to the input sentences. As
                               written in the task description, this could serve as a dropout like
                               mechanism, or a translation model from the previous iteration.
        :return: dict, loss at the current time step, produced by this iteration.
        """
        batch_size = targets.size(0)
        max_length = targets.size(1) - 1

        execute('clear', self._model.optimizers)

        outputs = self._step(inputs=inputs,
                             targets=targets,
                             lengths=lengths,
                             max_length=max_length,
                             noise_function=noise_function)

        outputs['loss'] = 0
        for step, step_output in enumerate(outputs['outputs']):
            outputs['loss'] += self.loss_function(step_output, targets[:, step + 1])

        # outputs['loss'] += self._reguralization(outputs['encoder_outputs'])

        lengths = torch.from_numpy(lengths).float()

        if self._use_cuda:
            lengths = lengths.cuda()

        outputs['loss'] = outputs['loss'] / torch.autograd.Variable(lengths)
        outputs['loss'] = outputs['loss'].sum() / batch_size
        outputs['loss'].backward()

        execute('step', self._model.optimizers)

        return outputs

    @property
    def state(self):
        """
        Property for the state of the task.
        :return: dict, containing the state of the model, and the embeddings.
        """
        return {
            'model': self._model.state,
            'embedding_fst': self.reader_fst.source_vocabulary.embedding.state,
            'embedding_snd': self.reader_snd.source_vocabulary.embedding.state
        }

    # noinspection PyMethodOverriding
    @state.setter
    def state(self, state):
        """
        Setter function for the state of the task, and the embeddings.
        :param state: dict, state, to be set as the current state.
        """
        self._model.state = state['model']
        self.reader_fst.source_vocabulary.embedding.state = state['embedding_fst']
        self.reader_snd.source_vocabulary.embedding.state = state['embedding_snd']

    @property
    def readers(self):
        """
        Property for the readers of the task. It is used by the session object,
        to manage the state of the readers, and switch between train, dev and
        evaluation mode.
        :return: list, containing the readers of the task.
        """
        return [self.reader_snd, self.reader_fst]


class SupervisedTranslation(Task):

    interface = OrderedDict(**{
            'use_cuda': None,
            'reader': reader.Reader,
            'model': models.Model
        })

    def __init__(self):
        pass
