from utils.reader import InputPipeline
from utils.reader import Monolingual
from models.models import Model

from modules.utils.utils import Noise
from modules.utils.utils import Discriminator
from modules.utils.utils import Layer

from utils.utils import Component
from utils.utils import call
from utils.utils import format_outputs
from utils.utils import ModelManager

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

    def inference(self):
        return NotImplementedError

    @property
    def input_pipelines(self):
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
        'use_cuda':         None,
        'tokens':           None,
        'input_pipelines':  InputPipeline,
        'model':            Model,
        'reguralization':   Discriminator
    })

    abstract = False

    @staticmethod
    def format_batch(batch, use_cuda):
        """
        The special batch format, that is required by the task. This function is passed to the input_pipeline,
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
        formatted_batch = {
            'inputs':   torch.from_numpy(batch[:, 1: -2]),
            'targets':  torch.from_numpy(numpy.hstack((batch[:, 0].reshape(-1, 1), batch[:, 2: -1]))),
            'lengths':  batch[:, -1] - 2
        }

        if use_cuda:
            formatted_batch['inputs'] = formatted_batch['inputs'].cuda()
            formatted_batch['targets'] = formatted_batch['targets'].cuda()

        formatted_batch['inputs'] = torch.autograd.Variable(formatted_batch['inputs'])
        formatted_batch['targets'] = torch.autograd.Variable(formatted_batch['targets'])

        return formatted_batch

    def __repr__(self):
        return

    def __init__(self,
                 model,
                 tokens,
                 use_cuda,
                 input_pipelines,
                 reguralization=None):
        """
        An instance of an unsupervised translation task.
        :param model: Model, that will be used to solve the task.
        :param use_cuda: bool, true if cuda support is enabled.
        :param input_pipelines: list, containing the input_pipelines for the task.
        :param reguralization: Reguralization, that will be used as an adversarial reguralizer during training.
        """
        self._input_pipelines = input_pipelines

        self._model = model
        self._translation_model = None

        self._reguralization = reguralization
        self._use_cuda = use_cuda

        self._embeddings = []
        self._output_layers = []
        self._tokens = []

        self._language_tokens = tokens

        for input_pipeline in input_pipelines:

            if not isinstance(input_pipeline.corpora, Monolingual):
                raise ValueError('Corpora of the input_pipelines must be Monolingual.')

            input_pipeline.batch_format = self.format_batch
            self._embeddings.append(input_pipeline.vocabulary[0].embedding)
            self._output_layers.append(
                Layer(input_size=self._model.output_size,
                      output_size=input_pipeline.vocabulary[0].vocab_size,
                      use_cuda=self._use_cuda))
            self._tokens.append(input_pipeline.vocabulary[0].tokens)

        self._model_manager = ModelManager(self._model)

        self._model_manager.init_table({
            'E_I':  self._embeddings,
            'D_I':  self._embeddings,
            'D_O':  self._output_layers
        })

        self.loss_function = torch.nn.NLLLoss(ignore_index=Vocabulary.PAD, reduce=False)
        self._reguralizer_loss_function = torch.nn.NLLLoss(reduce=False)

        self.noise_function = Noise()

    def train(self):
        """
        Training logic for the unsupervised translation task. The method iterates through
        the training corpora, updates the parameters of the model, based on the generated loss.
        :return: average loss during training, normalized by the sequence length, batch size and
        number of steps in an epoch.
        """
        losses = numpy.zeros(len(self._input_pipelines))
        steps = numpy.zeros(len(self._input_pipelines))

        for input_pipelines in zip(*list(map(lambda x: x.batch_generator(), self._input_pipelines))):

            # auto encoding
            for index in range(len(input_pipelines)):

                self._set_lookup({
                    'E_I':  index,
                    'D_I':  index,
                    'D_O':  index
                })

                input_pipelines[index]['inputs'] = self._substitute_tokens(
                    inputs=input_pipelines[index]['inputs'],
                    token=self._input_pipelines[index].vocabulary[0](self._language_tokens[index]))

                outputs = self._train_step(token_index=index,
                                           inputs=input_pipelines[index]['inputs'],
                                           targets=input_pipelines[index]['targets'],
                                           lengths=input_pipelines[index]['lengths'],
                                           noise_function=self.noise_function)

                steps[index] += 1
                losses[index] += outputs['loss']

            # translation TODO
            # for index in range(len(input_pipelines) - 1):
            #     self._set_lookup({
            #         'E_I':  index,
            #         'D_I':  index + 1,
            #         'D_O':  index + 1
            #     })
            #
            #     outputs = self._train_step(inputs=input_pipelines[index]['inputs'],
            #                                targets=input_pipelines[index]['targets'],
            #                                lengths=input_pipelines[index]['lengths'],
            #                                noise_function=self._translation_model)
            #
            #     steps += 1
            #     loss += outputs['loss']
            #
            # self._set_lookup({
            #     'E_I': -1,
            #     'D_I':  0,
            #     'D_O':  0
            # })
            #
            # outputs = self._train_step(inputs=input_pipelines[-1]['inputs'],
            #                            targets=input_pipelines[-1]['targets'],
            #                            lengths=input_pipelines[-1]['lengths'],
            #                            noise_function=self._translation_model)
            #
            # steps += 1
            # loss += outputs['loss']

        return losses / steps

    def evaluate(self):
        """
        Evaluation of the trained model. The parameters are not updated by this method.
        :return: outputs of the evaluation. The it is a list, that stores the encoder and
        decoder outputs, produced for a given sample.
        """
        outputs = []

        for input_pipelines in zip(*list(map(lambda x: x.batch_generator(), self._input_pipelines))):

            # auto encoding
            for index in range(len(input_pipelines)):

                self._set_lookup({
                    'E_I': index,
                    'D_I': index,
                    'D_O': index
                })

                input_pipelines[index]['inputs'] = self._substitute_tokens(
                    inputs=input_pipelines[index]['inputs'],
                    token=self._input_pipelines[index].vocabulary[0](self._language_tokens[index]))

                output = self._eval_step(token_index=index,
                                         inputs=input_pipelines[index]['inputs'],
                                         targets=input_pipelines[index]['targets'],
                                         lengths=input_pipelines[index]['lengths'],
                                         noise_function=self.noise_function)

                outputs.append({
                    'outputs': output,
                    'symbols': format_outputs(
                        (self._input_pipelines[index].vocabulary[0], input_pipelines[index]['inputs']),
                        (self._input_pipelines[index].vocabulary[0], input_pipelines[index]['targets']),
                        (self._input_pipelines[index].vocabulary[0], output['symbols'][0])
                    )
                })

            # translation TODO
            # for index in range(len(input_pipelines) - 1):
            #     self._set_lookup({
            #         'E_I': index,
            #         'D_I': index + 1,
            #         'D_O': index + 1
            #     })
            #
            #     output = self._step(inputs=input_pipelines[index]['inputs'],
            #                         targets=None,
            #                         lengths=input_pipelines[index]['lengths'],
            #                         max_length=max_length,
            #                         noise_function=self.noise_function)
            #
            #     outputs.append({
            #         'outputs': output,
            #         'symbols': format_outputs(
            #             (self._input_pipelines[index].vocabulary, input_pipelines[index]['inputs']),
            #             (self._input_pipelines[index].vocabulary, input_pipelines[index]['targets']),
            #             (self._input_pipelines[index].vocabulary, output['symbols'][0])
            #         )
            #     })
            #
            # self._set_lookup({
            #     'E_I': -1,
            #     'D_I':  0,
            #     'D_O':  0
            # })
            #
            # output = self._step(inputs=input_pipelines[index]['inputs'],
            #                     targets=None,
            #                     lengths=input_pipelines[index]['lengths'],
            #                     max_length=max_length,
            #                     noise_function=self.noise_function)
            #
            # outputs.append({
            #     'outputs': output,
            #     'symbols': format_outputs(
            #         (self._input_pipelines[index].vocabulary, input_pipelines[index]['inputs']),
            #         (self._input_pipelines[index].vocabulary, input_pipelines[index]['targets']),
            #         (self._input_pipelines[index].vocabulary, output['symbols'][0])
            #     )
            # })

        return outputs

    def inference(self):
        pass

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
                    token_index,
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

        call('clear', self._model.optimizers)

        outputs = self._step(inputs=inputs,
                             targets=targets,
                             lengths=lengths,
                             max_length=max_length,
                             noise_function=noise_function)

        outputs['loss'] = 0

        for step, step_output in enumerate(outputs['outputs']):
            # print(True if numpy.array(numpy.isnan(step_output.data), dtype=bool).any() else False)
            outputs['loss'] += self.loss_function(step_output, targets[:, step + 1])


        lengths = torch.from_numpy(lengths).float()

        if self._use_cuda:
            lengths = lengths.cuda()

        outputs['loss'] = outputs['loss'] / torch.autograd.Variable(lengths)
        outputs['loss'] = outputs['loss'].sum() / batch_size

        if self._reguralization is not None:
            r_loss = self._reguralize(outputs['encoder_outputs'], token_index, batch_size)
            outputs['loss'] += r_loss

        outputs['loss'].backward()

        call('step', self._model.optimizers)

        return outputs

    def _eval_step(self,
                   inputs,
                   targets,
                   lengths,
                   token_index,
                   noise_function):
        """

        :param inputs:
        :param targets:
        :param lengths:
        :param token_index:
        :param noise_function:
        :return:
        """
        max_length = targets.size(1) - 1

        call('clear', self._model.optimizers)

        outputs = self._step(inputs=inputs,
                             targets=targets,
                             lengths=lengths,
                             max_length=max_length,
                             noise_function=noise_function)

        outputs['loss'] = 0

        for step, step_output in enumerate(outputs['outputs']):
            # print(True if numpy.array(numpy.isnan(step_output.data), dtype=bool).any() else False)
            outputs['loss'] += self.loss_function(step_output, targets[:, step + 1])

        lengths = torch.from_numpy(lengths).float()

        if self._use_cuda:
            lengths = lengths.cuda()

        outputs['loss'] = outputs['loss'] / torch.autograd.Variable(lengths)

        if self._reguralization is not None:
            r_loss = self._reguralize(outputs['encoder_outputs'], token_index, 1)
            outputs['loss'] += r_loss

        return outputs

    def _reguralize(self, encoder_outputs, token_index, batch_size):
        """

        :param encoder_outputs:
        :param token_index:
        :return:
        """
        token_indexes = torch.from_numpy(numpy.array([token_index]*batch_size))

        if self._use_cuda:
            token_indexes = token_indexes.cuda()

        token_indexes = torch.autograd.Variable(token_indexes)
        outputs = self._reguralization(encoder_outputs)
        loss = self._reguralizer_loss_function(outputs, token_indexes)

        return loss.sum() / batch_size

    def _set_lookup(self, lookups):
        """
        Sets the lookups (embeddings) for the encoder and decoder.
        :param lookups: dict, that yields the new embeddings for the decoder and encoder.
        """
        self._model_manager.switch_lookups(lookups)
        self._model.decoder_tokens = self._tokens[lookups['D_I']]

    def _substitute_tokens(self, inputs, token):
        """

        :param inputs:
        :param token:
        :return:
        """
        tokens = torch.from_numpy(numpy.array([token] * inputs.size(0)))

        if self._use_cuda:
            tokens = tokens.cuda()

        inputs[:, 0] = tokens

        return inputs

    @property
    def state(self):
        """
        Property for the state of the task.
        :return: dict, containing the state of the model, and the embeddings.
        """
        return {
            'model':            self._model.state,
            'embeddings':       [embedding.state for embedding in self._embeddings],
            'output_layers':    [layer.state for layer in self._output_layers]
        }

    # noinspection PyMethodOverriding
    @state.setter
    def state(self, state):
        """
        Setter function for the state of the task, and the embeddings.
        :param state: dict, state, to be set as the current state.
        """
        self._model.state = state['model']

        for index, embedding_state in enumerate(state['embeddings']):
            self._embeddings[index].state = embedding_state

        for index, layer_state in enumerate(state['output_layers']):
            self._output_layers[index].state = layer_state

    @property
    def input_pipelines(self):
        """
        Property for the input_pipelines of the task. It is used by the session object,
        to manage the state of the input_pipelines, and switch between train, dev and
        evaluation mode.
        :return: list, containing the input_pipelines of the task.
        """
        return self._input_pipelines


class SupervisedTranslation(Task):

    interface = OrderedDict(**{
        'use_cuda':         None,
        'input_pipeline':   InputPipeline,
        'model':            Model
    })

    def __init__(self):
        pass
