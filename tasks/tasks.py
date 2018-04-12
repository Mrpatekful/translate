from utils.reader import InputPipeline
from utils.reader import Monolingual
from models.models import Model

from analysis.analysis import DataLog
from analysis.analysis import ScalarData
from analysis.analysis import TextData

from modules.utils.utils import Noise
from modules.utils.utils import Discriminator
from modules.utils.utils import Layer
from modules.utils.utils import Translator

from utils.utils import Component
from utils.utils import call
from utils.utils import format_outputs
from utils.utils import ModelManager
from utils.utils import Policy

import numpy
import torch
import torch.autograd

import copy

from collections import OrderedDict


class Task(Component):
    """
    Abstract base class for the tasks.
    """
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

    The main goal of this task is to train a denoising auto-encoder, that learns to map sentences to
    sentences in two ways. The first way is to transform a noisy version of the source sentence to it's
    original form, and the second way is to transform a translated version of a sentence to it's original form.
    There is an additional factor during training, which is an adversarial reguralization, that learns to
    discriminate the hidden representations of the source and target languages.
    """

    interface = OrderedDict(**{
        'policy':               Policy,
        'tokens':               None,
        'input_pipelines':      InputPipeline,
        'model':                Model,
        'translation_model':    Translator,
        'reguralization':       Discriminator
    })

    abstract = False

    def __init__(self,
                 model,
                 tokens,
                 policy,
                 input_pipelines,
                 translation_model,
                 reguralization=None):
        """
        Initialization of an unsupervised translation task. The embedding and output layers for the model
        are created in this function as well. These modules have to be changeable during training,
        so their references are kept in a list, where each index corresponds to the index of language.

        Args:
            model:
                A Model type instance, that will be used to solve the task.

            tokens:
                A list of strings, where each value represents the token, that identifies
                a language. The tokens should be provided in the same order, as the corresponding
                input pipelines are provided.

            input_pipelines:
                A list of InputPipeline type objects, which have to be created with a Monolingual
                type Corpora instance.

            policy:
                An UNMTPolicy object, that contains specific information about this particular task.
                The information is divided into three segments, the train, validation and test policy.
                The data contained in the segments are the following.

                    tf_ratio:
                        A float scalar, that determines the rate of teacher forcing during training phase.
                        A value of 0 will prevent teacher forcing, so the model will use predictive decoding.
                        A value of 1 will force the model to use the targets as previous outputs and a value of
                        0.5 will create a 50% chance of using either techniques. Default value is 1.

                    noise:
                        A boolean value signaling the presence of noise in the input data. The characteristics
                        of the noise function is ...

            reguralization:
                Reguralization, that will be used as an adversarial reguralizer during training. Default
                value is None, meaning there won't be any reguralization used during training.

        Raises:
            ValueError: If the corpora was not created with a Monolingual type Corpora object an exception is raised.
        """
        self._input_pipelines = input_pipelines
        self._reguralization = reguralization
        self._language_tokens = tokens
        self._policy = policy
        self._model = model

        self._translation_model = translation_model

        self._loss_functions = []
        self._output_layers = []
        self._embeddings = []
        self._tokens = []

        for input_pipeline in input_pipelines:

            if not isinstance(input_pipeline.corpora, Monolingual):
                raise ValueError('Corpora of the input_pipelines must be Monolingual.')

            self._embeddings.append(input_pipeline.vocabulary[0].embedding)

            self._output_layers.append(
                Layer(input_size=self._model.output_size,
                      output_size=input_pipeline.vocabulary[0].vocab_size,
                      use_cuda=self._policy.cuda))

            self._tokens.append(input_pipeline.vocabulary[0].tokens)

            self._loss_functions.append(torch.nn.NLLLoss(
                ignore_index=input_pipeline.vocabulary[0].tokens['<PAD>'],
                reduce=False)
            )

        self._model_manager = ModelManager(self._model)

        self._model_manager.init_table({
            'E_I':  self._embeddings,
            'D_I':  self._embeddings,
            'D_O':  self._output_layers
        })

        self._num_languages = len(self._input_pipelines)

        self._reguralizer_loss_function = torch.nn.CrossEntropyLoss(reduce=False)

        self._noise_function = Noise(use_cuda=self._policy.cuda)

    def _format_batch(self, batch):
        """
        The special batch format, that is required by the task. This function is passed to the input_pipeline,
        and will be used to produce batches and targets, in a way, that is convenient for this particular task.

        Args:
            batch:
                An unprocessed batch, that contains an <SOS> at the 0. index, <LNG> at 1. index
                and an <EOS> token at the -2. index. The element at the last index is the length of
                the sequence.

        Returns:
            Formatted batch:
                A dictionary, that contains different types of formatted inputs for the model.
                The batch is created from a monolingual corpora, so the only difference between
                the inputs and targets, are the shifting, and the tokens.

                    inputs:
                        A torch Variable, that is the input of the model. The <SOS> and <EOS> tokens are
                        cut from the original input.

                    targets:
                        A torch Variable, which will be the target of the model. The <LNG> token is removed
                        from the original batch.

                    lengths:
                        A NumPy Array, the lengths of the inputs provided to the encoder. These are required
                        by the PaddedSequence PyTorch utility.
        """
        formatted_batch = {
            'inputs':   torch.from_numpy(batch[:, 1: -2]),
            'targets':  torch.from_numpy(batch[:, : -1]),
            'lengths':  batch[:, -1] - 1
        }

        if self._policy.cuda:
            formatted_batch['inputs'] = formatted_batch['inputs']
            formatted_batch['targets'] = formatted_batch['targets'].cuda()

        formatted_batch['targets'] = torch.autograd.Variable(formatted_batch['targets'])

        return formatted_batch

    def train(self):
        """
        A single training iteration/epoch of the task. The method iterates through the training
        corpora once and updates the parameters of the model, based on the generated loss. The iteration
        has 2 main steps, the model and the discriminator training. During the model training, the inputs
        are propagated through an auto encoding, translation, and reguralization phase. The losses are
        calculated after each step, and summed with a specific weight. The weights are tuneable hyper
        parameters.  The sum of the losses are minimized, where auto encoding and translation losses are
        calculated by a negative log likelihood loss, and the reguralization is calculated by a cross
        entropy loss.

        Raises:
            RuntimeError:
                In case of an occurrence of NaN values a runtime exception is raised.

        Returns:
            total_iteration_loss:
                Loss of the model, including the auto encoding, translation and reguralization loss.
                The value is normalized, so this value represents the sum of average loss of a word
                after translation,

            tr_loss:
                Average loss of the translation phase of the model for an iteration. This value is
                a NumPy Array, with a dimension of (num_languages). A value at a given index
                corresponds to the average loss of a word prediction for the language of that index.

            ae_loss:
                Average loss of the auto encoding phase of the model.

            reg_loss:
                Average loss, created by the reguralization term, that contributes to the total model loss.

            dsc_loss:
                Average loss that is created by the discriminator, during its training phase.
        """
        def clear_optimizers(optimizers):
            """
            Convenience function for the execution of the clear function on the provided optimizer.
            Clear will reset the gradients of the optimized parameters.

            Args:
                optimizers: A list, containing the optimizer type objects.
            """
            call('clear', optimizers)

        def step_optimizers(optimizers):
            """
            Convenience function for the execution of the step function on the provided optimizer.
            Step will modify the values of the required parameters.

            Args:
                optimizers: A list, containing the optimizer type objects.
            """
            call('step', optimizers)

        def freeze(task_components):
            """
            Convenience function for the freezing the given components of the task. The frozen
            components won't receive any updates by the optimizers.

            Args:
                task_components: A list, containing Modules.
            """
            call('freeze', task_components)

        def unfreeze(task_components):
            """
            Convenience function for unfreezing the weights of the provided components. The
            optimizers will be able to modify the weights of these components.

            Args:
                task_components: A list, containing Modules.
            """
            call('unfreeze', task_components)

        logs = [DataLog({
            'total_loss':           ScalarData,
            'translation_loss':     ScalarData,
            'auto_encoding_loss':   ScalarData,
            'reguralization_loss':  ScalarData,
            'discriminator_loss':   ScalarData,
        }) for _ in range(self._num_languages)]

        model_optimizers = [
            *self._model.optimizers,
            *[embedding.optimizer for embedding in self._embeddings],
            *[layer.optimizer for layer in self._output_layers]
        ]

        model_components = [
            self._model,
            *self._embeddings,
            *self._output_layers
        ]

        if self._reguralization is not None:
            discriminator_components = [self._reguralization]
            discriminator_optimizers = [self._reguralization.optimizer]

        noise = self._policy.train_noise

        translate = self._num_languages == 2 and False

        for batches in zip(*list(map(lambda x: x.batch_generator(), self._input_pipelines))):

            batches = list(map(lambda x: self._format_batch(x), batches))

            iteration_loss = 0
            total_loss = [0]*self._num_languages
            reguralization_loss = [0]*self._num_languages

            if self._reguralization is not None:

                dsc_loss = 0

                freeze(model_components)
                unfreeze(discriminator_components)

                clear_optimizers(discriminator_optimizers)

                for index in range(self._num_languages):

                    loss = self._discriminate(lang_index=index, batches=batches)

                    dsc_loss += loss
                    logs[index].add(0, 'discriminator_loss', loss.data)

                dsc_loss.backward()

                step_optimizers(discriminator_optimizers)

                unfreeze(model_components)
                freeze(discriminator_components)

            clear_optimizers(model_optimizers)

            forced_targets = numpy.random.random() < self._policy.train_tf_ratio

            if not forced_targets:
                freeze(self._embeddings)

            for index in range(self._num_languages):

                loss, _, _ = self._auto_encode(noise=noise,
                                               lang_index=index,
                                               batches=batches,
                                               forced_targets=forced_targets)

                iteration_loss += loss[0]
                total_loss[index] += loss[0].data

                logs[index].add(0, 'auto_encoding_loss', loss[0].data)

                if self._reguralization is not None:
                    iteration_loss += loss[1]
                    total_loss[index] += loss[1].data
                    reguralization_loss[index] += loss[1].data

            if translate:

                loss, _, _ = self._translate(source_lang_index=0,
                                             target_lang_index=1,
                                             batches=batches,
                                             forced_targets=forced_targets)

                iteration_loss += loss[0]
                total_loss[0] += loss[0].data

                logs[0].add(0, 'translation_loss', loss[0].data)

                if self._reguralization is not None:
                    iteration_loss += loss[1]
                    total_loss[0] += loss[1].data
                    reguralization_loss[0] += loss[1].data

                loss, _, _ = self._translate(source_lang_index=1,
                                             target_lang_index=0,
                                             batches=batches,
                                             forced_targets=forced_targets)

                iteration_loss += loss[0]
                total_loss[1] += loss[0].data

                logs[1].add(0, 'translation_loss', loss[0].data)

                if self._reguralization is not None:
                    iteration_loss += loss[1]
                    total_loss[1] += loss[1].data
                    reguralization_loss[1] += loss[1].data

            for index in range(self._num_languages):
                logs[index].add(0, 'total_loss', total_loss[index])
                logs[index].add(0, 'reguralization_loss', reguralization_loss[index])

            iteration_loss.backward()

            step_optimizers(model_optimizers)

            if not forced_targets:
                unfreeze(self._embeddings)

            if translate:
                self._translation_model = copy.deepcopy(self)

        return logs

    def evaluate(self):
        """
        This function evaluates the model. Input data is propagated forward, and then the loss calculated
        based on the same loss function which was used during training. The weights however, are not modified
        in this function.

        Returns:
            logs:
                A list of DataLog type objects, that contain the logging data for the languages. The number of
                data logs equal to the number of languages, and each data log contains information about the
                produced output for the whole data set of a language.

                    total_loss:
                        The total loss of the iteration, which is the same as the model loss during training.
                        The value contains the loss of translation, auto-encoding and reguralization loss. The
                        individual error of the discriminator is not included.

                    translation_loss:
                        The error, that is produced by the model, when translating a sentence.

                    auto_encoding_loss:
                        The error, that is produced by the model,
                        when restoring (auto-encoding) a sentence.

                    reguralization_loss:
                        The reguralization loss, that is produced by the discriminator.

                    discriminator_loss:
                        The error of the discriminator, which is the loss that is produced, when the
                        discriminator identifies a given latent vector.

                    translation_text:
                        The textual representation of the input, target and output symbols at the
                        translation phase. These texts are produced by the format outputs
                        utility function.

                    auto_encoding_text:
                        The textual representation of the input, target and output symbols at the
                        auto encoding phase. These texts are produced by the format outputs
                        utility function.

                Additional outputs depend on the chosen model.
        """
        def freeze(task_components):
            """
            Convenience function for the freezing the given components of the task. The frozen
            components won't receive any updates by the optimizers.

            Args:
                task_components: A list, containing Modules.
            """
            call('freeze', task_components)

        def unfreeze(task_components):
            """
            Convenience function for unfreezing the weights of the provided components. The
            optimizers will be able to modify the weights of these components.

            Args:
                task_components: A list, containing Modules.
            """
            call('unfreeze', task_components)

        logs = [DataLog({
            'total_loss':           ScalarData,
            'translation_loss':     ScalarData,
            'auto_encoding_loss':   ScalarData,
            'reguralization_loss':  ScalarData,
            'discriminator_loss':   ScalarData,
            'translation_text':     TextData,
            'auto_encoding_text':   TextData,
            **self._model.output_types
        }) for _ in range(self._num_languages)]

        model_components = [
            self._model,
            *self._embeddings,
            *self._output_layers
        ]

        if self._reguralization is not None:
            discriminator_components = [self._reguralization]

        noise = self._policy.validation_noise

        translate = self._num_languages == 2 and False

        for identifier, batches in enumerate(zip(*list(map(lambda x: x.batch_generator(),
                                                           self._input_pipelines)))):

            batches = list(map(lambda x: self._format_batch(x), batches))

            freeze(model_components)

            if self._reguralization is not None:
                freeze(discriminator_components)

                for index in range(self._num_languages):
                    loss = self._discriminate(lang_index=index, batches=batches)

                    logs[index].add(identifier, 'discriminator_loss', loss.data)

            total_loss = [0]*self._num_languages
            reguralization_loss = [0]*self._num_languages

            forced_targets = False

            for index in range(self._num_languages):
                loss, outputs, inputs = self._auto_encode(noise=noise,
                                                          lang_index=index,
                                                          batches=batches,
                                                          forced_targets=forced_targets)

                logs[index].add(identifier, 'auto_encoding_loss', loss[0].data)

                total_loss[index] += loss[0].data

                logs[index].add(identifier, 'auto_encoding_text', format_outputs(
                        (self._input_pipelines[index].vocabulary[0], inputs),
                        (self._input_pipelines[index].vocabulary[0], batches[index]['targets']),
                        (self._input_pipelines[index].vocabulary[0], outputs['symbols'][0])
                    )
                )

                if self._reguralization is not None:
                    total_loss[index] += loss[1].data
                    reguralization_loss[index] += loss[1].data

                for key in outputs:
                    logs[index].add(identifier, key, outputs[key])

            if translate:

                loss, outputs = self._translate(source_lang_index=0,
                                                target_lang_index=1,
                                                forced_targets=forced_targets,
                                                batches=batches)

                logs[0].add(identifier, 'translation_loss', loss[0].data)

                total_loss[0] += loss[0].data

                logs[0].add(identifier, 'translation_text', format_outputs(
                        (self._input_pipelines[0].vocabulary[0], batches[0]['inputs']),
                        (self._input_pipelines[0].vocabulary[0], batches[0]['targets']),
                        (self._input_pipelines[0].vocabulary[0], outputs['symbols'][0])
                    )
                )

                if self._reguralization is not None:
                    total_loss[0] += loss[1].data
                    reguralization_loss[0] += loss[1].data

                for key in outputs:
                    logs[0].add(identifier, key, outputs[key])

                loss, outputs = self._translate(source_lang_index=1,
                                                target_lang_index=0,
                                                forced_targets=forced_targets,
                                                batches=batches)

                logs[1].add(identifier, 'translation_loss', loss[0].data)

                total_loss[1] += loss[0].data

                logs[1].add(identifier, 'translation_text', format_outputs(
                        (self._input_pipelines[1].vocabulary[0], batches[1]['inputs']),
                        (self._input_pipelines[1].vocabulary[0], batches[1]['targets']),
                        (self._input_pipelines[1].vocabulary[0], outputs['symbols'][0])
                    )
                )

                if self._reguralization is not None:
                    total_loss[1] += loss[1].data
                    reguralization_loss[1] += loss[1].data

                for key in outputs:
                    logs[1].add(identifier, key, outputs[key])

            for index in range(self._num_languages):
                logs[index].add(identifier, 'total_loss', total_loss[index])
                logs[index].add(identifier, 'reguralization_loss', reguralization_loss[index])

            unfreeze(model_components)

            if self._reguralization is not None:
                unfreeze(discriminator_components)

        return logs

    def inference(self):
        return None

    def translate(self, inputs, lengths, input_lang_index, target_lang_index):
        """


        Args:
            inputs:

            lengths:

            input_lang_index:

            target_lang_index:

        Returns:
            outputs:

            lengths:

        """

        def substitute_eos_for_padding(symbols, lang_index):
            for index in range(symbols.shape[0]):
                eos = numpy.argwhere(symbols[index] == 0)
                if eos.shape[0] > 0:
                    size = int(symbols.shape[1] - eos[0])
                    if size == 1:
                        symbols[index, -1] = lang_index
                    else:
                        symbols[index, eos[0]:] = [lang_index] * size

            return symbols, lengths

        self._set_lookup({
            'E_I': input_lang_index,
            'D_I': target_lang_index,
            'D_O': target_lang_index
        })

        max_length = inputs.size(1) - 1

        outputs = self._model(inputs=inputs,
                              lengths=lengths,
                              targets=None,
                              max_length=max_length)

        outputs, lengths = substitute_eos_for_padding(
            outputs['symbols'],
            self._tokens[target_lang_index]['<PAD>']
        )

        return outputs, lengths

    def _auto_encode(self, lang_index, batches, noise=True, forced_targets=True):
        """
        Implementation of a step of auto-encoding. The look up tables of the model are fitted to the
        provided inputs, and the <LNG> are substituted with the appropriate token. In this case the token
        is the source language token. The inputs are then transformed by a noise function, and then fed
        through the model. If reguralization is applied, the encoder outputs are fetched from the output
        of the model, which is used by the discriminator to apply an adversarial reguralization on these
        outputs.

        Args:
            lang_index:
                An int value, that represents the index of the language. This value will serve as
                the index of the substitution token for the input batch.

            batches:
                A list, containing the batches from the input pipelines.

        Returns:
            loss:
                A scalar loss value, indicating the average loss of the auto encoder.

            outputs:
                A dictionary, that contains the outputs of the model. The types (keys) contained
                by this dictionary depends on the model specifications.
        """
        self._set_lookup({
            'E_I': lang_index,
            'D_I': lang_index,
            'D_O': lang_index
        })

        if noise:
            inputs, lengths = self._noise_function(batches[lang_index]['inputs'],
                                                   self._tokens[lang_index]['<PAD>'])
        else:
            inputs = batches[lang_index]['inputs']
            lengths = batches[lang_index]['lengths']

        inputs = self._add_lng_token(
            inputs=inputs,
            token=self._input_pipelines[lang_index].vocabulary[0](self._language_tokens[lang_index])
        )

        loss, outputs = self._iterate_model(inputs=inputs,
                                            input_lengths=lengths,
                                            forced_targets=forced_targets,
                                            targets=batches[lang_index]['targets'],
                                            target_index=lang_index,
                                            target_lengths=batches[lang_index]['lengths'])

        reg_loss = 0

        if self._reguralization is not None:
            reg_loss = self._reguralize(outputs['encoder_outputs'], lang_index)

        return [loss, reg_loss], outputs, inputs

    def _translate(self,
                   source_lang_index,
                   target_lang_index,
                   batches,
                   forced_targets=True):
        """
        Implementation of a step of auto-encoding. The look up tables of the model are fitted to the
        provided inputs, and the <LNG> are substituted with the appropriate token. In this case the token
        is the source language token. The inputs are then transformed by a noise function, and then fed
        through the model. If reguralization is applied, the encoder outputs are fetched from the output
        of the model, which is used by the discriminator to apply an adversarial reguralization on these
        outputs.

        Args:
            source_lang_index:
                An int value, that represents the index of the language. This value will serve as
                the index of the substitution token for the input batch.

            target_lang_index:
                An int value, that represents the index of the language. This value will serve as
                the index of the substitution token for the input batch.

            batches:
                A list, containing the batches from the input pipelines.

        Returns:
            loss:
                A scalar loss value, indicating the average loss of the auto encoder.

            outputs:
                A dictionary, that contains the outputs of the model. The types (keys) contained
                by this dictionary depends on the model specifications.
        """
        inputs = self._add_lng_token(
            inputs=batches[source_lang_index]['inputs'],
            token=self._input_pipelines[source_lang_index].vocabulary[0](self._language_tokens[target_lang_index])
        )

        translated_inputs, translated_lengths = self._translation_model.translate(inputs=inputs,
                                                                                  source_lang_index=source_lang_index,
                                                                                  target_lang_index=target_lang_index)

        self._set_lookup({
            'E_I': target_lang_index,
            'D_I': source_lang_index,
            'D_O': source_lang_index
        })

        inputs = self._add_lng_token(
            inputs=translated_inputs,
            token=self._input_pipelines[target_lang_index].vocabulary[0](self._language_tokens[source_lang_index])
        )

        loss, outputs = self._iterate_model(inputs=inputs,
                                            input_lengths=translated_lengths,
                                            forced_targets=forced_targets,
                                            targets=batches[source_lang_index]['targets'],
                                            target_index=source_lang_index,
                                            target_lengths=batches[source_lang_index]['lengths'])

        reg_loss = 0

        if self._reguralization is not None:
            reg_loss = self._reguralize(outputs['encoder_outputs'], source_lang_index)

        return [loss, reg_loss], outputs, translated_inputs

    def _discriminate(self, lang_index, batches):
        """
        This function implements the discrimination mechanism, where the inputs and the targets - which
        are required for the evaluation and training of the discriminator - are created. The inputs are
        fed into the discriminator and evaluated based on the cross entropy loss, that is defined in the
        init function. The targets are either one-hot coded vectors, or their inverse. This depends on
        whether the loss is calculated for the discriminator or model loss.

        Args:
            lang_index:
                An int value, that represents the index of the encoder's input language.

            batches:
                A list, containing the batches from the input pipelines.

        Returns:
            loss:
                A scalar loss value, indicating the average loss of the discriminator for either the
                inverse or normal target vector.
        """
        loss = 0

        for index in range(self._num_languages):

            self._set_lookup({
                'E_I': lang_index,
            })

            inputs = self._add_lng_token(
                inputs=batches[lang_index]['inputs'],
                token=self._input_pipelines[lang_index].vocabulary[0](self._language_tokens[index])
            )

            encoder_outputs = self._model.encoder(inputs, batches[lang_index]['lengths'])['encoder_outputs']

            targets = self._create_targets(inputs.size(0), index)
            loss += self._iterate_discriminator(encoder_outputs, targets)

        return loss / self._num_languages

    def _reguralize(self, encoder_outputs, lang_index):
        """
        This function implements the reguralization mechanism. The inputs are
        fed into the discriminator and evaluated based on the cross entropy loss, that is defined in the
        init function. The targets are either one-hot coded vectors, or their inverse. This depends on
        whether the loss is calculated for the discriminator or model loss.

        Args:
            lang_index:
                An int value, that represents the index of the target language. This value will serve as
                the index of the substitution token for the input batch.

            encoder_outputs:
                PyTorch Variable, containing the outputs of the encoder.

        Returns:
            loss:
                A scalar loss value, indicating the average loss of the discriminator for either the
                inverse or normal target vector.
        """
        targets = self._create_targets(encoder_outputs.size(0), lang_index, True)
        loss = self._iterate_discriminator(encoder_outputs, targets)

        return loss


    def _iterate_model(self,
                       inputs,
                       input_lengths,
                       targets,
                       target_index,
                       target_lengths,
                       forced_targets):
        """
        Performs a single iteration on the model of the task. Inputs are propagated forward, and the
        losses are produced according to the provided targets, by the defined loss function. This method
        can be used during training phase, but not for inference. Back propagation is not done by this method.

        Args:
            inputs:
                A Variable type object, that contains a batch of ids, which will be processed by the model.
                The tensor must have a shape of (batch_size, sequence_length). Each input contain a special
                language token, that indicates the target language of the decoding. In case of different sequence
                lengths, the inputs to this function must already be padded.

            targets:
                A Variable type object, that contains the target values for the corresponding input.

            input_lengths:
                A NumPy Array object, that contains the lengths of each line of the input sequence.
                This parameter is required for the padded sequence object creation.

            target_lengths:

            target_index:

            forced_targets:
                A boolean value, that represents the chance of using teacher forcing. If
                this value is not given, then teacher is not used.

        Returns:
            Loss:
                A scalar float value, which is the normalized loss of the model in this iteration.
                It is computed by the negative log likelihood loss, which defined in the
                init function. The value is normalized, meaning it is the average loss of a predicted word.

            Outputs:
                Dictionary containing multiple output metrics of the model. The type of metrics
                which are present in the dictionary, depend on the model output interface.
        """
        batch_size = targets.size(0)
        max_length = targets.size(1) - 1

        if forced_targets:
            outputs = self._model(inputs=inputs,
                                  lengths=input_lengths,
                                  targets=targets,
                                  max_length=max_length)
        else:
            outputs = self._model(inputs=inputs,
                                  lengths=input_lengths,
                                  targets=None,
                                  max_length=max_length)

        loss = 0

        for step, step_output in enumerate(outputs['outputs']):
            loss += self._loss_functions[target_index](step_output, targets[:, step + 1])

        lengths = torch.from_numpy(target_lengths).float()

        if self._policy.cuda:
            lengths = lengths.cuda()

        loss = loss / torch.autograd.Variable(lengths)
        loss = loss.sum() / batch_size

        return loss, outputs

    def _iterate_discriminator(self, inputs, targets):
        """
        Performs a single iteration step on the discriminator. The provided inputs are processed, and
        the loss is calculated with regards to the targets parameter. The loss returned, is the average
        loss calculate over the batch.

        Args:
            inputs:
                A PyTorch Variable type object, that is the encoder outputs. The first dimension of the
                variable tensor is the batch size, and the second dimension is the hidden size of the
                recurrent layer of the encoder.

            targets:
                A NumPy Array type object, that contains a matrix with dimensions of (batch_size,
                num_languages). The matrix contains one hot coded rows in the case of discriminator
                training, and the inverse of these rows in the case of model training (reguralization).

        Returns:
            loss:
                Scalar float value containing the normalized loss of the discriminator on the provided
                input-target pairs.
        """
        batch_size = inputs.size(0)

        loss = 0

        for target in targets:

            token_indexes = torch.from_numpy(target).long()

            if self._policy.cuda:
                token_indexes = token_indexes.cuda()

            token_indexes = torch.autograd.Variable(token_indexes)
            outputs = self._reguralization(inputs)
            loss += self._reguralizer_loss_function(outputs, token_indexes)

        loss = loss.sum() / (len(targets) * batch_size)

        return loss

    def _create_targets(self, batch_size, target_lang, inverse=False):
        """
        Creates a target tensor for the cross entropy loss of the discriminator. The targets are
        a one-hot coded vector by default, which will be used for loss calculations during discriminator
        training, or its inverse, which will be used for the total loss of the model. The behaviour
        is decided by the 'inverse' parameter of this function.

        Args:
            batch_size:
                An int value, indicating the first dimension of the target tensor. This value should
                be the same as the currently used batch size.

            target_lang:
                An int value, the index of the target language, which will be labeled as 1 during
                discriminator training, and 0 during model training.

            inverse:
                A boolean value, representing the required target tensor. False by default,
                meaning it will produce a one hot coded vector, which is required for discriminator training
                phase.

        Returns:
            target_matrix:
                A NumPy Array type object, with dimensions of (size, lang_num), which is the same as
                (batch_size, num_languages).
        """
        if inverse:
            return [numpy.array([target_index for _ in range(batch_size)])
                    for target_index in range(self._num_languages) if target_index != target_lang]
        else:
            return [numpy.array([target_lang for _ in range(batch_size)])]

    def _set_lookup(self, lookups):
        """
        Sets the lookups (embeddings) for the encoder and decoder.

        Args:
            lookups:
                A dictionary, that yields the new embeddings for the decoder and encoder.
                The dictionary has to contain 3 keys, E_I, D_I, and D_O. The values of the keys
                are ints, which represent the index of the languages.
        """
        self._model_manager.switch_lookups(lookups)

        if 'D_I' in lookups:
            self._model.decoder_tokens = self._tokens[lookups['D_I']]

    def _add_lng_token(self, inputs, token):
        """
        Adds the provided tokens into the inputs. The inputs yield an <LNG> token, which
        will be replaced by the one that is provided in the parameter.

        Args:
            inputs:
                A NumPy Array type object, with dimensions of (batch_size, sequence_length + 1). The elements
                of the matrix are word ids.

            token:
                A language token that represents the targets language of the model. In case of auto-encoding
                this token is the same as the source language, and in case of translation it the token that
                corresponds to the target language of the translation.

        Returns:
            inputs:
                The same NumPy Array that has been provided as parameter, but the <LNG> tokens have been
                replaced by the correct token.
        """
        tokens = torch.from_numpy(numpy.array([token] * inputs.size(0))).view(-1, 1)
        tokenized_inputs = torch.cat((tokens, inputs), 1)

        if self._policy.cuda:
            tokenized_inputs = tokenized_inputs.cuda()

        return torch.autograd.Variable(tokenized_inputs)

    @property
    def state(self):
        """
        Property for the state of the task.
        """
        return {
            'model':                self._model.state,
            'translation_model':    self._translation_model,
            'embeddings':           [embedding.state for embedding in self._embeddings],
            'output_layers':        [layer.state for layer in self._output_layers]
        }

    # noinspection PyMethodOverriding
    @state.setter
    def state(self, state):
        """
        Setter function for the state of the task, and the embeddings.
        """
        self._model.state = state['model']

        self._translation_model = state['translation_model']

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
