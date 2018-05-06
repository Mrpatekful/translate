"""

"""

__author__ = "Patrik Purgai"
__copyright__ = "Copyright 2018, Patrik Purgai"
__date__ = "23 Apr 2018"
__version__ = "0.1"


import numpy

import logging

import torch
import torch.autograd

import copy

import tqdm

from src.components.utils.utils import Classifier, Layer

from src.models.models import Model

from src.modules.modules import AutoEncoder, Translator, Discriminator, WordTranslator, NoiseModel

from src.utils.analysis import DataLog, TextData, ScalarData

from src.utils.reader import Language

from src.utils.utils import Component, ModelWrapper, Policy, call, format_outputs, sentence_from_ids, UNMTPolicy, \
    Interface


class Experiment(Component):
    """
    Abstract base class for the experiments.
    """

    def train(self, epoch):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError

    @state.setter
    def state(self, value):
        raise NotImplementedError


class UnsupervisedTranslation(Experiment):
    """
    Translation experiment, without parallel corpus. The method follows the main principles described
    in this article:

        https://arxiv.org/abs/1711.00043

    The main goal of this experiment is to train a denoising auto-encoder, that learns to map sentences to
    sentences in two ways. The first way is to transform a noisy version of the source sentence to it's
    original form, and the second way is to transform a translated version of a sentence to it's original form.
    There is an additional factor during training, which is an adversarial reguralization, that learns to
    discriminate the hidden representations of the source and target languages.
    """

    interface = Interface(**{
        'policy':               (0, Policy),
        'language_identifiers': (1, None),
        'languages':            (2, Language),
        'model':                (3, Model),
        'initial_translator':   (4, WordTranslator),
        'reguralizer':          (5, Classifier)
    })

    abstract = False

    @staticmethod
    def clear_optimizers(optimizers: list):
        """
        Convenience function for the execution of the clear function on the provided optimizer.
        Clear will reset the gradients of the optimized parameters.

        Args:
            optimizers: A list, containing the optimizer type objects.
        """
        call('clear', optimizers)

    @staticmethod
    def step_optimizers(optimizers: list):
        """
        Convenience function for the execution of the step function on the provided optimizer.
        Step will modify the values of the required parameters.

        Args:
            optimizers: A list, containing the optimizer type objects.
        """
        call('step', optimizers)

    @staticmethod
    def freeze(task_components: list):
        """
        Convenience function for the freezing the given components of the task. The frozen
        components won't receive any updates by the optimizers.

        Args:
            task_components: A list, containing Modules.
        """
        call('freeze', task_components)

    @staticmethod
    def unfreeze(task_components: list):
        """
        Convenience function for unfreezing the weights of the provided components. The
        optimizers will be able to modify the weights of these components.

        Args:
            task_components: A list, containing Modules.
        """
        call('unfreeze', task_components)

    def __init__(self,
                 model:                Model,
                 policy:               UNMTPolicy,
                 language_identifiers: list,
                 languages:            list,
                 initial_translator:   WordTranslator,
                 reguralizer:          Classifier = None):
        """
        Initialization of an unsupervised translation task. The embedding and output layers for the model
        are created in this function as well. These modules have to be changeable during training,
        so their references are kept in a list, where each index corresponds to the index of language.

        Args:
            model:
                A Model type instance, that will be used during the experiment.

            languages:

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

            reguralizer:
                Reguralization, that will be used as an adversarial reguralizer during training. Default
                value is None, meaning there won't be any reguralization used during training.

        Raises:
            ValueError:
                If the corpora was not created with a Monolingual type Corpora object an
                exception is raised.
        """
        def initialize_embeddings() -> list:
            """
            Initializer function for the embeddings of different languages. Each language uses a different
            embedding layer, which have to be switched during training and evaluation.
            """
            nonlocal languages

            embeddings = []
            for language in languages:
                embeddings.append(language.vocabulary.embedding)

            return embeddings

        def initialize_loss_functions() -> list:
            """
            Initializer function for the loss functions of different languages. Each loss is a negative loss
            likelihood function. The difference is the padding value, that differs for the languages.
            """
            nonlocal languages

            loss_functions = []
            for language in languages:
                loss_functions.append(torch.nn.NLLLoss(
                    ignore_index=language.vocabulary.tokens['<PAD>'],
                    reduce=False))

            return loss_functions

        def initialize_output_layers() -> list:
            """
            Initializer function for the output layers of different languages. Each language uses a different
            output layer, which have to be switched during training and evaluation.
            """
            nonlocal languages
            nonlocal self

            output_layers = []
            for language in languages:
                output_layers.append(Layer(
                    input_size=self._model.output_size,
                    output_size=language.vocabulary.vocab_size,
                    use_cuda=self._policy.cuda))

            return output_layers

        def initialize_tokens() -> list:
            """
            Initializer function for the tokens of the different languages. These tokens are the <EOS>, <SOS>
            and <UNK> tokens. The returned 'tokens' list contains their ID representation, that is retrieved
            from the vocabulary of the corresponding language.
            """
            nonlocal languages

            tokens = []
            for language in languages:
                tokens.append(language.vocabulary.tokens)

            return tokens

        def initialize_input_pipelines() -> tuple:
            """
            Initializer function for the tokens of the different languages. These tokens are the <EOS>, <SOS>
            and <UNK> tokens. The returned 'tokens' list contains their ID representation, that is retrieved
            from the vocabulary of the corresponding language.
            """
            nonlocal languages

            train_pipelines = []
            validation_pipelines = []
            test_pipelines = []
            for language in languages:
                train_pipelines.append(language.input_pipelines['train'])
                validation_pipelines.append(language.input_pipelines['dev'])
                test_pipelines.append(language.input_pipelines['test'])

            assert all(list(map(lambda x: x.batch_size == train_pipelines[0].batch_size, train_pipelines))), \
                'Invalid batch size'

            return train_pipelines, validation_pipelines, test_pipelines

        self._policy = policy
        self._model = model
        self._reguralizer = reguralizer

        self.reguralize = False

        self._noise_model = NoiseModel(use_cuda=self._policy.cuda)

        self._language_identifiers = language_identifiers
        self._add_language_token = self._policy.add_language_token
        self._vocabularies = [l.vocabulary for l in languages]

        self._initial_translator = initial_translator
        self._initial_translator.vocabs = self._vocabularies
        self._initial_translator.cuda = self._policy.cuda
        self._initial_translator.language_tokens_required = self._add_language_token

        self._previous_translator = self._initial_translator

        # Initialization of the parameters, which will be different for each language used in the experiment.

        self._tokens = initialize_tokens()
        self._embeddings = initialize_embeddings()
        self._loss_functions = initialize_loss_functions()
        self._output_layers = initialize_output_layers()

        self._train_input, self._dev_input, self._test_input = initialize_input_pipelines()

        self._discriminator_loss_function = torch.nn.CrossEntropyLoss(reduce=False)

        # Initialization of the model wrapper object, that will be used by the modules, defined below.
        # The modules do not have full control over the model, so they use this interface, to set the
        # correct look up tables for the given input.

        self._model_wrapper = ModelWrapper(self._model, self._tokens)
        self._model_wrapper.init_table({
            'encoder_inputs':   self._embeddings,
            'decoder_inputs':   self._embeddings,
            'decoder_outputs':  self._output_layers
        })

        self._num_languages = len(languages)

        # Initialization of the modules, which will be used during the experiment. These objects (modules)
        # are at a higher abstraction level than the model, their responsibility is to iterate the given
        # batch through the model, with the correct configuration of the model look up tables.

        self._auto_encoder = AutoEncoder(
            # --OPTIONAL PARAMS--
            cuda=self._policy.cuda,
            noise_model=self._noise_model,
            add_language_token=self._add_language_token,
            language_identifiers=self._language_identifiers,

            # --REQUIRED PARAMS--
            model=self._model_wrapper,
            tokens=self._tokens,
            loss_functions=self._loss_functions,
            vocabularies=self._vocabularies
        )

        self._translator = Translator(
            # --OPTIONAL PARAMS--
            cuda=self._policy.cuda,
            add_language_token=self._add_language_token,
            language_identifiers=self._language_identifiers,

            # --REQUIRED PARAMS--
            model=self._model_wrapper,
            tokens=self._tokens,
            loss_functions=self._loss_functions,
            vocabularies=self._vocabularies
        )

        self._discriminator = Discriminator(
            # --OPTIONAL PARAMS--
            cuda=self._policy.cuda,

            # --REQUIRED PARAMS--
            model=self._reguralizer,
            loss_function=self._discriminator_loss_function,
        )

        self._iteration = 0
        self._batch_size = self._train_input[0].batch_size
        self._total_length = min(list(map(lambda x: x.total_length, self._train_input)))

        # Convenience attributes, that will help freezing and unfreezing the parameters of the model
        # or the discriminator during specific phases of the training or evaluation.

        self._auto_encoder_outputs = dict(
            zip(list(map(lambda x: f'auto_encoding_{str(x)}', self._model.output_types.keys())),
                self._model.output_types.values())
        )

        self._translator_outputs = dict(
            zip(list(map(lambda x: f'translation_{str(x)}', self._model.output_types.keys())),
                self._model.output_types.values())
        )

        self._model_optimizers = [
            *self._model.optimizers,
            *[embedding.optimizer for embedding in self._embeddings],
            *[layer.optimizer for layer in self._output_layers]
        ]

        self._model_components = [
            self._model,
            *self._embeddings,
            *self._output_layers
        ]

    def _format_auto_encoder_batch(self, batch: dict) -> dict:
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
            'inputs':           torch.from_numpy(batch[:, 1: -2]),
            'targets':          torch.from_numpy(batch[:, : -1]),
            'input_lengths':    batch[:, -1]
        }

        if self._add_language_token:
            formatted_batch['input_lengths'] = formatted_batch['input_lengths'] - 1
        else:
            formatted_batch['input_lengths'] = formatted_batch['input_lengths'] - 2

        if self._policy.cuda:
            formatted_batch['targets'] = formatted_batch['targets'].cuda()

        formatted_batch['targets'] = torch.autograd.Variable(formatted_batch['targets'])

        return formatted_batch

    def train(self, epoch: int):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def _train_discriminator(self, batches, logs):
        """


        Args:
            batches:

            logs:

        """
        discriminator_loss = 0

        discriminator_inputs = self._create_discriminator_inputs(batches)

        for batch in discriminator_inputs:
            inputs = self._numpy_to_variable(batch[:, 0])
            loss = self._discriminator(inputs=inputs, targets=batch[:, 1])

            discriminator_loss += loss

        discriminator_loss.backward()

        discriminator_loss /= len(discriminator_inputs)

        logs.add(DataLog.TRAIN_DATA_ID, 'discriminator_loss', discriminator_loss.data)

    def _eval_discriminator(self, batches, logs, identifier):
        """


        Args:
            batches:

            logs:

            identifier:

        """
        discriminator_loss = 0

        discriminator_inputs = self._create_discriminator_inputs(batches)

        for index, batch in enumerate(discriminator_inputs):
            inputs = self._numpy_to_variable(batch[:, 0])
            loss = self._discriminator(inputs=inputs, targets=batch[:, 1])

            discriminator_loss += loss

        discriminator_loss /= len(discriminator_inputs)

        logs.add(identifier, 'discriminator_loss', discriminator_loss.data)

    def _train_auto_encoder(self, batches, logs, forced_targets=True):
        """
        Implementation of a step of auto-encoding. The look up tables of the model are fitted to the
        provided inputs, and the <LNG> are substituted with the appropriate token. In this case the token
        is the source language token. The inputs are then transformed by a noise function, and then fed
        through the model. If reguralization is applied, the encoder outputs are fetched from the output
        of the model, which is used by the discriminator to apply an adversarial reguralization on these
        outputs.

        Args:
            batches:
                A list, containing the batches from the input pipelines.

        Returns:
            loss:
                A scalar loss value, indicating the average loss of the auto encoder.

            outputs:
                A dictionary, that contains the outputs of the model. The types (keys) contained
                by this dictionary depends on the model specifications.
        """
        auto_encoding_loss = 0
        reguralization_loss = 0

        for language_index, batch in enumerate(batches):
            loss, outputs, _ = self._auto_encoder(batch=batch,
                                                  lang_index=language_index,
                                                  forced_targets=forced_targets,
                                                  denoising=self._policy.train_noise)

            auto_encoding_loss += loss

            logs[language_index].add(DataLog.TRAIN_DATA_ID, 'auto_encoding_loss', loss.data)

            if self._reguralizer is not None and self.reguralize:
                for _language_index in range(self._num_languages):
                    if _language_index != language_index:
                        reguralization_loss += self._reguralize(outputs['encoder_outputs'], _language_index)

        return auto_encoding_loss, reguralization_loss

    def _validate_auto_encoder(self, batches, logs, identifier, forced_targets=True):
        """
        Implementation of a step of auto-encoding. The look up tables of the model are fitted to the
        provided inputs, and the <LNG> are substituted with the appropriate token. In this case the token
        is the source language token. The inputs are then transformed by a noise function, and then fed
        through the model. If reguralization is applied, the encoder outputs are fetched from the output
        of the model, which is used by the discriminator to apply an adversarial reguralization on these
        outputs.

        Args:
            batches:
                A list, containing the batches from the input pipelines.

        Returns:
            loss:
                A scalar loss value, indicating the average loss of the auto encoder.

            outputs:
                A dictionary, that contains the outputs of the model. The types (keys) contained
                by this dictionary depends on the model specifications.
        """
        auto_encoding_loss = 0
        reguralization_loss = 0

        for language_index, batch in enumerate(batches):
            loss, outputs, inputs = self._auto_encoder(batch=batch,
                                                       lang_index=language_index,
                                                       forced_targets=forced_targets,
                                                       denoising=self._policy.validation_noise)

            vocabulary = self._vocabularies[language_index]

            outputs['input_text'] = sentence_from_ids(vocabulary=vocabulary, ids=inputs)
            outputs['output_text'] = sentence_from_ids(vocabulary=vocabulary, ids=outputs['symbols'][0])

            auto_encoding_loss += loss

            logs[language_index].add(identifier, 'auto_encoding_loss', loss.data)

            logs[language_index].add(identifier, 'auto_encoding_text', {
                'input_text': outputs['input_text'],
                'target_text': sentence_from_ids(vocabulary=vocabulary, ids=batches[language_index]['targets']
                                                 .data.cpu().squeeze(0)[1:].numpy()),
                'output_text': outputs['output_text']
            })

            for key in self._auto_encoder_outputs.keys():
                logs[language_index].add(identifier, key,
                                         {key: outputs[key] for key in logs[language_index].get_required_keys(key)})

            if self._reguralizer is not None and self.reguralize:
                for _language_index in range(self._num_languages):
                    if _language_index != language_index:
                        reguralization_loss += self._reguralize(outputs['encoder_outputs'], _language_index)

        return auto_encoding_loss, reguralization_loss

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
        targets = numpy.array([lang_index]*encoder_outputs.size(0))
        loss = self._discriminator(inputs=encoder_outputs[:, -1, :], targets=targets)

        return loss

    def _create_discriminator_inputs(self, batches):
        """

        """
        batch_size = batches[0]['inputs'].size(0)

        concat_input = self._create_encoder_output(batches[0], lang_index=0)

        for index in range(1, self._num_languages):
            concat_input = [
                *concat_input,
                *self._create_encoder_output(batches[index], lang_index=index)
            ]

        concat_input = numpy.array(concat_input)

        numpy.random.shuffle(concat_input)

        return numpy.array([concat_input[index * batch_size:index * batch_size + batch_size]
                            for index in range(len(batches))])

    def _create_encoder_output(self, batch, lang_index):
        """

        """
        self._model_wrapper.set_lookup({'source': lang_index})

        if self._language_identifiers is not None and self._add_language_token:
            inputs = self._add_random_language_token(batch['inputs'], lang_index)
        else:
            inputs = batch['inputs']

        if self._policy.cuda:
            inputs = inputs.cuda()

        inputs = torch.autograd.Variable(inputs)

        outputs = self._model.encoder(
            inputs=inputs,
            lengths=batch['input_lengths'])['encoder_outputs'].data.cpu().numpy()

        return [(outputs[index, -1, :], lang_index) for index in range(len(outputs))]

    def _add_random_language_token(self, batch, lang_index):
        """

        """
        lang_tokens = numpy.random.uniform(0, self._num_languages, size=(batch.size(0)))

        tokens = torch.from_numpy(numpy.array([
            self._vocabularies[lang_index](self._language_identifiers[int(lang_tokens[token])])
            for token in range(len(lang_tokens))
        ])).view(-1, 1)

        return torch.cat((tokens, batch), dim=1)

    def _numpy_to_variable(self, inputs):
        """

        """
        shaped_inputs = numpy.zeros((inputs.shape[0], inputs[0].shape[0]))
        for index in range(shaped_inputs.shape[0]):
            shaped_inputs[index, :] = inputs[index]

        shaped_inputs = torch.from_numpy(shaped_inputs).float()

        if self._policy.cuda:
            shaped_inputs = shaped_inputs.cuda()

        shaped_inputs = torch.autograd.Variable(shaped_inputs)

        return shaped_inputs

    @property
    def state(self):
        raise NotImplementedError

    @state.setter
    def state(self, value):
        raise NotImplementedError


class DividedCurriculumTranslation(UnsupervisedTranslation):  # TODO
    """

    """

    interface = UnsupervisedTranslation.interface

    abstract = False

    def __init__(self,
                 policy:                UNMTPolicy,
                 model:                 Model,
                 language_identifiers:  list,
                 languages:             list,
                 initial_translator:    WordTranslator,
                 reguralizer:           Classifier = None):
        """


        Args:
            policy:

            model:

            language_identifiers:

            languages:

            initial_translator:

            reguralizer:

        """
        super().__init__(model=model,
                         policy=policy,
                         language_identifiers=language_identifiers,
                         languages=languages,
                         initial_translator=initial_translator,
                         reguralizer=reguralizer)

        def initialize_input_pipelines() -> tuple:
            """
            Initializer function for the tokens of the different languages. These tokens are the <EOS>, <SOS>
            and <UNK> tokens. The returned 'tokens' list contains their ID representation, that is retrieved
            from the vocabulary of the corresponding language.
            """
            nonlocal languages

            translated_train_pipelines = []
            translated_dev_pipelines = []
            for language in languages:
                translated_train_pipelines.append(language.input_pipelines['translated_train'])
                translated_dev_pipelines.append(language.input_pipelines['translated_dev'])

            return translated_train_pipelines, translated_dev_pipelines

        assert 'translated' in languages[0].input_pipelines, 'InputPipeline dictionary of the ' \
                                                             'languages must contain \'translated\' key'

        self._translated_train_input, self._translated_dev_input = initialize_input_pipelines()

    def _format_translator_batch(self, batch: dict) -> dict:
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
            'inputs':           torch.from_numpy(batch[:, 1: -2]),
            'targets':          torch.from_numpy(batch[:, : -1]),
            'input_lengths':    batch[:, -1] - 1
        }

        if self._policy.cuda:
            formatted_batch['targets'] = formatted_batch['targets'].cuda()

        formatted_batch['targets'] = torch.autograd.Variable(formatted_batch['targets'])

        return formatted_batch

    def train(self, epoch: int) -> dict:
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
        language_logs = [DataLog({
            'translation_loss':     ScalarData,
            'auto_encoding_loss':   ScalarData,
        }) for _ in range(self._num_languages)]

        mutual_logs = DataLog({
            'total_loss':           ScalarData,
            'discriminator_loss':   ScalarData,
            'reguralization_loss':  ScalarData
        })

        self.reguralize = (epoch + 1) % 2 == 0

        with tqdm.tqdm() as p_bar:

            p_bar.set_description('Translating corpora')

            for batches in zip(*list(map(lambda x: x.batch_generator(), self._train_input))):

                p_bar.update()


        with tqdm.tqdm() as p_bar:

            p_bar.set_description(f'Processing epoch {epoch}')

            for batches in zip(*list(map(lambda x: x.batch_generator(),
                                         [*self._train_input, *self._translated_train_input]))):

                p_bar.update()

                # Batches are generated from the InputPipeline object. In this experiment each language
                # has its own pipeline, with its vocabulary. The number of languages, however, may differ.
                # The generated 'batches' object contains the input, target, and length data for the model.

                auto_encoder_batches = list(map(self._format_auto_encoder_batch, batches[:len(batches) // 2]))

                translator_batches = list(map(self._format_translator_batch, batches[:len(batches) // 2]))

                iteration_loss = 0
                total_reguralization_loss = 0

                # Discriminator training or reguralization is not used by default, only if it has been explicitly
                # defined for the experiment.

                if self._reguralizer is not None and not self.reguralize:

                    self.freeze(self._model_components)
                    self.unfreeze([self._reguralizer])

                    self.clear_optimizers([self._reguralizer.optimizer])

                    self._train_discriminator(logs=mutual_logs, batches=auto_encoder_batches)

                    self.step_optimizers([self._reguralizer.optimizer])

                    self.unfreeze(self._model_components)

                if self._reguralizer is not None:
                    self.freeze([self._reguralizer])

                self.clear_optimizers(self._model_optimizers)

                # Choosing the mode of decoding for the iteration. During predictive decoding (when teacher
                # forcing is not used), the embeddings of the model must be set to frozen state.

                forced_targets = numpy.random.random() < self._policy.train_tf_ratio

                if not forced_targets:
                    self.freeze(self._embeddings)

                auto_encoding_loss, reguralization_loss = self._train_auto_encoder(logs=language_logs,
                                                                                   batches=batches,
                                                                                   forced_targets=forced_targets)

                iteration_loss += auto_encoding_loss
                iteration_loss += reguralization_loss

                if self._reguralizer is not None and self.reguralize:
                    total_reguralization_loss += reguralization_loss.data


                translation_loss, reguralization_loss = self._train_translator(logs=language_logs,
                                                                               batches=batches,
                                                                               forced_targets=forced_targets)

                iteration_loss += translation_loss
                iteration_loss += reguralization_loss

                if self._reguralizer is not None and self.reguralize:
                    total_reguralization_loss += reguralization_loss.data

                mutual_logs.add(DataLog.TRAIN_DATA_ID, 'total_loss', iteration_loss.data)
                mutual_logs.add(DataLog.TRAIN_DATA_ID, 'reguralization_loss', total_reguralization_loss)

                iteration_loss.backward()

                self.step_optimizers(self._model_optimizers)

                if not forced_targets:
                    self.unfreeze(self._embeddings)

        return {**dict(zip(self._language_identifiers, language_logs)), DataLog.MUTUAL_TOKEN_ID: mutual_logs}

    def validate(self) -> dict:
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
        language_logs = [DataLog({
            'translation_loss':     ScalarData,
            'auto_encoding_loss':   ScalarData,
            'translation_text':     TextData,
            'auto_encoding_text':   TextData,
            **self._auto_encoder_outputs,
            **self._translator_outputs
        }) for _ in range(self._num_languages)]

        mutual_logs = DataLog({
            'total_loss':           ScalarData,
            'discriminator_loss':   ScalarData,
            'reguralization_loss':  ScalarData,
        })

        with tqdm.tqdm() as p_bar:

            p_bar.set_description('Translating corpora')

            for batches in zip(*list(map(lambda x: x.batch_generator(), self._train_input))):

                p_bar.update()


        with tqdm.tqdm() as p_bar:

            p_bar.set_description('Validating')

            for identifier, batches in enumerate(zip(*list(
                    map(lambda x: x.batch_generator(), self._dev_input)))):

                p_bar.update()

                batches = list(map(self._format_auto_encoder_batch, batches))

                iteration_loss = 0
                full_reguralization_loss = 0

                self.freeze(self._model_components)

                if self._reguralizer is not None:
                    self.freeze([self._reguralizer])

                    self._eval_discriminator(logs=mutual_logs,
                                             batches=batches,
                                             identifier=identifier)

                auto_encoding_loss, reguralization_loss = self._validate_auto_encoder(logs=language_logs,
                                                                                      batches=batches,
                                                                                      identifier=identifier)

                iteration_loss += auto_encoding_loss
                iteration_loss += reguralization_loss

                if self._reguralizer is not None and self.reguralize:
                    full_reguralization_loss += reguralization_loss.data

                translation_loss, reguralization_loss = self._validate_translator(logs=language_logs,
                                                                                  batches=batches,
                                                                                  identifier=identifier)

                iteration_loss += auto_encoding_loss
                iteration_loss += reguralization_loss

                mutual_logs.add(identifier, 'total_loss', iteration_loss.data)
                mutual_logs.add(identifier, 'reguralization_loss', full_reguralization_loss)

                self.unfreeze(self._model_components)

                if self._reguralizer is not None and self.reguralize:
                    self.unfreeze([self._reguralizer])

        return {**dict(zip(self._language_identifiers, language_logs)), DataLog.MUTUAL_TOKEN_ID: mutual_logs}

    def test(self):
        pass

    def evaluate(self):
        pass

    def _train_translator(self, batches, logs, forced_targets=True):
        """


        Args:
            batches:

            logs:

            forced_targets:

        Returns:
            total_translation_loss:

            total_reguralization_loss:

        """
        total_translation_loss = 0
        total_reguralization_loss = 0

        translation_loss, reguralization_loss, outputs, _ = self._translate(
            batch=batches[0],
            logs=logs,
            input_lang_index=0,
            target_lang_index=1,
            identifier=DataLog.TRAIN_DATA_ID,
            forced_targets=forced_targets)

        total_translation_loss += translation_loss
        total_reguralization_loss += reguralization_loss

        translation_loss, reguralization_loss, outputs, _ = self._translate(
            batch=batches[1],
            logs=logs,
            input_lang_index=1,
            target_lang_index=0,
            identifier=DataLog.TRAIN_DATA_ID,
            forced_targets=forced_targets)

        total_translation_loss += translation_loss
        total_reguralization_loss += reguralization_loss

        return total_translation_loss, total_reguralization_loss

    def _validate_translator(self,  batches, logs, identifier, forced_targets=False):
        """


        Args:
            batches:

            logs:

            identifier:

            forced_targets:

        Returns:
            translation_loss:

            reguralization_loss:

        """
        total_translation_loss = 0
        total_reguralization_loss = 0

        translation_loss, reguralization_loss, outputs, translated_symbols = self._translate(
            batch=batches[0],
            logs=logs,
            input_lang_index=0,
            target_lang_index=1,
            identifier=identifier,
            forced_targets=forced_targets)

        total_translation_loss += translation_loss
        total_reguralization_loss += reguralization_loss

        source_vocabulary = self._vocabularies[0]
        target_vocabulary = self._vocabularies[1]

        outputs['input_text'] = sentence_from_ids(vocabulary=source_vocabulary, ids=translated_symbols)
        outputs['output_text'] = sentence_from_ids(vocabulary=target_vocabulary, ids=outputs['symbols'][0])

        logs[0].add(identifier, 'translation_text', format_outputs(
                (source_vocabulary, translated_symbols),
                (target_vocabulary, batches[1]['inputs']),
                (target_vocabulary, outputs['symbols'][0])
            )
        )

        logs[0].add(identifier, 'translation_text', {
            'input_text': outputs['input_text'],
            'target_text': sentence_from_ids(vocabulary=target_vocabulary, ids=batches[1]['inputs']
                                             .data.cpu().squeeze(0)[1:].numpy()),
            'output_text': outputs['output_text']
        })

        for key in self._translator_outputs.keys():
            logs[0].add(identifier, key, {key: outputs[key] for key in logs[0].get_required_keys(key)})

        translation_loss, reguralization_loss, outputs, translated_symbols = self._translate(
            batch=batches[1],
            logs=logs,
            input_lang_index=1,
            target_lang_index=0,
            identifier=identifier,
            forced_targets=forced_targets)

        total_translation_loss += translation_loss
        total_reguralization_loss += reguralization_loss

        source_vocabulary = self._vocabularies[1]
        target_vocabulary = self._vocabularies[0]

        outputs['input_text'] = sentence_from_ids(vocabulary=source_vocabulary, ids=translated_symbols)
        outputs['output_text'] = sentence_from_ids(vocabulary=target_vocabulary, ids=outputs['symbols'][0])

        logs[1].add(identifier, 'translation_text', format_outputs(
                (source_vocabulary, translated_symbols),
                (target_vocabulary, batches[0]['inputs']),
                (target_vocabulary, outputs['symbols'][0])
            )
        )

        for key in self._translator_outputs.keys():
            logs[1].add(identifier, key, {key: outputs[key] for key in logs[1].get_required_keys(key)})

        return translation_loss, reguralization_loss

    def _translate(self, batch, logs, input_lang_index, target_lang_index, identifier, forced_targets):
        """


        Args:
            batch:

            logs:

            input_lang_index:

            target_lang_index:

            identifier:

            forced_targets:


        Returns:
             translation_loss:

             reguralization_loss:

             outputs:

             translated_symbols:


        """
        reguralization_loss = 0

        # Loss will only be calculated by the translator, if targets, and targets_lengths are both provided.
        # During this step, the lengths of the targets are not provided, so loss will not be calculated.

        translation_loss, _, outputs, _, _, = self._translator(
            input_lang_index=input_lang_index,
            target_lang_index=target_lang_index,
            batch=batch,
            forced_targets=forced_targets)

        if self._reguralizer is not None and self.reguralize:
            for _language_index in range(self._num_languages):
                if _language_index != target_lang_index:
                    reguralization_loss += self._reguralize(outputs['encoder_outputs'], _language_index)

        if logs is not None:
            logs[input_lang_index].add(identifier, 'translation_loss', translation_loss.data)

        return translation_loss, reguralization_loss, outputs, batch['inputs']

    @property
    def state(self):
        """
        Property for the state of the task.
        """
        return {
            'model':            self._model.state,
            'embeddings':       [embedding.state for embedding in self._embeddings],
            'output_layers':    [layer.state for layer in self._output_layers],
        }

    # noinspection PyMethodOverriding
    @state.setter
    def state(self, state):
        """
        Setter function for the state of the task, and the embeddings.
        """
        self._model.state = state['model']

        for index, embedding_state in enumerate(state['embeddings']):
            self._embeddings[index].state = embedding_state

        for index, layer_state in enumerate(state['output_layers']):
            self._output_layers[index].state = layer_state


class MergedCurriculumTranslation(UnsupervisedTranslation):
    """

    """

    interface = UnsupervisedTranslation.interface

    abstract = False

    def __init__(self,
                 model:                 Model,
                 policy:                UNMTPolicy,
                 language_identifiers:  list,
                 languages:             list,
                 initial_translator:    WordTranslator,
                 reguralizer:           Classifier = None):
        """


        Args:
            model:

            policy:

            language_identifiers:

            languages:

            initial_translator:

            reguralizer:

        """
        super().__init__(model=model,
                         policy=policy,
                         language_identifiers=language_identifiers,
                         languages=languages,
                         initial_translator=initial_translator,
                         reguralizer=reguralizer)

        self._previous_model = copy.deepcopy(self._model)
        self._previous_embeddings = copy.deepcopy(self._embeddings)
        self._previous_output_layers = copy.deepcopy(self._output_layers)

        self._previous_model_wrapper = ModelWrapper(self._previous_model, self._tokens)
        self._previous_model_wrapper.init_table({
            'encoder_inputs':   self._previous_embeddings,
            'decoder_inputs':   self._previous_embeddings,
            'decoder_outputs':  self._previous_output_layers
        })

        self._previous_model_components = [
            self._previous_model,
            *self._previous_embeddings,
            *self._previous_output_layers
        ]

    def train(self, epoch: int) -> dict:
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
        language_logs = [DataLog({
            'translation_loss':     ScalarData,
            'auto_encoding_loss':   ScalarData,
        }) for _ in range(self._num_languages)]

        mutual_logs = DataLog({
            'total_loss':           ScalarData,
            'discriminator_loss':   ScalarData,
            'reguralization_loss':  ScalarData
        })

        self._previous_model.eval()
        self.freeze(self._previous_model_components)

        self.reguralize = True

        with tqdm.tqdm(total=self._total_length) as p_bar:

            p_bar.set_description(f'Processing epoch {epoch}')

            for iteration, batches in enumerate(zip(*list(map(lambda x: x.batch_generator(), self._train_input)))):

                p_bar.update()

                if iteration*self._batch_size < self._iteration:
                    continue
                else:
                    self._iteration = iteration*self._batch_size

                # Batches are generated from the InputPipeline object. In this experiment each language
                # has its own pipeline, with its vocabulary. The number of languages, however, may differ.
                # The generated 'batches' object contains the input, target, and length data for the model.

                batches = list(map(self._format_auto_encoder_batch, batches))

                iteration_loss = 0
                total_reguralization_loss = 0

                # Discriminator training or reguralization is not used by default, only if it has been explicitly
                # defined for the experiment.

                if self._reguralizer is not None:

                    self._model.eval()
                    self._reguralizer.train()

                    self.freeze(self._model_components)
                    self.unfreeze([self._reguralizer])

                    self.clear_optimizers([self._reguralizer.optimizer])

                    self._train_discriminator(logs=mutual_logs, batches=batches)

                    self.step_optimizers([self._reguralizer.optimizer])

                    self.unfreeze(self._model_components)

                if self._reguralizer is not None:
                    self.freeze([self._reguralizer])
                    self._reguralizer.eval()

                self.clear_optimizers(self._model_optimizers)

                # Choosing the mode of decoding for the iteration. During predictive decoding (when teacher
                # forcing is not used), the embeddings of the model must be set to frozen state.

                forced_targets = numpy.random.random() < self._policy.train_tf_ratio

                self._model.train()

                if not forced_targets:
                    self.freeze(self._embeddings)

                auto_encoding_loss, reguralization_loss = self._train_auto_encoder(logs=language_logs,
                                                                                   batches=batches,
                                                                                   forced_targets=forced_targets)

                iteration_loss += auto_encoding_loss
                iteration_loss += reguralization_loss

                del auto_encoding_loss

                if self._reguralizer is not None and self.reguralize:
                    total_reguralization_loss += reguralization_loss.data

                del reguralization_loss

                translation_loss, reguralization_loss = self._train_translator(logs=language_logs,
                                                                               batches=batches,
                                                                               forced_targets=forced_targets)

                iteration_loss += translation_loss
                iteration_loss += reguralization_loss

                del translation_loss

                if self._reguralizer is not None and self.reguralize:
                    total_reguralization_loss += reguralization_loss.data

                del reguralization_loss

                mutual_logs.add(DataLog.TRAIN_DATA_ID, 'total_loss', iteration_loss.data)
                mutual_logs.add(DataLog.TRAIN_DATA_ID, 'reguralization_loss', total_reguralization_loss)

                iteration_loss.backward()

                del iteration_loss
                del total_reguralization_loss

                self.step_optimizers(self._model_optimizers)

                if not forced_targets:
                    self.unfreeze(self._embeddings)

        self._iteration = 0

        return {**dict(zip(self._language_identifiers, language_logs)), DataLog.MUTUAL_TOKEN_ID: mutual_logs}

    def validate(self) -> dict:
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
        language_logs = [DataLog({
            'translation_loss':     ScalarData,
            'auto_encoding_loss':   ScalarData,
            'translation_text':     TextData,
            'auto_encoding_text':   TextData,
            **self._auto_encoder_outputs,
            **self._translator_outputs
        }) for _ in range(self._num_languages)]

        mutual_logs = DataLog({
            'total_loss':           ScalarData,
            'discriminator_loss':   ScalarData,
            'reguralization_loss':  ScalarData,
        })

        self._model.eval()
        self._previous_model.eval()

        if self._reguralizer is not None:
            self._reguralizer.eval()

        self.reguralize = True

        with tqdm.tqdm() as p_bar:

            p_bar.set_description('Validating')

            for identifier, batches in enumerate(zip(*list(map(lambda x: x.batch_generator(), self._dev_input)))):

                p_bar.update()

                batches = list(map(self._format_auto_encoder_batch, batches))

                iteration_loss = 0
                full_reguralization_loss = 0

                self.freeze(self._model_components)

                if self._reguralizer is not None:
                    self.freeze([self._reguralizer])

                    self._eval_discriminator(logs=mutual_logs,
                                             batches=batches,
                                             identifier=identifier)

                auto_encoding_loss, reguralization_loss = self._validate_auto_encoder(logs=language_logs,
                                                                                      batches=batches,
                                                                                      identifier=identifier)

                iteration_loss += auto_encoding_loss
                iteration_loss += reguralization_loss

                if self._reguralizer is not None and self.reguralize:
                    full_reguralization_loss += reguralization_loss.data

                translation_loss, reguralization_loss = self._validate_translator(logs=language_logs,
                                                                                  batches=batches,
                                                                                  identifier=identifier)

                iteration_loss += auto_encoding_loss
                iteration_loss += reguralization_loss

                mutual_logs.add(identifier, 'total_loss', iteration_loss.data)
                mutual_logs.add(identifier, 'reguralization_loss', full_reguralization_loss)

                self.unfreeze(self._model_components)

                if self._reguralizer is not None and self.reguralize:
                    self.unfreeze([self._reguralizer])

        self._previous_model.state = self._model.state

        for index, embedding_state in enumerate(self._embeddings):
            self._previous_embeddings[index].state = embedding_state.state

        for index, layer_state in enumerate(self._output_layers):
            self._previous_output_layers[index].state = layer_state.state

        self._previous_translator = Translator(
            # --OPTIONAL PARAMS--
            cuda=self._policy.cuda,
            language_identifiers=self._language_identifiers,

            # --REQUIRED PARAMS--
            model=self._previous_model_wrapper,
            tokens=self._tokens,
            add_language_token=self._add_language_token,
            loss_functions=self._loss_functions,
            vocabularies=self._vocabularies
        )

        return {**dict(zip(self._language_identifiers, language_logs)), DataLog.MUTUAL_TOKEN_ID: mutual_logs}

    def test(self) -> dict:
        """
        This function evaluates the model. Input data is propagated forward, and then the loss calculated
        based on the same loss function which was used during training. The weights however, are not modified
        in this function.

        Returns:
            logs:
                A list of DataLog type objects, that contain the logging data for the languages. The number of
                data logs equal to the number of languages, and each data log contains information about the
                produced output for the whole data set of a language.

                Additional outputs depend on the chosen model.
        """
        language_logs = [DataLog({
            'translation_loss': ScalarData,
            'translation_text': TextData,
            **self._translator_outputs
        }) for _ in range(self._num_languages)]

        mutual_logs = DataLog({
            'discriminator_loss': ScalarData,
        })

        self._model.eval()
        self._previous_model.eval()
        self._reguralizer.eval()

        self.freeze(self._model_components)
        self.freeze(self._previous_model_components)

        self.reguralize = False

        with tqdm.tqdm() as p_bar:

            p_bar.set_description('Testing')

            for identifier, batches in enumerate(zip(*list(map(lambda x: x.batch_generator(), self._test_input)))):

                p_bar.update()

                batches = list(map(self._format_auto_encoder_batch, batches))

                if self._reguralizer is not None:
                    self.freeze([self._reguralizer])
                    self._eval_discriminator(logs=mutual_logs,
                                             batches=batches,
                                             identifier=identifier)

                self._validate_translator(logs=language_logs,
                                          batches=batches,
                                          identifier=identifier)

        self.reguralize = True

        return {**dict(zip(self._language_identifiers, language_logs)), DataLog.MUTUAL_TOKEN_ID: mutual_logs}

    def evaluate(self) -> dict:
        """
        This function evaluates the model. Input data is propagated forward, and then the loss calculated
        based on the same loss function which was used during training. The weights however, are not modified
        in this function.

        Returns:
            logs:
                A list of DataLog type objects, that contain the logging data for the languages. The number of
                data logs equal to the number of languages, and each data log contains information about the
                produced output for the whole data set of a language.

                Additional outputs depend on the chosen model.
        """
        language_logs = [DataLog({
            'translation_text': TextData,
            **self._translator_outputs
        }) for _ in range(self._num_languages)]

        self._model.eval()
        self._previous_model.eval()

        self.freeze(self._model_components)

        with tqdm.tqdm() as p_bar:

            p_bar.set_description('Inference')

            outputs = []

            for identifier, batch in enumerate(self._test_input[0].batch_generator()):
                p_bar.update()

                batch = self._format_auto_encoder_batch(batch)

                input_text, output_text = self._eval_translator(batch=batch,
                                                                input_lang_index=0,
                                                                target_lang_index=1,
                                                                logs=language_logs,
                                                                identifier=identifier)

                outputs.append((input_text, output_text))

            for identifier, batch in enumerate(self._test_input[1].batch_generator()):
                p_bar.update()

                batch = self._format_auto_encoder_batch(batch)

                input_text, output_text = self._eval_translator(batch=batch,
                                                                input_lang_index=1,
                                                                target_lang_index=0,
                                                                logs=language_logs,
                                                                identifier=identifier)

                outputs.append((input_text, output_text))

            logging.info('\n\n'.join(list(map(lambda x: f'Input: {x[0]}\nOutput: {x[1]}', outputs))))

        return dict(zip(self._language_identifiers, language_logs))

    def _train_translator(self, batches, logs, forced_targets=True):
        """


        Args:
            batches:

            logs:

            forced_targets:

        Returns:
            total_translation_loss:

            total_reguralization_loss:

        """
        total_translation_loss = 0
        total_reguralization_loss = 0

        translation_loss, reguralization_loss, outputs, _ = self._translate(
            batch=batches[0],
            logs=logs,
            input_lang_index=0,
            target_lang_index=1,
            identifier=DataLog.TRAIN_DATA_ID,
            forced_targets=forced_targets)

        total_translation_loss += translation_loss
        total_reguralization_loss += reguralization_loss

        translation_loss, reguralization_loss, outputs, _ = self._translate(
            batch=batches[1],
            logs=logs,
            input_lang_index=1,
            target_lang_index=0,
            identifier=DataLog.TRAIN_DATA_ID,
            forced_targets=forced_targets)

        total_translation_loss += translation_loss
        total_reguralization_loss += reguralization_loss

        return total_translation_loss, total_reguralization_loss

    def _validate_translator(self,  batches, logs, identifier, forced_targets=False):
        """


        Args:
            batches:

            logs:

            identifier:

            forced_targets:

        Returns:
            translation_loss:

            reguralization_loss:

        """
        total_translation_loss = 0
        total_reguralization_loss = 0

        translation_loss, reguralization_loss, outputs, translated_symbols = self._translate(
            batch=batches[0],
            logs=logs,
            input_lang_index=0,
            target_lang_index=1,
            identifier=identifier,
            forced_targets=forced_targets)

        total_translation_loss += translation_loss
        total_reguralization_loss += reguralization_loss

        source_vocabulary = self._vocabularies[1]
        target_vocabulary = self._vocabularies[0]

        outputs['input_text'] = sentence_from_ids(vocabulary=source_vocabulary, ids=translated_symbols)
        outputs['output_text'] = sentence_from_ids(vocabulary=target_vocabulary, ids=outputs['symbols'][0])

        targets = batches[0]['inputs'].cpu().squeeze(0)[1:].numpy()
        targets = sentence_from_ids(vocabulary=target_vocabulary, ids=targets)

        logs[1].add(identifier, 'translation_text', {
            'input_text': outputs['input_text'],
            'target_text': targets,
            'output_text': outputs['output_text']
        })

        for key in self._translator_outputs.keys():
            logs[1].add(identifier, key, {key: outputs[key] for key in logs[1].get_required_keys(key)})

        translation_loss, reguralization_loss, outputs, translated_symbols = self._translate(
            batch=batches[1],
            logs=logs,
            input_lang_index=1,
            target_lang_index=0,
            identifier=identifier,
            forced_targets=forced_targets)

        total_translation_loss += translation_loss
        total_reguralization_loss += reguralization_loss

        source_vocabulary = self._vocabularies[0]
        target_vocabulary = self._vocabularies[1]

        outputs['input_text'] = sentence_from_ids(vocabulary=source_vocabulary, ids=translated_symbols)
        outputs['output_text'] = sentence_from_ids(vocabulary=target_vocabulary, ids=outputs['symbols'][0])

        targets = batches[1]['inputs'].cpu().squeeze(0)[1:].numpy()
        targets = sentence_from_ids(vocabulary=target_vocabulary, ids=targets)

        logs[0].add(identifier, 'translation_text', {
            'input_text': outputs['input_text'],
            'target_text': targets,
            'output_text': outputs['output_text']
        })

        for key in self._translator_outputs.keys():
            logs[0].add(identifier, key, {key: outputs[key] for key in logs[0].get_required_keys(key)})

        return translation_loss, reguralization_loss

    def _eval_translator(self, batch, input_lang_index, target_lang_index, logs, identifier):
        """


        Args:
            batch:

            logs:

            identifier:

        Returns:
            translation_loss:

            reguralization_loss:

        """
        _, translated_symbols, outputs, inputs, _, = self._translator(
            input_lang_index=input_lang_index,
            target_lang_index=target_lang_index,
            batch=batch,
            forced_targets=False)

        source_vocabulary = self._vocabularies[input_lang_index]
        target_vocabulary = self._vocabularies[target_lang_index]

        outputs['input_text'] = sentence_from_ids(vocabulary=source_vocabulary, ids=inputs.data.cpu().squeeze(0))
        outputs['output_text'] = sentence_from_ids(vocabulary=target_vocabulary, ids=translated_symbols.squeeze(0))

        logs[input_lang_index].add(identifier, 'translation_text', {
            'input_text':  outputs['input_text'],
            'output_text': outputs['output_text']
        })

        for key in self._translator_outputs.keys():
            logs[input_lang_index].add(identifier, key, {key: outputs[key] for key in
                                                         logs[input_lang_index].get_required_keys(key)})

        return ' '.join(outputs['input_text']), ' '.join(outputs['output_text'])

    def _translate(self, batch, logs, input_lang_index, target_lang_index, identifier, forced_targets):
        """


        Args:
            batch:

            logs:

            input_lang_index:

            target_lang_index:

            identifier:

            forced_targets:


        Returns:
             translation_loss:

             reguralization_loss:

             outputs:

             translated_symbols:


        """
        reguralization_loss = 0

        # Loss will only be calculated by the translator, if targets, and targets_lengths are both provided.
        # During this step, the lengths of the targets are not provided, so loss will not be calculated.

        _, translated_symbols, _, inputs, translated_lengths = self._previous_translator(
            input_lang_index=input_lang_index,
            target_lang_index=target_lang_index,
            batch=batch,
            forced_targets=False)

        # During 'back translation' loss can be calculated, because the lengths of the targets are known.

        translated_batch = {
            'inputs':           translated_symbols,
            'input_lengths':    translated_lengths,
            'targets':          batch['inputs'],
            'target_lengths':   batch['input_lengths']
        }

        translation_loss, _, outputs, _, _, = self._translator(
            input_lang_index=target_lang_index,
            target_lang_index=input_lang_index,
            batch=translated_batch,
            forced_targets=forced_targets)

        if self._reguralizer is not None and self.reguralize:
            for _language_index in range(self._num_languages):
                if _language_index != target_lang_index:
                    reguralization_loss += self._reguralize(outputs['encoder_outputs'], _language_index)

        logs[target_lang_index].add(identifier, 'translation_loss', translation_loss.data)

        translated_symbols = translated_symbols.squeeze(0).cpu().numpy()

        return translation_loss, reguralization_loss, outputs, translated_symbols

    @property
    def state(self):
        """
        Property for the state of the task.
        """
        return {
            'model':                    self._model.state,
            'iteration':                self._iteration,
            'previous_model':           self._previous_model.state,
            'previous_translator':      type(self._previous_translator),
            'embeddings':               [embedding.state for embedding in self._embeddings],
            'output_layers':            [layer.state for layer in self._output_layers],
            'previous_embeddings':      [embedding.state for embedding in self._previous_embeddings],
            'previous_output_layers':   [layer.state for layer in self._previous_output_layers],
        }

    @state.setter
    def state(self, state):
        """
        Setter function for the state of the task, and the embeddings.
        """
        self._model.state = state['model']

        self._previous_model.state = state['previous_model']

        self._iteration = state['iteration']

        for index, embedding_state in enumerate(state['embeddings']):
            self._embeddings[index].state = embedding_state

        for index, layer_state in enumerate(state['output_layers']):
            self._output_layers[index].state = layer_state

        for index, embedding_state in enumerate(state['previous_embeddings']):
            self._previous_embeddings[index].state = embedding_state

        for index, layer_state in enumerate(state['previous_output_layers']):
            self._previous_output_layers[index].state = layer_state

        if isinstance(state['previous_translator'], WordTranslator):
            self._previous_translator = self._initial_translator
        else:
            self._previous_translator = Translator(
                # --OPTIONAL PARAMS--
                cuda=self._policy.cuda,
                language_identifiers=self._language_identifiers,

                # --REQUIRED PARAMS--
                model=self._previous_model_wrapper,
                tokens=self._tokens,
                add_language_token=self._add_language_token,
                loss_functions=self._loss_functions,
                vocabularies=self._vocabularies
            )
