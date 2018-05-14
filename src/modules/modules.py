"""

"""

import torch
import numpy
import pickle

from src.components.utils.utils import Classifier

from src.utils.utils import Component, ModelWrapper, Interface


class NoiseModel:

    def __init__(self, use_cuda, p=0.1, k=3):
        self._use_cuda = use_cuda
        self._p = p
        self._k = k

    def __call__(self, inputs, padding):
        return self._drop_out(inputs, padding)

    def _drop_out(self, inputs, padding_value):
        inputs = inputs.cpu().numpy()
        noisy_inputs = numpy.zeros((inputs.shape[0], inputs.shape[1] + 1))
        mask = numpy.array(numpy.random.rand(inputs.shape[0], inputs.shape[1] - 1) > self._p, dtype=numpy.int32)
        noisy_inputs[:, 1:-1] = mask * inputs[:, 1:]
        noisy_inputs[:, 0] = inputs[:, 0]
        for index in range(inputs.shape[0]):
            remaining = noisy_inputs[index][noisy_inputs[index] != 0]
            padding = numpy.array([padding_value]*len(noisy_inputs[index][noisy_inputs[index] == 0]))
            padding[-1] = remaining.shape[0]
            noisy_inputs[index, :] = numpy.concatenate((remaining, padding))

        noisy_inputs = noisy_inputs[numpy.argsort(-noisy_inputs[:, -1])]

        return torch.from_numpy(noisy_inputs[:, :-1]).long(), numpy.array(noisy_inputs[:, -1], dtype=int)


class _STSModule:
    """
    A base class for the sequence-to-sequence type modules.
    """

    def __init__(self,
                 model:                 ModelWrapper,
                 vocabularies:          list,
                 loss_functions:        list,
                 tokens:                list,
                 add_language_token:    bool,
                 cuda:                  bool,
                 language_identifiers:  list):
        """
        An instance of an sts module.

        Arguments:
            model:
                ModelWrapper, that holds a sequence-to-sequence type model.

            vocabularies:
                list, containing the vocabularies for the languages.

            loss_functions:
                list, containing the loss functions for the languages.

            tokens:
                list, containing the <EOS>, <PAD> .. token ids for the languages.

            add_language_token:
                bool, determines, whether the language identifier should be added to the inputs sequence.

            cuda:
                bool, signals the availability of CUDA.

            language_identifiers:
                list, containing the identifiers of the languages.
        """
        self._model = model
        self._vocabularies = vocabularies
        self._loss_functions = loss_functions
        self._tokens = tokens
        self._language_token_required = add_language_token

        self._cuda = cuda
        self._language_identifiers = language_identifiers

    def _iterate_model(self, inputs: dict, targets: dict = None, forced_targets: bool = False):
        """
        Performs a single iteration on the model of the task. Inputs are propagated forward, and the
        losses are produced according to the provided targets, by the defined loss function. This method
        can be used during training phase, but not for inference. Back propagation is not done by this method.

        Args:
            inputs:
                A dict object, that contains a batch of ids, which will be processed by the model.
                The tensor must have a shape of (batch_size, sequence_length). Each input contain a special
                language token, that indicates the target language of the decoding. In case of different sequence
                lengths, the inputs to this function must already be padded.

            targets:
                A dict object, that contains the target values for the corresponding input.

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
        batch_size = inputs['data'].size(0)

        if targets is not None and targets.get('data', None) is not None:
            max_length = targets['data'].size(1) - 1
        else:
            max_length = None

        if forced_targets:
            outputs, predictions = self._model(inputs=inputs['data'],
                                               lengths=inputs['lengths'],
                                               targets=targets['data'],
                                               max_length=max_length)
        else:
            outputs, predictions = self._model(inputs=inputs['data'],
                                               lengths=inputs['lengths'],
                                               targets=None,
                                               max_length=max_length)

        loss = 0

        if targets is not None:
            for step, step_output in enumerate(predictions):
                loss += self._loss_functions[targets['language_index']](step_output, targets['data'][:, step + 1])

            del predictions

            lengths = torch.from_numpy(targets['lengths']).float()

            if self._cuda:
                lengths = lengths.cuda()

            loss = loss / torch.autograd.Variable(lengths)
            loss = loss.sum() / batch_size

        return loss, outputs

    def _add_language_token(self, batch: numpy.ndarray, token: int):
        """
        Adds the provided tokens into the inputs. The inputs yield an <LNG> token, which
        will be replaced by the one that is provided in the parameter.

        Args:
            batch:
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
        tokens = torch.from_numpy(numpy.array([token] * batch.size(0))).view(-1, 1)
        tokenized_batch = torch.cat((tokens, batch), 1)

        if self._cuda:
            tokenized_batch = tokenized_batch.cuda()

        return torch.autograd.Variable(tokenized_batch)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


class AutoEncoder(_STSModule):
    """
    Auto-encoder module for a sequence-to-sequence type model.
    """

    def __init__(self,
                 model:                 ModelWrapper,
                 vocabularies:          list,
                 loss_functions:        list,
                 tokens:                list,
                 cuda:                  bool = False,
                 add_language_token:    bool = False,
                 language_identifiers:  list = None,
                 noise_model:           NoiseModel = None):
        """
        An instance of an auto encoder module.

        Arguments:
            model:
                ModelWrapper, that holds a sequence-to-sequence type model.

            vocabularies:
                list, containing the vocabularies for the languages.

            loss_functions:
                list, containing the loss functions for the languages.

            tokens:
                list, containing the <EOS>, <PAD> .. token ids for the languages.

            add_language_token:
                bool, determines, whether the language identifier should be added to the inputs sequence.

            cuda:
                bool, signals the availability of CUDA.

            language_identifiers:
                list, containing the identifiers of the languages.
        """
        super().__init__(model=model,
                         vocabularies=vocabularies,
                         loss_functions=loss_functions,
                         tokens=tokens,
                         cuda=cuda,
                         add_language_token=add_language_token,
                         language_identifiers=language_identifiers)

        self._noise_model = noise_model

    def __call__(self,
                 batch:             dict,
                 lang_index:        int,
                 denoising:         bool = False,
                 forced_targets:    bool = True):
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

            batch:
                A dictionary

            denoising:

            forced_targets:


        Returns:
            loss:
                A scalar loss value, indicating the average loss of the auto encoder.

            outputs:
                A dictionary, that contains the outputs of the model. The types (keys) contained
                by this dictionary depends on the model specifications.
        """
        self._model.set_lookup({'source': lang_index, 'target': lang_index})

        if denoising:
            assert self._noise_model is not None, 'Noise model has not been defined for the AutoEncoder.'
            inputs, lengths = self._noise_model(batch['inputs'], self._tokens[lang_index]['<PAD>'])
        else:
            inputs, lengths = batch['inputs'], batch['input_lengths']

        if self._language_identifiers is not None and self._language_token_required:
            inputs = self._add_language_token(
                batch=inputs,
                token=self._vocabularies[lang_index](self._language_identifiers[lang_index])
            )
        else:
            if self._cuda:
                inputs = inputs.cuda()
            inputs = torch.autograd.Variable(inputs)

        loss, outputs = self._iterate_model(
            inputs={'data': inputs, 'lengths': lengths},
            targets={
                'data': batch['targets'],
                'lengths': batch['input_lengths'],
                'language_index': lang_index
            },
            forced_targets=forced_targets
        )

        inputs = inputs.cpu().data.squeeze(0).numpy()

        return loss, outputs, inputs


class Translator(_STSModule):
    """

    """

    @staticmethod
    def _substitute_eos_for_padding(symbols, eos_value, padding_value):
        """


        Args:
            symbols:

            eos_value:

            padding_value:

        Returns:
            symbols:

            new_lengths:

        """
        new_lengths = numpy.empty((symbols.shape[0]))
        for index in range(symbols.shape[0]):
            eos = numpy.argwhere(symbols[index] == int(eos_value)).reshape(-1)
            new_lengths[index] = symbols.shape[1]
            if len(eos) > 0:
                size = int(symbols.shape[1] - eos[0]) - 1
                symbols[index, eos[0] + 1:] = [padding_value] * size
                new_lengths[index] = new_lengths[index] - (new_lengths[index] - eos[0] - 1)

        symbols = symbols[:, :int(new_lengths.max(0))]
        symbols = symbols[new_lengths[::-1].argsort()]
        new_lengths[::-1].sort()

        return symbols, new_lengths.astype(numpy.int32)

    def __init__(self,
                 model:                 ModelWrapper,
                 vocabularies:          list,
                 loss_functions:        list,
                 tokens:                list,
                 add_language_token:    bool = False,
                 cuda:                  bool = False,
                 language_identifiers:  list = None):
        """


        Args:
            model:

            vocabularies:

            loss_functions:

            tokens:

            cuda:

            language_identifiers:

        """
        super().__init__(model=model,
                         vocabularies=vocabularies,
                         loss_functions=loss_functions,
                         tokens=tokens,
                         cuda=cuda,
                         add_language_token=add_language_token,
                         language_identifiers=language_identifiers)

    def __call__(self, batch, input_lang_index, target_lang_index, forced_targets=True):
        """
        Implementation of a step of auto-encoding. The look up tables of the model are fitted to the
        provided inputs, and the <LNG> are substituted with the appropriate token. In this case the token
        is the source language token. The inputs are then transformed by a noise function, and then fed
        through the model. If reguralization is applied, the encoder outputs are fetched from the output
        of the model, which is used by the discriminator to apply an adversarial reguralization on these
        outputs.

        Args:
            input_lang_index:
                An int value, that represents the index of the language. This value will serve as
                the index of the substitution token for the input batch.

            target_lang_index:

            batch:
                A list, containing the batches from the input pipelines.

            forced_targets:

        Returns:
            loss:
                A scalar loss value, indicating the average loss of the auto encoder.

            outputs:
                A dictionary, that contains the outputs of the model. The types (keys) contained
                by this dictionary depends on the model specifications.
        """
        self._model.set_lookup({'source': input_lang_index, 'target': target_lang_index})

        inputs = batch['inputs']

        if self._language_identifiers is not None and self._language_token_required:
            inputs = self._add_language_token(
                batch=inputs,
                token=self._vocabularies[input_lang_index](self._language_identifiers[target_lang_index])
            )
        else:
            if self._cuda:
                inputs = inputs.cuda()
            inputs = torch.autograd.Variable(inputs)

        targets = self._create_targets(batch, target_lang_index)

        loss, outputs = self._iterate_model(inputs={'data': inputs, 'lengths': batch['input_lengths']},
                                            targets=targets,
                                            forced_targets=forced_targets)

        # noinspection PyTypeChecker
        translated_symbols, translated_lengths = self._substitute_eos_for_padding(
            outputs['symbols'],
            self._tokens[target_lang_index]['<EOS>'],
            self._tokens[target_lang_index]['<PAD>'])

        translated_symbols = torch.from_numpy(translated_symbols)

        return loss, translated_symbols, outputs, inputs, translated_lengths

    def _create_targets(self, batch, target_lang_index):
        """

        Args:
            batch:

            target_lang_index:

        Returns:
            targets:

        """
        if batch.get('targets', None) is None or batch.get('target_lengths', None) is None:
            targets = None
        else:
            tgts = batch['targets']

            if self._cuda:
                tgts = tgts.cuda()

            targets = {
                'data': torch.autograd.Variable(tgts),
                'lengths': batch['target_lengths'],
                'language_index': target_lang_index
            }

        return targets


class Discriminator:
    """

    """

    def __init__(self,
                 model:            Classifier,
                 loss_function:    torch.nn.CrossEntropyLoss,
                 cuda:             bool = False):
        """


        Args:
            model:

            loss_function:

            cuda:

        """
        self._model = model
        self._loss_function = loss_function
        self._cuda = cuda

    def __call__(self, *args, inputs, targets, **kwargs):
        """
        This function implements the discrimination mechanism, where the inputs and the targets - which
        are required for the evaluation and training of the discriminator - are created. The inputs are
        fed into the discriminator and evaluated based on the cross entropy loss, that is defined in the
        init function. The targets are either one-hot coded vectors, or their inverse. This depends on
        whether the loss is calculated for the discriminator or model loss.

        Args:
            inputs:
                A list, containing the batches from the input pipelines.

            targets:
                An int value, that represents the index of the encoder's input language.

            lengths:

        Returns:
            loss:
                A scalar loss value, indicating the average loss of the discriminator for either the
                inverse or normal target vector.
        """
        return self._iterate_model(inputs, targets)

    def _iterate_model(self, inputs, targets):
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
        batch_size = inputs.shape[0]
        token_indexes = torch.from_numpy(numpy.array(targets, dtype=numpy.int32)).long()

        if self._cuda:
            token_indexes = token_indexes.cuda()

        token_indexes = torch.autograd.Variable(token_indexes)
        outputs, softmax = self._model(inputs=inputs)

        loss = self._loss_function(outputs, token_indexes)

        loss = loss.sum() / batch_size

        return loss


class WordTranslator(Component):

    abstract = False

    interface = Interface(**{
        'dictionaries':   (0, None)
    })

    @staticmethod
    def _load_dict(paths):
        dictionaries = []
        for path in paths:
            dictionaries.append(pickle.load(open(path, 'rb')))
        return dictionaries

    def __init__(self, dictionaries):
        self._dictionaries = self._load_dict(dictionaries)
        self._vocabs = None
        self._cuda = None
        self._language_tokens_required = True
        self._id_dictionaries = []

    def __call__(self, batch, input_lang_index, target_lang_index, forced_targets=True):

        if self._language_tokens_required:
            starting_index = 1
        else:
            starting_index = 0

        translated_symbols = numpy.empty((batch['inputs'].size(0), batch['inputs'].size(1) - starting_index))

        for index, line in enumerate(batch['inputs']):
            translated_symbols[index, :] = numpy.array(
                list(map(lambda x: self._id_dictionaries[input_lang_index][x],
                     line[starting_index:]))
            )

        translated_symbols = torch.from_numpy(translated_symbols.astype(numpy.int64))

        return None, translated_symbols, None, batch['inputs'], batch['input_lengths'] - starting_index

    def _convert_dictionaries_to_id(self):
        self._id_dictionaries.append({self._vocabs[0](word): self._vocabs[1](self._dictionaries[0][word])
                                      for word in self._dictionaries[0]})

        self._id_dictionaries.append({self._vocabs[1](word): self._vocabs[0](self._dictionaries[1][word])
                                      for word in self._dictionaries[1]})

        for token in self._vocabs[0].tokens:
            self._id_dictionaries[0][self._vocabs[0].tokens[token]] = self._vocabs[1].tokens[token]
            self._id_dictionaries[1][self._vocabs[1].tokens[token]] = self._vocabs[0].tokens[token]

        del self._dictionaries

    @property
    def vocabs(self):
        return self._vocabs

    @vocabs.setter
    def vocabs(self, value):
        self._vocabs = value
        self._convert_dictionaries_to_id()

    @property
    def language_tokens_required(self):
        return self._language_tokens_required

    @language_tokens_required.setter
    def language_tokens_required(self, value):
        self._language_tokens_required = value

    @property
    def cuda(self):
        return self._cuda

    @cuda.setter
    def cuda(self, value):
        self._cuda = value
