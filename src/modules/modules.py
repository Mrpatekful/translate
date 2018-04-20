import torch
import numpy


class _STSModule:

    def __init__(self, **params):
        """

        """
        self._cuda = params.get('cuda', False)
        self._language_tokens = params.get('language_tokens', None)

        try:
            self._model = params['model']
            self._tokens = params['tokens']
            self._loss_functions = params['loss_functions']
            self._input_pipelines = params['input_pipelines']

        except KeyError as error:
            raise RuntimeError(f'Value {error} has not been defined for the {self.__class__}.')

    def _iterate_model(self, inputs, targets=None, forced_targets=False):
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
            outputs = self._model(inputs=inputs['data'],
                                  lengths=inputs['lengths'],
                                  targets=targets['data'],
                                  max_length=max_length)
        else:
            outputs = self._model(inputs=inputs['data'],
                                  lengths=inputs['lengths'],
                                  targets=None,
                                  max_length=max_length)

        loss = 0

        if targets is not None:
            for step, step_output in enumerate(outputs['outputs']):
                loss += self._loss_functions[targets['language_index']](step_output, targets['data'][:, step + 1])

            lengths = torch.from_numpy(targets['lengths']).float()

            if self._cuda:
                lengths = lengths.cuda()

            loss = loss / torch.autograd.Variable(lengths)
            loss = loss.sum() / batch_size

        return loss, outputs

    def _add_language_token(self, batch, token):
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

    """
    def __init__(self, noise_model=None, **params):
        """

        """
        super().__init__(**params)

        self._noise_model = noise_model

    def __call__(self, batch, lang_index, denoising=False, forced_targets=True):
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

        if self._language_tokens is not None:
            inputs = self._add_language_token(
                batch=inputs,
                token=self._input_pipelines[lang_index].vocabulary[0](self._language_tokens[lang_index])
            )

        loss, outputs = self._iterate_model(
            inputs={'data': inputs, 'lengths': lengths},
            targets={
                'data': batch['targets'],
                'lengths': batch['input_lengths'],
                'language_index': lang_index
            },
            forced_targets=forced_targets
        )

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
        new_lengths = numpy.empty(symbols.shape[0], 1)
        for index in range(symbols.shape[0]):
            eos = numpy.argwhere(symbols[index] == int(eos_value))
            if eos.shape[0] > 0:
                size = int(symbols.shape[1] - eos[0])
                if size == 1:
                    symbols[index, -1] = padding_value
                else:
                    symbols[index, eos[0]:] = [padding_value] * size

        return symbols, new_lengths

    @staticmethod
    def _create_targets(batch, target_lang_index):
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
            targets = {
                'data': batch['targets'],
                'lengths': batch['target_lengths'],
                'language_index': target_lang_index
            }

        return targets

    def __init__(self, **params):
        super().__init__(**params)

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

        if self._language_tokens is not None:
            inputs = self._add_language_token(batch=inputs, token=self._input_pipelines[input_lang_index]
                                              .vocabulary[0](self._language_tokens[target_lang_index]))

        targets = self._create_targets(batch, target_lang_index)

        loss, outputs = self._iterate_model(inputs={'data': inputs, 'lengths': batch['input_lengths']},
                                            targets=targets,
                                            forced_targets=forced_targets)

        outputs, lengths = self._substitute_eos_for_padding(outputs['symbols'],
                                                            self._tokens[target_lang_index]['<EOS>'],
                                                            self._tokens[target_lang_index]['<PAD>'])

        return loss, outputs, inputs, lengths


class Discriminator:

    def __init__(self, **params):
        """

        """
        self._cuda = params.get('cuda', False)

        try:
            self._model = params['model']
            self._num_languages = params['num_languages']
            self._loss_function = params['loss_function']

        except KeyError as error:
            raise RuntimeError(f'Value {error} has not been defined for the {self.__class__}.')

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
        shaped_inputs = numpy.zeros((inputs.shape[0], inputs[0].shape[0]))
        for index in range(shaped_inputs.shape[0]):
            shaped_inputs[index, :] = inputs[index]

        inputs = torch.from_numpy(shaped_inputs).float()

        if self._cuda:
            inputs = inputs.cuda()

        inputs = torch.autograd.Variable(inputs)

        loss = self._iterate_model(inputs, targets)

        return loss

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


class RNNDiscriminator:  # TODO

    def __init__(self, **params):
        """

        """
        self._cuda = params.get('cuda', False)

        try:
            self._model = params['model']
            self._num_languages = params['num_languages']
            self._loss_function = params['loss_function']

        except KeyError as error:
            raise RuntimeError(f'Value {error} has not been defined for the {self.__class__}.')

    def classify(self, inputs, targets, lengths):

        shaped_inputs = numpy.zeros((inputs.shape[0], inputs[0].shape[0], inputs[0].shape[1]))
        for index in range(shaped_inputs.shape[0]):
            shaped_inputs[index, :lengths[index], :] = inputs[index][:lengths[index], :]

        inputs = torch.from_numpy(shaped_inputs).float()

        if self._cuda:
            inputs = inputs.cuda()

        inputs = torch.autograd.Variable(inputs)

        loss = self._iterate_model(inputs, targets, lengths)

        return loss

    def reguralize(self, inputs, lang_index):

        loss = 0

        for target_lang_index in [index for index in range(self._num_languages) if index != lang_index]:
            loss += self._iterate_model(inputs, numpy.array([target_lang_index]*inputs.size(0)))

        return loss

    def _iterate_model(self, inputs, targets, lengths):

        batch_size = inputs.shape[0]

        token_indexes = torch.from_numpy(numpy.array(targets, dtype=numpy.int32)).long()

        if self._cuda:
            token_indexes = token_indexes.cuda()

        token_indexes = torch.autograd.Variable(token_indexes)
        outputs, softmax = self._model(inputs=inputs, lengths=lengths)

        loss = self._loss_function(outputs, token_indexes)

        loss = loss.sum() / batch_size

        return loss
