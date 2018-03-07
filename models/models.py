import torch.nn as nn

from modules.rnn import decoder, encoder
from utils import utils


class Model(nn.Module):
    """
    Abstract base class for the models of the application.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError

    def step(self):
        return NotImplementedError

    def zero_grad(self):
        return NotImplementedError

    @classmethod
    def assemble(cls, params):
        return NotImplementedError

    @classmethod
    def abstract(cls):
        return True


class SeqToSeq(Model):
    """
    Sequence to sequence model according to the one described in:

        https://arxiv.org/abs/1409.3215

    The model has two main components, an encoder and a decoder, which may be
    implemented as recurrent or convolutional units. The main principle of this technique
    is to map a sequence - in case of translation - a sentence to another sentence,
    by encoding it to a fixed size representation, and then decoding this latent meaning
    vector to the desired sequence.
    """

    def __init__(self,
                 encoder_type,
                 decoder_type,
                 encoder_params,
                 decoder_params):
        """
        An instance of ta sequence to sequence model.
        :param encoder_type: Encoder, an encoder instance.
        :param decoder_type: Decoder, a decoder instance.
        :param encoder_params: ParameterSetter, containing the parameter dict for the encoder.
        :param decoder_params: ParameterSetter, containing the parameter dict for the decoder.
        """
        super().__init__()

        self._encoder = encoder_type(encoder_params).init_parameters().init_optimizer()
        self._decoder = decoder_type(decoder_params).init_parameters().init_optimizer()

    def forward(self,
                inputs,
                targets,
                lengths,
                loss_function):
        """
        Forward step of the sequence to sequence model.
        :param inputs: Variable, containing the ids for the tokens of the input sequence.
        :param targets: Variable, containing the ids for the tokens of the target sequence.
        :param lengths: Ndarray, containing the lengths of the original sequences.
        :param loss_function: loss function for the calculations of the error.
        :return decoder_outputs: dict, containing the output values of the decoder.
        :return encoder_outputs: dict, containing the output values of the encoder.
        """
        encoder_outputs = self._encoder.forward(inputs=inputs, lengths=lengths)

        decoder_outputs = self._decoder.forward(targets=targets,
                                                lengths=lengths,
                                                loss_function=loss_function,
                                                **encoder_outputs)

        return {**decoder_outputs, **encoder_outputs}

    def step(self):
        """
        Updates the parameters of the encoder and decoder modules.
        """
        self._encoder.optimizer.step()
        self._decoder.optimizer.step()

    def zero_grad(self):
        """
        Refreshes the gradients of the optimizer.
        """
        self._encoder.optimizer.zero_grad()
        self._decoder.optimizer.zero_grad()

    @classmethod
    def abstract(cls):
        return False

    @classmethod
    def assemble(cls, params):
        """
        Assembler function for the encoder and decoder units, and its parameters
        for the sequence to sequence model. The parameters are
        :param params: dict, the merged dictionary of the model and task configuration parameters.
        :return: dict, processed parameters wrapped in ParameterSetter object for both the encoder
                 and decoder.
        """
        encoders = utils.subclasses(encoder.Encoder)
        decoders = utils.subclasses(decoder.Decoder)

        enc = encoders[params['encoder']['type']]
        encoder_params = utils.ParameterSetter(enc.assemble({
            **params['encoder']['params'],
            'language': params['source_reader'].language,
            'use_cuda': params['use_cuda']
        }))

        dec = decoders[params['decoder']['type']]
        decoder_params = utils.ParameterSetter(dec.assemble({
            **params['decoder']['params'],
            'language': params['target_reader'].language,
            'use_cuda': params['use_cuda']
        }))

        return {
            'encoder_type': enc,
            'decoder_type': dec,
            'encoder_params': encoder_params,
            'decoder_params': decoder_params
        }

    @property
    def decoder_embedding(self):
        """
        Property for the embedding of the decoder.
        :return embedding: Embedding, the currently used embedding of the decoder.
        """
        return self._decoder.embedding

    @decoder_embedding.setter
    def decoder_embedding(self, embedding):
        """
        Setter for the embedding of the decoder.
        """
        self._decoder.embedding = embedding

    @property
    def encoder_embedding(self):
        """
        Property for the embedding of the decoder.
        :return embedding: Embedding, the currently used embedding of the encoder.
        """
        return self._encoder.embedding

    @encoder_embedding.setter
    def encoder_embedding(self, embedding):
        """
        Setter for the embedding of the encoder.
        """
        self._encoder.embedding = embedding
