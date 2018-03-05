import torch.nn as nn

from modules import decoder
from modules import encoder

from utils import utils


class Model(nn.Module):

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
    Sequence to sequence model for translation.
    """
    _encoders = utils.subclasses(encoder.Encoder)
    _decoders = utils.subclasses(decoder.Decoder)

    def __init__(self,
                 encoder_type,
                 decoder_type,
                 encoder_params,
                 decoder_params):
        """

        :param encoder_type:
        :param decoder_type:
        :param encoder_params:
        :param decoder_params:
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
        :param inputs:
        :param targets:
        :param lengths:
        :param loss_function:
        :return:
        """
        encoder_outputs = self._encoder.forward(inputs=inputs,
                                                lengths=lengths)

        decoder_outputs = self._decoder.forward(targets=targets,
                                                lengths=lengths,
                                                loss_function=loss_function,
                                                **encoder_outputs)

        return decoder_outputs, encoder_outputs

    def step(self):
        """

        """
        self._encoder.optimizer.step()
        self._decoder.optimizer.step()

    def zero_grad(self):
        """

        """
        self._encoder.optimizer.zero_grad()
        self._decoder.optimizer.zero_grad()

    @classmethod
    def abstract(cls):
        return False

    @classmethod
    def assemble(cls, params):
        """

        :param params:
        :return:
        """
        enc = cls._encoders[params['encoder']['encoder_type']]
        encoder_params = utils.ParameterSetter(enc.assemble({
            **params['encoder']['encoder_params'],
            'source_language': params['source_language'],
            'use_cuda': params['use_cuda']
        }))

        dec = cls._decoders[params['decoder']['decoder_type']]
        decoder_params = utils.ParameterSetter(dec.assemble({
            **params['decoder']['decoder_params'],
            'target_language': params['target_language'],
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

        :return:
        """
        return self._decoder.embedding

    @decoder_embedding.setter
    def decoder_embedding(self, embedding):
        """

        :return:
        """
        self._decoder.embedding = embedding

    @property
    def encoder_embedding(self):
        """

        :return:
        """
        return self._encoder.embedding

    @encoder_embedding.setter
    def encoder_embedding(self, embedding):
        """

        :return:
        """
        self._encoder.embedding = embedding


class Transformer(Model):
    pass
