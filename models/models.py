import torch.nn as nn

from modules import decoder
from modules import encoder

from utils import utils


class Models(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError

    def step(self):
        return NotImplementedError

    def zero_grad(self):
        return NotImplementedError

    @classmethod
    def descriptor(cls, components):
        return NotImplementedError

    @classmethod
    def abstract(cls):
        return True


class SeqToSeq(Models):
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

        self._encoder = self._encoders[encoder_type](encoder_params).init_parameters().init_optimizer()
        self._decoder = self._decoders[decoder_type](decoder_params).init_parameters().init_optimizer()

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
                                                encoder_outputs=encoder_outputs['hidden_states'],
                                                hidden_state=encoder_outputs['final_state'],
                                                loss_function=loss_function,
                                                tf_ratio=1)

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
    def descriptor(cls, components):
        encoder_type = components['encoder']['encoder_type']
        decoder_type = components['decoder']['decoder_type']
        encoder_params = components['encoder']['encoder_params']
        decoder_params = components['decoder']['decoder_params']
        return {
            'encoder_params': cls._encoders[encoder_type].desciptor(encoder_params),
            'decoder_params': cls._decoders[decoder_type].desciptor(decoder_params),
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


class Transformer(Models):
    pass
