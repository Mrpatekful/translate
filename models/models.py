from torch.nn import Module

from modules.rnn import encoders
from modules.rnn import decoders
from utils.utils import Component
from utils.utils import reduce_parameters

from collections import OrderedDict


class Model(Module, Component):
    """
    Abstract base class for the models of the application.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return NotImplementedError

    @property
    def optimizers(self):
        return NotImplementedError


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

    @staticmethod
    def interface():
        return OrderedDict(**{
            'encoder': encoders.Encoder,
            'decoder': decoders.Decoder
        })

    @classmethod
    def abstract(cls):
        return False

    def __init__(self, encoder, decoder):
        """
        An instance of ta sequence to sequence model.
        :param encoder: Encoder, an encoder instance.
        :param decoder: Decoder, a decoder instance.
        """
        super().__init__()

        self._encoder = encoder.init_parameters().init_optimizer()
        self._decoder = decoder.init_parameters().init_optimizer()

    def forward(self,
                inputs,
                targets,
                max_length,
                lengths):
        """
        Forward step of the sequence to sequence model.
        :param inputs: Variable, containing the ids of the tokens for the input sequence.
        :param targets: Variable, containing the ids of the tokens for the target sequence.
        :param max_length: int, the maximum length of the decoded sequence.
        :param lengths: Ndarray, containing the lengths of the original sequences.
        :return decoder_outputs: dict, containing the concatenated outputs of the encoder and decoder.
        """
        encoder_outputs = self._encoder.forward(inputs=inputs, lengths=lengths)
        decoder_outputs = self._decoder.forward(targets=targets, max_length=max_length, **encoder_outputs)

        return {**decoder_outputs, **encoder_outputs}

    def set_embeddings(self, encoder_embedding, decoder_embedding):
        """
        Sets the embedding layers for the encoder and decoder module.
        :param encoder_embedding: Embedding, layer type object, that will be used by the encoder.
        :param decoder_embedding: Embedding, layer type object, that will be used by the decoder.
        """
        self._encoder.embedding = encoder_embedding
        self._decoder.embedding = decoder_embedding

    @property
    def optimizers(self):
        """
        Convenience function for the optimizers of the encoder and decoder.
        :return: dict, containing the names and instances of optimizers for the encoder/decoder
                 and the currently used embeddings.
        """
        return [*self._encoder.optimizers, *self._decoder.optimizers]

    @property
    def decoder_tokens(self):
        """
        Tokens used by the decoder, for special outputs.
        """
        return self._decoder.tokens

    @decoder_tokens.setter
    def decoder_tokens(self, tokens):
        """
        Setter for the tokens, that will be used by the decoder.
        :param tokens: dict, tokens from the lut of decoding target.
        """
        self._decoder.tokens = tokens

    @property
    def state(self):
        return {'encoder': self._encoder.state, 'decoder': self._decoder.state}

    @state.setter
    def state(self, state):
        self._encoder.state = state['encoder']
        self._decoder.state = state['decoder']
