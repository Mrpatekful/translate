from torch.nn import Module

from modules.rnn import encoders
from modules.rnn import decoders
from utils.utils import Component

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

    @property
    def state(self):
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

    interface = OrderedDict(**{
        'encoder':      encoders.Encoder,
        'decoder':      decoders.Decoder,
    })

    abstract = False

    def __init__(self, encoder, decoder):
        """
        An instance of ta sequence to sequence model.
        :param encoder: Encoder, an encoder instance.
        :param decoder: Decoder, a decoder instance.
        """
        super().__init__()

        self.encoder = encoder.init_parameters().init_optimizer()
        self.decoder = decoder.init_parameters().init_optimizer()

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
        encoder_outputs = self.encoder.forward(inputs=inputs, lengths=lengths)
        decoder_outputs = self.decoder.forward(targets=targets, max_length=max_length, **encoder_outputs)

        return {**decoder_outputs, **encoder_outputs}

    @property
    def optimizers(self):
        """
        Convenience function for the optimizers of the encoder and decoder.
        :return: dict, containing the names and instances of optimizers for the encoder/decoder
                 and the currently used embeddings.
        """
        return [*self.encoder.optimizers, *self.decoder.optimizers]

    @property
    def output_size(self):
        """
        THe dimension of the decoder's output layer.
        """
        return self.decoder.output_size

    @property
    def decoder_tokens(self):
        """
        Tokens used by the decoder, for special outputs.
        """
        return self.decoder.tokens

    @decoder_tokens.setter
    def decoder_tokens(self, tokens):
        """
        Setter for the tokens, that will be used by the decoder.
        :param tokens: dict, tokens from the lut of decoding target.
        """
        self.decoder.tokens = tokens

    @property
    def state(self):
        """

        :return:
        """
        return {'encoder': self.encoder.state, 'decoder': self.decoder.state}

    # noinspection PyMethodOverriding
    @state.setter
    def state(self, state):
        """

        :param state:
        :return:
        """
        self.encoder.state = state['encoder']
        self.decoder.state = state['decoder']
