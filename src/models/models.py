from collections import OrderedDict

from torch.nn import Module

from src.components.base import Encoder
from src.components.base import Decoder

from src.utils.utils import Component


class Model(Module, Component):
    """
    Abstract base class for the models of the application.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def optimizers(self):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError

    @state.setter
    def state(self, value):
        raise NotImplementedError


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
        'encoder':    Encoder,
        'decoder':    Decoder
    })

    abstract = False

    def __init__(self, encoder, decoder):
        """
        An instance of ta sequence to sequence model.

        Args:
            encoder:
                Encoder, an encoder instance.

            decoder:
                Decoder, a decoder instance.
        """
        super().__init__()

        self.encoder = encoder.init_parameters().init_optimizer()
        self.decoder = decoder.init_parameters().init_optimizer()

        self._parameter_names = [name for name, _ in self.named_parameters()]

    def forward(self,
                inputs,
                lengths,
                targets,
                max_length):
        """
        Forward step of the sequence to sequence model.

        Args:
            inputs:
                Variable, containing the ids of the tokens for the input sequence.

            targets:
                Variable, containing the ids of the tokens for the target sequence.

            max_length:
                int, the maximum length of the decoded sequence.

            lengths:
                Ndarray, containing the lengths of the original sequences.

        Returns:
            outputs:
                dict, containing the concatenated outputs of the encoder and decoder.
        """
        encoder_outputs = self.encoder(inputs=inputs, lengths=lengths)
        decoder_outputs = self.decoder(targets=targets, max_length=max_length, **encoder_outputs)

        return {**decoder_outputs, **encoder_outputs}

    def freeze(self):
        """

        """
        for param in [param for name, param in self.named_parameters() if name in self._parameter_names]:
            param.requires_grad = False

    def unfreeze(self):
        """

        """
        for param in [param for name, param in self.named_parameters() if name in self._parameter_names]:
            param.requires_grad = True

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
        """
        self.decoder.tokens = tokens

    @property
    def output_types(self):
        """

        """
        return {
            **self.encoder.output_types,
            **self.decoder.output_types
        }

    @property
    def state(self):
        """

        """
        return {
            'encoder': self.encoder.state,
            'decoder': self.decoder.state
        }

    @state.setter
    def state(self, state):
        """

        """
        self.encoder.state = state['encoder']
        self.decoder.state = state['decoder']
