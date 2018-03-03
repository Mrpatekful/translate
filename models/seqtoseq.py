import torch.nn as nn

from modules import attention
from modules import encoder
from modules import decoder


class SeqToSeq(nn.Module):
    """
    Sequence to sequence model for translation.
    """
    _encoders = {
        'RNNEncoder': encoder.RNNEncoder,
    }

    _decoders = {
        'RNNDecoder': decoder.RNNDecoder,
        'BahdanauRNNDecoder': attention.BahdanauAttentionRNNDecoder,
        'GeneralAttentionRNNDecoder': attention.GeneralAttentionRNNDecoder,
        'DotAttentionRNNDecoder': attention.DotAttentionRNNDecoder,
        'ConcatAttentionRNNDecoder': attention.ConcatAttentionRNNDecoder,
    }

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
        initial_state = self._encoder.init_hidden(inputs.shape[0])

        encoder_outputs = self._encoder.forward(inputs=inputs,
                                                lengths=lengths,
                                                hidden_state=initial_state)

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
