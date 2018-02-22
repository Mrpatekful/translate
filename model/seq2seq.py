import torch

from modules import encoder, decoder, discriminator, attention
from utils import reader, utils

USE_CUDA = torch.cuda.is_available()

SRC_DATA_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_seg'
SRC_VOCAB_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_voc'

TGT_DATA_PATH = ''
TGT_VOCAB_PATH = ''


class Model:
    """
    Sequence to sequence model for translation.
    """

    def __init__(self):
        self._logger = Logger()

        self._src = reader.Language()
        self._src.load_vocab(SRC_VOCAB_PATH)

        # self._tgt = reader.Language()
        # self._tgt.load_vocab(SRC_VOCAB_PATH)

        self._reader_src = reader.FastReader(language=self._src, data_path=SRC_DATA_PATH,
                                             batch_size=32, use_cuda=USE_CUDA)

        # self.reader_tgt = reader.FastReader(language=self._tgt, data_path=TGT_DATA_PATH,
        #                                     batch_size=32, use_cuda=USE_CUDA)

        self._encoder = encoder.RNNEncoder(embedding_dim=self._src.embedding_dim, use_cuda=USE_CUDA,
                                           hidden_size=50, learning_rate=0.001, recurrent_layer='GRU')

        self._decoder = decoder.RNNDecoder(embedding_dim=self._src.embedding_dim, use_cuda=USE_CUDA,
                                           hidden_size=50, output_dim=self._src.vocab_size,
                                           learning_rate=0.001, recurrent_layer='GRU')

        # self._decoder.attention = attention.LuongAttention

        self.discriminator = discriminator.Discriminator(hidden_dim=1024, input_dim=2,
                                                         learning_rate=0.0005, use_cuda=USE_CUDA)

    def _train_step(self, input_batch, lengths, noise_function, loss_function):
        """
        :param input_batch:
        :param loss_function:
        :param lengths:
        :return:
        """
        encoder_state = self._encoder.init_hidden(input_batch.shape[0])

        self._encoder.optimizer.zero_grad()
        self._decoder.optimizer.zero_grad()

        noisy_input = noise_function(input_batch)

        encoder_outputs, encoder_state = self._encoder.forward(inputs=noisy_input,
                                                               lengths=lengths,
                                                               hidden_state=encoder_state)
        decoder_state = encoder_state

        # auto encoding -> inputs and targets are the same
        loss, symbols = self._decoder.forward(inputs=input_batch,
                                              encoder_outputs=encoder_outputs,
                                              lengths=lengths,
                                              hidden_state=decoder_state,
                                              loss_function=loss_function)

        loss.backward()

        self._encoder.optimizer.step()
        self._decoder.optimizer.step()

        return loss

    def fit(self, epochs):
        """
        :param epochs:
        :return:
        """
        loss_function = torch.nn.NLLLoss(ignore_index=0)

        for epoch in range(epochs):
            loss = 0
            self._encoder.embedding = self._src.embedding
            self._decoder.embedding = self._src.embedding

            for batch, lengths in self._reader_src.batch_generator():
                loss += self._train_step(input_batch=batch,
                                         lengths=lengths,
                                         loss_function=loss_function,
                                         noise_function=utils.apply_noise)

            print(epoch, loss)

            # self.encoder.embedding = self._tgt.embedding
            # self.decoder.embedding = self._tgt.embedding
            #
            # for batch, lengths in self.reader_tgt.batch_generator():
            #     loss += self._train_step(input_batch=batch,
            #                              lengths=lengths,
            #                              loss_function=loss_function,
            #                              noise_function=utils.apply_noise)
            #
            # self.encoder.embedding = self._src.embedding
            # self.decoder.embedding = self._src.embedding
            #
            # for batch, lengths in self.reader_src.batch_generator():
            #     loss += self._train_step(input_batch=batch,
            #                              lengths=lengths,
            #                              loss_function=loss_function,
            #                              noise_function=self.translate)
            #
            # self.encoder.embedding = self._tgt.embedding
            # self.decoder.embedding = self._tgt.embedding
            #
            # for batch, lengths in self.reader_src.batch_generator():
            #     loss += self._train_step(input_batch=batch,
            #                              lengths=lengths,
            #                              loss_function=loss_function,
            #                              noise_function=self.translate)

            self._logger.save_log(loss)

    def translate(self, inputs, lengths):
        """

        :param inputs:
        :param lengths:
        :return:
        """


class Logger:
    """

    """

    def __init__(self):
        pass

    def save_log(self, loss):
        pass

    def create_checkpoint(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
