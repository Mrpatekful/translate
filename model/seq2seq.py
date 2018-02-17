import torch
import copy

from modules import encoder, decoder
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

        self._tgt = reader.Language()
        self._src.load_vocab(SRC_VOCAB_PATH)

        self.reader_src = reader.FastReader(language=self._src, data_path=SRC_DATA_PATH,
                                            batch_size=32, use_cuda=USE_CUDA)

        self.reader_tgt = reader.FastReader(language=self._tgt, data_path=TGT_DATA_PATH,
                                            batch_size=32, use_cuda=USE_CUDA)

        self.encoder = encoder.Encoder(embedding_dim=self._src.embedding_size, use_cuda=USE_CUDA,
                                       hidden_dim=32, learning_rate=0.001)

        self.decoder = decoder.Decoder(embedding_dim=self._src.embedding_size, use_cuda=USE_CUDA,
                                       hidden_dim=32, learning_rate=0.001)

        # self.discriminator = modules.Discriminator()

    def _train_step(self, input_batch, lengths, noise_function, loss_function):
        """
        :param input_batch:
        :param loss_function:
        :param lengths:
        :return:
        """
        encoder_hidden = self.encoder.init_hidden()

        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()

        noisy_input = noise_function(input_batch, )

        encoder_output, encoder_hidden = self.encoder.forward(inputs=noisy_input,
                                                              lengths=lengths,
                                                              hidden=encoder_hidden)

        decoder_hidden = encoder_hidden

        # auto encoding -> inputs and targets are the same
        decoder_output, decoder_hidden, = self.decoder.forward(inputs=input_batch,
                                                               lengths=lengths,
                                                               hidden=decoder_hidden)

        loss = loss_function(decoder_output, input_batch)

        loss.backward()

        self.encoder.optimizer.step()
        self.decoder.optimizer.step()

        return loss

    def fit(self, epochs):
        """
        :param epochs:
        :return:
        """
        loss_function = torch.nn.NLLLoss()

        for epoch in range(epochs):
            loss = 0
            self.encoder.embedding = self._src.embedding
            self.decoder.embedding = self._src.embedding

            for batch, lengths in self.reader_src.batch_generator():
                loss += self._train_step(input_batch=batch,
                                         lengths=lengths,
                                         loss_function=loss_function,
                                         noise_function=utils.apply_noise)

            self.encoder.embedding = self._tgt.embedding
            self.decoder.embedding = self._tgt.embedding

            for batch, lengths in self.reader_tgt.batch_generator():
                loss += self._train_step(input_batch=batch,
                                         lengths=lengths,
                                         loss_function=loss_function,
                                         noise_function=utils.apply_noise)

            self.encoder.embedding = self._src.embedding
            self.decoder.embedding = self._src.embedding

            for batch, lengths in self.reader_src.batch_generator():
                loss += self._train_step(input_batch=batch,
                                         lengths=lengths,
                                         loss_function=loss_function,
                                         noise_function=self.translate)

            self.encoder.embedding = self._tgt.embedding
            self.decoder.embedding = self._tgt.embedding

            for batch, lengths in self.reader_src.batch_generator():
                loss += self._train_step(input_batch=batch,
                                         lengths=lengths,
                                         loss_function=loss_function,
                                         noise_function=self.translate)

            self._logger.save_log(loss)

    def translate(self, inputs, lengths):
        pass


class Logger:

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
