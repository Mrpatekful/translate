import torch

from data import reader
from model import modules

USE_CUDA = torch.cuda.is_available()

SRC_DATA_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_seg'
SRC_VOCAB_PATH = '/home/patrik/GitHub/nmt-BMEVIAUAL01/data/eng_voc'

TGT_DATA_PATH = ''


class Model:
    """
    Sequence to sequence model for translation.
    """
    def __init__(self):
        self._src = reader.Language()
        self._src.load_vocab(SRC_VOCAB_PATH)
        self._logger = Logger()

        # self._tgt = reader.Language()

        self.reader_src = reader.Reader(language=self._src, data_path=SRC_DATA_PATH,
                                        batch_size=32, full_load=False, use_cuda=USE_CUDA)

        # self.reader_tgt = reader.Reader(tgt, TGT_DATA_PATH, USE_CUDA)

        self.encoder = modules.Encoder(embedding_dim=self._src.embedding_size, use_cuda=USE_CUDA,
                                       hidden_dim=32, learning_rate=0.001)

        self.decoder = modules.Decoder(embedding_dim=self._src.embedding_size, use_cuda=USE_CUDA,
                                       hidden_dim=32, learning_rate=0.001)

        # self.discriminator = modules.Discriminator()

    def _train_step(self, input_sequence, target_sequence, loss_function):
        """
        :param input_sequence:
        :param target_sequence:
        :param loss_function
        :return:
        """
        encoder_hidden = self.encoder.init_hidden()

        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()

        encoder_output, encoder_hidden = self.encoder.forward(input_sequence, encoder_hidden)

        decoder_hidden = encoder_hidden

        decoder_output, decoder_hidden, = self.decoder.forward(target_sequence, decoder_hidden)

        loss = loss_function(decoder_output, target_sequence)

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

        self.encoder.embedding = self._src.embedding

        for _ in range(epochs):
            loss = 0
            for batch in self.reader_src.batch_generator():
                input_variable = batch[0]
                target_variable = batch[1]

                loss += self._train_step(input_variable, target_variable, loss_function)

            self._logger.save_log(loss)


class Logger:

    def __init__(self):
        pass

    def save_log(self, loss):
        pass
