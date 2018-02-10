import random
import time
import torch

from torch.autograd import Variable
from model import reader
from model import modules

USE_CUDA = torch.cuda.is_available()


class Model:
    """
    Sequence to sequence model for translation.
    """
    def __init__(self):
        src = reader.Language()
        tgt = reader.Language()

        self.auto_encoder_reader_l1 = reader.Reader(src, src, USE_CUDA)
        self.auto_encoder_reader_l2 = reader.Reader(tgt, tgt, USE_CUDA)
        self.translator_reader_l1 = reader.Reader(src, tgt, USE_CUDA)
        self.translator_reader_l2 = reader.Reader(tgt, src, USE_CUDA)

        self.encoder = modules.Encoder(embedding_dim=src.embedding_size, use_cuda=USE_CUDA,
                                       hidden_dim=32, learning_rate=0.001)

        self.decoder = modules.Decoder(embedding_dim=tgt.embedding_size, use_cuda=USE_CUDA,
                                       hidden_dim=32, learning_rate=0.001)

        self.discriminator = modules.Discriminator()

    def _train_step(self, input_sequence, target_sequence, loss_function):
        """

        :param input_sequence:
        :param target_sequence:
        :return:
        """
        encoder_hidden = self.encoder.initHidden()

        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()

        input_length = input_sequence.size()[0]
        target_length = target_sequence.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder.forward(input_sequence[ei],
                                                                  encoder_hidden)

            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([['<SOS>']]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden, = self.decoder.forward(decoder_input,
                                                                   decoder_hidden)

            loss += loss_function(decoder_output, target_sequence[di])
            decoder_input = target_sequence[di]  # Teacher forcing

        loss.backward()

        self.encoder.optimizer.step()
        self.decoder.optimizer.step()

    def fit(self, epochs,
            print_every=1000, plot_every=100):
        """
        :param epochs:
        :param print_every:
        :param plot_every:
        :return:
        """
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]

        criterion = torch.nn.NLLLoss()

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = self.train(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)


class Logger:
    pass
