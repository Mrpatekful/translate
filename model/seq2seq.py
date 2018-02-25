from modules.attention import *
from modules.encoder import *
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
        torch.manual_seed(0)
        np.random.seed(0)

        self.__src = reader.Language()
        self.__src.load_vocab(SRC_VOCAB_PATH)

        embedding_size = self.__src.embedding_size
        vocab_size = self.__src.vocab_size

        self.__reader_src = reader.FastReader(language=self.__src,
                                              data_path=SRC_DATA_PATH,
                                              batch_size=32,
                                              use_cuda=USE_CUDA)

        encoder_params = utils.ParameterSetter({
            '_hidden_size': 50,
            '_embedding_size': embedding_size,
            '_recurrent_type': 'LSTM',
            '_num_layers': 2,
            '_learning_rate': 0.01,
            '_use_cuda': USE_CUDA
        })

        decoder_params = utils.ParameterSetter({
            '_hidden_size': 50,
            '_embedding_size': embedding_size,
            '_output_size': vocab_size,
            '_recurrent_type': 'LSTM',
            '_num_layers': 2,
            '_learning_rate': 0.01,
            '_max_length': 15,
            '_tf_ratio': 0,
            '_use_cuda': USE_CUDA
        })

        self.__encoder = RNNEncoder(encoder_params).init_parameters().init_optimizer()

        self.__decoder = RNNDecoder(decoder_params).init_parameters().init_optimizer()

    def fit(self, epochs):
        """
        :param epochs:
        :return:
        """
        loss_function = torch.nn.NLLLoss(ignore_index=0)

        for epoch in range(epochs):
            loss = 0
            self.__encoder.embedding = self.__src.embedding
            self.__decoder.embedding = self.__src.embedding

            for batch, lengths in self.__reader_src.batch_generator():
                loss += self._train_step(input_batch=batch,
                                         lengths=lengths,
                                         loss_function=loss_function,
                                         noise_function=utils.apply_noise)

            print(epoch, loss)

    def _train_step(self, input_batch, lengths, noise_function, loss_function):
        """
        :param input_batch:
        :param loss_function:
        :param lengths:
        :return:
        """
        encoder_state = self.__encoder.init_hidden(input_batch.shape[0])

        self.__encoder.optimizer.zero_grad()
        self.__decoder.optimizer.zero_grad()

        noisy_input = noise_function(input_batch)

        encoder_outputs, encoder_state = self.__encoder.forward(inputs=noisy_input,
                                                                lengths=lengths,
                                                                hidden_state=encoder_state)

        loss, symbols = self.__decoder.forward(inputs=input_batch,
                                               encoder_outputs=encoder_outputs,
                                               lengths=lengths,
                                               hidden_state=encoder_state,
                                               loss_function=loss_function,
                                               tf_ratio=1)

        loss.backward()

        self.__encoder.optimizer.step()
        self.__decoder.optimizer.step()

        return loss
