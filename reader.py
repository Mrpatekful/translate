import numpy as np


EMBEDDING_DIM = 300


class Lang:

    def __init__(self):
        self.word_to_id = {0: '<SOS>', 1: '<EOS>', 2: '<UNK>'}
        self.word_to_count = {}
        self.n_words = 3

        self.vocab = None
        self.embedding = None

    @staticmethod
    def load_vocab(path):
        """
        Loader function for the embedding. Path is assumed to be a text
        file, where each line contains a word and its corresponding embedding weights.
        Args:
            path: string, the absolute path of the vocab.
        Returns:
            A numpy array containing the embeddings
            of dim (vocab_size, embedding_dim).



        """
        with open(path, 'r') as file:
            embedding = np.empty((len(list(file)), EMBEDDING_DIM))
            for line in file:
                

