from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from functools import wraps

import time
import pickle
import os.path
import numpy
import torch
import sys


def batch_to_padded_sequence(batch, lengths):
    """

    :param batch:
    :param lengths:
    :return:
    """
    return pack_padded_sequence(batch, lengths=lengths, batch_first=True)


def padded_sequence_to_batch(padded_sequence):
    """

    :param padded_sequence:
    :return:
    """
    return pad_packed_sequence(padded_sequence, batch_first=True)


def subclasses(base_cls):
    """
    Discovers the inheritance tree of a given class. The class must have an abstract() method. A class of the
    hierarchy will only be the part of the returned classes, if it's abstract() method returns False.
    :param base_cls: type, root of the inheritance hierarchy.
    :return: dict, classes that are part of the hierarchy, with their str names as keys, and type references as values.
    """
    def get_hierarchy(cls):
        sub_classes = {sub_cls.__name__: sub_cls for sub_cls in cls.__subclasses__()}
        hierarchy = {}
        for sub_cls_name in sub_classes:
            hierarchy = {**hierarchy, **get_hierarchy(sub_classes[sub_cls_name])}
        return {**hierarchy, **{name: sub_classes[name] for name in sub_classes if not sub_classes[name].abstract()}}

    return get_hierarchy(base_cls)


def logging(logger):
    """
    Decorator for the functions to get logs from. The function which is decorated
    with logging must return its values as a dictionary.
    :param logger: Logger object, which handles the data.
    :return: wrapper of the function.
    """
    def log_wrapper(func):
        """
        Wraps the wrapper of the function.
        :param func: the decorated function.
        :return: wrapper
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return logger(*args, func=func, **kwargs)
        return wrapper
    return log_wrapper


class Logger:
    """
    Logger class for saving the progress of training.
    """

    def __init__(self,
                 params,
                 dump_interval=1000):
        """
        A logger instance. Instantiation should happen as a parameter of logging decorator.
        :param params: tuple, name of the input parameters, which will be logged.
        :param dump_interval: int, number of iteration, between two log dumps.
        """
        self._log_dir = None
        self._id = 1
        self._params = params
        self._dump_interval = dump_interval
        self._log_dict = {}

    def __call__(self, *args, func, **kwargs):
        """
        Invocation of a logger object will execute the given function, record the time required
        for this operation, and then save the results and given input parameters to the log dictionary.
        :param args: arguments of the function, which will be executed.
        :param func: function to be executed.
        :param kwargs: keyword arguments of the function to be executed.
        :return: result of the execution.
        """
        exec_time = time.time()
        result = func(*args, **kwargs)
        exec_time = time.time() - exec_time
        self._save_log({
            'exec_time': exec_time,
            **{param: kwargs[param] for param in self._params},
            **result
        })

        return result

    def _save_log(self, log):
        """
        Saves the log to the log dictionary. When the log id reaches the dump interval, the
        dictionary is serialized, and then deleted from the memory.
        :param log: dict, the log to be saved.
        """
        self._log_dict[self._id] = log
        if self._id % self._dump_interval == 0:
            self._dump_log()
        self._id += 1

    def _dump_log(self):
        """
        Serializes the log dictionary.
        """
        pickle.dump(obj=self._log_dict,
                    file=open(os.path.join(self.log_dir, 'iter_%d-%d' %
                                           (self._id-self._dump_interval, self._id)), 'wb'))
        del self._log_dict
        self._log_dict = {}

    @property
    def log_dir(self):
        """
        Property for the logging directory.
        :return: str, location of the logs.
        """
        if self._log_dir is None:
            raise ValueError('Log directory has not been defined.')
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        """
        Setter for the directory of the logging directory.
        """
        self._log_dir = log_dir


class Parameter:
    """
    Parameter object for storing information about the contained variable.
    Attributes of large classes may pack their variables into Parameter objects,
    so they can be set with a single ParameterSetter object at initialization.
    """

    def __init__(self, name, doc):
        """
        An instance of a parameter object.
        :param name: str, name of the parameter.
        :param doc: str, description of the parameter.
        """
        self._name = name
        self._doc = doc
        self._value = None

    def __repr__(self):
        return self._name + ': { ' + self._doc + ' }'

    def __doc__(self):
        return self._doc

    @property
    def name(self):
        """
        Property for the name of the parameter.
        :return: str, name of the parameter.
        """
        return self._name

    @property
    def value(self):
        """
        Property for the value of the parameter.
        :return: value of the parameter.
        """
        if self._value is None:
            raise ValueError('Parameter value has not been set.')
        return self._value

    @value.setter
    def value(self, value):
        """
        Setter for the parameter value.
        :param value: value of the parameter.
        """
        self._value = value


class ParameterSetter:
    """
    This class handles the initialization of the given object's parameters.
    """

    def __init__(self, param_dict):
        """
        An instance of a ParameterSetter class.
        :param param_dict: dict, containing the key value pairs of the object's parameters.
                           The keys are the exact name of the attributes, which are initialized
                           as Parameter objects in the given class. Only those attributes will be
                           set, which are created this way.
        """
        try:
            for key in param_dict:
                if isinstance(param_dict[key], str) and param_dict[key] in param_dict.keys():
                    raise ValueError('Parameter initialization can not contain reference to other parameters.')

        except IndexError:
            print('Invalid value for parameter.')
            return

        self._param_dict = param_dict

    def __call__(self, obj_dict):
        """
        Invocation of a parameter object will initialize the object's Parameter type attributes.
        :param obj_dict: dict, a reference to the object's attribute dictionary (obj.__dict__).
        """
        obj_parameters = [parameter for parameter in obj_dict.keys()
                          if isinstance(obj_dict[parameter], Parameter)]

        try:
            for parameter in self._param_dict:
                if parameter in obj_parameters:
                    obj_dict[parameter].value = self._param_dict[parameter]
                else:
                    raise ValueError('Parameter is not a member of the object parameters.')

        except KeyError:
            print('Object requires a parameter (name: < {0} >), which hasn\'t '
                  'been given to the parameter dictionary.'.format(parameter))
            sys.exit()

    def __add__(self, new_parameters):
        """
        Adds parameters to the ParameterSetter object.
        By adding a dict with a value of a string, that is the key of an already existing value in the dictionary,
        the value of the new key will be set to be the same as the referenced key's value.
        :param new_parameters: dict, parameters to be added to the parameter list.
        :return: ParameterSetter, the new instance.
        """
        try:

            for key in new_parameters:
                if isinstance(new_parameters[key], str):
                    refs = new_parameters[key].split('+')
                    if len(refs) > 1:
                        new_parameters[key] = 0
                        for ref_key in refs:
                            if ref_key in self._param_dict.keys():
                                new_parameters[key] += self._param_dict[ref_key]
                            else:
                                raise KeyError()
                    else:
                        new_parameters[key] = self._param_dict[new_parameters[key]]

        except IndexError:
            print('Invalid value for parameter.')
            return

        except KeyError:
            print('The referenced parameter is not an element of the dictionary.')
            return

        self._param_dict.update(new_parameters)

        return self


class Language:
    """
    Wrapper class for the lookup tables of the languages.
    """

    def __init__(self,
                 path,
                 token,
                 trained,
                 use_cuda):
        """
        A language instance for storing the embedding and vocabulary.
        :param path: str, path of the embedding/vocabulary for the language.
        :param token: str, token, that identifies the language.
        :param trained: bool, true, if the embeddings have been pre-trained.
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._word_to_count = {}
        self._language_token = token
        self._use_cuda = use_cuda

        self.requires_grad = not trained

        self._embedding = None
        self._load_vocab(path)

    def _load_vocab(self, path):
        """
        Loads the vocabulary from a file. Path is assumed to be a text
        file, where each line contains a word and its corresponding embedding weights, separated by spaces.
        :param path: string, the absolute path of the vocab.
        """
        with open(path, 'r') as file:
            first_line = file.readline().split(' ')
            num_of_words = int(first_line[0])
            embedding_dim = int(first_line[1])
            self._embedding = numpy.empty((num_of_words + 5, embedding_dim), dtype='float')

            for index, line in enumerate(file):
                line_as_list = list(line.split(' '))
                self._word_to_id[line_as_list[0]] = index + 1  # all values are incremented by 1 because 0 is <PAD>
                self._embedding[index + 1, :] = numpy.array([float(element) for element in line_as_list[1:]],
                                                            dtype=float)

            self._word_to_id['<PAD>'] = 0
            self._word_to_id[self._language_token] = len(self._word_to_id)
            self._word_to_id['<SOS>'] = len(self._word_to_id)
            self._word_to_id['<EOS>'] = len(self._word_to_id)
            self._word_to_id['<UNK>'] = len(self._word_to_id)

            self._id_to_word = dict(zip(self._word_to_id.values(),
                                        self._word_to_id.keys()))

            self._embedding[0, :] = numpy.zeros(embedding_dim)
            self._embedding[-1, :] = numpy.zeros(embedding_dim)
            self._embedding[-2, :] = numpy.zeros(embedding_dim)
            self._embedding[-3, :] = numpy.zeros(embedding_dim)
            self._embedding[-4, :] = numpy.random.rand(embedding_dim)

            self._embedding = torch.from_numpy(self._embedding).float()

            if self._use_cuda:
                self._embedding = self._embedding.cuda()

    @property
    def embedding(self):
        """
        Property for the embedding matrix.
        :return: A PyTorch Variable object, that contains the embedding matrix
                for the language.
        """
        if self._embedding is None:
            raise ValueError('The vocabulary has not been initialized for the language.')
        return self._embedding

    @property
    def embedding_size(self):
        """
        Property for the dimension of the embeddings.
        :return: int, length of the embedding vectors (dim 1 of the embedding matrix).
        """
        if self._embedding is None:
            raise ValueError('The vocabulary has not been initialized for the language.')
        return self._embedding.shape[1]

    @property
    def vocab_size(self):
        """
        Property for the dimension of the embeddings.
        :return: int, length of the vocabulary (dim 1 of the embedding matrix).
        """
        if self._embedding is None:
            raise ValueError('The vocabulary has not been initialized for the language.')
        return self._embedding.shape[0]

    @property
    def word_to_id(self):
        """
        Property for the word to id dictionary.
        :return: dict, containing the word-id pairs.
        """
        return self._word_to_id

    @property
    def id_to_word(self):
        """
        Property for the word to id dictionary.
        :return: dict, containing the word-id pairs.
        """
        return self._id_to_word
