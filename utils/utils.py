import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


def apply_noise(input_batch):
    return input_batch


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


class Parser:
    pass


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
        self.__name = name
        self.__doc = doc
        self.__value = None

    def __repr__(self):
        return self.__name + ': { ' + self.__doc + ' }'

    def __doc__(self):
        return self.__doc

    @property
    def name(self):
        """
        Property for the name of the parameter.
        :return: str, name of the parameter.
        """
        return self.__name

    @property
    def value(self):
        """
        Property for the value of the parameter.
        :return: value of the parameter.
        """
        if self.__value is None:
            raise ValueError('Parameter value has not been set.')
        return self.__value

    @value.setter
    def value(self, value):
        """
        Setter for the parameter value.
        :param value: value of the parameter.
        """
        self.__value = value


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
        self.__param_dict = param_dict

    def __call__(self, obj_dict):
        """
        Invocation of a parameter object will initialize the object's Parameter type
        attributes.
        :param obj_dict: dict, a reference to the object's attribute dictionary (obj.__dict__).
        """
        obj_parameters = [parameter for parameter in obj_dict.keys()
                          if isinstance(obj_dict[parameter], Parameter)]

        try:

            for parameter in obj_parameters:
                obj_dict[parameter].value = self.__param_dict[parameter]

        except KeyError:
            print('Object requires a parameter (name: < {0} >), which hasn\'t '
                  'been given to the parameter dictionary.'.format(parameter))
            return
