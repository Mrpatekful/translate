import json

from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.utils.utils import Discriminator

from utils.reader import Reader
from utils.reader import Corpora
from models.models import Model
from tasks.tasks import Task

from utils import utils

import re
import sys


class Config:
    """
    Class for handling configurations of models and tasks.
    The configs are defined in JSON format files, which are parsed,
    and instantiated with the help of interface definitions of the components.
    Each node of the JSON file, that has a 'type' and 'params' key are Component type
    objects.
    """
    _base_nodes = [Encoder, Decoder, Discriminator, Reader, Corpora, Model]

    _tasks = utils.subclasses(Task)

    _modules = utils.merge_dicts(utils.subclasses, _base_nodes)

    @staticmethod
    def _apply_operator(args, op):
        """
        Applies the operator on the given arguments.
        :param args: tuple, which elements will serve as the operands.
        :param op: str, the operator.
        :raises: ValueError: if the operator is not defined.
        :return: result of the operation.
        """
        if op == '+':
            return args[0] + args[1]
        elif op == '-':
            return args[0] - args[1]
        elif op == '*':
            return args[0] * args[1]
        elif op == '/':
            return args[0] / args[1]
        else:
            raise ValueError('Undefined operand: %s.' % op)

    def __init__(self, config):
        """
        An instance of a configuration parser. The provided file is
        parsed and stored as a dictionary object.
        :param config: str, path of the task configuration file.
        """
        self._config = json.load(open(config, 'r'))
        self._registered_params = {}

    def assemble(self):
        """
        Assembles the components, described by the interface of the task from the configuration file.
        :except ValueError: Invalid JSON configuration file.
        :return task: Task, instance of the task, that was described in the configuration file.
        """
        try:

            log_dir = self._config['log_dir']
            task_type = self._tasks[self._config['type']]
            task = self._create_node(task_type, self._config['params'], 'Task')

        except ValueError as error:
            print('Error: %s' % error)
            sys.exit()

        return task, log_dir

    def _build_params(self, param_dict, interface_dict, config, lookup_id):
        """
        Builds a level of the interface dict. The function may do 4 different operations, based on the
        type of the value at a given key. If the value is a dictionary, the function calls itself,
        with that dictionary, and continues it's operation in that context. If the value of the interface is a
        module type class, the create_node function is called, that creates that specific type. If the value is
        none of the previous, but the key is present in the configuration file, the parameter is simply initialized
        with the value of the key from the config file. Finally, if the parameter is not in the config file, then it
        is an aggregated value, that must either be calculated, or derived from a previously created node instance.
        :param param_dict: dict, an dictionary with the same structure of the entity's interface dictionary.
        :param interface_dict: dict, the dictionary, that defines the interface of a component. Every component/module
                               that is a node in the JSON file, must have a class scope method, that
                               defines it's interface.
        :param config: dict, the currently visited node of the JSON configuration file.
        :param lookup_id: str, an identifier, that creates the keys for the shared parameter values.
        :return: dict, a fully initialized dictionary, with the same structure of the interface.
        """
        for key in interface_dict:
            if isinstance(interface_dict[key], dict):
                param_dict[key] = self._build_params(param_dict[key],
                                                     interface_dict[key],
                                                     config[key],
                                                     lookup_id)

            elif interface_dict[key] in self._base_nodes:
                module_dict = config[key]
                if isinstance(module_dict, str):
                    module_dict = json.load(open(module_dict, 'r'))

                module_type = self._modules[module_dict['type']]
                param_dict[key] = self._create_node(module_type,
                                                    module_dict['params'],
                                                    lookup_id + ':%s' % interface_dict[key].__name__)

            elif key in config.keys():
                param_dict[key] = config[key]
                self._registered_params[lookup_id + ':%s' % key] = param_dict[key]

            else:
                param_dict[key] = self._resolve_aggregation(interface_dict[key])
                self._registered_params[lookup_id + ':%s' % key] = param_dict[key]

        return param_dict

    def _create_node(self, entity, config, lookup_id):
        """
        Creates a given entity. The entity's interface is called, and passed to the build_params function,
        that recursively creates the required parameters for the entity. _registered_parameters
        dictionary is updated with the created entity's property values, so entities, that will
        be created later in the instantiation tree can reference these values.
        :param entity: Component, type class, that must have an interface() method.
        :param config: dict, configuration file of the entity.
        :param lookup_id: str, an identifier, that creates the keys for the shared parameter values.
        :return: Component, an instance of the passed entity.
        """
        interface = entity.interface
        param_dict = utils.copy_dict_hierarchy(interface)

        instance = entity(**utils.create_leaf_dict(self._build_params(param_dict, interface, config, lookup_id)))
        instance_properties = instance.properties()

        self._registered_params = {
            **self._registered_params,
            **{'%s:%s' % (lookup_id, name): instance_properties[name] for name in instance_properties}
        }

        return instance

    def _get_key(self, pattern):
        """
        Returns the key from the dictionary of registered parameters, that matches the given pattern.
        :param pattern: str, regexp that matches a single key from the dictionary.
        :raises: ValueError: if the regexp matches too many, or zero keys from the dictionary.
        :return: str, the matching key
        """
        keys = [key for key in list(self._registered_params.keys()) if re.search(pattern, key) is not None]
        if len(keys) > 1:
            raise ValueError('Ambiguous look up expression \' %s \'' % pattern)

        if len(keys) == 0:
            raise ValueError('No currently registered keys match the given look up expression \' %s \'' % pattern)
        return keys[0]

    def _resolve_aggregation(self, description):
        """
        Resolves the parameter references. This method is called, when a parameter is not present
        in the configuration file, so it must be calculated based on the provided description.
        :param description: str, the method of creating the parameter's value. The description may
                            reference parameters, that have already been created, and may also apply
                            operators on them. The operators are defined in the apply_operator function.
        :raises ValueError: if the description was incorrect.
        :return: The value of the parameter, that was created by the interpretation of the description.
        """
        def is_number(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        if isinstance(description, str):
            expressions = description.split(' ')

            if len(expressions) == 3:
                x = int(expressions[0]) if is_number(expressions[0]) \
                    else self._registered_params[self._get_key(expressions[0])]

                y = int(expressions[2]) if is_number(expressions[2]) \
                    else self._registered_params[self._get_key(expressions[2])]

                return self._apply_operator((x, y), expressions[1])

            elif len(expressions) == 1:
                return self._registered_params[self._get_key(expressions[0])]

            else:
                raise ValueError('Invalid description \' %s \'' % description)

        else:
            raise ValueError('Description must be str.')
