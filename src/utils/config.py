"""

"""

import json
import logging
import re
import sys
import os

from src.components.encoders.rnn import Encoder

from src.components.decoders.rnn import Decoder

from src.components.utils.utils import Classifier

from src.modules.modules import WordTranslator

from src.experiments.experiments import Experiment

from src.models.models import Model

from src.utils.reader import Corpora, InputPipeline, Language, Vocabulary

from src.utils.utils import Policy, copy_dict_hierarchy, merge_dicts, subclasses

from src.utils.session import Session

from collections import OrderedDict


class Config:
    """
    Class for handling configurations of models and tasks.
    The configs are defined in JSON format files, which are parsed,
    and instantiated with the help of interface definitions of the components.
    Each node of the JSON file, that has a 'type' and 'params' key are Component type
    objects.
    """
    _base_nodes = [Encoder, Decoder, Classifier, InputPipeline,
                   Corpora, Model, Policy, WordTranslator, Language, Vocabulary]

    _experiments = subclasses(Experiment)

    _modules = {**merge_dicts(subclasses, _base_nodes), 'WordTranslator': WordTranslator,
                'Language': Language, 'Vocabulary': Vocabulary}

    @staticmethod
    def _apply_operator(args, op):
        """
        Applies the operator on the given arguments.

        :param args:
            tuple, which elements will serve as the operands.

        :param op:
            str, the operator.

        :raises:
            ValueError:
                if the operator is not defined.

        :return:
            result of the operation.
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
            raise ValueError(f'Undefined operand: {op}.')

    def __init__(self,
                 config_path:   str,
                 logging_level: int):
        """
        An instance of a configuration parser. The provided file is
        parsed and stored as a dictionary object.

        :param config_path:
            str, path of the task configuration file.
        """
        self._config = json.load(open(config_path, 'r'))
        self._registered_params = OrderedDict()
        self._logging_level = logging_level

    def assemble(self) -> tuple:
        """
        Assembles the components, described by the interface of the task from the configuration file.

        :return experiment:
            The initialized experiment object.

        :return model_dir:
            The location of the experiment outputs.
        """
        try:

            model_dir = self._config['model_dir']

            assert os.path.exists(model_dir), f'{model_dir} does not exist'

            logging.basicConfig(level=self._logging_level,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename=os.path.join(model_dir, Session.LOG_DIR),
                                filemode='w')

            experiment_type = self._experiments[self._config['type']]
            experiment = self._create_node(experiment_type, self._config['params'], 'Experiment')

        except RuntimeError as error:
            logging.error(f'{error}')
            sys.exit()

        return experiment, model_dir

    def _build_params(self, param_dict, interface_dict, config, lookup_id):
        """
        Builds a level of the interface dict. The function may do 4 different operations, based on the
        type of the value at a given key. If the value is a dictionary, the function calls itself,
        with that dictionary, and continues it's operation in that context. If the value of the interface is a
        module type class, the create_node function is called, that creates that specific type. If the value is
        none of the previous, but the key is present in the configuration file, the parameter is simply initialized
        with the value of the key from the config file. Finally, if the parameter is not in the config file, then it
        is an aggregated value, that must either be calculated, or derived from a previously created node instance.

        :param param_dict:
            dict, an dictionary with the same structure of the entity's interface dictionary.

        :param interface_dict:
            dict, the dictionary, that defines the interface of a component. Every component/module
            that is a node in the JSON file, must have a class scope method, that
            defines it's interface.

        :param config:
            dict, the currently visited node of the JSON configuration file.

        :param lookup_id:
            str, an identifier, that creates the keys for the shared parameter values.

        :return:
            dict, a fully initialized dictionary, with the same structure of the interface.
        """
        def resolve_link(link) -> dict:
            try:
                if isinstance(link, str):
                    resolved_link = json.load(open(link, 'r'))
                else:
                    return link
            except FileNotFoundError:
                logging.error(f'File not found {link}')
                return link

            return resolved_link

        for key in interface_dict:
            try:
                if isinstance(interface_dict[key], dict):
                    param_dict[key] = self._build_params(param_dict[key], interface_dict[key],
                                                         config[key], lookup_id)

                elif interface_dict[key] in self._base_nodes:
                    config[key] = resolve_link(config.get(key, None))

                    if isinstance(config.get(key, None), list):
                        param_dict[key] = []
                        for index, element in enumerate(config[key]):
                            module_dict = resolve_link(element)

                            module_type = self._modules[module_dict['type']]
                            param_dict[key].append(self._create_node(module_type, module_dict['params'],
                                                                     f'{lookup_id}:{interface_dict[key].__name__}'
                                                                     f'/{index}'))

                        logging.debug(f'Created list of {interface_dict[key].__name__} for {lookup_id}:{key}')

                    elif isinstance(config.get(key, None), dict) and 'type' not in config[key]:
                        param_dict[key] = {}

                        for index, element in enumerate(config[key]):
                            module_dict = resolve_link(config[key][element])

                            module_type = self._modules[module_dict['type']]
                            param_dict[key][element] = self._create_node(module_type, module_dict['params'],
                                                                         f'{lookup_id}:'
                                                                         f'{interface_dict[key].__name__}'
                                                                         f'/{element}')

                        logging.debug(f'Created dict of {interface_dict[key].__name__} for {lookup_id}:{key} with %s'
                                      % ', '.join(list(param_dict[key].keys())))

                    else:
                        if config.get(key, None) is not None:
                            module_dict = config[key]

                            module_type = self._modules[module_dict['type']]
                            param_dict[key] = self._create_node(module_type, module_dict['params'],
                                                                f'{lookup_id}:{interface_dict[key].__name__}')

                            logging.debug(f'Created node of {interface_dict[key].__name__} for {lookup_id}:{key}')

                elif key in config.keys():
                    param_dict[key] = config[key]
                    self._registered_params[lookup_id + f':{key}'] = param_dict[key]

                elif interface_dict[key] is not None:
                    param_dict[key] = self._resolve_aggregation(interface_dict[key])
                    self._registered_params[lookup_id + f':{key}'] = param_dict[key]
                else:
                    del param_dict[key]

            except ValueError as error:
                raise RuntimeError(f'Error occurred during construction of {lookup_id}:{key} ({error})')

        return param_dict

    def _create_node(self, entity, config, lookup_id):
        """
        Creates a given entity. The entity's interface is called, and passed to the build_params function,
        that recursively creates the required parameters for the entity. _registered_parameters
        dictionary is updated with the created entity's property values, so entities, that will
        be created later in the instantiation tree can reference these values.

        :param entity:
            Component, type class, that must have an interface() method.

        :param config:
            dict, configuration file of the entity.

        :param lookup_id:
            str, an identifier, that creates the keys for the shared parameter values.

        :return:
             Component, an instance of the passed entity.
        """
        interface = entity.interface
        param_dict = copy_dict_hierarchy(interface.dictionary)

        try:

            instance = entity(**self._build_params(param_dict, interface, config, lookup_id))

        except TypeError as error:
            raise RuntimeError(f'Error occurred during construction of {lookup_id}:{entity.__name__} ({error})')

        instance_properties = instance.properties()

        self._registered_params = {
            **self._registered_params,
            f'{lookup_id}': instance,
            **{f'{lookup_id}:{name}': instance_properties[name] for name in instance_properties}
        }

        return instance

    def _get_key(self, pattern):
        """
        Returns the key from the dictionary of registered parameters, that matches the given pattern.

        :param pattern:
            str, regexp that matches a single key from the dictionary.

        :return:
            str, the matching key
        """
        keys = [key for key in list(self._registered_params.keys()) if re.search(pattern, key) is not None]
        if len(keys) > 1:
            logging.warning(f'Ambiguous look up expression \' {pattern} \', the first matching '
                            f'key has been chosen as default \' {keys[-1]} \'')
            return keys[-1]

        if len(keys) == 0:
            raise ValueError(f'No currently registered keys match the given look up expression \' {pattern} \'')

        return keys[-1]

    def _resolve_aggregation(self, description):
        """
        Resolves the parameter references. This method is called, when a parameter is not present
        in the configuration file, so it must be calculated based on the provided description.

        :param description:
            str, the method of creating the parameter's value. The description may
            reference parameters, that have already been created, and may also apply
            operators on them. The operators are defined in the apply_operator function.

        :raise:
            ValueError:
                if the description was incorrect.

        :return:
            The value of the parameter, that was created by the interpretation of the description.
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
                raise ValueError(f'Invalid description \' {description} \'')

        else:
            raise ValueError(f'Invalid description \' {description} \'')
