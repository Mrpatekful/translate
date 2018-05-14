# Configs

The application has to be provided with two main components, a model, that will be used to solve a given a problem, 
and task, which describes the problem, and the way that the model will be used to solve it.

According to the previously defined structure, the application currently accepts the model and task
descriptors in the form of **JSON** files.

There are several types of descriptors:

1. Model
2. Experiment
3. Language
4. Policy
5. Translator
6. Reguralizer

The first step to start an experiment, is to create it's configuration file.
The specific parameters, which must be provided in the configuration are described in the class-level attribute `interface` of the experiment.
All classes, which take part in the configuration assembly mechanism, must inherit from `Component` abstract base class.
This provides the required class level fields, which are the previously mentioned `interface` (`Interface` type object) and `abstract` (bool value).
During the assembly mechanism the builder object parses the provided configuration file, and detects the type of the node, which is given in the `type` entry of the **JSON**.
It then iterates through the `interface` attribute of the detected class, and fetches the required parameters from the `params` entry of the configuration file.
During this step, the builder may find an entry, that can't be given as a single parameter, since it may also be a complex component type object, such as the currently assembled experiment.
There are multiple ways to define these objects, but the most common is to create a **JSON** object, that contains the type and its parameters.
If the list of parameters is too long, or this object also contains a complex type, then it can also be a symbolic link to a file, that contains the configuration.
Another way to create a complex parameter, is to define multiple instances in the configuration file, by either packing them in a list,
or a **JSON** object. If the currently assembled object does not require any identifier paired with the complex instances,
then it is enough to pass it as a list, otherwise the builder object will create a python dictionary, containing the entries with the defined name.


### Experiment configurations

Experiment configurations must follow the format, which is described in the following examples.

```JSON
{
    "type": "<Experiment Type>",
    "params": {
 
    },
    "model_dir": "<Output Path>"
}
```

The previous **JSON** file is a general scheme for an experiment configuration file. The value of `type` entry defines the experiment,
and the `params` entry contains a **JSON** object for declaring the parameters. `model_dir` contains the location of the outputs and logs for the experiment.

The following configuration, with the corresponding interface definition may be a viable description for an experiment.

```Python
interface = Interface(**{
        'policy':               (0, Policy),
        'language_identifiers': (1, None),
        'languages':            (2, Language),
        'model':                (3, Model),
        'initial_translator':   (4, WordTranslator),
        'reguralizer':          (5, Classifier)
})
```

```JSON
{
    "type": "MergedCurriculumTranslation",
    "params": {
        "policy": "configs/utils/policies/policy.json",
        "language_identifiers": [
            "<ENG>",
            "<FRA>"
        ],
        "languages": [
            "configs/utils/languages/english.json",
            "configs/utils/languages/french.json"
        ],
        "model": "configs/models/sts.json",
        "initial_translator": "configs/utils/translators/word.json",
        "reguralizer": "configs/components/reguralizers/mlp.json"
    },
    "model_dir": "model_outputs/unmt_3"
}
```

The type of the experiment is ```MergedCurriculumTranslation```, that requires a `policy`, which is a complex parameter, 
that defines specific behaviours for the model during training, validation or testing phase. `language_identifiers` is a primitive,
that should contain a list of strings, which will identify the languages, that are used in the experiment. `languages` is a 
complex parameter, that yields its values as a list. A `Language` object defines the input pipelines and the vocabulary for a language.
`model` is the configuration for the model, that will be in the experiment. 
For further examples see the pre-defined experiment configurations in the */configs/tasks* directory.


### Model configurations

Model configurations must follow the format, which is described in the following examples.

```JSON
{
    "type": "<Model Type>",
    "params": "<Model Components"
    
}
```

The following example shows a possible configuration for the model.

```JSON
{
    "type": "SeqToSeq",
    "params": {
        "encoder":  {
            "type": "UnidirectionalRNNEncoder",
            "params": {
                "hidden_size": 100,
                "recurrent_type": "LSTM",
                "num_layers": 3,
                "optimizer_type": "Adam",
                "learning_rate": 0.01
            }
        },
        "decoder": {
            "type": "RNNDecoder",
            "params": {
                "hidden_size": 100,
                "recurrent_type": "LSTM",
                "num_layers": 3,
                "optimizer_type": "Adam",
                "learning_rate": 0.01,
                "max_length": 15
            }
        }
    }
}
```
The type for the model was defined as sequence to sequence, which requires an encoder and decoder as it's components.
*(To learn more about the required parameters for a given model, see the corresponding README.md files in the modules 
directories.)* In case of the ```SeqToSeq``` model, ```encoder``` and ```encoder``` are required nodes, and there will be an error message
indicating their absence. After parsing the defined type of components, the application will look for the parameters
required for the instantiation. The parameters shown in the example are *(currently)* sufficient for any of the 
```encoder``` or ```encoder``` type modules. 
