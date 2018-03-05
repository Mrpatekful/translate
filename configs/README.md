# Configs

The application has to be provided with two main components, a model, that will be used to solve a given a problem, 
and task, which describes the problem, and the way that the model will be used to solve it.

According to the previously defined structure, the application currently accepts the model and task
descriptors in the form of **JSON** files.

There are two types of descriptors:

1. Model configuration
2. Task configuration


### Model configurations

Model configurations must follow the format, which is described in the following examples.

```JSON
{
    "model_type": "<Model Type>",
    "components": {
        
    }
}
```

The only nodes, which are strictly shared between any kind of model configuration is the ```model_type```, and ```components```.
This scheme provides a more general definition, so the type of models can later be easily expanded, without changing 
the format of configuration files. 

The following example shows a possible configuration for the model.

```JSON
{
    "model_type": "SeqToSeq",
    "components": {
        "encoder":  {
            "encoder_type": "RNNEncoder",
            "encoder_params": {
                "hidden_size": 10,
                "recurrent_type": "GRU",
                "num_layers": 8,
                "learning_rate": 0.01
            }
        },
        "decoder": {
            "decoder_type": "RNNDecoder",
            "decoder_params": {
                "hidden_size": 10,
                "recurrent_type": "GRU",
                "num_layers": 8,
                "learning_rate": 0.01,
                "max_length": 15,
                "tf_ratio": 0
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
```encoder``` or ```encoder``` type modules. For further guidance see the pre-defined model configurations
in the */configs/models* directory.


### Task configurations

Task configurations must follow the format, which is described in the following examples.

```JSON
{
    "task_type": "<Task Type>",
    "components": {
 
    },
    "use_cuda": "<true/false>"
}
```

The required nodes for the interpretation of the model are, the ```task_type```, ```components``` and ```use_cuda```.
Depending on the chosen task, the components may vary, but standard nodes, that probably must appear in the configuration are
the ```reader```, which describes the input stream of the data for the model, and ```data```, which provides the information
about the location of the data used for the task. ```use_cuda``` could be considered as a special parameter, since this
definition applies to the model as well.

The following configuration may be a viable description for a task.

```JSON
{
    "task_type": "UnsupervisedTranslation",
    "components": {
        "reader": {
            "source_reader": "FastReader",
            "target_reader": "FastReader",
            "max_segment_size": 1000,
            "batch_size": 32
        },
        "data": {
            "source_data": "data/eng/eng_seg",
            "source_vocab": "data/eng/eng_voc",
            "target_data": "data/eng/eng_seg",
            "target_vocab": "data/eng/eng_voc"
        },
        "regularization": {
            "regularization_type": "MLPDiscriminator",
            "discriminator_params": {
                "hidden_size": 50,
                "optimalization": "RMSProp",
                "learning_rate": 0.01
            }
        }     
    },
    "use_cuda": true
}
```

The type of the problem is ```UnsupervisedTranslation```, that requires ```reader``` with a source and
target reader object. The segment and batch size are additional information, which are always required
in ```reader``` node. The ```regularization``` is a specific component for this task, that comes from the
description of ```UnsupervisedTranslation```. For further examples see the pre-defined task configurations
in the */configs/tasks* directory.