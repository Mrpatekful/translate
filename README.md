# Pytorch-MT

## Introduction

Pytroch-MT is a framework for creation and evaluation of neural machine translation algorithms. The purpose of this project is to provide a general interface for experiments with different methods, but with an emphasis on implementing a [particular translation approach](https://arxiv.org/abs/1711.00043). 


## Data

For training these machine translation models, large text corpora is required. The scripts provided for preprocessing data were tested on WMT-2014 english and french corpora. generate.py contains the sequence of functions, which creates the data with the vocabulary and alignment files that are required for the already implemented translation experiment.


## Usage

The entry point of the application is the main.py file.

After constructing the experiment configuration files (described in this [page](https://github.com/Mrpatekful/nmt-BMEVIAUAL01/blob/master/configs/README.md)), the training will start by `python main.py <config> -t`. Upon interruption, the model will start from the latest epoch, or in case of a non memory related error, the training will continue from the latest state in an epoch. 

By `python main.py <config> -t -c` the training will always start from an untrained state, and deletes all previous state and output files.

After training the model for sufficient number of steps, it can be evaluated or tested `python main.py <config> --test` or `python main.py <config> --evaluate`.

For visualization of the outputs, see [model-evaluation](https://github.com/Mrpatekful/nmt-BMEVIAUAL01/blob/master/model-evaluation.ipynb) notebook.


## Dependencies

* Python 3.6
* [PyTorch v0.4](https://pytorch.org/)
* [NumPy v1.14](https://www.scipy.org/scipylib/download.html)
* [NLTK v3.3](https://www.nltk.org/install.html)
* [TQDM v4.23](https://pypi.org/project/tqdm/)
* [Matplotlib v2.2.2](https://matplotlib.org/users/installing.html)

## Documentation

For more detailed information, see [Wiki](https://github.com/Mrpatekful/nmt-BMEVIAUAL01/wiki) or the [Documentation](https://github.com/Mrpatekful/nmt-BMEVIAUAL01/blob/master/docs.pdf).


