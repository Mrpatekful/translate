#!/usr/bin/env bash

# tokenization and length filtering of the lines
python process.py

# synchronization of the vocabularies and the corpus
python synchronize.py

# validation and correction of the provided vocabularies
python validate.py
