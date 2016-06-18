#!/bin/bash

# define $1 (raw text directory)
# change your python directory for the sentence splitter and tokenized text
python sentence_spliter.py $1 senSplit.txt
python tokenize_Text.py senSplit.txt tokenize.txt

