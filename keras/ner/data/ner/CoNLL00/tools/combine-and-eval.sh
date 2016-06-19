#!/bin/bash

# Combine given predictions with CoNLL development data and run evaluation.

set -e
set -u 

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONLL_DIR="$SCRIPT_DIR/../original-data/"

python "$SCRIPT_DIR"/combine.py "$CONLL_DIR"/devel.txt $1 | \
    "$SCRIPT_DIR/conlleval"
