#!/bin/bash

# srun CoNLL'00 data with various word vectors.
# Run this script with sbatch.


SCRIPT="python mlp.py --test"
WORDVECS=wordvecs-RepEval-final/*.bin
DATA=data/ner/CoNLL00

set -e
set -u

for w in $WORDVECS; do
    sleep 1
    CMD="$SCRIPT $DATA $w --verbosity 0"
    echo "Executing $CMD"
    srun --exclusive -n 1 bash -c \
"
echo \"--- START $DATA $w \$(date) ---\"
$CMD
echo \" --- END $DATA $w \$(date) ---\"
" &
done

wait
