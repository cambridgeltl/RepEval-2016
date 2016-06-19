#!/bin/bash

# srun PTB data with various word vectors.
# Run this script with sbatch.


SCRIPT="python mlp.py --test"
WORDVECS=wordvecs-RepEval-final/*.bin
DATA=data/ner/PTB-pos

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
