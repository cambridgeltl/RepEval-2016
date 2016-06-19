#!/bin/bash

source ~/env/theano/bin/activate

for i in `seq 10`; do
    echo $i
    for d in CoNLL00 CoNLL03 PTB; do 
	sbatch RepEval-${d}.sh
	sleep 60
    done
done
