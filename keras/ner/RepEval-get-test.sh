#!/bin/bash

for d in CoNLL00 CoNLL03 PTB-pos; do 
    echo; echo $d; 
    for w in 1 2 4 5 8 16 20 25 30; do 
	echo -n "$w"$'\t'
	files=$(egrep '"wordvecs":.*win'"${w}"'\.bin' logs/mlp--${d}--* | cut -d ':' -f 1)
	cat $files | egrep 'TEST .* acc ' | perl -pe 's/.*? (\S+)\% f .*/$1/' | perl -pe 's/.*? (\S+)\% acc .*/$1/' | awk '{ s+=$1; t++ } END { print s/t }'
    done
done
