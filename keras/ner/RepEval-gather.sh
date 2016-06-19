#!/bin/bash

set -e
set -u

datetime="2016-05-09"

for d in CoNLL00 CoNLL03 PTB-pos; do
    # get word vector windows
    wins=$(cat logs/mlp--${d}--${datetime}--* | egrep '"wordvecs"' | perl -pe 's/.*?"wordvecs":\s+//' | egrep 'win[0-9]+\.bin' | perl -pe 's/.*win(\d+)\.bin.*/$1/' | sort -n | uniq | tr '\n' ' ')
    echo "Win values for $d: $wins" >&2
    for w in $wins; do
	# get files with word vector window
	files=$(egrep '"wordvecs":.*win'"${w}"'\.bin' logs/mlp--${d}--${datetime}--* | cut -d ':' -f 1)
	fnum=$(echo $files | egrep -o '[^[:space:]]+' | wc -l)
	# get epochs
	epochs=$(cat $files | egrep 'Ep [0-9]+ devel .*\bacc\b' | perl -pe 's/.*Ep (\d+).*/$1/' | sort -n | uniq)
	enum=$(echo $epochs | egrep -o '[^[:space:]]+' | wc -l)
	echo "$d win $w ($enum epochs/$fnum repetitions)"
	for e in $epochs; do
	    # sanity-check
	    for f in $files; do
		c=$(cat $f | egrep "Ep $e devel .*\bacc\b" | wc -l)
		if [ $c -ne 1 ]; then
		    echo "Got $c results for epoch $e in $f" >&2
		fi
	    done
	    # pick out f-score if found, accuracy otherwise
	    results=$(cat $files | egrep "Ep $e devel .*\bacc\b" | perl -pe 's/.*? (\S+)\% f .*/$1/' | perl -pe 's/.*? (\S+)\% acc .*/$1/')
	    # take average and corrected sample stdandard deviation
	    avgstdev=$(echo $results | python -c 'import numpy as np; a=map(float, raw_input().split()); print "{:.4f} {:.4f}".format(np.mean(a), np.std(a, ddof=1))')
	    echo "$e$(echo $results $avgstdev | perl -pe 's/(\S+)/\t$1/g')"
	done
    done
    #egrep '"wordvecs":.*win1.bin' logs/mlp--PTB-pos--2016-05-06--* | cut -d ':' -f
done
