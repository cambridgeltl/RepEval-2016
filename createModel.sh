#!/bin/bash

# $1 = corpus dir e.g. Big-data_tokenise_text/Big-data_tokenise.txt

mkdir result_vector

data=$1
inFilestr=${data##*/}
echo $inFilestr
# default setting from word2vec, except iter (time issue)
word2vec/word2vec -train $data -output result_vector/$inFilestr-default-skipGram.bin -size 100 -window 5 -sample 1e-3 -negative 5 -hs 0 -binary 1 -cbow 0 -iter 1 -threads 12 -min-count 5 -alpha 0.025 

#cbow vs skip-gram
word2vec/word2vec -train $data -output result_vector/$inFilestr-default-cbow.bin -size 100 -window 5 -sample 1e-3 -negative 5 -hs 0 -binary 1 -cbow 1 -iter 1 -threads 12 -min-count 5 -alpha 0.025 

#various parameter
win=("1" "2" "4" "5" "8" "16" "20" "25" "30")

for i in "${win[@]}" ; do
	echo 'window size:' $i
	echo -train $data -output result_vector/$inFilestr-win$i.bin -size 100 -window $i -sample 1e-3 -negative 5 -hs 0 -binary 1 -cbow 0 -iter 1 -threads 12 -min-count 5 -alpha 0.025
	word2vec/word2vec -train $data -output result_vector/$inFilestr-win$i.bin -size 100 -window $i -sample 1e-3 -negative 5 -hs 0 -binary 1 -cbow 0 -iter 1 -threads 12 -min-count 5 -alpha 0.025 

done

