#!/bin/bash

# $1 = folder for testing word2vec vector.bin

home=$(pwd)/keras/
dataPath=$home'/ner/data/ner'
echo $home

#DIR=$home'/ner/evaluation' #"$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#echo $DIR
FILES=$(find $1 -type f -name '*.bin')
for file in $FILES
do
	for f in $dataPath/CoNLL03 $dataPath/CoNLL00 $dataPath/PTB-pos
	do 
	echo $f
	python $home/ner/mlp.py $f $file  
	done
done

# result located in log and prediction file
wait