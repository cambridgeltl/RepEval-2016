# RepEval-2016
Here contain the scripts and code used in Repeval 2016 paper: <br />
Intrinsic Evaluation of Word Vectors Fails to Predict Extrinsic Performance

## API Package
word2vec: original word2vec from Mikolov (https://code.google.com/archive/p/word2vec/) <br />
wvlib: lib to read word2vec file (https://github.com/spyysalo/wvlib) <br />

## Scripts
createRawText.sh: download file for creating raw corpus <br />
createCorpus.sh: Pre-process text (input: raw corpus directory) <br />
createModel.sh: Create word2vec.bin file with different window size <br />
intrinsicEva.sh: run intrinsic evaluation on 8 benchmark data-set (input: Dir. for testing vector) <br />
ExtrinsicEva.sh: run extrinsic evaluation <br />

## Code
Pre-processing: <br />
tokenize_text.py: tokenized text (need NLTK installed) <br />
sentence_spliter.py: segment sentence <br />

Intrinsic evaluation: <br />
evaluate.py: perform intrinisic evaluation <br />

Extrinsic evaluation: (Keras folder: Need either tensorflow or theano installed): <br />
mlp.py: simple feed-forward Neural Network <br />
setting.py: parameters for the Neual Network <br />

## Remark

https://drive.google.com/open?id=0BzMCqpcgEJgic0ttWTlyLWZOSVk

