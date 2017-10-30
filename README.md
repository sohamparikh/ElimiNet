# ElimiNet
This repo contains the code for [ElimiNet](https://openreview.net/forum?id=B1bgpzZAZ&noteId=B1bgpzZAZ) paper submitted to ICLR 2018. 

## Dependencies
* Python 2.7
* Theano >= 0.7
* Lasagne 0.2.dev1

## Datasets
* RACE:
    Please submit a data request [here](http://www.cs.cmu.edu/~glai1/data/race/). The data will be automatically sent to you. Create a "data" directory alongside "src" directory and download the data.

* Word embeddings:
    * glove.6B.zip: [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)

## Usage
### Preprocessing
    * python preprocess.py
### Pre-trained ElimiNet
    *For training: bash train_PRE.sh
    *For testing: bash test.sh
### End-to-end ElimiNet
    *For training: bash train_E2E.sh
    *For testing bash test.sh
## Acknowledgement
*This code is adapted from  repo for [RACE baseline](https://github.com/qizhex/RACE_AR_baselines)

## Contact
*Please contact Soham Parikh (sohamp AT cse DOT iitm DOT ac DOT in) for clarification/bugs in the code.
