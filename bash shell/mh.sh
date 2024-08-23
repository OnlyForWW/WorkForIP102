#!/bin/bash

source /data/miniconda/etc/profile.d/conda.sh

conda activate mh

export MODELVSHUMANDIR=/data/demo/model-vs-human-master/

python model-vs-human-master/examples/evaluate.py

cp /data/demo/model-vs-human-master/figures/example-figures/cue-conflict_shape-bias_matrixplot.pdf ./