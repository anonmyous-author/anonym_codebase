#!/bin/bash
RUN_SETUP=false
NUM_REPLACE=1000
MODEL_NAME=final-models/code2seq/sri/py150/normal/model
bash experiments/adv_attack_c2s.sh 0 2 1 true 1 1 true 1 false false false test-transforms.Replace-py150 transforms.Replace 0.5 0.5 0.01 sri/py150 1 $MODEL_NAME $NUM_REPLACE 1 $RUN_SETUP

