
# MODELS_IN=final-models/code2seq/sri/py150/normal GPU=0 DATASET=sri/py150 DEPTH=1 IS_ADV=false IS_AUG=false ./scripts/eval-model-code2seq.sh
DATASET=sri/py150

GPU=1 \
ARGS="--no-attack" \
DATASET_NAME=datasets/preprocessed/ast-paths/sri/py150/ \
RESULTS_OUT=final-results/code2seq/${DATASET}/normal-model/no-attack \
DATASET_SHORT_NAME=py150 \
MODELS_IN=final-models/code2seq/$DATASET/normal/model  \
  time make test-model-code2seq
