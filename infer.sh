#!/bin/bash

TEST_DATASET_PATH=$1
TEST_OUT_PATH=$2
$STRATEGY = $3

# Assign script folder based on strategy value
if [ "$STRATEGY" == "s1" ]; then
    SCRIPT_FOLDER="train_s1"
elif [ "$STRATEGY" == "s2" ]; then
    SCRIPT_FOLDER="train_s2"
elif [ "$STRATEGY" == "s3" ]; then
    SCRIPT_FOLDER="train_s3"
else
    SCRIPT_FOLDER="train_s1"
fi

#TEST_DATASET_PATH="./datasets/split/exec_aware_test.csv"
#TEST_OUT_PATH="./test_data"

# consts
NUM_PROC=32
DATA_NUM=-1
MAX_SOURCE_LEN=512
MAX_TARGET_LEN=512
BATCH_SIZE=6

CHECKPOINT="${TEST_OUT_PATH}/saved_models/summarize_python/checkpoint-3232"

mkdir -p logs

# paths
CACHE_DATA="${TEST_OUT_PATH}/cache_data/summarize_python"
SAVE_DIR="${TEST_OUT_PATH}/saved_models/summarize_python"

echo "Running inference ..."

sbatch launcher_inference.sh src/$SCRIPT_FOLDER/predict.py \
      --data-num $DATA_NUM \
      --max-source-len $MAX_SOURCE_LEN \
      --max-target-len $MAX_TARGET_LEN \
      --cache-data $CACHE_DATA \
      --save-dir $SAVE_DIR \
      --load $CHECKPOINT \
      --ds-test-path $TEST_DATASET_PATH \
      --batch-size $BATCH_SIZE \
      --num-proc $NUM_PROC \
      --remove-long-samples --debug --fp16
