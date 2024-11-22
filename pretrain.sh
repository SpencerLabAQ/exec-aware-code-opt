#!/bin/bash

DATASET_PATH=$1
TRAIN_OUT_PATH=$2
# ENV_PATH=$3
TASK=$3
$STRATEGY = $4

# consts
NUM_PROC=32
DATA_NUM=-1
MAX_SOURCE_LEN=512
MAX_TARGET_LEN=512

# training
EPOCHS=1
LR=5e-5
BATCH_SIZE=16
LOCAL_RANK=-1
DEEPSPEED=None

MODEL="Salesforce/codet5p-220m"

# Create log dir
mkdir -p logs

DATASET_DIR=$(dirname "$DATASET_PATH")
DATASET_NAME=$(basename "$DATASET_PATH")
DATASET_NAME=${DATASET_NAME%.*}

# paths
mkdir -p $TRAIN_OUT_PATH
CACHE_DATA="${TRAIN_OUT_PATH}/cache_data/summarize_python"
SAVE_DIR="${TRAIN_OUT_PATH}/saved_models/summarize_python"

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


echo "Splitting ..."
# generate splits
python src/$SCRIPT_FOLDER/split.py \
      --filepath $DATASET_PATH \
      --ignore-test

TRAIN_DS="${DATASET_DIR}/split/${DATASET_NAME}_train.csv"
VALIDATION_DS="${DATASET_DIR}/split/${DATASET_NAME}_val.csv"

echo "Running training ..."
python src/$SCRIPT_FOLDER/train.py \
      --data-num $DATA_NUM \
      --max-source-len $MAX_SOURCE_LEN \
      --max-target-len $MAX_TARGET_LEN \
      --cache-data $CACHE_DATA \
      --save-dir $SAVE_DIR \
      --load $MODEL \
      --epochs $EPOCHS \
      --lr $LR \
      --ds-train-path $TRAIN_DS \
      --ds-val-path $VALIDATION_DS \
      --batch-size-per-replica $BATCH_SIZE \
      --local_rank $LOCAL_RANK \
      --deepspeed $DEEPSPEED \
      --num-proc $NUM_PROC \
      --remove-long-samples \
      --fp16 \
      --task $TASK \
      --debug