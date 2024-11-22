#!/bin/bash

DATASET_PATH=$1

# Create log dir
mkdir -p logs

echo "Splitting ..."
# generate splits
python src/data/split.py \
      --filepath $DATASET_PATH \
      --ignore-test