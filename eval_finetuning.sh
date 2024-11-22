#!/bin/bash

DATASET=$1 #"predictions/strategy_3_LC_pred.csv"
SANDBOX_NAME=$1 #"strategy_3_LC_pred"

python src/eval/eval_finetuning_input.py --pred-data $DATASET --sandbox-name $SANDBOX_NAME --remove-tokens