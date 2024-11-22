#!/bin/bash


# Build pre-training dataset for S1
bash strat_1_dataset_pretraining.sh

# Build fine-tuning dataset for S1
bash strat_1_2_dataset_finetuning.sh

# Build pre-training dataset for S2
bash strat_2_dataset_pretraining.sh

# Build fine-tuning dataset for S2 (no need to run the command if it has been already run for S1)
bash strat_1_2_dataset_finetuning.sh

# Build fine-tuning dataset for S3
bash strat_3_dataset_LE_finetuning.sh
bash strat_3_dataset_LC_BC_PS_finetuning.sh