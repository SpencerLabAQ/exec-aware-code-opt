#!/bin/bash

# Line Coverage | Strategy S1
bash pretrain.sh ./datasets/strategy_1_LC.csv ./s1_pt_LC LC s1
bash finetune.sh ./datasets/strategy_1_2_ft.csv ./s1_ft_LC CO s1
bash infer.sh ./datasets/split/strategy_1_2_ft_test.csv ./s1_ft_LC s1

# Line Coverage | Strategy S2
bash pretrain.sh ./datasets/strategy_2_LC.csv ./s2_pt_LC LC s2
bash finetune.sh ./datasets/strategy_1_2_ft.csv ./s2_ft_LC CO s2
bash infer.sh ./datasets/split/strategy_1_2_ft_test.csv ./s2_ft_LC s2

# Line Coverage | Strategy S3
bash finetune.sh ./datasets/strategy_3_ft.csv ./s3_ft_LC CO s3
bash infer.sh ./datasets/split/strategy_3_ft_test.csv ./s3_ft_LC s3