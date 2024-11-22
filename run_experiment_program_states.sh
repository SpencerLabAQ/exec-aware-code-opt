#!/bin/bash

# Program States | Strategy S1
bash pretrain.sh ./datasets/strategy_1_PS.csv ./s1_pt_PS PS s1
bash finetune.sh ./datasets/strategy_1_2_ft.csv ./s1_ft_PS CO s1
bash infer.sh ./datasets/split/strategy_1_2_ft_test.csv ./s1_ft_PS s1

# Program States | Strategy S2
bash pretrain.sh ./datasets/strategy_2_PS.csv ./s2_pt_PS PS s2
bash finetune.sh ./datasets/strategy_1_2_ft.csv ./s2_ft_PS CO s2
bash infer.sh ./datasets/split/strategy_1_2_ft_test.csv ./s2_ft_PS s2

# Program States | Strategy S3
bash finetune.sh ./datasets/strategy_3_ft.csv ./s3_ft_PS CO s3
bash infer.sh ./datasets/split/strategy_3_ft_test.csv ./s3_ft_PS s3