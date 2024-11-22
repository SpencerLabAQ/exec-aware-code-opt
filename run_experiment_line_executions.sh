#!/bin/bash

# Line Executions | Strategy S1
bash pretrain.sh ./datasets/strategy_1_LE.csv ./s1_pt_LE LE s1
bash finetune.sh ./datasets/strategy_1_2_ft.csv ./s1_ft_LE CO s1
bash infer.sh ./datasets/split/strategy_1_2_ft_test.csv ./s1_ft_LE s1

# Line Executions | Strategy S2
bash pretrain.sh ./datasets/strategy_2_LE.csv ./s2_pt_LE LE s2
bash finetune.sh ./datasets/strategy_1_2_ft.csv ./s2_ft_LE CO s2
bash infer.sh ./datasets/split/strategy_1_2_ft_test.csv ./s2_ft_LE s2

# Line Executions | Strategy S3
bash finetune.sh ./datasets/strategy_3_ft.csv ./s3_ft_LE CO s3
bash infer.sh ./datasets/split/strategy_3_ft_test.csv ./s3_ft_LE s3