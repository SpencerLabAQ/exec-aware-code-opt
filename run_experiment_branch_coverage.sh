#!/bin/bash

# Branch Coverage | Strategy S1
bash pretrain.sh ./datasets/strategy_1_BC.csv ./s1_pt_BC BC s1
bash finetune.sh ./datasets/strategy_1_2_ft.csv ./s1_ft_BC CO s1
bash infer.sh ./datasets/split/strategy_1_2_ft_test.csv ./s1_ft_BC s1

# Branch Coverage | Strategy S2
bash pretrain.sh ./datasets/strategy_2_BC.csv ./s2_pt_BC BC s2
bash finetune.sh ./datasets/strategy_1_2_ft.csv ./s2_ft_BC CO s2
bash infer.sh ./datasets/split/strategy_1_2_ft_test.csv ./s2_ft_BC s2

# Branch Coverage | Strategy S3
bash finetune.sh ./datasets/strategy_3_ft.csv ./s3_ft_BC CO s3
bash infer.sh ./datasets/split/strategy_3_ft_test.csv ./s3_ft_BC s3