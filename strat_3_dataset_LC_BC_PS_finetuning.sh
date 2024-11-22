#!/bin/bash

# exit on errors
# set -e

# activate your environment
source .env/bin/activate

SEQUENCE_PATH=$1
NAME=$2
OUTPUT_PATH=$3
TRACE_PATH=$4
PIE_PATH="./pie/$NAME.jsonl"

mkdir -p "$OUTPUT_PATH/tmp"
mkdir -p datasets

python src/data/s3_ft_exec_aware_format.py --pie-path $PIE_PATH --sequence-folder $SEQUENCE_PATH --trace-path $TRACE_PATH --output-path "$OUTPUT_PATH" --suffix $NAME

process_dataset() {

    # $1 is dataset path
    # $2 is temporary folder to store cpp sources
    # $3 is the csv output NAME (no ext)

    echo "Processing $1 dataset"

    mkdir -p $2
    
    python src/data/generate_c_sources.py \
        --dataset $1 \
        --src-col source \
        --tgt-col target \
        --folder $2

    # set max jobs in parallel
    N=30
    job_i=0

    for cpp_prog in $(find ./$2 -type f -name '*.cpp')
    do
      ((job_i=job_i%N)); ((job_i++==0)) && wait
        gcc -fpreprocessed -dD -E -P "$cpp_prog" -o "${cpp_prog%.*}_processed.cpp" &
    done

    wait

    # reset jobs cnt
    job_i=0

    for cpp_prog in $(find ./$2 -type f -name '*_processed.cpp' | sort -t '\0' -n)
    do
        ((job_i=job_i%N)); ((job_i++==0)) && wait
        (
            echo $cpp_prog
            clang-format -i --style=llvm $cpp_prog 
        ) &
    done

    wait
    sleep 5

    python src/ft/remove_comments/create_c_sources.py \
        --folder ./$2 \
        --src-suffix _src_processed.cpp \
        --tgt-suffix _tgt_processed.cpp \
        --full-dataset-path $1 \
        --save-as datasets/$3.csv
    
}

# Preprocess source code removing comments and custom format

echo "Processing exec aware ${NAME} set"
process_dataset "$3/execaware_${NAME}.jsonl" "tmp_execaware_${NAME}" "execaware_${NAME}_processed"

python src/data/s3_specialize.py \
    --dataset-path "datasets/execaware_${NAME}_processed.csv" \
    --output-path "datasets/LE_execaware_${NAME}.csv" \
    --id-col id \
    --source-col source_line_exec \
    --target-col target

echo "LE dataset generated"

python src/data/s3_specialize.py \
    --dataset-path "datasets/execaware_${NAME}_processed.csv" \
    --output-path "datasets/LC_execaware_${NAME}.csv" \
    --id-col id \
    --source-col source_line_cov \
    --target-col target

echo "LC dataset generated"

python src/data/s3_specialize.py \
    --dataset-path "datasets/execaware_${NAME}_processed.csv" \
    --output-path "datasets/BC_execaware_${NAME}.csv" \
    --id-col id \
    --source-col source_bran_cov \
    --target-col target

echo "BC dataset generated"

python src/data/s3_specialize.py \
    --dataset-path "datasets/execaware_${NAME}_processed.csv" \
    --output-path "datasets/PS_execaware_${NAME}.csv" \
    --id-col id \
    --source-col source_prog_states \
    --target-col target

echo "PS dataset generated"

python src/data/s3_specialize.py \
    --dataset-path "datasets/execaware_${NAME}_processed.csv" \
    --output-path "datasets/vanilla_format_${NAME}.csv" \
    --id-col id \
    --source-col source_vanilla \
    --target-col target

echo "VANILLA FORMAT dataset generated"

python src/data/s3_specialize.py \
    --dataset-path "datasets/execaware_${NAME}_processed.csv" \
    --output-path "datasets/vanilla_${NAME}.csv" \
    --id-col id \
    --source-col source \
    --target-col target

echo "VANILLA dataset generated"
# echo "Processing vanilla ${NAME} set"
# process_dataset "$3/vanilla_${NAME}.jsonl" "tmp_vanilla_${NAME}" "vanilla_${NAME}_processed"

# python src/ft/ft_merge_traced_pie.py \
#     --original-path /mnt/data/ExecAwarePT/dataset/exec_aware_ft/exec_aware_${NAME}.jsonl \
#     --processed-path datasets/exec_aware_${NAME}_processed.csv \
#     --output-file datasets/FT_exec_aware_${NAME}.csv

# cp datasets/vanilla_${NAME}_processed.csv datasets/FT_vanilla_${NAME}.csv

# # remove tmp files
# rm -f datasets/vanilla_${NAME}_processed.csv
# rm -f datasets/exec_aware_${NAME}_processed.csv

# generate datasets
