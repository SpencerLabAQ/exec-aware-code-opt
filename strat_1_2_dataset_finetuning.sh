mkdir -p logs
mkdir -p tmp/sources

MAX_PARALLEL_JOBS=30

process_dataset() {

    # $1 is dataset path
    # $2 is temporary folder to store cpp sources
    # $3 is the csv output NAME (no ext)

    echo "Processing $1 dataset"

    mkdir -p $2
    
    python src/pie_to_files.py \
        --dataset $1 \
        --src-col $4 \
        --tgt-col $5 \
        --folder $2

    # parallel jobs counter
    job_i=0

    for cpp_prog in $(find ./$2 -type f -name '*.cpp')
    do
      ((job_i=job_i%MAX_PARALLEL_JOBS)); ((job_i++==0)) && wait
        gcc -fpreprocessed -dD -E -P "$cpp_prog" -o "${cpp_prog%.*}_processed.cpp" &
    done

    wait

    # reset jobs cnt
    job_i=0

    for cpp_prog in $(find ./$2 -type f -name '*_processed.cpp' | sort -t '\0' -n)
    do
        ((job_i=job_i%MAX_PARALLEL_JOBS)); ((job_i++==0)) && wait
        (
            echo $cpp_prog
            clang-format -i --style=llvm $cpp_prog 
        ) &
    done

    wait
    sleep 5

    python src/clean_ft_dataset.py \
        --folder $2 \
        --src-suffix _src_processed.cpp \
        --tgt-suffix _tgt_processed.cpp \
        --full-dataset-path $1 \
        --save-as datasets/strategy_1/finetuning/$3.csv
    
}

PIE_FOLDER=./pie
SOURCES_FOLDER=./tmp/sources

process_dataset "${PIE_FOLDER}/train.jsonl" "${SOURCES_FOLDER}/train" "strategy_1_2_ft_train" "src_code" "tgt_code"

echo "Training set completed"

process_dataset "${PIE_FOLDER}/val.jsonl" "${SOURCES_FOLDER}/val" "strategy_1_2_ft_val" "src_code" "tgt_code"

echo "Eval set completed"

process_dataset "${PIE_FOLDER}/test.jsonl" "${SOURCES_FOLDER}/test" "strategy_1_2_ft_test" "src_code" "target"

echo "Testing set completed"