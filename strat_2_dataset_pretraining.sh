: '
Strategy #1: Execution Aware Pre-Training + Fine-Tuning

Model is pre-trained providing a submission code and its input trying to predict execution aware information.
Execution data is tailored to a specific test-case/input, so each instance is (code,input)-wise. The same code can appear multiple times with different input and execution data to learn.
 
The subsequent fine-tuning is performed using slow-fast pairs trying to teach the model how to generate faster code.
In this case each dataset is built considering all the test cases for each program (ref), so each instance is program-wise, i.e. each code appears one single.
'

mkdir -p logs

mkdir -p datasets/strategy_2/finetuning
mkdir -p datasets/strategy_2/pretraining

mkdir -p datasets/strategy_2/traces 

RAW_TRACES_PATH=./tracing/pie/tree
SEQ_FOLDER=./tracing/pie/sequences
PT_OUT_FOLDER=./datasets/strategy_2/pretraining
SUFFIX="pretraining"


# IF YOU RAN THIS CODE FOR THE STRATEGY 1, JUST RUN THE specialize_mlm.py SCRIPT
# python src/generate_exec_labels.py \
#     --sequence-folder $SEQ_FOLDER \
#     --trace-path $RAW_TRACES_PATH \
#     --output-path $PT_OUT_FOLDER \
#     --suffix $SUFFIX


# Generate specialized datasets

# Line Execution (LE)
python src/data/specialize_mlm.py \
    --dataset-path "${PT_OUT_FOLDER}/execaware_full_pretraining.jsonl" \
    --output-path "${PT_OUT_FOLDER}/strategy_2_LE.csv" \
    --input-prefix "line_exec: " \
    --source-col vanilla_format \
    --target-col code_exec_lbl

# Line Coverage (LC)
python src/data/specialize_mlm.py \
    --dataset-path "${PT_OUT_FOLDER}/execaware_full_pretraining.jsonl" \
    --output-path "${PT_OUT_FOLDER}/strategy_2_LC.csv" \
    --input-prefix "line_cov: " \
    --source-col vanilla_format \
    --target-col line_cov_lbl

# Branch Coverage (BC)
python src/data/specialize_mlm.py \
    --dataset-path "${PT_OUT_FOLDER}/execaware_full_pretraining.jsonl" \
    --output-path "${PT_OUT_FOLDER}/strategy_2_BC.csv" \
    --input-prefix "branch_cov: " \
    --source-col vanilla_format \
    --target-col branch_covered_in_trace

# Program States (PS)
python src/data/specialize_mlm.py \
    --dataset-path "${PT_OUT_FOLDER}/execaware_full_pretraining.jsonl" \
    --output-path "${PT_OUT_FOLDER}/strategy_2_PS.csv" \
    --input-prefix "p_states: " \
    --source-col vanilla_format \
    --target-col prog_states_lbl