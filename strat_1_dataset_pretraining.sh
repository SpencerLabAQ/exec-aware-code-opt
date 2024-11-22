: '
Strategy #1: Execution Aware Pre-Training + Fine-Tuning

Model is pre-trained providing a submission code and its input trying to predict execution aware information.
Execution data is tailored to a specific test-case/input, so each instance is (code,input)-wise. The same code can appear multiple times with different input and execution data to learn.
 
The subsequent fine-tuning is performed using slow-fast pairs trying to teach the model how to generate faster code.
In this case each dataset is built considering all the test cases for each program (ref), so each instance is program-wise, i.e. each code appears one single.
'

mkdir -p logs

mkdir -p datasets/strategy_1/finetuning
mkdir -p datasets/strategy_1/pretraining

mkdir -p datasets/strategy_1/traces 

RAW_TRACES_PATH=./tracing/pretraining/tree
SEQ_FOLDER=./tracing/pretraining/sequences
PT_OUT_FOLDER=./datasets/strategy_1/pretraining
SUFFIX="pretraining"

python src/data/generate_exec_labels.py \
    --sequence-folder $SEQ_FOLDER \
    --trace-path $RAW_TRACES_PATH \
    --output-path $PT_OUT_FOLDER \
    --suffix $SUFFIX


# Generate specialized datasets

# Line Execution (LE)
python src/data/specialize.py \
    --dataset-path "${PT_OUT_FOLDER}/execaware_full_pretraining.jsonl" \
    --output-path "${PT_OUT_FOLDER}/strategy_1_LE.csv" \
    --source-col vanilla_format \
    --target-col code_exec_lbl

# Line Coverage (LC)
python src/data/specialize.py \
    --dataset-path "${PT_OUT_FOLDER}/execaware_full_pretraining.jsonl" \
    --output-path "${PT_OUT_FOLDER}/strategy_1_LC.csv" \
    --source-col vanilla_format \
    --target-col line_cov_lbl

# Branch Coverage (BC)
python src/data/specialize.py \
    --dataset-path "${PT_OUT_FOLDER}/execaware_full_pretraining.jsonl" \
    --output-path "${PT_OUT_FOLDER}/strategy_1_BC.csv" \
    --source-col vanilla_format \
    --target-col branch_covered_in_trace

# Program States (PS)
python src/data/specialize.py \
    --dataset-path "${PT_OUT_FOLDER}/execaware_full_pretraining.jsonl" \
    --output-path "${PT_OUT_FOLDER}/strategy_1_PS.csv" \
    --source-col vanilla_format \
    --target-col prog_states_lbl