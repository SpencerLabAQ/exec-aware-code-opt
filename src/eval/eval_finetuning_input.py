"""
Author: Federico Di Menna
Year:   2024

Definition of the evaluation environment for the model generated output.
The script only requires metrics.py and simulate.py files to compute the evaluation metrics.

This _input version simulates also the source of the input program.

python packages:
- pathlib, tqdm, pandas, jsonlines

Each output will be check in terms of:
- (i)   Correctness -> compilation
- (ii)  Correctness -> expected output
- (iii) Fully correctness -> (i) and (ii) 
- (iv)  Execution Time -> %OPT: percentage of test set improved at least of 10%
- (v)   Execution Time -> Speedup: input_time / output_time (only correct ones) 
- (vi)  String similarity with labels
"""

import os
import logging
from pathlib import Path 
from tqdm import tqdm
import pandas as pd
import configparser
import jsonlines
import subprocess
import threading
import re
import argparse
import multiprocessing
import json

from metrics import add_correctness_col, generate_submission_dataset

KEYS_TO_SAVE = [
    "submission_id",
    "problem_id",
    "user_id",
    "simulation_results"
]

TOKENS_TO_REMOVE = ["<ne>", "<e>", "<E>", "<e+>", "<E+>", "<NB>", "<BC>", "<BNC>", "optimize: ", "// <ne>", "// <e>", "// <E>", "// <e+>", "// <E+>", "// <NB>", "// <BC>", "// <BNC>"]

class SafeThread(threading.Thread):
    def __init__(self, cmd, timeout, redirection = False):
        threading.Thread.__init__(self)
        self.cmd = cmd
        self.timeout = timeout
        self.redirection = redirection

    def run(self):
        if self.redirection:
            # print("Running", self.cmd)
            self.p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            self.p = subprocess.Popen(self.cmd)
        self.p.wait()
    
    def Run(self):
        self.start()
        self.join(self.timeout)

        if self.is_alive():
            logging.info(f"Terminating process from command {str(self.cmd)}")
            self.p.terminate()
            self.join()

def __preprocess(pred_df):
    pred_df["problem_id"] = pred_df["id"].apply(lambda id: id.split("#")[0])
    pred_df["submission_id_v0"] = pred_df["id"].apply(lambda id: id.split("#")[-1].split("_")[0]) 
    pred_df["submission_id_v1"] = pred_df["id"].apply(lambda id: id.split("#")[-1].split("_")[1])
    return pred_df

def __generate_source_files(pred_data, src_code_path, code_col = 'prediction', mode = 'input'):
    filename = f"{pred_data['problem_id']}_{pred_data['submission_id_v0']}_{mode}.cpp"
    if not (src_code_path / filename).exists():
        with open(src_code_path / filename, "w") as prog_file:
            prog_file.write(pred_data[code_col])
    return filename

def __compile(save_as, compile_cmd, paths_dict):
    if not (paths_dict.get("compiled_code_path") / save_as).exists():
        # proc = subprocess.Popen(
        #     compile_cmd,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.STDOUT,
        # )
        SafeThread(compile_cmd, timeout=10, redirection=True).Run()
        # stdout_data, stderr_data = proc.communicate()
        # return_code = proc.wait()
        # return return_code
    return 0

def __check_stat_file(path):
    with open(path, "r") as f:
        content = f.read()
        if content != "":
            return True
        print(f"{path} is empty")
        return False

def __compile_source_files(params):
    pred_data, shared_list, paths_dict = params

    # C Version
    # c_std = "c99"
    # c_comp = "gcc"

    # C++ version
    c_std = "c++17"
    c_comp = "g++"
    compiled_name_input = f"{Path(pred_data['filename_input']).stem}.out"
    compile_cmd_input = [f"{c_comp}", f"-std={c_std}", "-O3", paths_dict.get("src_code_path") / pred_data['filename_input'], "-o", paths_dict.get("compiled_code_path") / compiled_name_input]
    
    compiled_name_tgt = f"{Path(pred_data['filename_tgt']).stem}.out"
    compile_cmd_tgt = [f"{c_comp}", f"-std={c_std}", "-O3", paths_dict.get("src_code_path") / pred_data['filename_tgt'], "-o", paths_dict.get("compiled_code_path") / compiled_name_tgt]
    
    ret_code_input = __compile(
        compiled_name_input,
        compile_cmd_input,
        paths_dict
        )
    ret_code_tgt = __compile(
        compiled_name_tgt,
        compile_cmd_tgt,
        paths_dict
        )
    
    pred_data.update({
        "compiled_name_input": compiled_name_input, 
        "compiled_succ_input" : (paths_dict.get("compiled_code_path") / compiled_name_input).exists(),
        "compiled_name_tgt": compiled_name_tgt, 
        "compiled_succ_tgt" : (paths_dict.get("compiled_code_path") / compiled_name_tgt).exists()
        })
    # if not (compiled_code_path / compiled_name).exists():
    #     print(f"Compilation failed for {compile_cmd}")
    shared_list.append(pred_data)

def __simulate(paths_dict, problem_id, submission_id, input_id, compiled_name, mode = "input"):
    prefix = f"{problem_id}_{submission_id}_{input_id}_{mode}"
    simulation_cmd = [f"{paths_dict.get('gem5_run_path')}",
            "-q", 
            f"--outdir={paths_dict.get('simul_code_path') / 'stats'}", 
            f"--stats-file={prefix}_gem5_stats.txt",
            "--silent-redirect", 
            "-r",
            f"--stdout-file={(paths_dict.get('simul_code_path') / 'output' / f'{prefix}_gem5_output.txt').resolve()}",
            "-e",
            f"--stderr-file={(paths_dict.get('simul_code_path') / 'output' / f'{prefix}_gem5_error.txt').resolve()}",
        #   f"--debug-file={paths_dict.get('simul_code_path') / debug_filename}",
            "./src/eval/simulate.py",
            # "./src/data/simulate_skylake.py",
            f"{paths_dict.get('compiled_code_path') / compiled_name}",
            f"{paths_dict.get('problems_input_path') / problem_id / f'input_{input_id}.txt'}"
            ]
    SafeThread(simulation_cmd, timeout=500, redirection=True).Run()

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def __filter_output_content(content):
    lines = content.split("\n")
    to_return = []
    for line in lines:
        if isinstance(line, str) and "Global freq" in line:
            continue
        # if is_float(line):
        #     to_return.append(str(float(line)))
        else:
            to_return.append(line)
    return "\n".join(to_return)

def __run_simulation(params):
    pred_data, shared_list, paths_dict = params
    
    problem_id = pred_data["problem_id"]
    submission_id_input = pred_data["submission_id_v0"]
    # submission_id_tgt = pred_data["submission_id_v1"]
    input_id = pred_data["input_id"]
    stats_filename_input = f"{problem_id}_{submission_id_input}_{input_id}_input_gem5_stats.txt"
    stats_filename_tgt = f"{problem_id}_{submission_id_input}_{input_id}_tgt_gem5_stats.txt"
    
    # Simulate input
    if pred_data['compiled_succ_input'] and (not (paths_dict.get("simul_code_path") / 'stats' / stats_filename_input).exists() or not __check_stat_file(paths_dict.get("simul_code_path") / 'stats' / stats_filename_input)):
        if (paths_dict.get("compiled_code_path") / pred_data["compiled_name_input"]).exists():
            # print(f"Simulating {problem_id}-{submission_id_input}-input")
            __simulate(paths_dict, problem_id, submission_id_input, input_id, pred_data["compiled_name_input"], mode = "input")
        else:
            print("[INPUT] file", paths_dict.get("compiled_code_path") / pred_data["compiled_name_input"], "does not exists")

    # Simulate target (optimized)
    if pred_data['compiled_succ_tgt'] and (not (paths_dict.get("simul_code_path") / 'stats' / stats_filename_tgt).exists()  or not __check_stat_file(paths_dict.get("simul_code_path") / 'stats' / stats_filename_tgt)):
        if (paths_dict.get("compiled_code_path") / pred_data["compiled_name_tgt"]).exists():
            # print(f"Simulating {problem_id}-{submission_id_input}-tgt")
            __simulate(paths_dict, problem_id, submission_id_input, input_id, pred_data["compiled_name_tgt"], mode = "tgt")
        else:
            print("[TGT] file", paths_dict.get("compiled_code_path") / pred_data["compiled_name_tgt"], "does not exists")

    input_stdout = None
    tgt_stdout = None
    
    try:
        if (paths_dict.get("simul_code_path") / 'output' / f"{problem_id}_{submission_id_input}_{input_id}_input_gem5_output.txt").exists():
            with open(paths_dict.get("simul_code_path") / 'output' / f"{problem_id}_{submission_id_input}_{input_id}_input_gem5_output.txt", "r") as input_stdout_file:
                input_stdout_original = input_stdout_file.read()
                input_stdout = __filter_output_content(input_stdout_original)
        if (paths_dict.get("simul_code_path") / 'output' / f"{problem_id}_{submission_id_input}_{input_id}_tgt_gem5_output.txt").exists():
            with open(paths_dict.get("simul_code_path") / 'output' / f"{problem_id}_{submission_id_input}_{input_id}_tgt_gem5_output.txt", "r") as tgt_stdout_file:
                tgt_stdout = tgt_stdout_file.read()
                tgt_stdout = __filter_output_content(tgt_stdout)
    except Exception as ex:
        print(str(ex))

    # if str(input_stdout).strip().replace("\n", "") != str(__filter_output_content(pred_data["exp_output"])).strip().replace("\n", ""):
    #     print("\n\nIncorrect", pred_data["id"], input_stdout_original, "Input STDOUT", input_stdout, "Expected STDOUT", pred_data["exp_output"])
    
    pred_data.update({
        "simul_name_input": stats_filename_input, 
        "simul_succ_input": (paths_dict.get("simul_code_path") / 'stats' / stats_filename_input).exists() and __check_stat_file(paths_dict.get("simul_code_path") / 'stats' / stats_filename_input),
        "simul_stdout_input": input_stdout,
        "simul_time_input": get_simulation_seconds(f"{paths_dict.get('simul_code_path') / 'stats' / stats_filename_input}") if (paths_dict.get('simul_code_path') / 'stats' / stats_filename_input).exists() else None
        })
    pred_data.update({
        "simul_name_tgt": stats_filename_tgt, 
        "simul_succ_tgt": (paths_dict.get("simul_code_path") / 'stats' / stats_filename_tgt).exists() and __check_stat_file(paths_dict.get("simul_code_path") / 'stats' / stats_filename_tgt),
        "simul_stdout_tgt": tgt_stdout,
        "simul_time_tgt": get_simulation_seconds(f"{paths_dict.get('simul_code_path') / 'stats' / stats_filename_tgt}") if (paths_dict.get('simul_code_path') / 'stats' / stats_filename_tgt).exists() else None
        })
    shared_list.append(pred_data)

def __parallel_fn(fn, shared_list, df_records, paths_dict, nproc = 1, desc="Parallel execution"):
    
    iterations = [(item, shared_list, paths_dict) for item in df_records]

    with multiprocessing.Pool(nproc) as p:
        with tqdm(total=len(iterations), desc=desc) as pbar:
            for _ in p.imap_unordered(fn, iterations):
                pbar.update()

def __print_metadata(df, df_name):
    print(f"Analysis of {df_name} dataset")
    print("Size: ", len(df))

def load_process_dataset(path_dict, remove_tokens = False, input_col = 'input'):

    pred_df = None

    if path_dict.get("df_test_path").suffix == ".csv":
        pred_df = pd.read_csv(path_dict.get("df_test_path"))
    elif path_dict.get("df_test_path").suffix == ".jsonl":
        data = []
        with jsonlines.open(path_dict.get("df_test_path"), "r") as pred_file:
            data = [row for row in pred_file]
        assert len(data) > 0, "No rows in jsonl dataset"
        pred_df = pd.DataFrame(data)

    assert pred_df is not None, f"Exception occured while loading evaluation dataset. No dataframe loaded"
    
    if remove_tokens:
        print("Removing tokens from input")
        for token in TOKENS_TO_REMOVE:
            pred_df[input_col] = pred_df[input_col].astype(str)
            escaped_token = re.escape(token)
            pred_df[input_col] = pred_df[input_col].str.replace(escaped_token, "", regex=True)
    return __preprocess(pred_df)

def get_simulation_seconds(results_file_path):
    try:
        with open(results_file_path, "r") as res_file:
            lines = res_file.readlines()
            if len(lines):
                key, val = lines[2].strip().split(maxsplit=1)
                if key == "simSeconds":
                    # return simSeconds value
                    return val.split(" ")[0]
            else:
                return -1
    except Exception as ex:
        # logging.error("Exception during opening file", results_file_path)
        # -1 value means error during simulation results parsing
        return -2

def generate_source_code_files(pred_df_lst, paths_dict, input_col = 'input', tgt_col = 'prediction'):
    for row in tqdm(pred_df_lst, total = len(pred_df), desc="Generating source code files"):
        row[f"filename_input"] = __generate_source_files(row, src_code_path = paths_dict.get("src_code_path"), code_col = input_col, mode = "input")
        row[f"filename_tgt"] = __generate_source_files(row, src_code_path = paths_dict.get("src_code_path"), code_col = tgt_col, mode = "tgt")

def compile_all_programs(pred_df_lst, paths_dict, nproc = 35):
    # # parallel compilation of source files
    shared_list = multiprocessing.Manager().list()

    __parallel_fn(
        fn=__compile_source_files,
        shared_list=shared_list, 
        df_records=pred_df_lst, 
        paths_dict = paths_dict,
        nproc=nproc,
        desc="Compiling generated source codes"
        )

    return list(shared_list)

def get_input_output(paths_dict):
    data = None
    with open(paths_dict.get("input_output_path"), "r") as in_out_file:
        data = json.load(in_out_file)

    assert data is not None, f"Error during reading input output file"
    return data  

def generate_dataset_with_inputs(pred_df_lst, paths_dict):
    # load inputs / outputs
    io_data = get_input_output(paths_dict=paths_dict)
    pred_df_inputs = []
    for row in tqdm(pred_df_lst, total = len(pred_df_lst), desc="Building dataset with inputs"):
        problem_id = row["problem_id"]
        for input_id, input_data in enumerate(io_data[problem_id]["sample_input"]):
            item = row.copy()
            item["input_id"] = input_id
            item["input_data"] = input_data
            item["exp_output"] = io_data[problem_id]["sample_output"][input_id]
            pred_df_inputs.append(item)
    return pred_df_inputs

def simulate_all_programs(pred_df_lst, nproc = 35):
    shared_list = multiprocessing.Manager().list()

    # simulation
    __parallel_fn(
        fn=__run_simulation,
        shared_list=shared_list, 
        df_records=pred_df_lst, 
        paths_dict = paths_dict,
        nproc=nproc,
        desc="Simulating generated source codes"
        )
    
    return list(shared_list)

def setup():
    # logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f"logs/{Path(__file__).stem}.log", 
        filemode='w', 
        encoding='utf-8', 
        level=logging.INFO)
    
    # args
    parser = argparse.ArgumentParser(description="Evaluation of generated optimized code")
    parser.add_argument('--pred-data', default='./predictions.csv', type=str)
    parser.add_argument('--sandbox-name', default='ft_eval', type=str)
    parser.add_argument("--remove-tokens", default=False, action='store_true')

    args = parser.parse_args()

    # -- config paths -- 
    config = configparser.ConfigParser()
    config.read("./config/config.ini")

    paths_dict = {
        "problems_input_path": Path(config["paths"]["all_input_output_path"]),
        "gem5_run_path": Path(config["paths"]["gem5_path"]),
        "eval_sandbox_path": Path(config["paths"]["eval_sandbox_path"]) / args.sandbox_name,
        "input_output_path": Path(config["paths"]["all_input_output_path"]).parent / "all_input_output.json"
    }

    paths_dict.update({
        "src_code_path": paths_dict.get("eval_sandbox_path") / "src",
        "compiled_code_path": paths_dict.get("eval_sandbox_path") / "compiled",
        "simul_code_path": paths_dict.get("eval_sandbox_path") / "simul"
    })

    paths_dict.get("src_code_path").mkdir(parents=True, exist_ok=True)
    paths_dict.get("compiled_code_path").mkdir(parents=True, exist_ok=True)
    paths_dict.get("simul_code_path").mkdir(parents=True, exist_ok=True)
    (paths_dict.get("simul_code_path") / "stats").mkdir(parents=True, exist_ok=True)
    (paths_dict.get("simul_code_path") / "output").mkdir(parents=True, exist_ok=True)

    paths_dict["df_test_path"] = Path(args.pred_data)
    # print(paths_dict)
    # input()
    
    return paths_dict, args

if __name__ == "__main__":

    # Setup of all the paths required for the evaluation folder
    paths_dict, args = setup()

    print("Started evaluating process ...")

    # load and process df
    pred_df = load_process_dataset(paths_dict, remove_tokens = args.remove_tokens, input_col = 'input')
    print(f"Length of the dataset: {len(pred_df)}")

    # sort vals
    pred_df = pred_df.sort_values(by = ["id"])
    
    # generate the 'list of dicts' version of the dataset
    problem_list = pred_df["problem_id"].unique()
    pred_df_lst = pred_df.to_dict('records')

    # generate source code files
    generate_source_code_files(pred_df_lst, paths_dict = paths_dict, input_col = 'input', tgt_col = 'prediction') 

    # compile source files
    pred_df_lst = compile_all_programs(pred_df_lst, paths_dict = paths_dict, nproc = 30)
    
    print("Generating dataset with test cases")
    pred_df_lst = generate_dataset_with_inputs(pred_df_lst, paths_dict = paths_dict)

    print(f"Generated dataset with all the inputs {len(pred_df_lst)}")

    # run simulations and save times
    pred_df_lst = simulate_all_programs(pred_df_lst, nproc = 35)

    # save processed df 
    processed_df = pd.DataFrame(pred_df_lst)
    
    processed_df.to_csv(paths_dict.get("eval_sandbox_path") / f"simulated_tc_{paths_dict.get('df_test_path').stem}.csv", index=None) # df has id, input, target and predictions columns
    
    '''
    Starting evaluation 
    '''
    # # # load processed df
    # processed_df = pd.read_csv(paths_dict.get("eval_sandbox_path") / f"simulated_tc_{paths_dict.get('df_test_path').stem}.csv") # df has id, input, target and predictions columns
    print(f"Length of processed dataframe: {len(processed_df)}")

    # cast simulation time columns
    processed_df["simul_time_input"] = processed_df["simul_time_input"].astype(float)
    processed_df["simul_time_tgt"] = processed_df["simul_time_tgt"].astype(float)

    # add evaluations columns: (correctness, )
    processed_df["correctness_input"] = add_correctness_col(processed_df, simul_succ_col = "simul_succ_input", exp_col="exp_output", simul_col="simul_stdout_input")
    processed_df["correctness_tgt"] = add_correctness_col(processed_df, simul_succ_col = "simul_succ_tgt", exp_col="simul_stdout_input", simul_col="simul_stdout_tgt")

    '''
    Generate dataset for single submission aggregating the metrics for all the test cases:
    - speedup: average speedup over all the test cases
    - correctness: correct output for all the test cases
    - perc opt: consider the avg speedup  
    '''
    evaluation_pair_df = generate_submission_dataset(processed_df=processed_df)
    print("Total number of problems: ", len(evaluation_pair_df["problem_id"].unique()))
    print("Total number of pairs: ", len(evaluation_pair_df))
    print("Num of correct pairs: ", len(evaluation_pair_df[evaluation_pair_df["correctness"]]), (len(evaluation_pair_df[evaluation_pair_df["correctness"]]) / len(evaluation_pair_df)) * 100, '%')
    print("Num of uncorrect pairs: ", len(evaluation_pair_df[~evaluation_pair_df["correctness"]]), (len(evaluation_pair_df[~evaluation_pair_df["correctness"]]) / len(evaluation_pair_df)) * 100, '%')
    # evaluation_pair_df[evaluation_pair_df['correctness']].to_csv(paths_dict.get("eval_sandbox_path") / f"correct_pairs_{paths_dict.get('df_test_path').stem}.csv", index=None)

    evaluation_pair_df.to_csv(paths_dict.get("eval_sandbox_path") / f"pairs_{paths_dict.get('df_test_path').stem}.csv", index=None)

    '''
    Speedup
    (predictions that are not correct or slower will have a speedup = 1)
    '''    
    evaluation_pair_df.loc[~evaluation_pair_df['correctness'], "speedup"] = 1
    evaluation_pair_df.loc[evaluation_pair_df['speedup'] < 1, "speedup"] = 1
    
    avg_speedup = evaluation_pair_df['speedup'].mean()
    print(f"Speedup: {avg_speedup:.2f}")

    correct_df = evaluation_pair_df[evaluation_pair_df['correctness']]
    correct_faster = correct_df[correct_df["speedup"] > 1]

    print(f"Num of problems optimized: {len(correct_df['problem_id'].unique())}")

    print(f"{len(correct_faster)=}")
    print(f"{len(correct_faster[correct_faster['speedup'] > 1.10])=}")

    # perc_opt_progs = len(correct_faster[correct_faster["speedup"] > 1.10]) / len(correct_faster) * 100
    # UPDATE dividing by 978 (test set length)
    perc_opt_progs = len(correct_faster[correct_faster["speedup"] > 1.10]) / len(evaluation_pair_df) * 100

    print(f"Percentage of optimized: {perc_opt_progs:.2f} %")

    results = {
        "total pairs": len(evaluation_pair_df),
        "correct pairs": len(evaluation_pair_df[evaluation_pair_df["correctness"]]),
        "uncorrect pairs": len(evaluation_pair_df[~evaluation_pair_df["correctness"]]),
        "correct faster": len(correct_faster),
        "problems_optimized": len(correct_df['problem_id'].unique()),
        "eval_correctness": (len(evaluation_pair_df[evaluation_pair_df["correctness"]]) / len(evaluation_pair_df)) * 100,
        "eval_speedup": avg_speedup,
        "eval_perc_opt": perc_opt_progs
    }


    with open(paths_dict.get("eval_sandbox_path") / 'eval_metrics.json', 'w') as results_file:
        json.dump(results, results_file)