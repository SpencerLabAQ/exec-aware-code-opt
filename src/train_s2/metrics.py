import numpy as np
from difflib import SequenceMatcher
from tqdm import tqdm
import pandas as pd

'''
Evaluation Metrics for the exexcution-aware pre-training.
'''
# Calculate the exact match line by line between the model output and the eval/test labels
def instance_exact_match(out, lbl, sep = "\n"):
    total_stats = 0
    matches = 0
    assert len(out) == len(lbl), "Output and labels have different lenghts"
    for _idx in range(len(out)):
        out_i, lbl_i = out[_idx].split(sep), lbl[_idx].split(sep)

        # add the correct labels to total
        total_stats += len(lbl_i)
        
        # cut the out and lbl to have the same length
        max_len = min(len(out_i), len(lbl_i))
        
        cut_out_i, cut_lbl_i = out_i[:max_len], lbl_i[:max_len]

        assert len(cut_out_i) == len(cut_lbl_i), "Output and labels instance have different lenghts, autocut has not been performed correctly"

        matches += sum(1 for out, lbl in zip(out_i, lbl_i) if out == lbl)
    
    return matches / total_stats

# Calculate the exact match line by line between the model output and the eval/test labels
def exact_match(out, lbl):
    total_stats = 0
    matches = 0
    assert len(out) == len(lbl), "Output and labels have different lenghts"
    for _idx in range(len(out)):
        if __check_outputs_equality(out[_idx], lbl[_idx]):
            matches += 1

        # add the correct labels to total
        total_stats += 1
    
    return matches / total_stats

'''
Evaluation Metrics for the C code-optimization finetuning.

These metrics mainly focus on evaluating the (i) correctness 
and (ii) execution time of the generated programs.
'''

# Helpers
def __check_outputs_equality(expected, simulated):
    # clean output from formatting amd check equality
    expected, simulated = str(expected).strip().replace("\n", ""), str(simulated).strip().replace("\n", "")
    return expected == simulated

# Metrics
def add_correctness_col(pred_df: pd.DataFrame):
    # correctness == True means that program has been simulated with success and it produced the expected output 
    return pred_df.apply(lambda pred: True if pred.simul_succ and __check_outputs_equality(expected=pred["problem_exp_output"], simulated=pred["problem_simul_output"]) else False, axis = 1)

def compilation_correctness(out):
    return len([pred for pred in out if pred["compiled_succ"]]) / len(out) * 100

def output_correctness(out):
    return len([pred for pred in out if pred["correctness"]]) / len(out) * 100

def speedup(out, improvement_only = False):
    '''
    Two modes:
    (i)     consider only correct outputs that have been optimized WRT input program (improvement_only == True)
    (ii)    consider all the correct outputs
    '''

    # filter the correct predictions
    correct_preds = [pred for pred in out if pred["correctness"]]

    ret_lst = []
    for i in range(len(correct_preds)):    
        exec_time_diff = correct_preds[i]["simul_time_v0"] / correct_preds[i]["simul_time_v1"]
        if improvement_only:
            if exec_time_diff >= 1:
                ret_lst.append(exec_time_diff)
        else:
            ret_lst.append(exec_time_diff)
    return np.mean(ret_lst)

def perc_opt(out, threshold = 1.2, verbose = False):
    # print("perc_opt details")
    optimized = 0
    for i in range(len(out)):    
        if out[i]["simul_succ"] and isinstance(out[i]["simul_time_v1"], float) and out[i]["simul_time_v1"] >= 0 and out[i]["simul_time_v0"] / out[i]["simul_time_v1"] >= threshold:
            if verbose and not optimized:
                print("Example of optimized")
                print("SLOW")
                print(out[i]["input"])
                print("FAST")
                print(out[i]["prediction"])
            optimized += 1
    if verbose: 
        print("Total len:", len(out), "optimized: ", optimized)
    
    return optimized / len(out) * 100

def submission_correctness(pred_df, verbose = False):
    sub_pred_df = pred_df.groupby("submission_id_v0")
    
    total_sub = len(sub_pred_df)
    correct_sub = 0
    for _, sub_data in sub_pred_df:
        if sub_data["correctness"].all():
            correct_sub += 1
    if verbose:
        print("Total submissions: ", total_sub)
        print("Correct submissions: ", correct_sub)
    return correct_sub / total_sub * 100

def simulate_exec_time(out):
    pass

def string_similarity(out):
    simil_lst = []
    for pred in tqdm(out, total = len(out), desc = "Computing string similarity"):
        simil_lst.append(SequenceMatcher(None, pred["input"], pred["target"]).ratio())
    return np.mean(simil_lst) * 100