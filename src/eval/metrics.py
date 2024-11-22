import numpy as np
from difflib import SequenceMatcher
from tqdm import tqdm
import pandas as pd
import math

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

def exact_match(out, lbl):
    total_stats = len(out)
    matches = 0
    assert len(out) == len(lbl), "Output and labels have different lenghts"
    for _idx in range(len(out)):
        if str(out[_idx]) == str(lbl[_idx]):
            matches += 1
    
    return matches / total_stats

def classif_performance_metrics(df, target_col = "target", pred_col = "prediction", tokens : list = ["<NB>", "<BC>", "<BNC>"]):
    from sklearn.metrics import precision_score, recall_score, f1_score

    assert target_col in df.columns and pred_col in df.columns, f"{target_col} and {pred_col} are not both in dataset columns ({df.columns})"

    precisions, recalls, f1_scores = [], [], []

    targets = df[target_col].tolist()
    predictions = df[pred_col].tolist()

    diff_len_cnt = 0

    for i in tqdm(range(len(df)), total = len(df), desc=f"Computing precision, recall and f1 score"):
        labels = targets[i].split("\n")
        pred = predictions[i].split("\n")
        if len(labels) == len(pred):
            precision_scores_local = precision_score(labels, pred, labels = tokens, average = None, zero_division=0).tolist()
            recall_scores_local = recall_score(labels, pred, labels = tokens, average = None, zero_division=0).tolist()
            f1_scores_local = f1_score(labels, pred, labels = tokens, average = None, zero_division=0).tolist()
            for tkn_idx, tkn in enumerate(tokens):
                if len(["_" for lbl in labels if lbl == tkn]) == 0 and len(["_" for p in pred if p == tkn]) == 0:
                    precision_scores_local[tkn_idx] = 1
                    recall_scores_local[tkn_idx] = 1
                    f1_scores_local[tkn_idx] = 1
            precisions.append(precision_scores_local)
            recalls.append(recall_scores_local)
            f1_scores.append(f1_scores_local)
        else:
            # original program and predicted tokens have different length
            precisions.append((0, 0, 0))
            recalls.append((0, 0, 0))
            f1_scores.append((0, 0, 0))
            diff_len_cnt += 1

    print("Num of predictions with a wrong lenght", diff_len_cnt)


    return pd.DataFrame(precisions, columns=[f"precision_{token}" for token in tokens]), pd.DataFrame(recalls, columns=[f"recall_{token}" for token in tokens]), pd.DataFrame(f1_scores, columns=[f"f1_score_{token}" for token in tokens])

'''
Evaluation Metrics for the C code-optimization finetuning.

These metrics mainly focus on evaluating the (i) correctness 
and (ii) execution time of the generated programs.
'''

def generate_submission_dataset(processed_df):
    final_df = []
    for submission_id_v0, submission_data in processed_df.groupby('submission_id_v0'):
        correctness = submission_data["correctness_tgt"].all()
        src = submission_data['input'].unique()
        pred = submission_data['prediction'].unique()
        assert len(submission_data['problem_id'].unique()) == 1
        final_df.append({
            'correctness': correctness,
            'problem_id': submission_data['problem_id'].unique()[0],
            'submission_id_v0': submission_id_v0,
            'speedup': (submission_data['simul_time_input'] / submission_data['simul_time_tgt']).mean() if correctness else None,
            'all_correctness_input': submission_data["correctness_input"].tolist(),
            'all_correctness_tgt': submission_data["correctness_tgt"].tolist(),
            'all_simul_time_input': submission_data['simul_time_input'].tolist(),
            'all_simul_time_tgt': submission_data['simul_time_tgt'].tolist(),
            'src': src[0],
            'pred': pred[0]
        })
    return pd.DataFrame(final_df)
        
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def __filter_output_content(content):
    lines = content.split("\n")
    to_return = []
    for line in lines:
        if line == "":
            continue
        if is_float(line):
            to_return.append(str(float(line)))
        else:
            to_return.append(str(line).strip().replace("\n", ""))
    return to_return

# inherited
def get_accuracy(output: str, ground_truth: str) -> float:
    """
    Compare the output of the code with the ground truth.
    """
    num_correct = 0
    ground_truth_lines = ground_truth.strip().splitlines()
    output_truth_lines = output.strip().splitlines()
    for gen_output, ground_truth_output in zip(output_truth_lines, ground_truth_lines):
        is_corr = gen_output == ground_truth_output
        if not is_corr:
            try:
                gen_output = float(gen_output)
                ground_truth_output = float(ground_truth_output)
                is_corr = abs(gen_output - ground_truth_output) < 1e-3
            except:
                pass
        num_correct += int(is_corr)

    return num_correct == len(ground_truth_lines)

# Helpers
def __check_outputs_equality(expected, simulated):
    # clean output from formatting amd check equality
    if isinstance(expected, str) and isinstance(simulated, str):
        return expected.strip() == simulated.strip()
    # print(f"Not string: {expected}, {simulated}")
    # print(expected, type(expected), simulated, type(simulated))
    if (isinstance(expected, float) and math.isnan(expected)) or (isinstance(simulated, float) and math.isnan(simulated)):
        # print(expected, simulated, "return false")
        return False
    return expected == simulated

    # # float case
    # if is_float(expected) and is_float(simulated):
    #     return float(expected) == float(simulated)

    # exp_rows = __filter_output_content(expected)
    # sim_rows = __filter_output_content(simulated)

    # if len(exp_rows) != len(sim_rows):
    #     return False

    # for i in range(len(exp_rows)):
    #     if exp_rows[i] != sim_rows[i]:
    #         return False

    # return True
    

    # str case
    # expected, simulated = str(expected).strip().replace("\n", ""), str(simulated).strip().replace("\n", "")
    # # if not expected == simulated:
    # #     print(f"Wrong execution output {expected=} {simulated=}")
    # return expected == simulated

# Metrics
def add_correctness_col(pred_df: pd.DataFrame, simul_succ_col : str = "simul_succ", exp_col : str = "problem_exp_output", simul_col : str = "problem_simul_output"):
    # correctness == True means that program has been simulated with success and it produced the expected output 
    return pred_df.apply(lambda pred: True if pred[simul_succ_col] and get_accuracy(ground_truth=str(pred[exp_col]), output=str(pred[simul_col])) else False, axis = 1)

def compilation_correctness(out, col = "compiled_succ"):
    return len([pred for pred in out if pred[col]]) / len(out) * 100

def output_correctness(out, correctness_col = "correctness"):
    return len([pred for pred in out if pred[correctness_col]]) / len(out) * 100

def speedup(out, simul_time_input_col = "simul_time_v0", simul_time_tgt_col = "simul_time_v1", to_check = ["simul_succ"], improvement_only = False):
    '''
    Two modes:
    (i)     consider only correct outputs that have been optimized WRT input program (improvement_only == True)
    (ii)    consider all the correct outputs
    '''

    # filter the correct predictions
    correct_preds = [pred for pred in out if not False in [pred[col] for col in to_check]]
    
    ret_lst = []
    for i in range(len(correct_preds)):    
        try:
            exec_time_diff = float(correct_preds[i][simul_time_input_col]) / float(correct_preds[i][simul_time_tgt_col])
            if np.isnan(exec_time_diff):
                print(exec_time_diff, correct_preds[i])
            if improvement_only:
                if exec_time_diff >= 1:
                    ret_lst.append(exec_time_diff)
            else:
                ret_lst.append(exec_time_diff)
        except Exception as e:
            print(correct_preds[i])
            print(str(e))

    # print("Len of speedup let_lst", len(ret_lst))
    # print(ret_lst)
    return np.mean(ret_lst)

def __check_cols(dct, to_check):
    for col in to_check:
        if not dct[col]:
            return False
    return True

def perc_opt(out, to_check = ["simul_succ"], simul_time_input_col = "simul_time_v0", simul_time_tgt_col = "simul_time_v1", input_col = "input", tgt_col = "prediction", threshold = 1.2, verbose = False):
    # print("perc_opt details")
    optimized = 0
    for i in range(len(out)):
        if __check_cols(out[i], to_check) and float(out[i][simul_time_tgt_col]) >= 0 and float(out[i][simul_time_input_col]) / float(out[i][simul_time_tgt_col]) >= threshold:
            if verbose and not optimized:
                print("Example of optimized")
                print("SLOW")
                print(out[i][input_col])
                print("FAST")
                print(out[i][tgt_col])
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

def string_similarity(out, input_col = "input", tgt_col = "target"):
    simil_lst = []
    for pred in tqdm(out, total = len(out), desc = "Computing string similarity"):
        simil_lst.append(SequenceMatcher(None, pred[input_col], pred[tgt_col]).ratio())
    return np.mean(simil_lst) * 100
