'''
Generate the execution aware information by
merging multiple available test cases with a specific criteria
'''

import argparse
import configparser
import jsonlines
import logging
import os
import random
import re

import pandas as pd
from pathlib import Path
from tqdm import tqdm

import xml.etree.ElementTree as ET
from text_utils import *

EXECUTION_TOKENS = ["<ne>", "<e>", "<e+>", "<E>", "<E+>"]
FINALSTATES_KEYS = ['age', 'type', 'name', 'value']

def label_line_cov(count):
    if count > 0:
        return "<e>"
    else:
        return ""

def quantize_exec_count(count):
    '''
    Quantization of line execution counts. The criteria we used considers the 
    distribution of the value counts.

    Stats: 
    Total num of lines: 3833623
    Min executions: 0 | Max executions: 1755
    Q1 (without 0 and 1): 3.0
    MEDIAN (without 0 and 1): 5.0
    Q3 (without 0 and 1): 8.0
    IQR (without 0 and 1): 5.0
    Outlier threshold (Q3 + 2,5 * IQR): 20.5

    Tokens:
    -       : Range = (-1.0, 0.0],      No of lines: 1832267 (48%)  [Line not executed]
    <e>     : Range = (0.0, 1.0],       No of lines: 1393089 (36%)  [Line executed exactly once]
    <e+>    : Range = (1.0, 5.0],       No of lines: 351667 (9%)
    <E>     : Range = (5.0, 20.5],      No of lines: 193992 (5%)
    <E+>    : Range = (20.5, 1755.0],   No of lines: 62608 (2%)
    '''

    assert count >= 0, "The count of line executions has to be a positive number"
    
    if count == 1: 
        return "<e>"
    elif count > 1 and count <= 5:
        return "<e+>"
    elif count > 5 and count <= 20.5:
        return "<E>"
    elif count > 20.5:
        return "<E+>"
    # zero case 
    return ""

def label_branch_exec(branch_exec_val):
    '''
    Three special tokens have been used to label branch coverage:
    - <NB>: Not a branch
    - <BC>: Branch covered in trace
    - <BNC>: Branch not covered in trace
    '''

    if branch_exec_val == None:
        return ""
    elif branch_exec_val == True:
        return "<BC>"
    elif branch_exec_val == False:
        return "<BNC>"
    else:
        raise ValueError(f"'{branch_exec_val}' value not allowed for branch coverage")

def len_unique_problems(json_list):
    unique_texts = set()
    for obj in json_list:
        if 'filepath' in obj:
            unique_texts.add(obj['filepath'].split("/")[0])
    return len(unique_texts)

def remove_comments(text):
    # The pattern /\*.*?\*/ matches /* followed by any characters including newlines, ending with */
    pattern = r'/\*.+?\*/'
    # re.sub removes the pattern from the text, re.DOTALL allows . to match newlines
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text

def read_seq_file(args, regex = "*_LINE.jsonl"):
    df = []
    try:
        for sequence_file in Path(args.sequence_folder).glob(regex):
            logging.info(f"Reading {sequence_file}...")
            with jsonlines.open(sequence_file, "r") as seq_file:
                to_add = [item for item in seq_file]
                df += to_add
                logging.info(f"Adding {len(to_add)} sequences")
        assert len(df) > 0, f"No sequence LINES found."
    except Exception as e:
        logging.error(f"An error occurred opening sequence file: {str(e)}")
        exit()

    logging.info(f"Sequence {regex} full dataset (from 0 to 4052) LENGTH: {len(df)}")
    logging.info(f"Number of considered challenges: {len_unique_problems(df)}")

    return df

def merge_sequences(seq_lines, seq_branch):
    joined_list = []
    
    # Convert seq_branch to a dictionary keyed by the join fields for fast lookup
    branch_dict = {}
    for item_branch in seq_branch:
        join_key = (item_branch["filepath"], item_branch["input_no"])
        if join_key not in branch_dict:
            branch_dict[join_key] = []
        branch_dict[join_key].append(item_branch)
    
    # Perform the join
    for item_lines in seq_lines:
        join_key = (item_lines["filepath"], item_lines["input_no"])
        if join_key in branch_dict:
            assert len(branch_dict[join_key]) == 1, f"{len(branch_dict[join_key])}, {branch_dict[join_key]}"
            # Merge dictionaries
            merged_item = {**item_lines}
            merged_item.update({"branch_covered_in_trace" : branch_dict[join_key][0]["covered_in_trace"]})
            joined_list.append(merged_item)
    
    return joined_list

if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f"logs/{Path(__file__).stem}.log", 
        filemode='w', 
        encoding='utf-8', 
        level=logging.INFO)

    random.seed(42)

    # newline_token = config["tokens"]["newline"]
    newline_token = "\n"

    parser = argparse.ArgumentParser(description="Read tracing sequences jsonl files and build line execution csv pretraining dataset")
    parser.add_argument('--sequence-folder', default="./sequences/", type=str)
    parser.add_argument('--output-path', default="./output/", type=str)
    parser.add_argument('--processed-path', default="./processed/", type=str)
    parser.add_argument('--suffix', default="train", type=str)
    parser.add_argument('--trace-path', default="./tree/", type=str)
    args = parser.parse_args()

    # join sequences
    seq_lines_df = read_seq_file(args, regex = "*_LINE.jsonl")
    seq_brn_df = read_seq_file(args, regex = "*_BRANCH.jsonl")

    logging.info(f"{len(seq_lines_df)=}, {len(seq_brn_df)=}")

    # join branch and lines sequences
    seq_df = merge_sequences(seq_lines = seq_lines_df, seq_branch = seq_brn_df)
    logging.info(f"{len(seq_df)}") 

    # transform jsonl to pandas
    pd_seq_df = pd.DataFrame(seq_df)

    # Build execution lines dataset
    execaware_df = {}

    for filepath, submission_df in pd_seq_df.groupby("filepath"):

        # print(f"Analyzing {filepath=}, {len(submission_df)=}")
        filepath_data = filepath.split("/")

        problem_id, submission_id = filepath_data[0], filepath_data[2].replace(".cpp", "")

        # horizontal assertion
        for key, item in submission_df.iterrows():
            assert len(item["src_lines"]) == len(item["src_linenos"]) == len(item["count_in_trace"]), "Source lines and coverage labels have different lenghts"

        # vertical assertion
        list_lengths = submission_df.count_in_trace.apply(len)
        if list_lengths.nunique() != 1:
            raise ValueError("Not all lists have the same length")
        
        # compute means
        counts_df = pd.DataFrame(submission_df.count_in_trace.tolist())

        mean_vals = counts_df.mean().tolist()
        median_vals = counts_df.median().tolist()
        max_vals = counts_df.max().tolist()

        # code is the same for all the submissions df
        src_lines = submission_df["src_lines"].tolist()[0]
        code = newline_token.join([elem.strip() for elem in src_lines])

        # skip all the submissions with the tokens in the code for avoiding ambiguity
        if any(token in code for token in EXECUTION_TOKENS):
            continue

        elem = {
            "problem_id": problem_id,
            "submission_id": submission_id,
            "traced_test_cases": len(submission_df),
            "original_code": code,
            "counts_df": counts_df.to_dict('records'),
            "mean_vals": mean_vals,
            "median_vals": median_vals,
            "max_vals": max_vals,
        }

        # Execution lines MEAN
        code_exec_label_mean = newline_token.join([elem + " // " + quantize_exec_count(mean_vals[i]) if quantize_exec_count(mean_vals[i]) else elem for i, elem in enumerate(src_lines)])
        code_exec_label_mean = remove_comments(code_exec_label_mean) # remove multi-line comments 
        code_exec_label_mean = newline_token.join([elem for elem in code_exec_label_mean.split(newline_token) if not elem.startswith("//")]) # remove single-line comments
        elem.update({'code_exec_lbl_mean': code_exec_label_mean})

        # Execution lines MEDIAN
        code_exec_label_median = newline_token.join([elem + " // " + quantize_exec_count(median_vals[i]) if quantize_exec_count(median_vals[i]) else elem for i, elem in enumerate(src_lines)])
        code_exec_label_median = remove_comments(code_exec_label_median) # remove multi-line comments 
        code_exec_label_median = newline_token.join([elem for elem in code_exec_label_median.split(newline_token) if not elem.startswith("//")]) # remove single-line comments
        elem.update({'code_exec_lbl_median': code_exec_label_median})

        # Execution lines MAX
        code_exec_label_max = newline_token.join([elem + " // " + quantize_exec_count(max_vals[i]) if quantize_exec_count(max_vals[i]) else elem for i, elem in enumerate(src_lines)])
        code_exec_label_max = remove_comments(code_exec_label_max) # remove multi-line comments 
        code_exec_label_max = newline_token.join([elem for elem in code_exec_label_max.split(newline_token) if not elem.startswith("//")]) # remove single-line comments
        elem.update({'code_exec_lbl_max': code_exec_label_max})

        execaware_df[submission_id] = elem
    

    # logging.info(f"Generating csv file (writing {len(execaware_df)} rows) ...")
    with jsonlines.open(Path(args.output_path) / f"execaware_multi_tc_{args.suffix}.jsonl", "w") as outfile:
        outfile.write_all(execaware_df.items())
        print(f'{Path(args.output_path) / f"execaware_multi_tc_{args.suffix}.jsonl"} saved')

    processed_df = pd.read_csv(Path(args.processed_path)).to_dict('records')

    print(f"{len(processed_df)=}")

    exec_aware_df = []

    for elem in processed_df:

        src_id = elem["id"].split("#")[-1].split("_")[0]

        exec_aware_df.append({
            "id": elem["id"],
            "traced_test_cases": execaware_df.get(src_id)["traced_test_cases"],
            "source": elem['source'],
            "source_le_mean": execaware_df.get(src_id)["code_exec_lbl_mean"],
            "source_le_median": execaware_df.get(src_id)["code_exec_lbl_median"],
            "source_le_max": execaware_df.get(src_id)["code_exec_lbl_max"],
            "target": elem['target']
        })

    with jsonlines.open(Path(args.output_path) / f"LE_multinput_{args.suffix}_full.jsonl", "w") as outfile:
        outfile.write_all(exec_aware_df)

    pd.DataFrame(exec_aware_df)[["id", "source_le_max", "target"]].rename(columns = {"source_le_max": "source"}).to_csv(Path(args.output_path) / f"LE_multinput_{args.suffix}.csv")

    