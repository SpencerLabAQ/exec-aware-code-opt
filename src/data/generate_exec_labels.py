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
from exec_format_utils import label_line_cov, quantize_exec_count, label_branch_exec, remove_comments
from pstates_from_trace import get_trace, classify_type, quantize_value

EXECUTION_TOKENS = ["<ne>", "<e>", "<e+>", "<E>", "<E+>"]
FINALSTATES_KEYS = ['age', 'type', 'name', 'value']


def len_unique_problems(json_list):
    unique_texts = set()
    for obj in json_list:
        if 'filepath' in obj:
            unique_texts.add(obj['filepath'].split("/")[0])
    return len(unique_texts)

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

def tuples_to_dicts(list_of_tuples, keys):
    list_of_dicts = []
    for tup in list_of_tuples:
        if len(tup[1]) > 0:
            # print(tup)
            for elem in tup[1]:
                dict_item = {'line': tup[0]}
                for i, key in enumerate(keys):
                    dict_item[key] = elem[i]
                list_of_dicts.append(dict_item)
    return list_of_dicts



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
    # parser.add_argument('--pie-path', default="./pie/", type=str)
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

    # dump merged df
    with jsonlines.open(Path(args.output_path) / f"merged_sequences_{args.suffix}.jsonl", "w") as outfile:
        outfile.write_all(seq_df)

    # Build execution lines dataset
    execaware_df = []
    for item in tqdm(seq_df, total=len(seq_df), desc="Extracting execution details from source code"):

        filepath_data = item["filepath"].split("/")

        problem_id, submission_id, input = filepath_data[0], filepath_data[2].replace(".c", "").replace("pp", ""), item["input_no"]

        assert len(item["src_lines"]) == len(item["src_linenos"]) == len(item["count_in_trace"]), "Source lines and coverage labels have different lenghts"

        code = newline_token.join([elem.strip() for elem in item["src_lines"]])

        # skip all the submissions with the tokens in the code for avoiding ambiguity
        if any(token in code for token in EXECUTION_TOKENS):
            continue

        elem = {
            "problem_id": problem_id,
            "submission_id": submission_id,
            "input_id": input,
            "input": item["input"],
            "code": code,
        }

        # Execution lines
        code_exec_label = newline_token.join([quantize_exec_count(elem) if quantize_exec_count(elem) else "" for i, elem in enumerate(item["count_in_trace"])])
        # code_exec_label = newline_token.join([elem + "" + quantize_exec_count(item["count_in_trace"][i]) if quantize_exec_count(item["count_in_trace"][i]) else elem for i, elem in enumerate(item["src_lines"])])
        # code_exec_label = remove_comments(code_exec_label) # remove multi-line comments 
        # code_exec_label = newline_token.join([elem for elem in code_exec_label.split(newline_token) if not elem.startswith("//")]) # remove single-line comments
        elem.update({'code_exec_lbl': code_exec_label})

        # Line coverage
        line_cov_lbl = newline_token.join([label_line_cov(elem) if label_line_cov(elem) else "" for i, elem in enumerate(item["count_in_trace"])])
        # line_cov_lbl = newline_token.join([elem + "" + label_line_cov(item["count_in_trace"][i]) if label_line_cov(item["count_in_trace"][i]) else elem for i, elem in enumerate(item["src_lines"])])
        # line_cov_lbl = remove_comments(line_cov_lbl) # remove multi-line comments 
        # line_cov_lbl = newline_token.join([elem for elem in line_cov_lbl.split(newline_token) if not elem.startswith("//")]) # remove single-line comments
        elem.update({'line_cov_lbl': line_cov_lbl})

        # Branch coverage
        branch_cov = newline_token.join([label_branch_exec(elem) if label_branch_exec(elem) else "" for i, elem in enumerate(item["branch_covered_in_trace"])])
        # branch_cov = newline_token.join([elem + "" + label_branch_exec(item["branch_covered_in_trace"][i]) if label_branch_exec(item["branch_covered_in_trace"][i]) else elem for i, elem in enumerate(item["src_lines"])])
        # branch_cov = remove_comments(branch_cov) # remove multi-line comments 
        # branch_cov = newline_token.join([elem for elem in branch_cov.split(newline_token) if not elem.startswith("//")]) # remove single-line comments
        elem.update({'branch_covered_in_trace': branch_cov})

        # Program states
        log_file = str(Path(args.trace_path) / problem_id / "C++" / submission_id / f"{input}.txt_log.xml")
        states, any_modified, timed_out = get_trace(log_file, lang = "cpp")
        states_dict = tuples_to_dicts(states, FINALSTATES_KEYS)
        if len(states_dict) == 0:
            logging.info(f"[{problem_id} | {submission_id} | {input}] Skipping: len(states) = 0")
            continue
        final_states = pd.DataFrame(states_dict).drop_duplicates(subset=["type", "name"], keep='last')

        final_states["value_type"] = final_states.apply(lambda val: val["type"].split(" ", maxsplit = 1)[0], axis = 1)
        final_states["var_type"] = final_states.apply(lambda val: classify_type(val["type"]), axis = 1)
        final_states["quantized_value"] = final_states.apply(lambda val: quantize_value(data_type=val["var_type"], value_type=val["value_type"], value=val["value"], age=val["age"]), axis = 1) 
        # # complete label
        # prog_states_lbl = "\n".join([f"// {var['name']} {var['var_type']} {var['value_type']} {var['quantized_value']}" for var in final_states.to_dict('records')])
        # partial
        prog_states_lbl = "\n".join([f"// {var['name']} {var['var_type']} {var['quantized_value']}" for var in final_states.to_dict('records')])


        ps_code = newline_token.join(item["src_lines"])
        ps_code = remove_comments(ps_code) # remove multi-line comments 
        ps_code = newline_token.join([elem for elem in ps_code.split(newline_token) if not elem.startswith("//")]) # remove single-line comments
        
        # build vanilla too
        elem.update({"vanilla_format": ps_code})
        # add complete PS label
        # elem.update({"prog_states_lbl": f"{ps_code}\n{prog_states_lbl}"})
        elem.update({"prog_states_lbl": prog_states_lbl})

        execaware_df.append(elem)

        # Log each 10000 instances
        # if _idx % 10000 == 0:
        #     logging.info(f"Processed {_idx} elements")

    # logging.info(f"Generating csv file (writing {len(execaware_df)} rows) ...")
    with jsonlines.open(Path(args.output_path) / f"execaware_full_{args.suffix}.jsonl", "w") as outfile:
        outfile.write_all(execaware_df)

    # pie_df = []
    # # read pie4perf
    # with jsonlines.open(Path(args.pie_path), "r") as pie_file:
    #     pie_df = [row for row in pie_file]

    # print(len(pie_df))


    # # execaware_dict_full = {item['submission_id']:item for item in execaware_df}
    
    # execaware_dict_full = {}
    # for item in execaware_df:
    #     submission_id = item['submission_id']
    #     input_id = item['input_id']
    #     if submission_id not in execaware_dict_full:
    #         execaware_dict_full[submission_id] = {}
    #     execaware_dict_full[submission_id][input_id] = item

    # selected_test_cases = []
    # execaware_dict_randtc = {}
    # for submission_id in execaware_dict_full:
    #     tcs = list(execaware_dict_full[submission_id].keys())
    #     random_tc = random.choice(tcs)
    #     execaware_dict_randtc[submission_id] = execaware_dict_full[submission_id][random_tc]
    #     selected_test_cases.append({
    #         "problem_id": execaware_dict_full[submission_id][random_tc]["problem_id"],
    #         "submission_id": submission_id,
    #         "selected_test_case": random_tc
    #     })

    # with jsonlines.open(Path(args.output_path) / f"test_cases_{args.suffix}.jsonl", "w") as tc_file:
    #     tc_file.write_all(selected_test_cases)

    # # print(f"dataset size with all test cases: {len(execaware_dict_full)}")
    # print(f"dataset size with a random test case: {len(execaware_dict_randtc)}")


    # exec_aware_df = []

    # for elem in pie_df:

    #     if elem['src_id'] not in execaware_dict_randtc.keys():
    #         # print("skipping")
    #         continue 
    #     # print("inserting")

    #     exec_aware_df.append({
    #         "id": f"{elem['problem_id']}#{elem['src_id']}_{elem['tgt_id']}",
    #         "problem_id": elem['problem_id'],
    #         "input_id": execaware_dict_randtc.get(elem['src_id'])["input_id"],
    #         "src_id": elem['src_id'],
    #         "tgt_id": elem['tgt_id'],
    #         "source": elem['src_code'],
    #         "source_vanilla": execaware_dict_randtc.get(elem['src_id'])["vanilla_format"],
    #         "source_line_exec": execaware_dict_randtc.get(elem['src_id'])["code_exec_lbl"],
    #         "source_line_cov": execaware_dict_randtc.get(elem['src_id'])["line_cov_lbl"],
    #         "source_bran_cov": execaware_dict_randtc.get(elem['src_id'])["branch_covered_in_trace"],
    #         "source_prog_states": execaware_dict_randtc.get(elem['src_id'])["prog_states_lbl"],
    #         "target": elem['tgt_code'],
    #     })

    # # pd.DataFrame(exec_aware_df).to_csv(Path(args.output_path) / "exec_aware_train.csv")
    # # pd.DataFrame(vanilla_train).to_csv(Path(args.output_path) / "vanilla_train.csv")
    # with jsonlines.open(Path(args.output_path) / f"execaware_{args.suffix}.jsonl", "w") as outfile:
    #     outfile.write_all(exec_aware_df)

    # pd.DataFrame(exec_aware_df).to_csv(Path(args.output_path) / f"execaware_{args.suffix}.csv")

    