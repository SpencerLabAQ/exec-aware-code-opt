'''
Python script to remove comments from C/C++
source codes contained in a dataset
'''

import argparse
import configparser
import logging
import jsonlines
import multiprocessing
import subprocess
import tempfile
import threading
import time

import pandas as pd
from tqdm import tqdm

from pathlib import Path

def find_files_by_pattern(directory, pattern):
    # Create a Path object for the directory
    path = Path(directory)
    
    # Use the glob method to find files matching the pattern
    matching_files = sorted(list(path.glob(f"*{pattern}")))
    
    return matching_files

def __generate_file(params):

    data, src_col, tgt_col, args = params

    pair_id = data["problem_id"] + "#" + data["src_id"] + "_" + data["tgt_id"]

    # src
    with open(f"./{args.folder}/{pair_id}_src.cpp", "w") as temp_file:
        temp_file.write(data[src_col])

    # tgt
    with open(f"./{args.folder}/{pair_id}_tgt.cpp", "w") as temp_file:
        temp_file.write(data[tgt_col])

def generate_sources(dataset, src_col, tgt_col, args, nproc = 35):

    iterations = [(item, src_col, tgt_col, args) for item in dataset]

    with multiprocessing.Pool(nproc) as p:
        with tqdm(total=len(iterations), desc="Generating C/C++ source code") as pbar:
            for _ in p.imap_unordered(__generate_file, iterations):
                pbar.update()

if __name__ == "__main__":
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f"logs/{Path(__file__).stem}.log",
        filemode='w',
        encoding='utf-8', 
        level=logging.DEBUG
        )
    
    # --- args ---
    parser = argparse.ArgumentParser(description="Evaluation of generated optimized code")
    parser.add_argument('--dataset', default='./dataset.jsonl', type=str, required=True)
    parser.add_argument('--src-col', default='source', type=str, required=False)
    parser.add_argument('--tgt-col', default='target', type=str, required=False)
    parser.add_argument('--folder', default='./tmp', type=str, required=False)
    args = parser.parse_args()

    # --- load data ---
    dataset_path = Path(args.dataset)
    dataset = []
    with jsonlines.open(dataset_path, "r") as dataset_file:
        dataset = [instance for instance in dataset_file]

        assert len(dataset), f"Empty dataset {dataset_path}"

        # preprocess column removing comments
        generate_sources(dataset, src_col = args.src_col, tgt_col = args.tgt_col, args = args, nproc = 25)

    generated_source_files = find_files_by_pattern(f"./{args.folder}/", "_src.cpp")
    generated_target_files = find_files_by_pattern(f"./{args.folder}/", "_tgt.cpp")

    print(f"Generated {len(generated_source_files)} src files, {len(generated_target_files)} tgt files")

    generated_files = pd.DataFrame()
    generated_files["source"] = pd.Series(generated_source_files)
    generated_files["target"] = pd.Series(generated_target_files)

    generated_files.to_csv(Path(args.folder) / "generated_files_list.csv", index = None)