'''
Python script to remove comments from C/C++
source codes contained in a dataset
'''

import argparse
import logging

import pandas as pd
import jsonlines
from tqdm import tqdm
from pathlib import Path

def find_files_by_pattern(directory, pattern):
    # Create a Path object for the directory
    path = Path(directory)
    
    # Use the glob method to find files matching the pattern
    matching_files = sorted(list(path.glob(f"*{pattern}")))
    
    return matching_files

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
    parser.add_argument('--full-dataset-path', default='dataset.jsonl', type=str, required=False)
    parser.add_argument('--folder', default='./', type=str, required=True)
    parser.add_argument('--src-suffix', default='_src_processed.cpp', type=str, required=False)
    parser.add_argument('--tgt-suffix', default='_tgt_processed.cpp', type=str, required=False)
    parser.add_argument('--save-as', default='./processed.csv', type=str, required=False)
    args = parser.parse_args()

    # --- load data ---
    folder_path = Path(args.folder)

    df = []

    processed_src_files = find_files_by_pattern(args.folder, args.src_suffix)
    processed_tgt_files = find_files_by_pattern(args.folder, args.tgt_suffix)

    # print(f"Processed src files '{len(processed_src_files)}'", args.folder, args.src_suffix)
    
    # there may be processed files that do not exist for parsing issues
    for src_file_path in tqdm(processed_src_files, total=len(processed_src_files), desc="Extracting processed source code"):
        pair_id_prefix = str(src_file_path).replace(args.src_suffix, "")
        # get target if exist

        tgt_file_path = Path(f"{pair_id_prefix}{args.tgt_suffix}")

        if tgt_file_path not in processed_tgt_files:
            # print(processed_tgt_files[0])
            print(f"{pair_id_prefix} - {tgt_file_path} TARGET not generated")
        else:
            # print("Processing", src_file_path, tgt_file_path)
            with open(src_file_path, "r") as src_file, open(tgt_file_path, "r") as tgt_file:
                src_pair_id = src_file_path.name.replace(args.src_suffix, "")
                tgt_pair_id = tgt_file_path.name.replace(args.tgt_suffix, "")

                # double check on names
                if src_pair_id != tgt_pair_id:
                    print("Pairs mismatch", src_pair_id, tgt_pair_id)
                    continue
                
                pair_id = src_pair_id

                df.append({
                    "id": pair_id,
                    "source": "\n".join([line.strip() for line in src_file.read().split("\n")]),
                    "target": "\n".join([line.strip() for line in tgt_file.read().split("\n")])
                })


    # read full dataset
    full_dataset = []
    with jsonlines.open(Path(args.full_dataset_path), "r") as fd_reader:
        full_dataset = [row for row in fd_reader]
    assert len(full_dataset)

    full_dataset = {instance["id"]:instance for instance in full_dataset}

    for instance in tqdm(df, total=len(df), desc="Generating processed dataset"):
        full_instance = full_dataset.get(instance["id"])

        instance.update({
            "source_line_exec": full_instance["source_line_exec"],
            "source_line_cov": full_instance["source_line_cov"],
            "source_bran_cov": full_instance["source_bran_cov"],
            "source_prog_states": full_instance["source_prog_states"],
            "source_vanilla": full_instance["source_vanilla"]
        })


    pd.DataFrame(df).to_csv(Path(args.save_as), index = None)
    print(f"Dataset saved as {args.save_as}")