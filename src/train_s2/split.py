import argparse
import configparser
import pandas as pd
import logging
import re
import os

from pathlib import Path
from sklearn.model_selection import train_test_split

def split(df, test_size, random_state):

    # check if there is problem_id, otherwise it is the first part of the id before the '#'
    if 'problem_id' not in df.columns:
        df["problem_id"] = df["id"].str.split("_").str[0]

    # get problems
    problems = df['problem_id'].unique()
    problems.sort()

    # create dataframe
    problems = pd.DataFrame(problems, columns=['problem_id'])

    # split into train and test problems
    train_probs, test_probs = train_test_split(problems, test_size=test_size, random_state=random_state)

    # split into train and test according to benchmarks split
    train = df[df['problem_id'].isin(train_probs['problem_id'])]
    test = df[df['problem_id'].isin(test_probs['problem_id'])]

    print("null samples in train", len(train[train['source'].isna()]))
    print("null samples in test", len(test[test['source'].isna()]))
    
    return train, test

if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f"logs/{Path(__file__).stem}.log", 
        filemode='w', 
        encoding='utf-8', 
        level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Dataset split by CodeNet challenges")

    parser.add_argument('--filepath', default='./datasets/dataset.csv', type=str)
    parser.add_argument('--random-state', default=42, type=int)
    parser.add_argument('--ignore-test', action='store_true', default=False)

    args = parser.parse_args()

    # read df and set paths
    df_path = Path(args.filepath)
    df_dir, df_name, df = df_path.parent, df_path.stem, pd.read_csv(df_path)
    (df_dir / "split").mkdir(exist_ok=True)

    if args.ignore_test:
        
        # IGNORE THE TEST SET: perform 80 (train) | 20 (validation) splitting
        train, val = split(df, test_size = 0.2, random_state = args.random_state) # 100% => 80% - 20%

        logging.info(f"Full dataset has been splitted (problem-based) in 80% train and 20% validation and test sets. Train samples : {len(train)}, val samples : {len(val)}")

        train.to_csv(df_dir / "split" / (df_name + "_train.csv"), index=None)
        val.to_csv(df_dir / "split" / (df_name + "_val.csv"), index=None)

    else:

        # perform 80 (train) | 10 (validation) | 10 (test) splitting
        train, test = split(df, test_size = 0.2, random_state = args.random_state) # 100% => 80% - 20%
        val, test = split(test, test_size = 0.5, random_state = args.random_state) # 20% => 10% - 10%

        logging.info(f"Full dataset has been splitted (problem-based) in 80% train, 10% validation and 10% test sets. Train samples : {len(train)}, val samples : {len(val)}, test samples : {len(test)}")

        train.to_csv(df_dir / "split" / (df_name + "_train.csv"), index=None)
        val.to_csv(df_dir / "split" / (df_name + "_val.csv"), index=None)
        test.to_csv(df_dir / "split" / (df_name + "_test.csv"), index=None)

