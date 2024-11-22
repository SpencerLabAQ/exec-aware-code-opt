import pandas as pd
import jsonlines
import logging 
import argparse

from pathlib import Path

def mlm_src_col(dataset: pd.DataFrame, prefix: str, src_col: str):
    return prefix + dataset[src_col]

def exec_aware_src_col(dataset: pd.DataFrame, prefix: str, src_col: str):
    return prefix + dataset["input"] + "<SEP>" + dataset[src_col]

if __name__ == "__main__":
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f"logs/{Path(__file__).stem}.log",
        filemode='w',
        encoding='utf-8', 
        level=logging.DEBUG
        )

    parser = argparse.ArgumentParser(description="Read tracing sequences jsonl files and build line execution csv pretraining dataset")
    parser.add_argument('--dataset-path', default="./dataset.csv", type=str)
    parser.add_argument('--output-path', default="./output/specialized.csv", type=str)
    # parser.add_argument('--id-col', default="id", type=str)
    parser.add_argument('--source-col', default="source", type=str)
    parser.add_argument('--target-col', default="target", type=str)
    parser.add_argument('--input-prefix', default="classify: ", type=str)
    args = parser.parse_args()

    dataset = []

    if Path(args.dataset_path).is_file():
        
        if Path(args.dataset_path).suffix == '.csv':
            dataset = pd.read_csv(Path(args.dataset_path))
        elif Path(args.dataset_path).suffix == '.jsonl':
            with jsonlines.open(Path(args.dataset_path), "r") as df_file:
                ls_dataset = [row for row in df_file]
                dataset = pd.DataFrame(ls_dataset)

    assert len(dataset), "Empty dataset"

    dataset = dataset.dropna()

    dataset["id"] = dataset["problem_id"] + "_" + dataset["submission_id"] + "_" + dataset["input_id"]
    
    assert "input" in dataset.columns, "Input field required to generate the pretraining dataset"

    # # random shuffling
    # dataset = dataset.sample(frac=1)
    # split_idx = int(len(dataset) / 2)

    # # half for mlm task
    # mlm_dataset = dataset[:split_idx]
    # mlm_dataset["task"] = "mlm"
    # print(f"{len(mlm_dataset)} instances for mlm")
    
    # # half for exec_aware task
    # cls_dataset = dataset[split_idx+1:]
    # mlm_dataset["task"] = "cls"
    # print(f"{len(cls_dataset)} instances for exec aware")

    # # add cols
    # cls_dataset[args.source_col] = exec_aware_src_col(dataset = cls_dataset, prefix = args.input_prefix, src_col = args.source_col)
    
    # mlm_dataset[args.target_col] = mlm_dataset[args.source_col]
    # mlm_dataset[args.source_col] = mlm_src_col(dataset = mlm_dataset, prefix = "mlm: ", src_col = args.source_col)
    

    # '''
    # Obfuscation of tokens for MLM objectives will be done
    # in the training script based on the 'task' column
    # in the dataset.
    # '''

    # dataset = pd.concat([cls_dataset, mlm_dataset])

    # # check if all specified columns are in the dataset
    # columns_subset = ["id", "task", args.source_col, args.target_col]

    dataset["source"] = args.input_prefix + dataset["input"] + "<SEP>" + dataset[args.source_col]
    dataset["code"] = "mlm: " + dataset[args.source_col]

    columns_subset = ["id", "code", "source", args.target_col]

    assert set(columns_subset).issubset(dataset.columns), "Specified columns are not in dataset"

    # rename
    dataset = dataset[columns_subset]
    dataset.rename(columns = {
        args.target_col: "target"
    }, inplace=True)

    dataset[["id", "code", "source", "target"]].to_csv(Path(args.output_path), index=None)