import pandas as pd
import jsonlines
import logging 
import argparse

from pathlib import Path

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
    parser.add_argument('--id-col', default="id", type=str)
    parser.add_argument('--source-col', default="source", type=str)
    parser.add_argument('--target-col', default="target", type=str)
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
    
    # check if all specified columns are in the dataset
    assert set([args.id_col, args.source_col, args.target_col]).issubset(dataset.columns), "Specified columns are not in dataset"

    # rename
    dataset = dataset[[args.id_col, args.source_col, args.target_col]]
    dataset.rename(columns = {
        args.id_col: "id",
        args.source_col: "source",
        args.target_col: "target"
    }, inplace=True)

    
    dataset[["id", "source", "target"]].to_csv(Path(args.output_path), index=None)