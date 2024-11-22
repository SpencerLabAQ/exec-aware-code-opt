
import argparse
import os
import pprint
import numpy as np
import pandas as pd
from pathlib import Path

import random
import torch

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, DataCollatorForSeq2Seq

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def save_prediction_stats(filepath: str, ids: list, inputs: list, targets: list, predictions: list):
    df = pd.DataFrame({'id': ids, \
                       'input': inputs, \
                       'target': targets, \
                       'prediction': predictions})
    df.to_csv(filepath, index=False)

def run_inference(args, model_folder, test_data, processed_test_data):
    """
    This function runs the inference on the test data using the model in the model folder.
    It loads the model from the model folder, runs the inference on the test data, and saves the predictions to disk.
    """
    # Load the model from the model folder
    model = AutoModelForSeq2SeqLM.from_pretrained(model_folder)
    print(f"  ==> Loaded model checkpoint from {model_folder}, model size {model.num_parameters()}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.load)

    # Create a data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding=True, label_pad_token_id=tokenizer.pad_token_id)

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_folder,
        overwrite_output_dir=False,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        fp16=args.fp16
    )

    # Create a Trainer instance and compute the metrics
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Run the inference
    results = trainer.predict(processed_test_data, max_length = args.max_target_len)
    decoded_result = np.where(results.predictions != -100, results.predictions, tokenizer.pad_token_id)
    batch_decoded = tokenizer.batch_decode(decoded_result, skip_special_tokens=True)

    assert len(processed_test_data) == len(batch_decoded)

    # Export prediction stats
    df = processed_test_data.to_pandas()
    inputs = df["source"].tolist()
    targets = df["target"].tolist()
    save_prediction_stats(Path(args.save_dir) / "predictions.csv", df["id"].tolist(), inputs, targets, batch_decoded)

    # # Save the predictions to disk
    # with open(os.path.join(model_folder, "predictions.txt"), "w") as f:
    #     for item in batch_decoded:
    #         f.write("%s\n" % item)

    return batch_decoded

def load_tokenize_data(tokenizer, args):

    # custom dataset
    data_files = {}
    data_files["test"] = args.ds_test_path

    datasets = load_dataset("csv", data_files=data_files)
    len_test_dataset = len(datasets['test'])

    test_data = datasets['test']

    # Let us print the initial length of the dataset
    print(f'Initial length of the test datasets: {len_test_dataset=}\n')

    # Load and tokenize data
    if os.path.exists(args.cache_data):
        processed_test_data = load_from_disk(args.cache_data + '_eval')
        return test_data, processed_test_data
    else:
        # The preprocess function prepares the data for the model.
        def preprocess_function(examples):
            nonlocal tokenizer
            source = [ex for ex in examples["source"]]
            target = [ex for ex in examples["target"]]

            # truncation = True
            truncation = False if args.remove_long_samples else True # if remove-long-samples is True, we will remove that samples later

            model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=truncation) 
            labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=truncation) 

            model_inputs["labels"] = labels["input_ids"].copy()
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
            ]
            return model_inputs
        
        def remove_long_samples(examples):
            nonlocal tokenizer
            filtered_data = examples.filter(lambda x: len(x['input_ids']) <= args.max_source_len and len(x['labels']) <= args.max_target_len)

            return filtered_data

        num_proc = args.num_proc

        # The map function applies the preprocess function to the entire dataset
        
        processed_test_data = test_data.map(
            preprocess_function,
            batched=True,
            # remove_columns=test_data.column_names,
            num_proc=num_proc,
            load_from_cache_file=False,
        )
        print(f'  ==> Loaded {len(processed_test_data)} test samples')

        processed_test_data = remove_long_samples(processed_test_data)
        len_test_dataset = len(processed_test_data)
        
        # Let us print the initial length of the dataset
        if args.debug:
            print(f'Length of the test datasets after long sequences filtering - test : {len_test_dataset=}\n')
        
        # save after filtering
        processed_test_data.save_to_disk(args.cache_data + '_test')
        return test_data, processed_test_data

def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    set_seeds(42)

    # Save command to file
    with open(os.path.join(args.save_dir, "command_predict.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and update tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.load)

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    test_data, processed_test_data = load_tokenize_data(tokenizer, args)

    if args.data_num != -1:
        processed_test_data = processed_test_data.select([i for i in range(args.data_num)])

    run_inference(args, args.load, test_data, processed_test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning of Salesforce/codet5p-220m for execution aware code-related tasks")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=320, type=int)
    parser.add_argument('--max-target-len', default=128, type=int)
    parser.add_argument('--cache-data', default='cache_data/summarize_python', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Load datasets
    parser.add_argument('--ds-test-path', default='./dataset/test.csv', type=str)

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/summarize_python", type=str)

    # custom argumetns
    parser.add_argument('--remove-long-samples', default=False, action='store_true')
    parser.add_argument('--num-proc', default=2, type=int)

    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)