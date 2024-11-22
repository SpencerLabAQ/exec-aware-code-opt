"""
Finetuning of Salesforce/codet5p-220m for execution aware code-related tasks
"""

import os
import pprint
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import random
import torch

from datetime import datetime

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, DataCollatorForSeq2Seq

from metrics import instance_exact_match, exact_match

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def dump_eval_predictions(decoded_preds, decoded_labels):
    df = pd.DataFrame({'target': decoded_labels, 'prediction': decoded_preds})
    ts_to_save = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(Path(__file__).parent.resolve() / f"{ts_to_save}_training_epoch_eval.csv", index=False)

def replace_tokens(list_ids, tokenizer, list = True):
    if list:
        return [np.where(np.array(ids) != -100, np.array(ids), tokenizer.pad_token_id) for ids in list_ids]
    else:
        return np.where(np.array(list_ids) != -100, np.array(list_ids), tokenizer.pad_token_id)

def prepare_compute_metrics(tokenizer, debug = False, task = "BC"):

    def compute_metrics(eval_pred):
        nonlocal tokenizer

        predictions_ids, labels_ids = eval_pred
        if isinstance(predictions_ids, tuple):
            predictions_ids = predictions_ids[0]

        # convert lists in numpys
        predictions_ids = np.array(predictions_ids)
        labels_ids = np.array(labels_ids)

        # Replace -100 in the labels as we can't decode them.
        predictions = np.where(predictions_ids != -100, predictions_ids, tokenizer.pad_token_id)
        labels = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)

        # Decode the predictions and the labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ---- begin logging eval ------

        if debug:
            print("decoded_preds", decoded_preds[:10])
            print("decoded_labels", decoded_labels[:10])
            print(f"{predictions_ids.shape=}")
            print(f"{labels_ids.shape=}")
            print(f"{len(decoded_preds)=}")
            print(f"{len(decoded_labels)=}")

            dump_eval_predictions(decoded_preds=decoded_preds, decoded_labels=decoded_labels)

        # ---- end logging eval ------

        # Compute the right metric for each task instance_exact_match score
        if task == "BC" or task == "LE" or task == "LC":
            instance_exact_match_score = instance_exact_match(out=decoded_preds, lbl=decoded_labels)
        elif task == "RD":
            instance_exact_match_score = exact_match(out=decoded_preds, lbl=decoded_labels)
        elif task == "PS":
            instance_exact_match_score = instance_exact_match(out=decoded_preds, lbl=decoded_labels, sep="\n")
        else:
            raise Exception(f"Task [{task}] not supported by this script.")
        return {f'{task}_metric': instance_exact_match_score}
    
    return compute_metrics

def run_training(args, model, tokenizer, train_data, eval_data):
    print(f"Starting main loop")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding=True, label_pad_token_id=tokenizer.pad_token_id)
    training_args = Seq2SeqTrainingArguments(
        # Training and evaluation settings
        output_dir=args.save_dir,
        overwrite_output_dir=False,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        per_device_eval_batch_size=args.batch_size_per_replica,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        
        # Optimizer
        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        # Logging
        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=100,

        # Misc
        local_rank=args.local_rank,
        fp16=args.fp16
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        compute_metrics=prepare_compute_metrics(tokenizer, args.debug, args.task) if args.task != "CO" else None
    )
    
    # Train the model
    trainer.train()
    model.eval()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def load_tokenize_data(tokenizer, args):
    # Load and tokenize data
    if os.path.exists(args.cache_data):
        train_data = load_from_disk(args.cache_data + '_train')
        eval_data = load_from_disk(args.cache_data + '_eval')
        return train_data, eval_data
    else:

        # custom dataset
        data_files = {}
        data_files["train"] = args.ds_train_path
        data_files["validation"] = args.ds_val_path

        datasets = load_dataset("csv", data_files=data_files)
        len_train_dataset = len(datasets['train'])

        # Let us print the initial length of the dataset
        print(f'Initial length of the datasets: {len_train_dataset=}\n')

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

            if args.debug:
                
                to_remove = examples.filter(lambda x: len(x['input_ids']) > args.max_source_len or len(x['labels']) > args.max_target_len)
                print(f"Removing {len(to_remove)} samples exceeding model input/output max lenght")
                
                if len(to_remove):
                    assert len(filtered_data) + len(to_remove) == len(examples), "Length of filtered + removed examples does not correspond to the original sample length"
                    
                    print("Example of too long ignored sample (decoded)", 
                          tokenizer.decode(replace_tokens(to_remove["input_ids"], tokenizer)[0], skip_special_tokens=True), 
                          tokenizer.decode(replace_tokens(to_remove["labels"], tokenizer)[0], skip_special_tokens=True)
                          )
                    print(f"Num of input tokens : {len(to_remove['input_ids'][0])}, num of output tokens : {len(to_remove['labels'][0])}")
                
                else:
                    print("No examples to ignore")

            return filtered_data

        num_proc = args.num_proc

        # The map function applies the preprocess function to the entire dataset
        train_data = datasets['train']
        train_data = train_data.map(
            preprocess_function,
            batched=True,
            remove_columns=train_data.column_names,
            num_proc=num_proc,
            load_from_cache_file=False,
        )
        print(f'  ==> Loaded {len(train_data)} train samples')

        train_data = remove_long_samples(train_data)
        len_train_dataset = len(train_data)

        eval_data = datasets['validation']
        eval_data = eval_data.map(
            preprocess_function,
            batched=True,
            remove_columns=eval_data.column_names,
            num_proc=num_proc,
            load_from_cache_file=False,
        )
        print(f'  ==> Loaded {len(eval_data)} validation samples')

        eval_data = remove_long_samples(eval_data)
        len_eval_dataset = len(eval_data)
        
        # Let us print the initial length of the dataset
        if args.debug:
            print(f'Length of the datasets after long sequences filtering - train : {len_train_dataset=}\n')
            print(f'Length of the datasets after long sequences filtering - validation : {len_eval_dataset=}\n')
        
        # save after filtering
        train_data.save_to_disk(args.cache_data + '_train')
        eval_data.save_to_disk(args.cache_data + '_eval')
        return train_data, eval_data


def load_update_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load, cache_dir = None)
    """
    Tokens to add:

    - <SEP> : special token that has been added between the program input and program code (both pre-training datasets)
    - <ne>, <e>, <e+>, <E>, <E+> : labels representing quantized values for the number of times a program line is executed (line execution dataset)
    - <varsep>, quantized_values: token added to divide the variables states in the program states dataset output
    - <NB>, <BC>, <BNC> : special tokens for the branch coverage (not branch, branch covered, branch not covered)
    """
    
    std_tokens = ["<SEP>", "classify: ", "optimize: "]

    # special tokens for program states execution
    task_tokens = {
        "BC": ["<NB>", "<BC>", "<BNC>"],
        "LE": ["<ne>", "<e>", "<E>", "<e+>", "<E+>"],
        "LC": ["<e>"],
        "PS": ["<varsep>", "basic_type", "array", "POSITIVE-REG", "POSITIVE-VL", "ZERO", "NEGATIVE-REG", "NEGATIVE-VL", "INIT", "NOT-INIT", "NOT-NULL"],
        "CO": ["<ne>", "<e>", "<E>", "<e+>", "<E+>", "<NB>", "<BC>", "<BNC>", "OTHER", "basic_type", "array", "POSITIVE-REG", "POSITIVE-VL", "ZERO", "NEGATIVE-REG", "NEGATIVE-VL", "INIT", "NOT-INIT", "NOT-NULL"], # direct fine-tuning strategy
        "RD": ["<1>", "<-1>", "<0>"]
    }

    if args.task in task_tokens.keys():
        tokenizer.add_tokens(std_tokens + task_tokens[args.task])
        print("Added tokens: ", std_tokens + task_tokens[args.task])
    else:
        tokenizer.add_tokens(std_tokens)
        print("Added tokens: ", std_tokens)

    return tokenizer


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    set_seeds(42)

    # Save command to file
    with open(os.path.join(args.save_dir, "command_train.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and update tokenizer
    tokenizer = load_update_tokenizer(args)

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    train_data, eval_data = load_tokenize_data(tokenizer, args)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])
        eval_data = eval_data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    model.resize_token_embeddings(len(tokenizer))    
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    if args.debug:
        print("Decoding the first 3 samples in the training and evaluation set")
        # train_lbl_ids, eval_lbl_ids = train_data["labels"], eval_data["labels"]
        # train_lbl_ids = np.where(train_lbl_ids != -100, train_lbl_ids, tokenizer.pad_token_id)
        # eval_lbl_ids = np.where(eval_lbl_ids != -100, eval_lbl_ids, tokenizer.pad_token_id)
        print("Train sources", tokenizer.batch_decode(train_data["input_ids"], skip_special_tokens=True)[:3])
        print("Train targets b", tokenizer.batch_decode(replace_tokens(train_data["labels"], tokenizer)[:3], skip_special_tokens=True))
        print("Eval sources", tokenizer.batch_decode(eval_data["input_ids"], skip_special_tokens=True)[:3])
        print("Eval targets", tokenizer.batch_decode(replace_tokens(eval_data["labels"], tokenizer)[:3], skip_special_tokens=True))

    run_training(args, model, tokenizer, train_data, eval_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning of Salesforce/codet5p-220m for execution aware code-related tasks")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=320, type=int)
    parser.add_argument('--max-target-len', default=128, type=int)
    parser.add_argument('--cache-data', default='cache_data/summarize_python', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)

    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')
    
    # Load datasets
    parser.add_argument('--ds-train-path', default='./dataset/train.csv', type=str)
    parser.add_argument('--ds-val-path', default='./dataset/val.csv', type=str)
    # parser.add_argument('--ds-test-path', default='./dataset/test.csv', type=str)

    '''
    Seq2seq code-related tasks supported by this script:
    - LE (Line Executions)
    - BC (Branch Coverage)
    - PS (Program Final States)
    - CO (Code Optimization)

    - RD (Regression Detection)
    '''
    parser.add_argument('--task', type=str, default='BC')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/summarize_python", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    # custom argumetns
    parser.add_argument('--remove-long-samples', default=False, action='store_true')
    parser.add_argument('--num-proc', default=2, type=int)

    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)