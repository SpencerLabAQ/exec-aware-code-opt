from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
import random

class MLMExecAwareDataCollator:
    def __init__(self, tokenizer, model, mlm_probability=0.15, padding = True, label_pad_token_id = -100):
        self.tokenizer = tokenizer
        self.model = model
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=mlm_probability
        )
        self.seq2seq_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            model=model,
            padding=padding,
            label_pad_token_id=label_pad_token_id
        )

    def __call__(self, features):
        batch = {}
        mlm_features = []
        seq2seq_features = []
        # print("inside collator features", features[0].keys())
        for feature in features:
            if random.random() < 0.5:
                # MLM task
                mlm_features.append({'input_ids': feature['code_input_ids']})
            else:
                # Text generation task
                seq2seq_features.append({
                    'input_ids': feature['exec_input_ids'],
                    'labels': feature['exec_labels'],
                    'attention_mask': feature['attention_mask']
                })
        # Prepare MLM batch
        if mlm_features:
            mlm_batch = self.mlm_collator(mlm_features)
            batch['mlm_input_ids'] = mlm_batch['input_ids']
            batch['mlm_labels'] = mlm_batch['labels']
            batch['mlm_attention_mask'] = mlm_batch['attention_mask']
        # Prepare Seq2Seq batch
        if seq2seq_features:
            seq2seq_batch = self.seq2seq_collator(seq2seq_features)
            batch.update(seq2seq_batch)
        return batch
