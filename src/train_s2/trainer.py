from transformers import Seq2SeqTrainer

class MLMExecAwareTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        total_loss = 0
        outputs = {}
        
        # MLM Task
        if 'mlm_input_ids' in inputs:
            mlm_outputs = model(
                input_ids=inputs['mlm_input_ids'],
                attention_mask=inputs['mlm_attention_mask'],
                labels=inputs['mlm_labels'],
            )
            total_loss += mlm_outputs.loss
        
        # Text Generation Task
        if 'input_ids' in inputs and 'labels' in inputs:
            seq2seq_outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels'],
            )
            total_loss += seq2seq_outputs.loss
        
        return (total_loss, outputs) if return_outputs else total_loss