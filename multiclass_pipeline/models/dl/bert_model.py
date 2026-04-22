# multiclass_pipeline/models/dl/bert_model.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

def get_bert_multiclass(num_labels=5):
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        problem_type='single_label_classification'
    )
    return model

def train_bert_step(model, ids, mask, y_batch, optimizer):
    # In training loop, use integer labels 0-4, not binary
    # Loss is computed internally by HuggingFace when labels are passed
    model.train()
    optimizer.zero_grad()
    outputs = model(input_ids=ids, attention_mask=mask, labels=y_batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return loss.item(), outputs.logits
