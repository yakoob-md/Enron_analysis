# binary_pipeline/src/models/bert_model.py
# FIXED: proper scheduler, warmup, gradient clipping, early stopping

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup


def get_model(name='bert'):
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=1,                           # binary → sigmoid
        problem_type='single_label_classification'
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=3,
                batch_size=16,
                lr=2e-5,
                results_dir='results',
                pos_weight_ratio=None):
    """
    FIXED vs original:
    - Linear warmup scheduler (was missing)
    - Gradient clipping (was missing)
    - Early stopping on val loss (was missing)
    - pos_weight for class imbalance (was missing)
    - Proper batch-level evaluation tracking
    """
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ── Class imbalance: weighted BCE loss ──────────────────────────────────
    if pos_weight_ratio:
        # pos_weight = neg_count / pos_count  (e.g. 2605/7381 ≈ 0.35)
        # For BERT we override the loss manually
        pos_weight = torch.tensor([pos_weight_ratio], dtype=torch.float).to(device)
    else:
        pos_weight = None

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_dataset = TensorDataset(
        X_train['input_ids'],
        X_train['attention_mask'],
        torch.tensor(y_train.values, dtype=torch.float)
    )
    val_dataset = TensorDataset(
        X_val['input_ids'],
        X_val['attention_mask'],
        torch.tensor(y_val.values, dtype=torch.float)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, counter = 2, 0

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        ep_losses = []
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask)
            logits  = outputs.logits.squeeze(-1)
            loss    = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # FIX: clip
            optimizer.step()
            scheduler.step()                                           # FIX: schedule
            ep_losses.append(loss.item())

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        v_losses = []
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                outputs = model(input_ids=ids, attention_mask=mask)
                logits  = outputs.logits.squeeze(-1)
                v_losses.append(criterion(logits, labels).item())

        tr_loss = np.mean(ep_losses)
        vl_loss = np.mean(v_losses)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f}")

        # ── Early stopping ───────────────────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            counter = 0
            torch.save(model.state_dict(), f'{results_dir}/bert_best_weights.pt')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best
    model.load_state_dict(torch.load(f'{results_dir}/bert_best_weights.pt', map_location=device))

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.title('BERT Training Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/bert_training.png', dpi=150)
    plt.close()

    return model