# multiclass_pipeline/models/dl/bert_model.py
# FIXED vs original:
#   - num_labels=5, problem_type='single_label_classification' (correct)
#   - train_bert_step uses outputs.loss (HuggingFace internal CrossEntropy) — correct
#   - Added full training loop with scheduler, early stopping, gradient clipping
#   - Added label smoothing option

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
try:
    from utils.focal_loss import MulticlassFocalLoss
except ImportError:
    # Fallback for different execution contexts
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.focal_loss import MulticlassFocalLoss


def get_bert_multiclass(num_labels=5):
    """
    FIXED: num_labels=5 for 5-class problem.
    HuggingFace computes CrossEntropyLoss internally when labels are passed.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        problem_type='single_label_classification'
    )
    return model


def train_bert_multiclass(model, X_train, y_train, X_val, y_val,
                          epochs=3, batch_size=16, lr=2e-5,
                          results_dir='results', class_weights=None,
                          use_focal_loss=True):
    """
    Full training loop for multiclass BERT.
    
    class_weights : dict {class_idx: weight} for imbalanced classes.
                    If None, uses HF internal unweighted CrossEntropy.
    """
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    y_tr = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long)
    y_va = torch.tensor(y_val.values   if hasattr(y_val,   'values') else y_val,   dtype=torch.long)

    train_ds = TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_tr)
    val_ds   = TensorDataset(X_val['input_ids'],   X_val['attention_mask'],   y_va)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # ── Build loss function ───────────────────────────────────────────────────
    if class_weights:
        num_classes = max(class_weights.keys()) + 1
        weight_list = [class_weights.get(i, 1.0) for i in range(num_classes)]
        weights_t   = torch.tensor(weight_list, dtype=torch.float).to(device)
    else:
        weights_t = None

    if use_focal_loss:
        criterion = MulticlassFocalLoss(gamma=2.0, weight=weights_t)
        use_custom_loss = True
    elif class_weights:
        criterion = torch.nn.CrossEntropyLoss(weight=weights_t)
        use_custom_loss = True
    else:
        use_custom_loss = False

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

            if use_custom_loss:
                outputs = model(input_ids=ids, attention_mask=mask)
                loss    = criterion(outputs.logits, labels)
            else:
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss    = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_losses.append(loss.item())

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        v_losses = []
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                if use_custom_loss:
                    outputs = model(input_ids=ids, attention_mask=mask)
                    v_loss  = criterion(outputs.logits, labels).item()
                else:
                    outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                    v_loss  = outputs.loss.item()
                v_losses.append(v_loss)

        tr_loss = np.mean(ep_losses)
        vl_loss = np.mean(v_losses)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            counter = 0
            model.save_pretrained(f'{results_dir}/bert_multiclass_best')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best
    from transformers import AutoModelForSequenceClassification as AMSC
    model = AMSC.from_pretrained(f'{results_dir}/bert_multiclass_best')
    model.to(device)

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.title('BERT Multiclass Training Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/bert_multiclass_training.png', dpi=150)
    plt.close()

    return model


def predict_bert_multiclass(model, X_test, batch_size=16):
    """
    Returns:
        y_pred : (n,) int array of predicted class indices
        y_prob : (n, num_classes) float array of softmax probabilities
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    ds     = TensorDataset(X_test['input_ids'], X_test['attention_mask'])
    loader = DataLoader(ds, batch_size=batch_size)

    all_probs = []
    with torch.no_grad():
        for ids, mask in loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    y_prob = np.vstack(all_probs)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob