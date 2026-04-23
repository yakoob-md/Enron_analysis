# multiclass_pipeline/training/trainer.py
# FIXED vs original:
#   - FocalLoss option added (replaces plain CrossEntropyLoss)
#   - Label smoothing option added
#   - class_weights now properly converted to tensor
#   - Early stopping correctly reloads best model
#   - Softmax probabilities returned for evaluation

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import json
from torch.utils.data import DataLoader, TensorDataset

from utils.focal_loss import MulticlassFocalLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_multiclass(model, train_loader, val_loader,
                     class_weights=None,
                     epochs=5, lr=0.001,
                     model_name="model",
                     results_dir="results",
                     use_focal_loss=True,
                     label_smoothing=0.1):
    """
    FIXED + EXTENDED training loop.
    
    Parameters
    ----------
    use_focal_loss   : bool  — use FocalLoss instead of CrossEntropyLoss
    label_smoothing  : float — 0.0 disables label smoothing
    class_weights    : dict  {int: float} — per-class weights
    """
    set_seed(42)
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ── Build loss function ───────────────────────────────────────────────────
    if class_weights is not None:
        num_classes = max(class_weights.keys()) + 1
        weight_list = [class_weights.get(i, 1.0) for i in range(num_classes)]
        weights_t   = torch.tensor(weight_list, dtype=torch.float).to(device)
    else:
        weights_t = None

    if use_focal_loss:
        criterion = MulticlassFocalLoss(gamma=2.0, weight=weights_t)
    elif label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(weight=weights_t, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_t)

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=1, factor=0.5
    )

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_loss = float('inf')
    patience_cfg  = 3
    counter       = 0
    best_path     = f"{results_dir}/{model_name}_best.pt"

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        batch_losses, batch_accs = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).long()
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            preds = torch.argmax(logits, dim=1)
            batch_losses.append(loss.item())
            batch_accs.append((preds == yb).float().mean().item())

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        v_losses, v_accs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).long()
                logits = model(xb)
                v_losses.append(criterion(logits, yb).item())
                v_accs.append((torch.argmax(logits, 1) == yb).float().mean().item())

        tr_loss, tr_acc = np.mean(batch_losses), np.mean(batch_accs)
        vl_loss, vl_acc = np.mean(v_losses),     np.mean(v_accs)
        train_losses.append(tr_loss);  val_losses.append(vl_loss)
        train_accs.append(tr_acc);     val_accs.append(vl_acc)
        scheduler.step(vl_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            counter += 1
            if counter >= patience_cfg:
                print("Early stopping triggered.")
                break

    # ── Load best ─────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device))

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train'); ax1.plot(val_losses, label='Val')
    ax1.set_title('Loss'); ax1.legend()
    ax2.plot(train_accs,  label='Train'); ax2.plot(val_accs,  label='Val')
    ax2.set_title('Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_{model_name}.png", dpi=150)
    plt.close()

    # ── Save history ─────────────────────────────────────────────────────────
    history = {
        "train_loss": train_losses, "val_loss": val_losses,
        "train_acc":  train_accs,   "val_acc":  val_accs
    }
    with open(f"{results_dir}/{model_name}_history.json", "w") as f:
        json.dump(history, f)

    return model


def predict_multiclass(model, X_test, batch_size=64):
    """
    Run inference and return (y_pred, y_prob).
    y_prob is softmax probabilities, shape (n, num_classes).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.long)),
        batch_size=batch_size
    )

    all_probs = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    y_prob = np.vstack(all_probs)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob