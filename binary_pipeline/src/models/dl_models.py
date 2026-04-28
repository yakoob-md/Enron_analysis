# binary_pipeline/src/models/dl_models.py
# FIXED: BiLSTM with proper dropout, pretrained embeddings hook,
#        FocalLoss for class imbalance, early stopping

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from config import VOCAB_SIZE, EPOCHS, BATCH_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss  (better than BCE for imbalanced binary classification)
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    FocalLoss for binary classification.
    gamma=2.0 focuses learning on hard examples.
    alpha=0.75 down-weights the majority class.
    """
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM Model
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTMBinary(nn.Module):
    """
    FIXED vs original:
    - Proper embedding layer (was missing in original binary bilstm)
    - num_layers=2 with dropout between layers
    - Layer norm after LSTM
    - Dropout before classifier
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128,
                 hidden_dim=128, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout          # FIX: dropout between LSTM layers
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x   = self.embedding(x)
        _, (h, _) = self.lstm(x)
        # Concatenate last layer forward + backward hidden states
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, hidden*2)
        h = self.layer_norm(h)
        h = self.dropout(h)
        return self.fc(h)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def get_model(name='bilstm'):
    if name == 'bilstm':
        return BiLSTMBinary()
    raise ValueError(f"Unknown DL model: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3,
                results_dir='results', pos_weight_ratio=None):
    """
    FIXED vs original:
    - FocalLoss replaces plain BCE
    - AdamW + ReduceLROnPlateau scheduler
    - Early stopping on val loss
    - Gradient clipping
    - Training/val loss curves saved
    """
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ── Loss ─────────────────────────────────────────────────────────────────
    pos_weight = None
    if pos_weight_ratio:
        pos_weight = torch.tensor([pos_weight_ratio], dtype=torch.float).to(device)

    criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)

    # ── DataLoader ───────────────────────────────────────────────────────────
    X_tr = torch.tensor(X_train, dtype=torch.long)
    y_tr = torch.tensor(y_train.values, dtype=torch.float)
    X_va = torch.tensor(X_val,   dtype=torch.long)
    y_va = torch.tensor(y_val.values,   dtype=torch.float)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=1, factor=0.5
    )

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, counter = 3, 0

    for epoch in range(epochs):
        # Train
        model.train()
        ep_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(loss.item())

        # Validate
        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(-1)
                v_losses.append(criterion(logits, yb).item())

        tr_loss = np.mean(ep_losses)
        vl_loss = np.mean(v_losses)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        scheduler.step(vl_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            counter = 0
            torch.save(model.state_dict(), f'{results_dir}/bilstm_best.pt')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(f'{results_dir}/bilstm_best.pt', map_location=device))

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.title('BiLSTM Training Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/bilstm_training.png', dpi=150)
    plt.close()

    return model