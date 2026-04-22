# multiclass_pipeline/training/trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_multiclass(model, train_loader, val_loader, class_weights=None, 
                     epochs=5, lr=0.001, model_name="model", results_dir="results"):
    
    set_seed(42)
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use weights if provided (for imbalanced multiclass)
    if class_weights is not None:
        # Convert dict to ordered list/tensor
        weight_list = [class_weights.get(i, 1.0) for i in range(max(class_weights.keys())+1)]
        weights_t = torch.tensor(weight_list, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_t)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    patience = 2
    counter = 0

    for epoch in range(epochs):
        model.train()
        batch_losses, batch_accs = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).long()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == yb).float().mean().item()
            batch_losses.append(loss.item())
            batch_accs.append(acc)

        model.eval()
        v_batch_losses, v_batch_accs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).long()
                logits = model(xb)
                v_loss = criterion(logits, yb).item()
                v_preds = torch.argmax(logits, dim=1)
                v_acc = (v_preds == yb).float().mean().item()
                v_batch_losses.append(v_loss)
                v_batch_accs.append(v_acc)
        
        val_loss = np.mean(v_batch_losses)
        val_acc = np.mean(v_batch_accs)
        train_losses.append(np.mean(batch_losses))
        val_losses.append(val_loss)
        train_accs.append(np.mean(batch_accs))
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f"{results_dir}/{model_name}_best.pt")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    # History Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_title('Loss')
    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Val')
    ax2.set_title('Accuracy')
    plt.savefig(f"{results_dir}/training_{model_name}.png")
    plt.close()

    history = {"train_loss": train_losses, "val_loss": val_losses, "train_acc": train_accs, "val_acc": val_accs}
    with open(f"{results_dir}/{model_name}_history.json", "w") as f:
        json.dump(history, f)

    return model
