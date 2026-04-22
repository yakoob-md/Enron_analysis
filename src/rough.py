import os
import torch
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

def train_with_diagnostics(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=5,
    batch_size=16,
    lr=2e-5,
    model_name="model",
    results_dir="results"
):
    set_seed(42)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # STEP 4: Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=1, factor=0.5
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # STEP 3: Early Stopping
    best_val_loss = float('inf')
    patience = 2
    counter = 0

    for epoch in range(epochs):
        model.train()
        batch_losses, batch_accs = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb).squeeze()
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == yb).float().mean().item()

            batch_losses.append(loss.item())
            batch_accs.append(acc)

        # Validation
        model.eval()
        with torch.no_grad():
            X_v = torch.tensor(X_val, dtype=torch.long).to(device)
            y_v = torch.tensor(y_val.values, dtype=torch.float).to(device)

            val_logits = model(X_v).squeeze()
            val_loss = criterion(val_logits, y_v).item()

            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_v).float().mean().item()

        train_losses.append(np.mean(batch_losses))
        val_losses.append(val_loss)
        train_accs.append(np.mean(batch_accs))
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_acc:.4f}")
        
        # STEP 4: Scheduler step
        scheduler.step(val_loss)
        
        # STEP 3: Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f"{results_dir}/{model_name}_best.pt")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    # Save final model
    torch.save(model.state_dict(), f"{results_dir}/{model_name}_final.pt")

    # Plot curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()

    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Accuracy Curves')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_{model_name}.png", dpi=150)
    plt.show()
    
    # STEP 6: Save Training Logs (History)
    history = {
        "train_loss": [float(l) for l in train_losses],
        "val_loss": [float(l) for l in val_losses],
        "train_acc": [float(a) for a in train_accs],
        "val_acc": [float(a) for a in val_accs]
    }

    with open(f"{results_dir}/{model_name}_history.json", "w") as f:
        json.dump(history, f)


    return model