def train_with_diagnostics(model, X_train, y_train, X_val, y_val,
                           epochs=5, batch_size=16, lr=2e-5):
    import torch.optim as optim

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    train_accs,  val_accs   = [], []

    for epoch in range(epochs):
        model.train()
        batch_losses, batch_accs = [], []

        for i in range(0, len(y_train), batch_size):
            xb = torch.tensor(X_train[i:i+batch_size], dtype=torch.long)
            yb = torch.tensor(y_train.iloc[i:i+batch_size].values, dtype=torch.float)

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
            X_v = torch.tensor(X_val, dtype=torch.long)
            y_v = torch.tensor(y_val.values, dtype=torch.float)
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

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses,   label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()

    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs,   label='Val Acc')
    ax2.set_title('Accuracy Curves')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/training_diagnostics.png', dpi=150)
    plt.show()

    return model