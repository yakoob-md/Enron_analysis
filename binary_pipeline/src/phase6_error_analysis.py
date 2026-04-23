import numpy as np
import torch
from transformers import PreTrainedModel
from torch.utils.data import DataLoader, TensorDataset


def error_analysis(model, X_test, y_test, X_text, threshold=0.5):

    print("\n--- Phase 6: Error Analysis ---")

    # ===============================
    # BERT (BATCHED)
    # ===============================
    if isinstance(model, PreTrainedModel):
        model.eval()

        dataset = TensorDataset(
            X_test['input_ids'],
            X_test['attention_mask']
        )

        loader = DataLoader(dataset, batch_size=8)

        all_preds = []
        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = [b.to(device) for b in batch]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits.squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > threshold).astype(int)

                all_preds.extend(preds)

        y_pred = np.array(all_preds)

    # ===============================
    # PyTorch LSTM
    # ===============================
    elif isinstance(model, torch.nn.Module):
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
            outputs = model(X_tensor).squeeze()

            y_prob = torch.sigmoid(outputs).cpu().numpy()
            y_pred = (y_prob > threshold).astype(int)

    # ===============================
    # ML models
    # ===============================
    else:
        y_pred = model.predict(X_test)

    y_pred = np.array(y_pred).astype(int)

    # ===============================
    # ERROR COLLECTION
    # ===============================
    false_pos = []
    false_neg = []

    for i in range(len(y_test)):

        if y_test.iloc[i] == 0 and y_pred[i] == 1:
            false_pos.append(X_text.iloc[i])

        if y_test.iloc[i] == 1 and y_pred[i] == 0:
            false_neg.append(X_text.iloc[i])

    # ===============================
    # PRINT
    # ===============================
    print(f"False Positives: {len(false_pos)}")
    if false_pos:
        print("Sample FP:", false_pos[0][:200])

    print(f"\nFalse Negatives: {len(false_neg)}")
    if false_neg:
        print("Sample FN:", false_neg[0][:200])