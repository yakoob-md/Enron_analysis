from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import numpy as np
import torch
from transformers import PreTrainedModel
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def evaluate(model, X_test, y_test, model_name='MODEL'):

    # ===============================
    # BERT MODEL (BATCHED)
    # ===============================
    if isinstance(model, PreTrainedModel):
        model.eval()

        dataset = TensorDataset(
            X_test['input_ids'],
            X_test['attention_mask']
        )

        loader = DataLoader(dataset, batch_size=8)

        all_probs = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                input_ids, attention_mask = batch

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits.squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.extend(probs)

                # 🔥 Progress log
                if i % 20 == 0:
                    print(f"Evaluating batch {i}/{len(loader)}")

        y_prob = np.array(all_probs)

        # ===============================
        # 📊 PROBABILITY ANALYSIS
        # ===============================
        print("\n=== PROBABILITY STATS ===")
        print("Min:", y_prob.min())
        print("Max:", y_prob.max())
        print("Mean:", y_prob.mean())

        print("\nPercentiles:")
        for p in [10, 25, 50, 75, 90]:
            print(f"{p}th:", np.percentile(y_prob, p))

        # 🔥 Histogram (overall)
        plt.figure(figsize=(6,4))
        plt.hist(y_prob, bins=50)
        plt.title("Prediction Probability Distribution")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.show()

        # 🔥 Class-wise histogram
        y_test_np = np.array(y_test)

        y_prob_0 = y_prob[y_test_np == 0]
        y_prob_1 = y_prob[y_test_np == 1]

        plt.figure(figsize=(6,4))
        plt.hist(y_prob_0, bins=50, alpha=0.5, label="Non-Disclosure")
        plt.hist(y_prob_1, bins=50, alpha=0.5, label="Disclosure")
        plt.legend()
        plt.title("Class-wise Probability Distribution")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.show()

        # 🔥 Threshold (adjust here)
        threshold = 0.66
        y_pred = (y_prob > threshold).astype(int)

    # ===============================
    # PyTorch LSTM
    # ===============================
    elif isinstance(model, torch.nn.Module):
        model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.long)
            outputs = model(X_tensor).squeeze()

            y_prob = torch.sigmoid(outputs).cpu().numpy()
            threshold = 0.7
            y_pred = (y_prob > threshold).astype(int)

    # ===============================
    # ML models
    # ===============================
    elif hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        threshold = 0.5
        y_pred = (y_prob > threshold).astype(int)

    else:
        y_pred = model.predict(X_test)
        y_prob = None

    y_pred = np.array(y_pred).astype(int)

    # ===============================
    # METRICS
    # ===============================
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except:
            pass

    # ===============================
    # PRINT RESULTS
    # ===============================
    print(f"\n=== {model_name} RESULTS ===")

    print(classification_report(
        y_test,
        y_pred,
        target_names=['Non-Disclosure', 'Disclosure'],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if roc_auc is not None:
        print("ROC-AUC:", round(roc_auc, 4))
    else:
        print("ROC-AUC: Not available")

    # ===============================
    # RETURN
    # ===============================
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'roc_auc': round(roc_auc, 4) if roc_auc is not None else None
    }