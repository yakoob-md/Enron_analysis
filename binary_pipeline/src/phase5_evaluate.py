import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    auc
)
from sklearn.model_selection import learning_curve
import torch
from transformers import PreTrainedModel
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from utils.table_visualizer import save_styled_table, export_classification_report


def find_optimal_threshold(y_test, y_prob, method='youden'):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    if method == 'youden':
        j = tpr - fpr
        idx = np.argmax(j)

    elif method == 'f1':
        precisions, recalls, pr_thresh = precision_recall_curve(y_test, y_prob)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        idx = np.argmax(f1s[:-1])
        return float(pr_thresh[idx])

    elif method == 'gmean':
        gmean = np.sqrt(tpr * (1 - fpr))
        idx = np.argmax(gmean)

    elif method == 'cost':
        cost_fn, cost_fp = 3, 1
        costs = cost_fn * (1 - tpr) + cost_fp * fpr
        idx = np.argmin(costs)

    return float(thresholds[idx])


def plot_roc_curve(y_test, y_prob, model_name, results_dir='results'):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', lw=2)
    plt.plot([0,1],[0,1],'--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/roc_{model_name.lower()}.png', dpi=150)
    plt.show()
    return auc


def plot_pr_curve(y_test, y_prob, model_name, results_dir='results'):
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, label=f'PR (AP = {ap:.3f})', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/pr_{model_name.lower()}.png', dpi=150)
    plt.show()
    return ap


def plot_threshold_f1(y_test, y_prob, model_name, chosen_threshold, results_dir='results'):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1s[:-1], label='F1 Score', color='purple', lw=2)
    
    # Mark the chosen threshold
    idx = np.argmin(np.abs(thresholds - chosen_threshold))
    plt.scatter([chosen_threshold], [f1s[idx]], color='red', s=100, label=f'Chosen ({chosen_threshold:.3f})', zorder=5)
    
    plt.axvline(x=chosen_threshold, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'Threshold vs F1 — {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/f1_threshold_{model_name.lower()}.png', dpi=150)
    plt.show()


def plot_learning_curves(model, X, y, model_name, results_dir):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=3, scoring='accuracy', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label='Train Accuracy')
    plt.plot(train_sizes, test_mean, label='Cross-Val Accuracy')
    plt.title(f'Learning Curves — {model_name}')
    plt.xlabel('Training Samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/learning_{model_name.lower()}.png", dpi=150)
    plt.show()


def plot_threshold_vs_metrics(y_test, y_prob, model_name, results_dir='results'):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.plot(thresholds, f1s[:-1], label='F1')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold vs Metrics — {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/threshold_{model_name.lower()}.png', dpi=150)
    plt.show()


def get_probabilities(model, X_test):
    """Unified probability extraction for ML, BERT, BiLSTM."""

    if isinstance(model, PreTrainedModel):
        model.eval()
        dataset = TensorDataset(X_test['input_ids'], X_test['attention_mask'])
        loader = DataLoader(dataset, batch_size=8)
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
                all_probs.extend(np.atleast_1d(probs))
        return np.array(all_probs)

    elif isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.long)
            outputs = model(X_tensor).squeeze()
            return torch.sigmoid(outputs).cpu().numpy()

    elif hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]

    else:
        return None


def plot_probability_distribution(y_prob, y_test):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    plt.hist(y_prob, bins=50)
    plt.title("Prediction Probability Distribution")
    plt.show()

    y_test = np.array(y_test)
    plt.figure()
    plt.hist(y_prob[y_test==0], bins=50, alpha=0.5, label='Non-Disclosure')
    plt.hist(y_prob[y_test==1], bins=50, alpha=0.5, label='Disclosure')
    plt.legend()
    plt.title("Class-wise Probability Distribution")
    plt.show()


def evaluate(model, X_test, y_test, model_name='MODEL',
             threshold_method='youden', results_dir='results',
             X_train=None, y_train=None):

    os.makedirs(results_dir, exist_ok=True)
    
    # Plot Learning Curves for ML models (if training data provided)
    if X_train is not None and y_train is not None and not isinstance(model, torch.nn.Module):
        print(f"--- Plotting Learning Curves for {model_name} ---")
        plot_learning_curves(model, X_train, y_train, model_name, results_dir)

    y_prob = get_probabilities(model, X_test)

    if y_prob is not None:
        print(f"\n--- Threshold Analysis for {model_name} ---")
        print(f"{'Method':<12} | {'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
        print("-" * 60)
        
        threshold_results = []
        for m in ['youden', 'f1', 'gmean', 'cost']:
            t = find_optimal_threshold(y_test, y_prob, method=m)
            y_pred_m = (y_prob > t).astype(int)
            p_m = precision_score(y_test, y_pred_m, zero_division=0)
            r_m = recall_score(y_test, y_pred_m, zero_division=0)
            f_m = f1_score(y_test, y_pred_m, zero_division=0)
            print(f"{m:<12} | {t:<10.4f} | {p_m:<10.4f} | {r_m:<10.4f} | {f_m:<10.4f}")
            threshold_results.append({
                'Method': m,
                'Threshold': t,
                'Precision': p_m,
                'Recall': r_m,
                'F1': f_m
            })
        
        # Save Threshold Table
        df_thresh = pd.DataFrame(threshold_results)
        table_dir = os.path.join(results_dir, "tables", "binary")
        save_styled_table(df_thresh, f"binary_threshold_{model_name.lower()}.png", 
                          table_dir, f"Threshold Analysis - {model_name}")

        plot_threshold_vs_metrics(y_test, y_prob, model_name, results_dir)
        
        threshold = find_optimal_threshold(y_test, y_prob, method=threshold_method)
        print(f"\nOptimal threshold ({threshold_method}): {threshold:.4f}")

        # New specific F1 plot with mark
        plot_threshold_f1(y_test, y_prob, model_name, threshold, results_dir)

        y_pred = (y_prob > threshold).astype(int)

        roc_auc = plot_roc_curve(y_test, y_prob, model_name, results_dir)
        pr_auc  = plot_pr_curve(y_test, y_prob, model_name, results_dir)
        plot_threshold_vs_metrics(y_test, y_prob, model_name, results_dir)

    else:
        y_pred = model.predict(X_test)
        y_prob = None
        roc_auc = None
        pr_auc = None
        threshold = 0.5

    y_pred = np.array(y_pred).astype(int)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n=== {model_name} RESULTS ===")
    print(classification_report(y_test, y_pred,
          target_names=['Non-Disclosure', 'Disclosure'], zero_division=0))
    
    # Save Classification Report Table
    report_dict = classification_report(y_test, y_pred, target_names=['Non-Disclosure', 'Disclosure'], output_dict=True, zero_division=0)
    table_dir = os.path.join(results_dir, "tables", "binary")
    export_classification_report(report_dict, f"binary_classification_{model_name.lower()}.png", 
                                 table_dir, f"Classification Report - {model_name}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    if roc_auc:
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Threshold: {threshold:.4f}")

    return {
        'accuracy':  round(accuracy, 4),
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'f1':        round(f1, 4),
        'roc_auc':   round(roc_auc, 4) if roc_auc else None,
        'pr_auc':    round(pr_auc, 4)  if pr_auc  else None,
        'threshold': round(threshold, 4)
    }