# multiclass_pipeline/evaluation/evaluator.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve

def evaluate_multiclass(y_test, y_pred, y_prob, model_name, results_dir, class_names):
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n=== {model_name} MULTICLASS RESULTS ===")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"{results_dir}/cm_{model_name.lower()}.png")
    plt.close()

    # Per-class ROC-AUC (One-vs-Rest)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    if y_prob is not None:
        y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
        for i, cls in enumerate(class_names):
            try:
                auc_val = roc_auc_score(y_bin[:, i], y_prob[:, i])
                print(f"ROC-AUC ({cls}): {auc_val:.4f}")
                metrics[f'auc_{cls.lower()}'] = auc_val
            except Exception:
                print(f"ROC-AUC ({cls}): N/A")
                metrics[f'auc_{cls.lower()}'] = None

        macro_auc = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
        print(f"Macro ROC-AUC: {macro_auc:.4f}")
        metrics['roc_auc_macro'] = macro_auc
        
        # Plot Multiclass ROC
        plot_multiclass_roc(y_bin, y_prob, class_names, model_name, results_dir)
    
    return metrics

def plot_multiclass_roc(y_bin, y_prob, class_names, model_name, results_dir):
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC (OvR) — {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f"{results_dir}/roc_multi_{model_name.lower()}.png", dpi=150)
    plt.close()

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
    plt.close()
