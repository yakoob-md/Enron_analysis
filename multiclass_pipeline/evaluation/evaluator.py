# multiclass_pipeline/evaluation/evaluator.py
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluate_multiclass(y_true, y_pred, y_prob, model_name, results_dir, class_names):
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n=== {model_name} MULTICLASS RESULTS ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"{results_dir}/cm_{model_name.lower()}.png")
    plt.close()
    
    # Macro/Micro Metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    
    # ROC AUC (OvR)
    try:
        metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except:
        metrics['roc_auc_ovr'] = 0.0
        
    return metrics
