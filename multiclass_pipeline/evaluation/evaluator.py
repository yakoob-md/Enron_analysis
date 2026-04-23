# multiclass_pipeline/evaluation/evaluator.py
# FIXED + EXTENDED:
#   - Missing sklearn imports added
#   - Per-class One-vs-Rest threshold optimization
#   - Macro/micro ROC-AUC curves
#   - Per-class confusion matrix (heatmap)
#   - Macro/weighted F1

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION — One-vs-Rest per class
# ─────────────────────────────────────────────────────────────────────────────
def find_best_threshold_ovr(y_true_bin, y_prob_col, method='f1'):
    """
    Find the best classification threshold for a single class (OvR).

    y_true_bin : binary array (1 = this class, 0 = all others)
    y_prob_col : predicted probability for this class
    method     : 'f1' | 'youden' | 'gmean'
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true_bin, y_prob_col)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)

    if method == 'f1':
        idx = np.argmax(f1s[:-1])
        return float(thresholds[idx]), float(f1s[idx])

    fpr, tpr, roc_thresh = roc_curve(y_true_bin, y_prob_col)
    if method == 'youden':
        idx = np.argmax(tpr - fpr)
    elif method == 'gmean':
        idx = np.argmax(np.sqrt(tpr * (1 - fpr)))
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(roc_thresh[idx]), 0.0


def optimize_multiclass_thresholds(y_true, y_prob, class_names, method='f1'):
    """
    Run OvR threshold optimization for every class.

    Returns
    -------
    thresholds : dict  {class_name: optimal_threshold}
    summary    : list of dicts for printing
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    thresholds = {}
    summary = []

    print(f"\n{'='*60}")
    print(f"  PER-CLASS THRESHOLD OPTIMIZATION  (method={method})")
    print(f"{'='*60}")
    print(f"{'Class':<20} {'Threshold':>10} {'OvR F1':>10}")
    print(f"{'-'*44}")

    for i, name in enumerate(class_names):
        t, f1 = find_best_threshold_ovr(y_bin[:, i], y_prob[:, i], method=method)
        thresholds[name] = t
        summary.append({'class': name, 'threshold': t, 'ovr_f1': f1})
        print(f"  {name:<18} {t:>10.4f} {f1:>10.4f}")

    return thresholds, summary


def apply_multiclass_thresholds(y_prob, thresholds, class_names):
    """
    Apply per-class thresholds (OvR) and return final predictions.
    Ties (no class exceeds its threshold) fall back to argmax.
    """
    n = y_prob.shape[0]
    y_pred = np.full(n, -1, dtype=int)

    for i in range(n):
        scores = [
            y_prob[i, j] - thresholds[class_names[j]]
            for j in range(len(class_names))
        ]
        best = int(np.argmax(scores))
        # Require the best class to actually exceed its threshold
        if y_prob[i, best] >= thresholds[class_names[best]]:
            y_pred[i] = best
        else:
            y_pred[i] = int(np.argmax(y_prob[i]))  # fallback: argmax

    return y_pred


# ─────────────────────────────────────────────────────────────────────────────
# ROC — One-vs-Rest (macro + per class)
# ─────────────────────────────────────────────────────────────────────────────
def plot_multiclass_roc(y_true, y_prob, class_names, model_name, results_dir):
    """
    Plots per-class OvR ROC curves + macro-average ROC curve.
    Returns dict of per-class AUC and macro AUC.
    """
    os.makedirs(results_dir, exist_ok=True)
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    plt.figure(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_dict[i], tpr_dict[i], auc_dict[name] = fpr, tpr, roc_auc
        plt.plot(fpr, tpr, color=color, lw=1.5,
                 label=f'{name} (AUC={roc_auc:.3f})')

    # Macro average (interpolated)
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    auc_dict['macro'] = macro_auc
    plt.plot(all_fpr, mean_tpr, 'k--', lw=2.5,
             label=f'Macro avg (AUC={macro_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'gray', linestyle=':', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass OvR ROC — {model_name}')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/roc_multiclass_{model_name.lower()}.png', dpi=150)
    plt.close()

    # Also compute sklearn macro / weighted AUC
    try:
        auc_dict['sklearn_macro'] = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='macro'
        )
        auc_dict['sklearn_weighted'] = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='weighted'
        )
    except Exception as e:
        print(f"[WARN] roc_auc_score failed: {e}")

    return auc_dict


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def plot_multiclass_cm(y_true, y_pred, class_names, model_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/cm_{model_name.lower()}.png', dpi=150)
    plt.close()
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_multiclass(y_true, y_pred, y_prob, model_name,
                        results_dir, class_names,
                        threshold_method='f1'):
    """
    Full evaluation:
    1. Standard classification report (argmax predictions)
    2. Per-class OvR threshold optimization → re-evaluate with tuned preds
    3. OvR ROC curves + macro AUC
    4. Confusion matrix heatmap
    5. Return metrics dict
    """
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {model_name} MULTICLASS RESULTS  (argmax baseline)")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # ── Threshold optimization ────────────────────────────────────────────────
    thresholds, _ = optimize_multiclass_thresholds(
        y_true, y_prob, class_names, method=threshold_method
    )
    y_pred_tuned = apply_multiclass_thresholds(y_prob, thresholds, class_names)

    print(f"\n{'='*60}")
    print(f"  {model_name} RESULTS  (OvR threshold-tuned)")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred_tuned, target_names=class_names, zero_division=0))

    # ── Confusion matrices ───────────────────────────────────────────────────
    plot_multiclass_cm(y_true, y_pred,       class_names, f'{model_name}_argmax', results_dir)
    plot_multiclass_cm(y_true, y_pred_tuned, class_names, f'{model_name}_tuned',  results_dir)

    # ── ROC curves ───────────────────────────────────────────────────────────
    auc_dict = plot_multiclass_roc(y_true, y_prob, class_names, model_name, results_dir)

    print("\n  Per-class AUC (OvR):")
    for name in class_names:
        print(f"    {name:<20} {auc_dict.get(name, 0):.4f}")
    print(f"  Macro AUC (manual)  : {auc_dict.get('macro', 0):.4f}")
    print(f"  Macro AUC (sklearn) : {auc_dict.get('sklearn_macro', 0):.4f}")
    print(f"  Weighted AUC        : {auc_dict.get('sklearn_weighted', 0):.4f}")

    # ── Aggregate metrics ────────────────────────────────────────────────────
    metrics = {
        'accuracy':        round(accuracy_score(y_true, y_pred), 4),
        'f1_macro':        round(f1_score(y_true, y_pred, average='macro',    zero_division=0), 4),
        'f1_weighted':     round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'f1_macro_tuned':  round(f1_score(y_true, y_pred_tuned, average='macro',    zero_division=0), 4),
        'precision_macro': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'recall_macro':    round(recall_score(y_true, y_pred,    average='macro', zero_division=0), 4),
        'roc_auc_macro':   round(auc_dict.get('sklearn_macro', 0), 4),
        'roc_auc_weighted':round(auc_dict.get('sklearn_weighted', 0), 4),
        'per_class_auc':   {k: round(v, 4) for k, v in auc_dict.items() if k in class_names},
        'per_class_thresholds': {k: round(v, 4) for k, v in thresholds.items()},
    }

    return metrics