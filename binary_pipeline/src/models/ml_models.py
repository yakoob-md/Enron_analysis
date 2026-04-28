# binary_pipeline/src/models/ml_models.py
# FIXED: class_weight='balanced' on all models, SVM probability=True, XGBoost scale_pos_weight

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import numpy as np


def get_model(name, class_ratio=None):
    """
    class_ratio: imbalance ratio (positive / negative count).
                 Pass it in so XGBoost scale_pos_weight is correct.
                 For your dataset: 7381/2605 ≈ 2.83  →  invert = 2605/7381 ≈ 0.35
                 scale_pos_weight should be neg/pos for XGBoost binary.
    """
    # For binary: scale_pos_weight = num_negatives / num_positives
    spw = (1.0 / class_ratio) if class_ratio else 0.35

    if name == 'lr':
        return LogisticRegression(
            C=1.0,                      # regularisation strength
            class_weight='balanced',    # FIX: was missing in original
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )

    elif name == 'rf':
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced_subsample',  # FIX: better for RF than 'balanced'
            n_jobs=-1,
            random_state=42
        )

    elif name == 'svm':
        # FIX: probability=True is required for ROC-AUC + threshold analysis
        # Wrap in Platt scaling via CalibratedClassifierCV for better calibration
        base = SVC(
            C=1.0,
            kernel='rbf',
            class_weight='balanced',    # FIX: was missing
            probability=True,           # FIX: was missing → caused "ROC-AUC: Not available"
            random_state=42
        )
        # Optional: uncomment for better probability calibration
        # return CalibratedClassifierCV(base, cv=3, method='sigmoid')
        return base

    elif name == 'xgb':
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,       # FIX: was None → now set correctly
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown model: {name}")


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model