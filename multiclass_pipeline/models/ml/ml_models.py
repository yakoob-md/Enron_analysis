# multiclass_pipeline/models/ml/ml_models.py
# FIXED vs original:
#   - XGBoost uses sample_weight in fit() instead of scale_pos_weight (which is binary-only)
#   - Logistic Regression: C tunable, added penalty
#   - RandomForest: balanced_subsample (better than 'balanced' for RF)
#   - Added train_model_ml that handles XGBoost sample weights

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


def get_model_ml(name, num_classes=5):
    if name == 'lr':
        return LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    elif name == 'rf':
        return RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced_subsample',  # FIX: better than 'balanced' for RF
            n_jobs=-1,
            random_state=42
        )
    elif name == 'xgb':
        # FIX: scale_pos_weight is binary-only.
        # For multiclass, pass sample_weight to fit() instead.
        return XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def train_model_ml(model, X_train, y_train):
    """
    FIXED: XGBoost multiclass imbalance handled via sample_weight.
    All other sklearn models use class_weight='balanced' internally.
    """
    if isinstance(model, XGBClassifier):
        # compute_sample_weight gives each sample a weight inverse to class freq
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    return model