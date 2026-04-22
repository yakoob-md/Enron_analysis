# multiclass_pipeline/models/ml/ml_models.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

def get_model_ml(name, num_classes=5):
    if name == 'lr':
        return LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            max_iter=1000
        )
    elif name == 'rf':
        return RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            n_jobs=-1
        )
    elif name == 'xgb':
        # XGBoost handles multiclass natively
        return XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            scale_pos_weight=None  # use sample_weight instead
        )
    else:
        raise ValueError(f"Unknown model: {name}")

def train_model_ml(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
