# multiclass_pipeline/models/ml/ml_models.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import numpy as np

def get_model_ml(name='lr', num_classes=5):
    if name == 'lr':
        return LogisticRegression(
            max_iter=3000,
            class_weight='balanced',
            multi_class='multinomial',
            solver='saga',
            random_state=42
        )
    elif name == 'rf':
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif name == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            objective='multi:softprob',
            num_class=num_classes,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model: {name}")

def train_model_ml(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
