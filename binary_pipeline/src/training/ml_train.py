# binary_pipeline/src/training/ml_train.py
# FIXED: passes class_ratio to get_model so XGBoost scale_pos_weight is correct

import numpy as np
from models.ml_models import get_model, train_model


def run_ml_training(model_names, X_train, y_train):
    """
    FIXED: computes class_ratio from y_train and passes to each model.
    class_ratio = pos_count / neg_count  (2.83 for your dataset)
    XGBoost needs scale_pos_weight = neg/pos = 1/2.83 ≈ 0.35
    """
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    class_ratio = pos / neg   # 2.83

    models = {}
    for name in model_names:
        print(f"\n--- Training {name.upper()} ---")
        model = get_model(name, class_ratio=class_ratio)
        model = train_model(model, X_train, y_train)
        models[name] = model

    return models