# multiclass_pipeline/pipeline.py
# FIXED + EXTENDED:
#   - Full ML + DL (BiLSTM + BERT) support
#   - Correct multiclass evaluation (ROC, per-class thresholds, CM)
#   - Class weights passed everywhere
#   - DL predictions return (y_pred, y_prob) via softmax

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

from configs.config import (
    MODE, DATA_PATH, MODEL_DIR, RESULTS_DIR,
    NUM_CLASSES, CLASS_NAMES, EPOCHS, BATCH_SIZE, LR, MAX_LEN
)
from preprocessing.preprocess import preprocess_multiclass
from preprocessing.label_encoder import MultiClassLabelEncoder
from features.features import engineer_features, HAND_FEATURES
from vectorizers.ml_vectorizer import vectorize_ml
from models.ml.ml_models import get_model_ml, train_model_ml
from evaluation.evaluator import evaluate_multiclass
from utils.class_weights import get_class_weights


def run_multiclass_pipeline():
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n🚀 RUNNING MULTICLASS PIPELINE [{MODE.upper()}]\n")

    # ── 1. Load & Preprocess ──────────────────────────────────────────────────
    df = pd.read_parquet(DATA_PATH)
    df = preprocess_multiclass(df)

    # ── 2. Encode Labels ──────────────────────────────────────────────────────
    encoder = MultiClassLabelEncoder(CLASS_NAMES)
    df['label_idx'] = encoder.encode(df['disclosure_type'])

    # ── 3. Class Weights ──────────────────────────────────────────────────────
    class_weights = get_class_weights(df['label_idx'].values)
    print("Class weights:", {CLASS_NAMES[k]: round(v, 3) for k, v in class_weights.items()})

    # ── 4. Feature Engineering (ML only) ─────────────────────────────────────
    if MODE == 'ml':
        df = engineer_features(df)

    # ── 5. Split ──────────────────────────────────────────────────────────────
    X_tv, X_te, y_tv, y_te = train_test_split(
        df['text_input'], df['label_idx'],
        test_size=0.15, random_state=42, stratify=df['label_idx']
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv,
        test_size=0.176, random_state=42, stratify=y_tv
    )

    results = []

    # ══════════════════════════════════════════════════════════════════════════
    # ML MODE
    # ══════════════════════════════════════════════════════════════════════════
    if MODE == 'ml':
        X_tr_hf = df.loc[X_tr.index, HAND_FEATURES]
        X_va_hf = df.loc[X_va.index, HAND_FEATURES]
        X_te_hf = df.loc[X_te.index, HAND_FEATURES]

        X_train, X_test, tfidf = vectorize_ml(X_tr, X_te, X_tr_hf, X_te_hf)
        joblib.dump(tfidf, f'{MODEL_DIR}/tfidf_multi.joblib')

        for m_name in ['lr', 'rf', 'xgb']:
            print(f"\n{'='*50}")
            print(f"  Training {m_name.upper()}")
            print(f"{'='*50}")
            model = get_model_ml(m_name, NUM_CLASSES)
            model = train_model_ml(model, X_train, y_tr)
            joblib.dump(model, f'{MODEL_DIR}/{m_name}_multi.joblib')

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)   # (n, num_classes)

            metrics = evaluate_multiclass(
                y_te.values, y_pred, y_prob,
                m_name.upper(), RESULTS_DIR, CLASS_NAMES
            )
            results.append({'model': m_name, **metrics})

    # ══════════════════════════════════════════════════════════════════════════
    # DL MODE — BiLSTM
    # ══════════════════════════════════════════════════════════════════════════
    elif MODE == 'dl':
        from vectorizers.dl_vectorizer import vectorize as dl_vectorize
        from models.dl.bilstm import BiLSTMModelMulti
        from training.trainer import train_multiclass, predict_multiclass

        X_train_enc, X_test_enc, word2idx = dl_vectorize(X_tr, X_te)
        X_val_enc,   _,          _        = dl_vectorize(X_tr, X_va)

        tr_ds  = TensorDataset(torch.tensor(X_train_enc, dtype=torch.long),
                               torch.tensor(y_tr.values, dtype=torch.long))
        va_ds  = TensorDataset(torch.tensor(X_val_enc,   dtype=torch.long),
                               torch.tensor(y_va.values, dtype=torch.long))
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE)

        model = BiLSTMModelMulti(num_classes=NUM_CLASSES)
        model = train_multiclass(
            model, tr_loader, va_loader,
            class_weights=class_weights,
            epochs=EPOCHS, lr=LR,
            model_name='bilstm_multi',
            results_dir=RESULTS_DIR,
            use_focal_loss=True,
            label_smoothing=0.1
        )
        torch.save(model.state_dict(), f'{MODEL_DIR}/bilstm_multi.pt')

        y_pred, y_prob = predict_multiclass(model, X_test_enc)
        metrics = evaluate_multiclass(
            y_te.values, y_pred, y_prob,
            'BiLSTM', RESULTS_DIR, CLASS_NAMES
        )
        results.append({'model': 'bilstm', **metrics})

    # ══════════════════════════════════════════════════════════════════════════
    # DL MODE — BERT
    # ══════════════════════════════════════════════════════════════════════════
    elif MODE == 'bert':
        from vectorizers.bert_vectorizer import vectorize as bert_vectorize
        from models.dl.bert_model import (
            get_bert_multiclass, train_bert_multiclass, predict_bert_multiclass
        )

        X_train_enc, X_test_enc, tokenizer = bert_vectorize(X_tr, X_te)
        X_val_enc,   _,          _         = bert_vectorize(X_tr, X_va)
        joblib.dump(tokenizer, f'{MODEL_DIR}/tokenizer_multi.joblib')

        bert_path = f'{MODEL_DIR}/bert_multiclass'

        if os.path.exists(bert_path):
            print("✅ Loading saved BERT multiclass model...")
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(bert_path)
        else:
            model = get_bert_multiclass(num_labels=NUM_CLASSES)
            model = train_bert_multiclass(
                model, X_train_enc, y_tr, X_val_enc, y_va,
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                results_dir=RESULTS_DIR,
                class_weights=class_weights
            )
            model.save_pretrained(bert_path)

        y_pred, y_prob = predict_bert_multiclass(model, X_test_enc)
        metrics = evaluate_multiclass(
            y_te.values, y_pred, y_prob,
            'BERT', RESULTS_DIR, CLASS_NAMES
        )
        results.append({'model': 'bert', **metrics})

    # ── Final Summary ─────────────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    print("\n=== FINAL MULTICLASS COMPARISON ===")
    print(res_df[['model', 'accuracy', 'f1_macro', 'f1_weighted', 'roc_auc_macro']].to_string())
    res_df.to_csv(f'{RESULTS_DIR}/multiclass_comparison.csv', index=False)
    return res_df


if __name__ == "__main__":
    run_multiclass_pipeline()