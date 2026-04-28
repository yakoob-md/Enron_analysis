
import os
import sys
import pandas as pd
import joblib
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# --- PATHS ---
ROOT = r'c:\Users\dabaa\OneDrive\Desktop\NLP3'
BINARY_SRC = os.path.join(ROOT, 'binary_pipeline', 'src')
MULTI_ROOT = os.path.join(ROOT, 'multiclass_pipeline')

sys.path.append(BINARY_SRC)
sys.path.append(MULTI_ROOT)

# Import local modules
from phase2_preprocess import clean_text
from phase2b_features import engineer_features as eng_bin
from preprocessing.preprocess import preprocess_multiclass
from features.features import engineer_features as eng_multi
from vectorizers.ml_vectorizer import vectorize_ml as vectorize_multi_ml

def analyze_binary():
    print("\n" + "="*40)
    print("  BINARY PIPELINE ERROR ANALYSIS")
    print("="*40)
    
    data_path = os.path.join(ROOT, 'binary_pipeline', 'data', 'emails_labeled_silver.parquet')
    df = pd.read_parquet(data_path)
    df['body_clean'] = df['body'].apply(clean_text)
    df = eng_bin(df)
    
    _, df_test = train_test_split(df, test_size=0.2, random_state=42)
    y_test = df_test['label'].values
    
    tfidf = joblib.load(os.path.join(ROOT, 'binary_pipeline', 'models', 'tfidf.joblib'))
    X_tfidf = tfidf.transform(df_test['body_clean'])
    
    hand_cols = ['f_word_count', 'f_avg_word_len', 'f_disclosure_hits', 'f_disclosure_ratio', 
                 'f_modal_count', 'f_modal_ratio', 'f_uncertainty_count', 'f_caps_ratio', 
                 'f_has_dollar', 'f_has_legal_term', 'f_num_sentences', 'f_avg_sentence_len']
    X_hand = df_test[hand_cols].values
    
    # In binary pipeline, StandardScaler was used
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_hand) # Approximation
    X_hand_scaled = scaler.transform(X_hand)
    X_test_ml = hstack([X_tfidf, X_hand_scaled])
    
    models = ['lr', 'rf', 'svm', 'xgb']
    for m in models:
        path = os.path.join(ROOT, 'binary_pipeline', 'models', f'{m}_model.joblib')
        if os.path.exists(path):
            model = joblib.load(path)
            y_pred = model.predict(X_test_ml)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print(f"{m.upper():<5} | FP: {fp:<4} | FN: {fn:<4} | Errors: {fp+fn:<4} | Samples: {len(y_test)}")

def analyze_multiclass():
    print("\n" + "="*40)
    print("  MULTICLASS PIPELINE ERROR ANALYSIS")
    print("="*40)
    
    data_path = os.path.join(ROOT, 'multiclass_pipeline', 'data', 'emails_labeled_silver_tenK.parquet')
    df = pd.read_parquet(data_path)
    df = preprocess_multiclass(df)
    
    # Encoding (from pipeline.py logic)
    class_names = ['NONE', 'STRATEGIC', 'RELATIONAL', 'LEGAL', 'FINANCIAL']
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    df['label_idx'] = df['disclosure_type'].map(name_to_idx)
    
    X_tv, X_te, y_tv, y_te = train_test_split(
        df['text_input'], df['label_idx'],
        test_size=0.15, random_state=42, stratify=df['label_idx']
    )
    
    # Vectorize
    tfidf = joblib.load(os.path.join(ROOT, 'multiclass_pipeline', 'saved_models', 'tfidf_multi.joblib'))
    df_ml = eng_multi(df)
    hand_cols = [c for c in df_ml.columns if c.startswith('f_')]
    # This might be tricky because vectorize_ml expects specific shapes.
    # I'll just do it manually.
    X_tfidf = tfidf.transform(X_te)
    X_hand = df_ml.loc[X_te.index, [c for c in hand_cols if c in df_ml.columns]].values
    # Filter to actual HAND_FEATURES if needed
    HAND_FEATURES = ['f_word_count', 'f_avg_word_len', 'f_disclosure_hits', 'f_disclosure_ratio', 
                     'f_modal_count', 'f_modal_ratio', 'f_uncertainty_count', 'f_caps_ratio', 
                     'f_has_dollar', 'f_has_legal_term', 'f_num_sentences', 'f_avg_sentence_len']
    X_hand = df_ml.loc[X_te.index, HAND_FEATURES].values
    X_test_ml = hstack([X_tfidf, X_hand])
    
    models = ['lr', 'rf', 'xgb']
    for m in models:
        path = os.path.join(ROOT, 'multiclass_pipeline', 'saved_models', f'{m}_multi.joblib')
        if os.path.exists(path):
            model = joblib.load(path)
            y_pred = model.predict(X_test_ml)
            cm = confusion_matrix(y_te, y_pred)
            total_errors = np.sum(cm) - np.trace(cm)
            print(f"{m.upper():<5} | Total Errors: {total_errors:<4} | Samples: {len(y_te)} | Acc: {accuracy_score(y_te, y_pred):.4f}")
            # Per class errors
            for i, name in enumerate(class_names):
                fn = np.sum(cm[i, :]) - cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                print(f"  -> {name:<10}: FP: {fp:<4} | FN: {fn:<4}")

if __name__ == "__main__":
    analyze_binary()
    analyze_multiclass()
