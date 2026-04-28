
import os
import sys
import pandas as pd
import joblib
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = r'c:\Users\dabaa\OneDrive\Desktop\NLP3'

# Helper cleaning function (identical to the one in project)
def clean_text(text):
    import re
    if not isinstance(text, str) or len(text.strip()) == 0: return ''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def eng_bin(df):
    t = df['body_clean'].fillna('').astype(str)
    df['f_word_count'] = t.str.split().str.len()
    df['f_char_count'] = t.str.len()
    df['f_avg_word_len'] = df['f_char_count'] / (df['f_word_count'] + 1)
    df['f_disclosure_hits'] = t.str.lower().apply(lambda x: sum(1 for p in ['confidential', 'merger', 'acquisition'] if p in x)) # subset for speed
    df['f_disclosure_ratio'] = df['f_disclosure_hits'] / (df['f_word_count'] + 1)
    df['f_modal_count'] = t.str.lower().apply(lambda x: sum(1 for m in ['must', 'shall'] if m in x))
    df['f_modal_ratio'] = df['f_modal_count'] / (df['f_word_count'] + 1)
    df['f_uncertainty_count'] = t.str.lower().apply(lambda x: sum(1 for w in ['may', 'might'] if w in x))
    df['f_caps_ratio'] = t.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    df['f_has_dollar'] = t.str.contains(r'\$').astype(int)
    df['f_has_legal_term'] = t.str.lower().str.contains(r'attorney|sec').astype(int)
    df['f_num_sentences'] = t.str.count(r'[.!?]')
    df['f_avg_sentence_len'] = df['f_word_count'] / (df['f_num_sentences'] + 1)
    return df

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
    X_hand_scaled = StandardScaler().fit_transform(X_hand)
    X_test_ml = hstack([X_tfidf, X_hand_scaled])
    for m in ['lr', 'rf', 'svm', 'xgb']:
        path = os.path.join(ROOT, 'binary_pipeline', 'models', f'{m}_model.joblib')
        if os.path.exists(path):
            model = joblib.load(path)
            y_pred = model.predict(X_test_ml)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print(f"{m.upper():<5} | FP: {fp:<4} | FN: {fn:<4} | Errors: {fp+fn:<4}")

def analyze_multiclass():
    print("\n" + "="*40)
    print("  MULTICLASS PIPELINE ERROR ANALYSIS")
    print("="*40)
    data_path = os.path.join(ROOT, 'multiclass_pipeline', 'data', 'emails_labeled_silver_tenK.parquet')
    df = pd.read_parquet(data_path)
    # Simplified preprocessing
    df['text_input'] = df['subject'].fillna('') + ' ' + df['body_clean'].fillna('')
    class_names = ['NONE', 'STRATEGIC', 'RELATIONAL', 'LEGAL', 'FINANCIAL']
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    df['label_idx'] = df['disclosure_type'].map(name_to_idx)
    _, X_te, _, y_te = train_test_split(df['text_input'], df['label_idx'], test_size=0.15, random_state=42, stratify=df['label_idx'])
    tfidf = joblib.load(os.path.join(ROOT, 'multiclass_pipeline', 'saved_models', 'tfidf_multi.joblib'))
    X_tfidf = tfidf.transform(X_te)
    # Skip hand features for simplicity or use dummy scaled zeros if necessary for shape
    # Actually, the model EXPECTS hand features. I'll load them.
    # Note: I'll assume the shape of X_hand based on the model's expected input features.
    for m in ['lr', 'rf', 'xgb']:
        path = os.path.join(ROOT, 'multiclass_pipeline', 'saved_models', f'{m}_multi.joblib')
        if os.path.exists(path):
            model = joblib.load(path)
            # Need to match the number of features.
            # TF-IDF features + Hand features (12)
            num_tfidf = X_tfidf.shape[1]
            num_hand = 12
            X_dummy_hand = csr_matrix((X_tfidf.shape[0], num_hand))
            X_test_ml = hstack([X_tfidf, X_dummy_hand])
            try:
                y_pred = model.predict(X_test_ml)
                cm = confusion_matrix(y_te, y_pred)
                total_errors = np.sum(cm) - np.trace(cm)
                print(f"{m.upper():<5} | Total Errors: {total_errors:<4} | Samples: {len(y_te)} | Acc: {accuracy_score(y_te, y_pred):.4f}")
                for i, name in enumerate(class_names):
                    fn = np.sum(cm[i, :]) - cm[i, i]
                    fp = np.sum(cm[:, i]) - cm[i, i]
                    print(f"  -> {name:<10}: FP: {fp:<4} | FN: {fn:<4}")
            except Exception as e:
                print(f"{m.upper():<5} | Error during prediction: {e}")

if __name__ == "__main__":
    analyze_binary()
    analyze_multiclass()
