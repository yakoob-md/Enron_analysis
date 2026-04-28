
import os
import sys
import pandas as pd
import joblib
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler

# --- PATH CONFIGURATION ---
ROOT = r'c:\Users\dabaa\OneDrive\Desktop\NLP3'
BINARY_SRC = os.path.join(ROOT, 'binary_pipeline', 'src')
MULTI_ROOT = os.path.join(ROOT, 'multiclass_pipeline')

sys.path.append(BINARY_SRC)
sys.path.append(MULTI_ROOT)

def clean_text(text):
    import re
    if not isinstance(text, str) or len(text.strip()) == 0: return ''
    text = text.lower()
    text = re.sub(r'http\S+| \S+@\S+|\d+|[^\w\s]', ' ', text)
    return ' '.join(text.split())

def run_diagnostics():
    print("\n" + ">>> STARTING DIAGNOSTIC ERROR ANALYSIS (MODEL BY MODEL) <<<")
    
    # 1. LOAD DATA
    data_path = os.path.join(ROOT, 'multiclass_pipeline', 'data', 'emails_labeled_silver_tenK.parquet')
    df = pd.read_parquet(data_path)
    
    # BINARY SETUP
    df['label_bin'] = (df['disclosure_type'] != 'NONE').astype(int)
    df['body_clean'] = df['body'].apply(clean_text)
    
    # MULTICLASS SETUP
    df['text_input_multi'] = df['subject'].fillna('') + ' ' + df['body_clean'].fillna('')
    class_names = ['NONE', 'STRATEGIC', 'RELATIONAL', 'LEGAL', 'FINANCIAL']
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    df['label_multi'] = df['disclosure_type'].map(name_to_idx)
    
    # SPLITS
    # Binary test set (15%)
    _, df_te_bin = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label_bin'])
    # Multiclass test set (15%)
    _, df_te_multi = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label_multi'])

    # ==========================================================================
    # PHASE 1: BINARY MODELS
    # ==========================================================================
    print("\n" + "="*80)
    print("  PHASE 1: BINARY ERROR ANALYSIS")
    print("="*80)
    
    tfidf_bin = joblib.load(os.path.join(ROOT, 'binary_pipeline', 'models', 'tfidf.joblib'))
    X_tfidf_bin = tfidf_bin.transform(df_te_bin['body_clean'])
    
    # Simple hand features logic (same as training)
    def get_bin_hand(df_t):
        t = df_t['body_clean'].astype(str)
        # Dummy features to match shape (12)
        return csr_matrix((len(df_t), 12))

    X_hand_bin = get_bin_hand(df_te_bin)
    X_test_bin = hstack([X_tfidf_bin, X_hand_bin])
    y_test_bin = df_te_bin['label_bin'].values

    for m_name in ['lr', 'rf', 'xgb']:
        path = os.path.join(ROOT, 'binary_pipeline', 'models', f'{m_name}_model.joblib')
        if os.path.exists(path):
            print(f"\n[ BINARY MODEL: {m_name.upper()} ]")
            model = joblib.load(path)
            y_pred = model.predict(X_test_bin)
            tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred).ravel()
            print(f"Stats -> FP: {fp} | FN: {fn} | Accuracy: {accuracy_score(y_test_bin, y_pred):.4f}")
            
            fp_idx = np.where((y_test_bin == 0) & (y_pred == 1))[0]
            fn_idx = np.where((y_test_bin == 1) & (y_pred == 0))[0]
            
            if len(fp_idx) > 0:
                print(f"Sample FP (Predicted Disclosure but was None):")
                print(f"  > {df_te_bin['body_clean'].iloc[fp_idx[0]][:150]}...")
            if len(fn_idx) > 0:
                print(f"Sample FN (Predicted None but was Disclosure):")
                print(f"  > {df_te_bin['body_clean'].iloc[fn_idx[0]][:150]}...")

    # ==========================================================================
    # PHASE 2: MULTICLASS MODELS
    # ==========================================================================
    print("\n" + "="*80)
    print("  PHASE 2: MULTICLASS ERROR ANALYSIS")
    print("="*80)
    
    tfidf_multi = joblib.load(os.path.join(ROOT, 'multiclass_pipeline', 'saved_models', 'tfidf_multi.joblib'))
    X_tfidf_multi = tfidf_multi.transform(df_te_multi['text_input_multi'])
    X_dummy_multi = csr_matrix((X_tfidf_multi.shape[0], 12))
    X_test_multi = hstack([X_tfidf_multi, X_dummy_multi])
    y_test_multi = df_te_multi['label_multi'].values

    for m_name in ['lr', 'rf', 'xgb']:
        path = os.path.join(ROOT, 'multiclass_pipeline', 'saved_models', f'{m_name}_multi.joblib')
        if os.path.exists(path):
            print(f"\n" + "-"*60)
            print(f"  MULTICLASS MODEL: {m_name.upper()}")
            print("-"*60)
            model = joblib.load(path)
            y_pred = model.predict(X_test_multi)
            cm = confusion_matrix(y_test_multi, y_pred)
            print(f"Global Accuracy: {accuracy_score(y_test_multi, y_pred):.4f}")
            
            for i, name in enumerate(class_names):
                print(f"\n[{name} Diagnostics]")
                fn_idx = np.where((y_test_multi == i) & (y_pred != i))[0]
                fp_idx = np.where((y_test_multi != i) & (y_pred == i))[0]
                print(f"  Errors -> FP: {len(fp_idx)} | FN: {len(fn_idx)}")
                
                if len(fn_idx) > 0:
                    p_name = class_names[y_pred[fn_idx[0]]]
                    print(f"  Sample FN (Was {name}, predicted {p_name}):")
                    print(f"    > {df_te_multi['text_input_multi'].iloc[fn_idx[0]][:150]}...")
                if len(fp_idx) > 0:
                    a_name = class_names[y_test_multi[fp_idx[0]]]
                    print(f"  Sample FP (Was {a_name}, predicted {name}):")
                    print(f"    > {df_te_multi['text_input_multi'].iloc[fp_idx[0]][:150]}...")

    print("\n" + ">>> ALL MODELS ANALYZED <<<")

if __name__ == "__main__":
    run_diagnostics()
