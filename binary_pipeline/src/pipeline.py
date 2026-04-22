import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split

# Config
from config import MODE, ML_MODELS, DL_MODEL, DATA_PATH, MODEL_DIR, RESULTS_DIR

# Common phases
from phase1_validate import validate_data
from phase2_preprocess import preprocess
from phase2b_features import engineer_features
from phase5_evaluate import evaluate
from phase6_error_analysis import error_analysis


# ===============================
# DYNAMIC IMPORTS
# ===============================
if MODE == 'ml':
    from vectorizers.ml_vectorizer import vectorize, HAND_FEATURES
    from models.ml_models import get_model, train_model

elif MODE == 'dl':
    import torch

    if DL_MODEL == 'bert':
        from vectorizers.bert_vectorizer import vectorize
        from models.bert_model import get_model, train_model
        from transformers import AutoModelForSequenceClassification
    else:
        from vectorizers.dl_vectorizer import vectorize
        from models.dl_models import get_model, train_model

else:
    raise ValueError("MODE must be 'ml' or 'dl'")


# ===============================
# MAIN PIPELINE
# ===============================
def run_pipeline():

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n🚀 RUNNING PIPELINE IN [{MODE.upper()} MODE]\n")

    # -----------------------
    # LOAD DATA
    # -----------------------
    print("--- Loading Data ---")
    df = pd.read_parquet(DATA_PATH)

    # -----------------------
    # LABEL CREATION
    # -----------------------
    print("--- Creating Label ---")
    df['label'] = df['disclosure_type'].apply(
        lambda x: 0 if x == 'NONE' else 1
    )

    # -----------------------
    # VALIDATION
    # -----------------------
    print("--- Phase 1: Validation ---")
    df = validate_data(df)

    # -----------------------
    # PREPROCESS
    # -----------------------
    print("--- Phase 2: Preprocessing ---")
    df = preprocess(df)

    # ML only
    if MODE == 'ml':
        df = engineer_features(df)

    # -----------------------
    # REMOVE DUPLICATES
    # -----------------------
    print("--- Removing duplicate texts ---")
    df = df.drop_duplicates(subset=['text_input']).reset_index(drop=True)
    print("New dataset size:", len(df))

    # -----------------------
    # SPLIT (Train: 70%, Val: 15%, Test: 15%)
    # -----------------------
    print("--- Splitting Data ---")

    if MODE == 'ml':
        # First split into train+val (85%) and test (15%)
        X_tv_txt, X_te_txt, X_tv_hf, X_te_hf, y_tv, y_te = train_test_split(
            df['text_input'],
            df[HAND_FEATURES],
            df['label'],
            test_size=0.15,
            random_state=42,
            stratify=df['label']
        )
        # Then split tv into train and val
        X_tr_txt, X_va_txt, X_tr_hf, X_va_hf, y_tr, y_va = train_test_split(
            X_tv_txt, X_tv_hf, y_tv,
            test_size=0.176, # 0.15 / 0.85
            random_state=42,
            stratify=y_tv
        )
    else:
        # First split into train+val (85%) and test (15%)
        X_tv_txt, X_te_txt, y_tv, y_te = train_test_split(
            df['text_input'],
            df['label'],
            test_size=0.15,
            random_state=42,
            stratify=df['label']
        )
        # Then split tv into train and val
        X_tr_txt, X_va_txt, y_tr, y_va = train_test_split(
            X_tv_txt, y_tv,
            test_size=0.176, # 0.15 / 0.85
            random_state=42,
            stratify=y_tv
        )

    # -----------------------
    # VECTORIZATION
    # -----------------------
    print("--- Phase 3: Vectorization ---")

    if MODE == 'ml':
        X_train, X_test, vectorizer = vectorize(
            X_tr_txt, X_te_txt, X_tr_hf, X_te_hf
        )
        # Vectorize validation set as well
        _, X_val, _ = vectorize(X_tr_txt, X_va_txt, X_tr_hf, X_va_hf)
        joblib.dump(vectorizer, f'{MODEL_DIR}/tfidf.joblib')

    else:
        X_train, X_test, tokenizer = vectorize(X_tr_txt, X_te_txt)
        # Vectorize validation set as well
        _, X_val, _ = vectorize(X_tr_txt, X_va_txt)
        joblib.dump(tokenizer, f'{MODEL_DIR}/tokenizer.joblib')

    results = []

    # ===============================
    # ML MODE
    # ===============================
    if MODE == 'ml':

        for model_name in ML_MODELS:

            print(f"\n==============================")
            print(f"Running Model: {model_name.upper()}")
            print(f"==============================")

            model = get_model(model_name)
            model = train_model(model, X_train, y_tr)

            joblib.dump(model, f'{MODEL_DIR}/{model_name}_model.joblib')

            print("--- Evaluation ---")
            for method in ['youden', 'f1', 'cost']:
                print(f"\n>>>> Threshold Method: {method.upper()}")
                metrics = evaluate(model, X_test, y_te, model_name.upper(), 
                                   threshold_method=method, X_train=X_train, y_train=y_tr)
                results.append({
                    'model': model_name,
                    'threshold_type': method,
                    **metrics
                })

            print("--- Error Analysis ---")
            error_analysis(model, X_test, y_te, X_te_txt)

    # ===============================
    # DL MODE
    # ===============================
    elif MODE == 'dl':

        print(f"\n==============================")
        print(f"Running DL Model: {DL_MODEL.upper()}")
        print(f"==============================")

        # 🔥 BERT LOGIC
        if DL_MODEL == 'bert':

            model_path = f"{MODEL_DIR}/bert_model"

            if os.path.exists(model_path):
                print("✅ Loading saved BERT model...")
                model = AutoModelForSequenceClassification.from_pretrained(model_path)

            else:
                print("🚀 Training BERT model with diagnostics...")
                model = get_model(DL_MODEL)
                model = train_model(model, X_train, y_tr, X_val, y_va)
                model.save_pretrained(model_path)

        # 🔥 OTHER DL MODELS
        else:
            model = get_model(DL_MODEL)
            model = train_model(model, X_train, y_tr, X_val, y_va)

            import torch
            torch.save(model.state_dict(), f"{MODEL_DIR}/{DL_MODEL}_model.pt")

        # -----------------------
        # EVALUATION
        # -----------------------
        print("--- Evaluation ---")
        for method in ['youden', 'f1', 'cost']:
            print(f"\n>>>> Threshold Method: {method.upper()}")
            metrics = evaluate(model, X_test, y_te, DL_MODEL.upper(), 
                               threshold_method=method, X_train=X_train, y_train=y_tr)
            results.append({
                'model': DL_MODEL,
                'threshold_type': method,
                **metrics
            })

        print("--- Error Analysis ---")
        error_analysis(model, X_test, y_te, X_te_txt)

    # -----------------------
    # SAVE RESULTS
    # -----------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{RESULTS_DIR}/model_comparison.csv', index=False)

    print("\n=== FINAL RESULTS ===")
    print(results_df)

    return results_df


# ===============================
# RUN
# ===============================
if __name__ == '__main__':
    import sys
    sys.path.append('src')

    run_pipeline()