# multiclass_pipeline/pipeline.py
import pandas as pd
import os
import joblib
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from configs.config import *
from preprocessing.preprocess import preprocess_multiclass
from preprocessing.label_encoder import MultiClassLabelEncoder
from features.features import engineer_features, HAND_FEATURES
from vectorizers.ml_vectorizer import vectorize_ml
from models.ml.ml_models import get_model_ml, train_model_ml
from evaluation.evaluator import evaluate_multiclass, plot_learning_curves

def run_multiclass_pipeline():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n🚀 RUNNING MULTICLASS PIPELINE [{MODE.upper()}]\n")

    # 1. Load & Preprocess
    df = pd.read_parquet(DATA_PATH)
    df = preprocess_multiclass(df)
    
    # 2. Encode Labels
    encoder = MultiClassLabelEncoder(CLASS_NAMES)
    df['label_idx'] = encoder.encode(df['disclosure_type'])
    
    # Compute class weights to handle imbalance (Step 1 from prompt)
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight('balanced', classes=np.unique(df['label_idx']), y=df['label_idx'])
    class_weights = dict(enumerate(weights))
    print("\nClass weights:", class_weights)
    
    # 3. Features
    if MODE == 'ml':
        df = engineer_features(df)
    
    # 4. Split
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df['text_input'], df['label_idx'], test_size=0.2, random_state=42, stratify=df['label_idx']
    )
    
    # ===============================
    # ML MODE
    # ===============================
    if MODE == 'ml':
        X_train_hf = df.loc[X_train_txt.index, HAND_FEATURES]
        X_test_hf = df.loc[X_test_txt.index, HAND_FEATURES]
        
        # 5. Vectorize
        X_train, X_test, tfidf = vectorize_ml(X_train_txt, X_test_txt, X_train_hf, X_test_hf)
        
        # 6. Train & Eval
        results = []
        for m_name in ['lr', 'rf', 'xgb']:
            print(f"\n--- Training {m_name.upper()} ---")
            model = get_model_ml(m_name, NUM_CLASSES)
            model = train_model_ml(model, X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            metrics = evaluate_multiclass(y_test, y_pred, y_prob, m_name, RESULTS_DIR, CLASS_NAMES)
            
            print(f"--- Plotting Learning Curves for {m_name.upper()} ---")
            plot_learning_curves(model, X_train, y_train, m_name, RESULTS_DIR)
            
            results.append({'model': m_name, **metrics})
            
        res_df = pd.DataFrame(results)
        print("\n=== FINAL COMPARISON ===")
        print(res_df)
        res_df.to_csv(f"{RESULTS_DIR}/multiclass_comparison.csv", index=False)

    # ===============================
    # DL MODE
    # ===============================
    elif MODE == 'dl':
        from vectorizers.dl_vectorizer import vectorize
        from models.dl.bilstm import BiLSTMModelMulti
        from training.trainer import train_multiclass
        
        print("--- Vectorizing (DL) ---")
        X_train, X_test, tokenizer = vectorize(X_train_txt, X_test_txt)
        
        # DataLoaders
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train.values, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test.values, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
        
        # Model & Train
        model = BiLSTMModelMulti(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
        model = train_multiclass(model, train_loader, test_loader, class_weights=class_weights, 
                                 epochs=EPOCHS, lr=LR, model_name="bilstm", results_dir=RESULTS_DIR)
        
        # Final Evaluation
        model.eval()
        all_preds, all_probs = [], []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)
                all_probs.extend(probs)
        
        metrics = evaluate_multiclass(y_test, np.array(all_preds), np.array(all_probs), "BILSTM", RESULTS_DIR, CLASS_NAMES)
        print("\n=== DL RESULTS ===")
        print(metrics)

if __name__ == "__main__":
    run_multiclass_pipeline()
