# multiclass_pipeline/vectorizers/bert_vectorizer.py
from transformers import AutoTokenizer
try:
    from configs.config import MAX_LEN
except ImportError:
    MAX_LEN = 200

def vectorize(X_train_txt, X_test_txt, X_val_txt=None):
    print("BERT Tokenization (Fast)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    def tokenize(texts):
        return tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

    train_encodings = tokenize(X_train_txt)
    test_encodings = tokenize(X_test_txt)
    
    if X_val_txt is not None:
        val_encodings = tokenize(X_val_txt)
        return train_encodings, test_encodings, val_encodings, tokenizer
        
    return train_encodings, test_encodings, tokenizer
