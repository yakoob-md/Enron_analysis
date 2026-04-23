# multiclass_pipeline/vectorizers/bert_vectorizer.py
from transformers import AutoTokenizer
try:
    from configs.config import MAX_LEN
except ImportError:
    MAX_LEN = 200

def vectorize(X_train_txt, X_test_txt):
    print("BERT Tokenization...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
    return train_encodings, test_encodings, tokenizer
