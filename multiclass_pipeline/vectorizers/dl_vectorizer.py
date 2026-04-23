# multiclass_pipeline/vectorizers/dl_vectorizer.py
from collections import Counter
import numpy as np
try:
    from configs.config import VOCAB_SIZE, MAX_LEN
except ImportError:
    # Fallback if run from a different context
    VOCAB_SIZE = 10000
    MAX_LEN = 200

def build_vocab(texts, vocab_size):
    counter = Counter()
    for text in texts:
        if isinstance(text, str):
            counter.update(text.split())
    
    most_common = counter.most_common(vocab_size - 2)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = i
    return word2idx

def encode(texts, word2idx, max_len):
    sequences = []
    for text in texts:
        if isinstance(text, str):
            seq = [word2idx.get(word, 1) for word in text.split()]
        else:
            seq = []
        seq = seq[:max_len]
        seq += [0] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences)

def vectorize(X_train_txt, X_test_txt):
    print("PyTorch Vectorization (BiLSTM)...")
    word2idx = build_vocab(X_train_txt, VOCAB_SIZE)
    X_train = encode(X_train_txt, word2idx, MAX_LEN)
    X_test = encode(X_test_txt, word2idx, MAX_LEN)
    return X_train, X_test, word2idx
