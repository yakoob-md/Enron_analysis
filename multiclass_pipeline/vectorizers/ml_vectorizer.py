# multiclass_pipeline/vectorizers/ml_vectorizer.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from features.features import HAND_FEATURES

def vectorize_ml(X_train_txt, X_test_txt, X_train_hf, X_test_hf):
    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True
    )
    X_tr_tfidf = tfidf.fit_transform(X_train_txt)
    X_te_tfidf = tfidf.transform(X_test_txt)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hf)
    X_test_scaled = scaler.transform(X_test_hf)
    X_tr = hstack([X_tr_tfidf, csr_matrix(X_train_scaled)])
    X_te = hstack([X_te_tfidf, csr_matrix(X_test_scaled)])
    return X_tr, X_te, tfidf
