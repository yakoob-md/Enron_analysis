from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

HAND_FEATURES = [
    'f_word_count', 'f_avg_word_len',
    'f_disclosure_hits', 'f_disclosure_ratio',
    'f_modal_count', 'f_modal_ratio',
    'f_uncertainty_count',
    'f_caps_ratio', 'f_has_dollar', 'f_has_legal_term',
    'f_num_sentences', 'f_avg_sentence_len'
]

def vectorize(X_train_txt, X_test_txt, X_train_hf, X_test_hf):
    tfidf = TfidfVectorizer(
        max_features=7000,              # slightly higher
        ngram_range=(1, 2),
        min_df=3,                        # more coverage
        max_df=0.95,
        sublinear_tf=True
    )

    X_tr_tfidf = tfidf.fit_transform(X_train_txt)
    X_te_tfidf = tfidf.transform(X_test_txt)

    # SCALE hand-crafted features (VERY IMPORTANT)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hf)
    X_test_scaled = scaler.transform(X_test_hf)

    # Convert to sparse
    X_tr = hstack([X_tr_tfidf, csr_matrix(X_train_scaled)])
    X_te = hstack([X_te_tfidf, csr_matrix(X_test_scaled)])

    return X_tr, X_te, tfidf