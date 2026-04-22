import re
import numpy as np

DISCLOSURE_PHRASES = [
    'confidential', 'do not forward', 'privileged', 'off the record',
    'not for distribution', 'merger', 'acquisition', 'write-down',
    'off-balance', 'special purpose entity', 'reserve', 'attorney',
    'sec filing', 'earnings', 'settlement', 'salary'
]

MODAL_VERBS = ['must', 'shall', 'should', 'cannot', 'may not', 'required to']
UNCERTAINTY_WORDS = ['may', 'might', 'possibly', 'likely', 'uncertain', 'risk']

def engineer_features(df):
    t = df['body_clean'].fillna('').astype(str)

    # Basic stats
    df['f_word_count'] = t.str.split().str.len()
    df['f_char_count'] = t.str.len()

    # Avoid division issues
    df['f_avg_word_len'] = df['f_char_count'] / (df['f_word_count'] + 1)

    # Normalize disclosure hits (IMPORTANT)
    df['f_disclosure_hits'] = t.str.lower().apply(
        lambda x: sum(1 for p in DISCLOSURE_PHRASES if p in x)
    )
    df['f_disclosure_ratio'] = df['f_disclosure_hits'] / (df['f_word_count'] + 1)

    # Modal verbs
    df['f_modal_count'] = t.str.lower().apply(
        lambda x: sum(1 for m in MODAL_VERBS if m in x)
    )
    df['f_modal_ratio'] = df['f_modal_count'] / (df['f_word_count'] + 1)

    # Uncertainty signal (VERY IMPORTANT for disclosures)
    df['f_uncertainty_count'] = t.str.lower().apply(
        lambda x: sum(1 for w in UNCERTAINTY_WORDS if w in x)
    )

    # Capitalization
    df['f_caps_ratio'] = t.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )

    # Financial signals
    df['f_has_dollar'] = t.str.contains(
        r'\$|\bUSD\b|\bmillion\b|\bbillion\b', regex=True
    ).astype(int)

    # Legal signals
    df['f_has_legal_term'] = t.str.lower().str.contains(
        r'attorney|counsel|litigation|sec|ferc', regex=True
    ).astype(int)

    # Email structure (VERY USEFUL)
    df['f_num_sentences'] = t.str.count(r'[.!?]')
    df['f_avg_sentence_len'] = df['f_word_count'] / (df['f_num_sentences'] + 1)

    return df