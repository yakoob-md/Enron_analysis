# multiclass_pipeline/preprocessing/preprocess.py
import re
import pandas as pd

def clean_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_multiclass(df):
    df = df[df['word_count'] <= 2000].copy()
    df['text_input'] = (
        df['subject'].fillna('').apply(clean_text) + ' ' +
        df['body_clean'].apply(clean_text)
    )
    df = df[df['text_input'].str.len() > 10]
    
    # We keep the original 'disclosure_type' as the label for multiclass
    return df
