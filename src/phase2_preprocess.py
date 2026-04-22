import re

def clean_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'\S+@\S+', '', text)          # remove emails
    text = re.sub(r'\d+', ' NUM ', text)          # normalize numbers
    text = re.sub(r'[^\w\s]', ' ', text)          # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(df):
    df = df[df['word_count'] <= 2000].copy()      # remove extreme outliers
    df['text_input'] = (
        df['subject'].fillna('').apply(clean_text) + ' ' +
        df['body_clean'].apply(clean_text)
    )
    df = df[df['text_input'].str.len() > 10]      # remove empty after cleaning
    return df