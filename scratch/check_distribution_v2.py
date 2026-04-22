import pandas as pd
import os

data_path = r'c:\Users\dabaa\OneDrive\Desktop\NLP3\data\emails_labeled_silver_tenK.parquet'

if os.path.exists(data_path):
    df = pd.read_parquet(data_path)
    
    print("Value counts for 'disclosure_type':")
    print(df['disclosure_type'].value_counts())
    print("\nNormalized 'disclosure_type':")
    print(df['disclosure_type'].value_counts(normalize=True))
    
    print("\nValue counts for 'sender_role':")
    print(df['sender_role'].value_counts())
    
    print("\nValue counts for 'confidence':")
    print(df['confidence'].value_counts())

    print("\nSample disclosure types:")
    print(df[['disclosure_type', 'body_excerpt']].head(10))
else:
    print(f"File not found at {data_path}")
