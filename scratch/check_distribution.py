import pandas as pd
import os

data_path = r'c:\Users\dabaa\OneDrive\Desktop\NLP3\data\emails_labeled_silver_tenK.parquet'

if os.path.exists(data_path):
    df = pd.read_parquet(data_path)
    print("Columns:", df.columns.tolist())
    print("\nClass Distribution (Target: label):")
    if 'label' in df.columns:
        print(df['label'].value_counts())
        print("\nNormalized Distribution:")
        print(df['label'].value_counts(normalize=True))
    else:
        print("Column 'label' not found. Available columns:", df.columns.tolist())
    
    # Check for other potential categorical columns for multiclass
    print("\nPotential Multiclass Columns (Unique Counts):")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
             print(f"{col}: {df[col].nunique()} unique values")
else:
    print(f"File not found at {data_path}")
