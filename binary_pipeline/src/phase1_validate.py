import pandas as pd

def validate_data(df):
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("Missing:\n", df.isnull().sum())
    print("Duplicates:", df.duplicated(subset='mid').sum())
    df['label'] = df['disclosure_type'].apply(lambda x: 0 if x == 'NONE' else 1)
    print("Class distribution:\n", df['label'].value_counts())
    print("Imbalance ratio:", round(df['label'].value_counts()[1] / df['label'].value_counts()[0], 2))
    return df