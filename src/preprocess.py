
import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data", "heart.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data", "processed_data.csv")

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Run download_dataset.py or add heart.csv to data/")
    return pd.read_csv(path)

def basic_preprocessing(df):
    # Example preprocessing based on fields commonly present in heart datasets
    # 1) Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    # 2) Fill missing numerical with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    # 3) Cap numeric outliers at 1st and 99th percentiles
    nums = df.select_dtypes(include=[np.number]).columns
    for col in nums:
        low = df[col].quantile(0.01)
        high = df[col].quantile(0.99)
        df[col] = df[col].clip(low, high)
    return df

def main():
    df = load_data()
    print("Loaded data shape:", df.shape)
    df = basic_preprocessing(df)
    df.to_csv(OUT_PATH, index=False)
    print("Saved processed data to", OUT_PATH)

if __name__ == '__main__':
    main()
