import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from pathlib import Path


def preprocess(
    input_path,
    output_suffix="_preprocessing"):

    df = pd.read_csv(input_path)

    df = df.drop(columns=["CustomerID"])

    df = df.drop_duplicates()

    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(exclude="number").columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    for col in num_cols:
        df[col] = df[col].clip(
            lower=lower_bound[col],
            upper=upper_bound[col]
        )


    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

if __name__ == "__main__":
    this = Path(__file__).resolve().parent
    input_path = this.parent / "Mall_Customers_raw.csv"
    df_processed = preprocess(input_path)
    output_path = this / "Mall_Customers_preprocessed.csv"
    df_processed.to_csv(output_path, index=False)
