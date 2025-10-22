from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def make_sequences(df, feature_cols, target_col, window=60):
    X, y = [], []
    arr = df[feature_cols].values
    tgt = df[target_col].values
    for i in range(window, len(df)):
        X.append(arr[i - window:i])
        y.append(tgt[i])
    return np.array(X), np.array(y)

def scale_fit(train_df: pd.DataFrame, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler

def scale_transform(df: pd.DataFrame, scaler, feature_cols):
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols].values)
    return out
