from __future__ import annotations
import numpy as np
import pandas as pd
from src.model.transformer import build_transformer
from src.utils.sequences import scale_fit, scale_transform, make_sequences

def walk_forward_train(df: pd.DataFrame, feature_cols, window=60,
                       init_train_days=504, step_days=21, epochs=8, batch_size=64):
    preds = []
    idxs = []
    mse_list = []
    total = len(df)
    steps = max((total - init_train_days - window) // step_days, 0)
    for i in range(steps):
        start = init_train_days + i * step_days
        tr = df.iloc[:start]
        te_end = min(total, start + step_days)
        te = df.iloc[:te_end]

        scaler = scale_fit(tr, feature_cols)
        tr_s = scale_transform(tr, scaler, feature_cols)
        te_s = scale_transform(te, scaler, feature_cols)

        X_tr, y_tr = make_sequences(tr_s, feature_cols, "ret_next", window)
        X_te, y_te = make_sequences(te_s, feature_cols, "ret_next", window)

        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        model = build_transformer(input_dim=len(feature_cols), window=window)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

        y_hat = model.predict(X_te[-step_days:], verbose=0).ravel()
        y_true = y_te[-step_days:]
        n = min(len(y_hat), len(y_true))
        if n == 0:
            continue
        mse = float(((y_hat[-n:] - y_true[-n:])**2).mean())
        mse_list.append(mse)

        preds.extend(y_hat[-n:])
        idxs.extend(df.index[start + window : start + window + n])

    pred_series = pd.Series(preds, index=idxs)
    metrics = {"wf_mse": float(np.mean(mse_list)) if mse_list else float("nan")}
    return pred_series, metrics
