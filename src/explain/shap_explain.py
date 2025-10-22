from __future__ import annotations
import shap
import numpy as np
import pandas as pd
from src.model.train import build_dataset, load_artifacts
from src.utils.sequences import scale_transform, make_sequences

def shap_summary(ticker: str, nsamples: int = 200):
    model, meta = load_artifacts(ticker)
    scaler = meta['scaler']
    feature_cols = meta['feature_cols']
    window = meta['window']

    df = build_dataset(ticker)
    df_scaled = scale_transform(df, scaler, feature_cols)
    X, y = make_sequences(df_scaled, feature_cols, 'ret_next', window)

    X2d = X.reshape((X.shape[0], -1))
    bg = shap.sample(pd.DataFrame(X2d), nsamples=min(100, X2d.shape[0]))

    f = lambda z: model.predict(z.values.reshape((-1, window, len(feature_cols))), verbose=0)
    explainer = shap.KernelExplainer(f, bg)
    sample = shap.sample(pd.DataFrame(X2d), nsamples=min(nsamples, X2d.shape[0]))
    sv = explainer.shap_values(sample, nsamples=100)

    shap.summary_plot(sv, sample, show=True)
    return sv, sample
