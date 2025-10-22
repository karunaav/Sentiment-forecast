from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.model.train import build_dataset

@dataclass
class Performance:
    cagr: float
    sharpe: float
    max_dd: float

def compute_metrics(equity: pd.Series, rf: float = 0.0) -> Performance:
    ret = equity.pct_change().dropna()
    ann = ret.mean() * 252
    vol = ret.std() * (252 ** 0.5)
    sharpe = (ann - rf) / (vol + 1e-9)
    n_years = len(equity) / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 1e-9)) - 1
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1).min()
    return Performance(cagr=cagr, sharpe=sharpe, max_dd=dd)

def signal_from_pred(df: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    sig = (df['pred_ret'] > threshold).astype(int)
    return sig

def backtest_daily(ticker: str, preds: pd.Series, cost_bps: float = 1.0):
    df = build_dataset(ticker)
    df = df.iloc[-len(preds):].copy()
    df['pred_ret'] = preds.values
    df['signal'] = signal_from_pred(df)
    strat_ret = df['signal'].shift(1).fillna(0) * df['ret_next']
    churn = (df['signal'].diff().abs().fillna(0)) * (cost_bps / 10000.0)
    strat_ret = strat_ret - churn
    eq = (1 + strat_ret).cumprod()
    bh = (1 + df['ret_next']).cumprod()
    return eq, bh, compute_metrics(eq)
