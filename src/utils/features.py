import pandas as pd
import numpy as np


def _to_series(prices):
    """Ensure input is a flat 1-D numeric Series."""
    if isinstance(prices, pd.DataFrame):
        # Take the first column if it's a DataFrame
        prices = prices.iloc[:, 0]
    if isinstance(prices, (list, np.ndarray)):
        prices = pd.Series(prices)
    return pd.to_numeric(prices.squeeze(), errors="coerce").ffill()


def _rsi(prices, window=14):
    """Compute RSI (Relative Strength Index) for a Series of prices."""
    prices = _to_series(prices)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def add_technical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.
    Handles MultiIndex columns and inconsistent yfinance outputs.
    """

    # 🧹 Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]).strip("_") for col in df.columns]

    # 🔍 Identify the Close & Volume columns robustly
    close_candidates = [c for c in df.columns if "Close" in c or "Adj Close" in c or "close" in c]
    volume_candidates = [c for c in df.columns if "Volume" in c or "volume" in c]

    if not close_candidates:
        raise ValueError(f"No 'Close' column found in columns: {df.columns}")
    if not volume_candidates:
        raise ValueError(f"No 'Volume' column found in columns: {df.columns}")

    close_col = close_candidates[0]
    volume_col = volume_candidates[0]

    # ✅ Convert to clean numeric Series
    close = _to_series(df[close_col])
    volume = _to_series(df[volume_col])

    # 📈 Add technical indicators
    df["ret1"] = close.pct_change(fill_method=None)
    df["rsi14"] = _rsi(close, 14)
    df["vol_chg"] = volume.pct_change(fill_method=None).fillna(0)
    df["ma5"] = close.rolling(window=5).mean()
    df["ma20"] = close.rolling(window=20).mean()

    df = df.dropna().reset_index(drop=True)
    return df
