import pandas as pd
import yfinance as yf

# ============================================================
# 📈 Load Stock Price Data
# ============================================================

def load_prices(ticker: str, start: str = "2017-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Load historical price data for a given ticker using yfinance.
    Returns a DataFrame with standardized column names.
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"No price data found for {ticker}")

        # Reset index and clean
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adj_Close",
            "Volume": "Volume"
        })

        # Keep only essential columns
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna().reset_index(drop=True)

        return df

    except Exception as e:
        print(f"❌ Error loading prices for {ticker}: {e}")
        return pd.DataFrame()


# ============================================================
# 🧩 Merge On Date (Safe for Single DF)
# ============================================================

def merge_on_date(df1: pd.DataFrame, df2: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge two DataFrames on the 'Date' column.
    If df2 is None, returns df1 unchanged.
    """
    if df2 is None or df2.empty:
        return df1

    try:
        df_merged = pd.merge(df1, df2, on="Date", how="left")
        for col in df_merged.columns:
            if df_merged[col].isnull().any():
                df_merged[col] = df_merged[col].ffill().bfill()
        return df_merged
    except Exception as e:
        print(f"❌ Error merging dataframes: {e}")
        return df1


# ============================================================
# 🏦 (Optional) FRED Economic Data (Disabled for now)
# ============================================================

def load_fred():
    """
    Placeholder for loading FRED macroeconomic data.
    You can later use fredapi or API keys to add unemployment, CPI, etc.
    """
    # Example skeleton (disabled until API key is added)
    return pd.DataFrame(columns=["Date"])
