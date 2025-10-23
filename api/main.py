# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def home():
    return {"message": "Sentiment Forecast API is running 🚀"}

# No uvicorn.run() here — Vercel handles it automatically!

app = FastAPI(title="Stock Forecasting API", version="1.0")

# Allow the Dash app (localhost and any dev origins). Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["http://127.0.0.1:8050", "http://localhost:8050"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAANG = ["AAPL", "AMZN", "META", "NFLX", "GOOGL"]

class ForecastOut(BaseModel):
    ticker: str
    predicted_return: float
    confidence: float
    as_of: str

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": dt.datetime.utcnow().isoformat()}

def _fetch_prices(ticker: str, days: int = 365) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=days * 2)  # cushion for market holidays
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df = df.rename(columns=str.title).reset_index()[["Date", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def _toy_model_next_return(prices: pd.Series) -> tuple[float, float]:
    """
    Simple, fast placeholder:
    - Signal = 5d momentum minus 20d momentum (scaled)
    - Confidence = absolute z-score of last 10d returns (bounded)
    Replace this with your real model later; API contract stays the same.
    """
    close = prices.dropna().astype(float)
    if len(close) < 25:
        return 0.0, 0.2
    ret = close.pct_change().dropna()
    mom5 = close.pct_change(5).iloc[-1]
    mom20 = close.pct_change(20).iloc[-1]
    signal = 0.5 * (mom5 - mom20)
    # crude confidence: higher when recent vol is low and momentum is strong
    z = (ret.iloc[-1] - ret.tail(50).mean()) / (ret.tail(50).std() + 1e-9)
    confidence = float(np.clip(abs(z) * 0.25 + min(0.9, abs(signal) * 10), 0.15, 0.95))
    return float(signal), confidence

@app.get("/predict/{ticker}", response_model=ForecastOut)
def predict_ticker(ticker: str):
    ticker = ticker.upper()
    if ticker not in FAANG:
        return ForecastOut(
            ticker=ticker, predicted_return=0.0, confidence=0.2,
            as_of=dt.datetime.utcnow().isoformat()
        )
    df = _fetch_prices(ticker)
    pred, conf = _toy_model_next_return(df["Close"])
    return ForecastOut(
        ticker=ticker,
        predicted_return=pred,
        confidence=conf,
        as_of=dt.datetime.utcnow().isoformat(),
    )

# Optional: run API directly (useful if you expose via ngrok)

