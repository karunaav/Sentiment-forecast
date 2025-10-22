from __future__ import annotations
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
USE_FINBERT = os.getenv("USE_FINBERT", "true").lower() == "true"

import requests
from bs4 import BeautifulSoup
from datetime import datetime

def fetch_headlines_finviz(ticker: str) -> pd.DataFrame:
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=15).text
    soup = BeautifulSoup(html, "html.parser")
    news_table = soup.find(id="news-table")
    rows = []
    if news_table:
        for tr in news_table.find_all("tr"):
            a = tr.find("a")
            if not a:
                continue
            headline = a.get_text(strip=True)
            td_time = tr.find("td", class_="nn-date")
            ts = td_time.get_text(strip=True) if td_time else ""
            rows.append({"date_time": ts, "headline": headline})
    df = pd.DataFrame(rows)
    def to_date(s: str):
        parts = s.split()
        try:
            if len(parts) == 2:
                dt = datetime.strptime(" ".join(parts), "%b-%d-%y %I:%M%p")
            else:
                dt = datetime.utcnow()
        except Exception:
            dt = datetime.utcnow()
        return pd.to_datetime(dt.date())
    if not df.empty:
        df["date"] = df["date_time"].apply(to_date)
    return df

if USE_FINBERT:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    _model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    _model.eval()
    def score_sentiment(texts: list[str]) -> list[float]:
        with torch.no_grad():
            toks = _tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
            logits = _model(**toks).logits
            probs = torch.softmax(logits, dim=-1)
            weights = torch.tensor([-1.0, 0.0, 1.0])
            val = (probs * weights).sum(dim=1)
            return val.cpu().numpy().tolist()
else:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _an = SentimentIntensityAnalyzer()
    def score_sentiment(texts: list[str]) -> list[float]:
        return [_an.polarity_scores(str(t))["compound"] for t in texts]

def daily_sentiment(ticker: str) -> pd.DataFrame:
    news = fetch_headlines_finviz(ticker)
    if news.empty:
        return pd.DataFrame({"date": [], "sentiment": []})
    news["sentiment"] = score_sentiment(news["headline"].astype(str).tolist())
    agg = news.groupby("date")["sentiment"].mean().reset_index()
    agg["date"] = pd.to_datetime(agg["date"])
    return agg
