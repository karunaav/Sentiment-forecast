# Sentiment‑Aware Stock Forecasting (Transformer + API + Dash + Backtesting + SHAP + MLflow)

**What’s inside**
- Transformer **encoder‑only** model for next‑day return
- **FAANG** universe (AAPL, AMZN, META, NFLX, GOOGL)
- **News sentiment** via FinBERT (fallback VADER)
- **Macroeconomics** (FRED: UNRATE, CPIAUCSL)
- **FastAPI** REST (train/predict)
- **Dash/Plotly** dashboard
- **Walk‑forward retraining** (expanding window)
- **Backtesting** (CAGR, Sharpe, Max DD)
- **SHAP** explanations
- **MLflow** experiment tracking

## Quickstart

```bash
pip install -r requirements.txt

# (optional) set env
copy .env.example .env  # Windows
# or
cp .env.example .env    # macOS/Linux

# Train FAANG (logs to MLflow)
export MLFLOW_TRACKING_URI="file:./mlruns"
python -c "from src.model.train import train_universe; print(train_universe(['AAPL','AMZN','META','NFLX','GOOGL']))"

# Launch MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5001

# API
uvicorn api.main:app --reload --port 8000
# test: http://127.0.0.1:8000/predict?ticker=AAPL

# Dashboard
python dashboard/app.py
# open: http://127.0.0.1:8050

# SHAP (heavy; uses KernelExplainer)
python -c "from src.explain.shap_explain import shap_summary; shap_summary('AAPL', nsamples=200)"
```

> Educational use only — **not financial advice**.
