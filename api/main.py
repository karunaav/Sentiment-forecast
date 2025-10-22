import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.model.train import predict_next_return

app = FastAPI(title="Stock Sentiment Forecast API")

# Allow CORS (so dashboard can call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for local use
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "📈 Stock Sentiment Forecast API is running"}

@app.get("/predict/{ticker}")
def get_prediction(ticker: str):
    try:
        result = predict_next_return(ticker)
        if result is None:
            return {"ticker": ticker, "prediction": None, "status": "No data or model issue"}
        return {"ticker": ticker, "prediction": float(result)}
    except Exception as e:
        return {"error": str(e)}

