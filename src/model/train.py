import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from src.utils.data import load_prices, merge_on_date
from src.utils.features import add_technical

# ============================================================
# 🧠 Transformer Model Builder
# ============================================================

def build_transformer(window: int, n_features: int):
    """
    Build a lightweight Transformer encoder model for time series forecasting.
    """
    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=n_features)(x, x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


# ============================================================
# 🧩 Dataset Builder
# ============================================================

def build_dataset(ticker: str, window: int = 30):
    """
    Create a time-series dataset from price + technical data.
    """
    print(f"📥 Loading data for {ticker} ...")
    df = load_prices(ticker)
    if df.empty:
        print(f"⚠️ Skipping {ticker} — no data found.")
        return None

    # Add technical indicators
    df = add_technical(df)
    df = merge_on_date(df)

    # Select features
    features = ["ret1", "rsi14", "vol_chg", "ma5", "ma20"]
    df = df.dropna(subset=features).reset_index(drop=True)

    if df.empty:
        print(f"⚠️ Skipping {ticker} — insufficient data.")
        return None

    # Prepare sliding window dataset
    X, y = [], []
    for i in range(window, len(df)):
        X.append(df[features].iloc[i - window:i].values)
        y.append(df["ret1"].iloc[i])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    return X, y


# ============================================================
# 🎯 Train and Save Model
# ============================================================

def train_and_save(ticker: str, window: int = 30, epochs: int = 5):
    """
    Train a Transformer model for one ticker and save it.
    """
    try:
        data = build_dataset(ticker, window)
        if data is None:
            print(f"⚠️ Skipping {ticker} — no valid dataset.")
            return None

        X, y = data
        model = build_transformer(window, X.shape[2])
        model.fit(X, y, epochs=epochs, batch_size=16, verbose=1)

        # Evaluate model
        mse = float(np.mean((model.predict(X).flatten() - y) ** 2))

        os.makedirs(f"models/{ticker}", exist_ok=True)
        model.save(f"models/{ticker}/model.h5")  # can also use .keras

        result = {"ticker": ticker, "mse": mse, "samples": len(X)}
        print(f"✅ Finished training {ticker}: MSE={mse:.6f}")
        return result

    except Exception as e:
        print(f"❌ Error training {ticker}: {e}")
        return None


# ============================================================
# 🪙 Train Universe (FAANG)
# ============================================================

def train_universe(tickers):
    """
    Train multiple tickers (FAANG etc.) and log results.
    """
    results = {}
    for t in tickers:
        print(f"\n🚀 Training {t} ...")
        result = train_and_save(t)
        results[t] = result if result else {"ticker": t, "mse": None, "samples": 0}

    os.makedirs("models", exist_ok=True)
    pd.DataFrame(results.values()).to_csv("models/training_results.csv", index=False)
    print("\n✅ All tickers processed. Results saved to models/training_results.csv\n")
    return results


# ============================================================
# 🔮 Predict Next Return for a Single Ticker
# ============================================================

def predict_next_return(ticker: str, window: int = 30):
    """
    Load trained model and predict the next-day return.
    """
    try:
        model_path = os.path.join("models", ticker, "model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found for {ticker}")

        # ✅ FIX: Load without compiling
        model = tf.keras.models.load_model(model_path, compile=False)

        df = load_prices(ticker)
        df = add_technical(df)

        features = ["ret1", "rsi14", "vol_chg", "ma5", "ma20"]
        df = df.dropna(subset=features).reset_index(drop=True)
        X = df[features].to_numpy(dtype=np.float32)[-window:]

        if len(X) < window:
            raise ValueError("Not enough recent data for prediction")

        pred = model.predict(X[np.newaxis, :, :])[0][0]
        print(f"🔮 Predicted next return for {ticker}: {pred:.5f}")
        return float(pred)

    except Exception as e:
        print(f"❌ Prediction error for {ticker}: {e}")
        return None
