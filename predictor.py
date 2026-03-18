"""
predictor.py
Handles data fetching, model loading, and predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
import yfinance as yf

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
WINDOW      = 60
FEATURES    = ["Close", "Volume", "SMA20", "EMA20", "RSI", "Upper", "Lower"]

_model_cache  = {}
_scaler_cache = {}


# ==========================
# BUILD MODEL ARCHITECTURE
# ==========================
def _build_model():
    inputs    = tf.keras.Input(shape=(WINDOW, len(FEATURES)))
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x         = LayerNormalization()(inputs + attention)
    x         = LSTM(64, return_sequences=True)(x)
    x         = Dropout(0.2)(x)
    x         = GlobalAveragePooling1D()(x)
    outputs   = Dense(1)(x)
    model     = tf.keras.Model(inputs, outputs)
    return model


# ==========================
# LOAD MODEL + SCALER
# ==========================
def load_assets(ticker):
    if ticker not in _model_cache:
        try:
            m = _build_model()
            m.load_weights(f"models/{ticker}_model.weights.h5")
            _model_cache[ticker] = m
        except Exception as e:
            print(f"Model load error for {ticker}: {e}")
            _model_cache[ticker] = None

    if ticker not in _scaler_cache:
        try:
            scaler = pickle.load(open(f"scalers/{ticker}_scaler.pkl", "rb"))
            _scaler_cache[ticker] = scaler
        except Exception as e:
            print(f"Scaler load error for {ticker}: {e}")
            _scaler_cache[ticker] = None

    return _model_cache[ticker], _scaler_cache[ticker]


# ==========================
# INDICATORS
# ==========================
def add_indicators(data):
    data = data.copy()
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["EMA20"] = data["Close"].ewm(span=20).mean()

    delta        = data["Close"].diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = -delta.clip(upper=0).rolling(14).mean()
    rs           = gain / loss
    data["RSI"]  = 100 - (100 / (1 + rs))

    data["MA20"]  = data["Close"].rolling(20).mean()
    data["STD"]   = data["Close"].rolling(20).std()
    data["Upper"] = data["MA20"] + (2 * data["STD"])
    data["Lower"] = data["MA20"] - (2 * data["STD"])

    data.dropna(inplace=True)
    return data


# ==========================
# FETCH HISTORICAL DAILY DATA
# ==========================
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = add_indicators(df)
        return df
    except Exception as e:
        print(f"fetch_data error for {ticker}: {e}")
        return None


# ==========================
# LIVE PRICE (Finnhub)
# ==========================
def get_live_price(ticker):
    try:
        res = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": ticker, "token": FINNHUB_KEY},
            timeout=5
        ).json()
        return res.get("c", None)
    except:
        return None


# ==========================
# PREDICT NEXT N DAYS
# ==========================
def predict_next_days(ticker, data, n_days=7):
    """
    Returns a list of predicted closing prices for the next n_days.
    Uses iterative prediction — each prediction feeds into the next.
    """
    model, scaler = load_assets(ticker)

    if model is None or scaler is None:
        return None

    try:
        features = data[FEATURES].values
        scaled   = scaler.transform(features)

        # Start with the last WINDOW rows
        sequence = list(scaled[-WINDOW:])
        preds_scaled = []

        for _ in range(n_days):
            x   = np.array(sequence[-WINDOW:])
            x   = np.expand_dims(x, axis=0)
            out = model.predict(x, verbose=0)[0][0]
            preds_scaled.append(out)

            # Build next row: use predicted Close, carry forward last indicators
            last_row      = sequence[-1].copy()
            last_row[0]   = out          # index 0 = Close (scaled)
            sequence.append(last_row)

        # Inverse transform — only Close column (index 0)
        dummy        = np.zeros((n_days, len(FEATURES)))
        dummy[:, 0]  = preds_scaled
        predictions  = scaler.inverse_transform(dummy)[:, 0]

        return predictions.tolist()

    except Exception as e:
        print(f"Prediction error for {ticker}: {e}")
        return None
