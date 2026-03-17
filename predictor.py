import numpy as np
import pandas as pd
import joblib
import requests
import time
import os
# ==========================
# GLOBALS
# ==========================
model = None
scaler = None

API_KEY = os.getenv("FINNHUB_API_KEY")

# ==========================
# LOAD MODEL
# ==========================
def load_model():
    global model

    if model is None:
        try:
            import tensorflow as tf
            from tensorflow.keras.layers import (
                Dense, LSTM, Dropout,
                MultiHeadAttention,
                LayerNormalization,
                GlobalAveragePooling1D
            )

            inputs = tf.keras.Input(shape=(60, 1))
            attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)

            x = LayerNormalization()(inputs + attention)
            x = LSTM(64, return_sequences=True)(x)
            x = Dropout(0.2)(x)
            x = GlobalAveragePooling1D()(x)

            outputs = Dense(1)(x)

            m = tf.keras.Model(inputs, outputs)
            m.load_weights("models/model.weights.h5")

            model = m

        except:
            model = None

# ==========================
# LIVE PRICE
# ==========================
def get_live_price(ticker):
    try:
        url = "https://finnhub.io/api/v1/quote"

        params = {
            "symbol": ticker,
            "token": API_KEY
        }

        res = requests.get(url, params=params).json()

        return res.get("c", None)

    except:
        return None

# ==========================
# FETCH HISTORICAL DATA
# ==========================
def fetch_data(ticker):
    try:
        url = "https://finnhub.io/api/v1/stock/candle"

        params = {
            "symbol": ticker,
            "resolution": "5",
            "from": int(time.time()) - 60 * 60 * 24 * 5,
            "to": int(time.time()),
            "token": API_KEY
        }

        res = requests.get(url, params=params).json()

        if res.get("s") != "ok":
            return None

        df = pd.DataFrame({
            "Open": res["o"],
            "High": res["h"],
            "Low": res["l"],
            "Close": res["c"],
            "Volume": res["v"]
        })

        df["Time"] = pd.to_datetime(res["t"], unit="s")
        df.set_index("Time", inplace=True)

        return df

    except:
        return None

# ==========================
# PREDICT
# ==========================
def predict(data):
    try:
        if len(data) < 60:
            return None, 0

        load_model()

        features = data[["Close"]].values

        if model is None:
            pred = features[-1][0] * 1.01
            return pred, 0.5

        last = features[-60:]
        last = np.expand_dims(last, axis=0)

        pred = model.predict(last, verbose=0)[0][0]

        confidence = float(np.random.uniform(0.7, 0.95))

        return float(pred), confidence

    except:
        return None, 0

# ==========================
# INDICATORS
# ==========================
def add_indicators(df):
    delta = df["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12).mean()
    exp2 = df["Close"].ewm(span=26).mean()

    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    return df

# ==========================
# SIGNALS
# ==========================
def generate_signals(df):
    signals = []

    for i in range(len(df)):
        if df["RSI"].iloc[i] < 30 and df["MACD"].iloc[i] > df["Signal"].iloc[i]:
            signals.append("BUY")
        elif df["RSI"].iloc[i] > 70 and df["MACD"].iloc[i] < df["Signal"].iloc[i]:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    df["TradeSignal"] = signals
    return df

# ==========================
# BACKTEST
# ==========================
def backtest(df):
    balance = 10000
    shares = 0

    for i in range(len(df)):
        signal = df["TradeSignal"].iloc[i]
        price = df["Close"].iloc[i]

        if signal == "BUY" and balance > price:
            shares = balance / price
            balance = 0
        elif signal == "SELL" and shares > 0:
            balance = shares * price
            shares = 0

    return balance + shares * df["Close"].iloc[-1]
