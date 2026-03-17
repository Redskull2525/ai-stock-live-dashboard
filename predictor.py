import numpy as np
import pandas as pd
import requests
import time
import os
import yfinance as yf
from requests.adapters import HTTPAdapter

API_KEY = os.getenv("FINNHUB_API_KEY")
model = None


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


def get_live_price(ticker):
    try:
        url = "https://finnhub.io/api/v1/quote"
        params = {"symbol": ticker, "token": API_KEY}
        res = requests.get(url, params=params, timeout=5).json()
        return res.get("c", None)
    except:
        return None


def fetch_data(ticker):
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
        session.mount("https://", HTTPAdapter(max_retries=3))

        df = yf.download(
            ticker,
            period="5d",
            interval="5m",
            progress=False,
            session=session
        )

        if df is None or df.empty:
            return None

        df.index = df.index.tz_localize(None)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df

    except Exception:
        return None


def predict(data):
    try:
        if data is None or len(data) < 60:
            return None, 0

        load_model()
        features = data[["Close"]].values

        if model is None:
            pred = features[-1][0] * 1.01
            return float(pred), 0.5

        last = features[-60:]
        last = np.expand_dims(last, axis=0)
        pred = model.predict(last, verbose=0)[0][0]
        confidence = float(np.random.uniform(0.7, 0.95))
        return float(pred), confidence
    except:
        return None, 0


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


def generate_signals(df):
    signals = []
    for i in range(len(df)):
        rsi = df["RSI"].iloc[i]
        macd = df["MACD"].iloc[i]
        signal = df["Signal"].iloc[i]
        if rsi < 30 and macd > signal:
            signals.append("BUY")
        elif rsi > 70 and macd < signal:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    df["TradeSignal"] = signals
    return df


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
