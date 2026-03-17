import numpy as np
import pandas as pd
import yfinance as yf
import joblib

# Lazy globals
model = None
scaler = None


# ==========================
# LOAD SCALER
# ==========================
def load_scaler():
    global scaler
    if scaler is None:
        try:
            scaler = joblib.load("models/scaler.pkl")
        except Exception as e:
            print("Scaler error:", e)
            scaler = None


# ==========================
# LOAD MODEL (SAFE)
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

            attention = MultiHeadAttention(
                num_heads=4,
                key_dim=32
            )(inputs, inputs)

            x = LayerNormalization()(inputs + attention)

            x = LSTM(64, return_sequences=True)(x)
            x = Dropout(0.2)(x)

            x = GlobalAveragePooling1D()(x)
            outputs = Dense(1)(x)

            m = tf.keras.Model(inputs, outputs)
            m.load_weights("models/model.weights.h5")

            model = m

        except Exception as e:
            print("Model error:", e)
            model = None


# ==========================
# FETCH DATA
# ==========================
def fetch_data(ticker):
    try:
        return yf.download(ticker, period="5d", interval="5m")
    except Exception as e:
        print("Fetch error:", e)
        return None


# ==========================
# PREDICT
# ==========================
def predict(data):
    try:
        if data is None or len(data) < 60:
            return None

        load_scaler()
        load_model()

        if model is None:
            return None

        features = data[["Close"]].values

        if scaler:
            features = scaler.transform(features)

        last = features[-60:]
        last = np.expand_dims(last, axis=0)

        pred = model.predict(last, verbose=0)
        return float(pred[0][0])

    except Exception as e:
        print("Prediction error:", e)
        return None


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
def backtest(df, initial_balance=10000):

    balance = initial_balance
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

    final_value = balance + shares * df["Close"].iloc[-1]

    return final_value
