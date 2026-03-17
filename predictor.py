import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D

# ==========================
# LOAD SCALER
# ==========================
try:
    scaler = joblib.load("models/scaler.pkl")
except:
    scaler = None

# ==========================
# MODEL
# ==========================
def build_model():
    inputs = tf.keras.Input(shape=(60, 1))

    attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x = LayerNormalization()(inputs + attention)

    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    x = GlobalAveragePooling1D()(x)

    outputs = Dense(1)(x)

    return tf.keras.Model(inputs, outputs)

try:
    model = build_model()
    model.load_weights("models/model.weights.h5")
except:
    model = None

# ==========================
# FETCH DATA
# ==========================
def fetch_data(ticker):
    try:
        return yf.download(ticker, period="1d", interval="5m")
    except:
        return None

# ==========================
# PREDICT
# ==========================
def predict(data):
    try:
        if data is None or len(data) < 60:
            return None

        features = data[["Close"]].values

        if scaler:
            features = scaler.transform(features)

        last = features[-60:]
        last = np.expand_dims(last, axis=0)

        pred = model.predict(last, verbose=0)

        return float(pred[0][0])
    except:
        return None

# ==========================
# RSI + MACD
# ==========================
def add_indicators(df):

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df
