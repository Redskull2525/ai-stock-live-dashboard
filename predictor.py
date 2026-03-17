import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D
)

# -------------------------------
# Load scaler safely
# -------------------------------
try:
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    print("Error loading scaler:", e)
    scaler = None


# -------------------------------
# Build model
# -------------------------------
def build_model():
    inputs = tf.keras.Input(shape=(60, 1))  # ⚠️ FIXED (was 7, but you only use Close)

    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=32
    )(inputs, inputs)

    x = LayerNormalization()(inputs + attention)

    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    x = GlobalAveragePooling1D()(x)

    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# -------------------------------
# Load model safely
# -------------------------------
try:
    model = build_model()
    model.load_weights("models/model.weights.h5")
except Exception as e:
    print("Error loading model:", e)
    model = None


# -------------------------------
# Fetch data
# -------------------------------
def fetch_data(ticker):
    try:
        data = yf.download(
            ticker,
            period="5d",   # ⚠️ safer than 1d (ensures enough rows)
            interval="1m"
        )
        return data
    except Exception as e:
        print("Error fetching data:", e)
        return None


# -------------------------------
# Predict
# -------------------------------
def predict(data):
    try:
        if data is None or len(data) < 60:
            return "Not enough data"

        features = data[["Close"]].values

        # Apply scaling if available
        if scaler:
            features = scaler.transform(features)

        last = features[-60:]
        last = np.expand_dims(last, axis=0)

        prediction = model.predict(last, verbose=0)

        return float(prediction[0][0])

    except Exception as e:
        print("Prediction error:", e)
        return None
