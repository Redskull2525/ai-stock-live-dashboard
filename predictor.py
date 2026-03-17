import numpy as np
import pandas as pd
import yfinance as yf
import pickle

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D
)

# Load scaler
scaler = pickle.load(open("models/scaler.pkl","rb"))

# Rebuild model architecture
def build_model():

    inputs = tf.keras.Input(shape=(60,7))

    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=32
    )(inputs,inputs)

    x = LayerNormalization()(inputs + attention)

    x = LSTM(64, return_sequences=True)(x)

    x = Dropout(0.2)(x)

    x = GlobalAveragePooling1D()(x)

    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs,outputs)

    return model

# Load model
model = build_model()
model.load_weights("models/model.weights.h5")

# Fetch data
def fetch_data(ticker):

    data = yf.download(
        ticker,
        period="1d",
        interval="1m"
    )

    return data

# Predict
def predict(data):

    features = data[["Close"]].values

    last = features[-60:]

    last = np.expand_dims(last, axis=0)

    prediction = model.predict(last)

    return float(prediction[0][0])
