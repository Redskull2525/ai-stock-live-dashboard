import yfinance as yf
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D
)

ticker = "AAPL"

data = yf.download(ticker, period="2y", interval="1d")

# Indicators
data["SMA20"] = data["Close"].rolling(20).mean()
data["EMA20"] = data["Close"].ewm(span=20).mean()

delta = data["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

data["RSI"] = 100 - (100 / (1 + rs))

data["MA20"] = data["Close"].rolling(20).mean()
data["STD"] = data["Close"].rolling(20).std()

data["Upper"] = data["MA20"] + (2 * data["STD"])
data["Lower"] = data["MA20"] - (2 * data["STD"])

data.dropna(inplace=True)

features = data[
[
"Close",
"Volume",
"SMA20",
"EMA20",
"RSI",
"Upper",
"Lower"
]
]

scaler = MinMaxScaler()

scaled = scaler.fit_transform(features)

X=[]
y=[]

window=60

for i in range(window,len(scaled)):
    X.append(scaled[i-window:i])
    y.append(scaled[i,0])

X=np.array(X)
y=np.array(y)

# Transformer block
inputs = tf.keras.Input(shape=(X.shape[1],X.shape[2]))

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

model.compile(
optimizer="adam",
loss="mse"
)

model.fit(
X,
y,
epochs=20,
batch_size=32
)

model.save("models/transformer_model.keras")

pickle.dump(
scaler,
open("models/scaler.pkl","wb")
)
