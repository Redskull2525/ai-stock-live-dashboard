import yfinance as yf
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.models import load_model

model = load_model("models/transformer_model.keras")

scaler = pickle.load(open("models/scaler.pkl","rb"))

def fetch_data(ticker):

    data = yf.download(
        ticker,
        period="1d",
        interval="1m"
    )

    return data


def add_indicators(data):

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

    return data


def predict(data):

    data = add_indicators(data)

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

    scaled = scaler.transform(features)

    last = scaled[-60:]

    pred = model.predict(
        last.reshape(1,60,last.shape[1])
    )

    output = scaler.inverse_transform(
        np.hstack([pred,np.zeros((1,6))])
    )

    return float(output[0][0])
