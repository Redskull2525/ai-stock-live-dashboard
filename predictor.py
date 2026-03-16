import numpy as np
import pandas as pd
import pickle
import yfinance as yf

from tensorflow.keras.models import load_model

model = load_model("models/lstm_model.keras")

scaler = pickle.load(open("models/scaler.pkl","rb"))

def fetch_live_data(ticker):

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

    data.dropna(inplace=True)

    return data


def predict_next_minute(data):

    data = add_indicators(data)

    features = data[["Close","Volume","SMA20","EMA20","RSI"]]

    scaled = scaler.transform(features)

    last_window = scaled[-60:]

    pred = model.predict(
        last_window.reshape(1,60,last_window.shape[1])
    )

    prediction = scaler.inverse_transform(
        np.hstack(
            [pred,
             np.zeros((1,4))]
        )
    )

    return float(prediction[0][0])
