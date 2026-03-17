import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from predictor import fetch_data, predict, add_indicators, generate_signals, backtest

st.set_page_config(page_title="FINAL AI TRADING TERMINAL", layout="wide")

st.title("🚀 FINAL AI TRADING TERMINAL")

stocks = {"Apple": "AAPL", "Google": "GOOGL", "Tesla": "TSLA"}
selected = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[selected]

# ==========================
# FETCH DATA
# ==========================
data = fetch_data(ticker)

# DEBUG
st.write("Raw Data Shape:", None if data is None else data.shape)

# ==========================
# SAFETY CHECK
# ==========================
if data is None or data.empty:
    st.error("Data not available")
    st.stop()

# Fix columns
data.columns = [col.capitalize() for col in data.columns]

# ==========================
# ADD INDICATORS
# ==========================
data = add_indicators(data)
data = generate_signals(data)

# ==========================
# CLEAN DATA (SAFE)
# ==========================
data = data.fillna(method="bfill").fillna(method="ffill")

# DEBUG
st.write("Processed Data Shape:", data.shape)

# ==========================
# FINAL CHECK
# ==========================
if len(data) < 20:
    st.error("Not enough data after processing")
    st.stop()

# ==========================
# METRICS
# ==========================
price = data["Close"].iloc[-1]
prediction, confidence = predict(data)

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", round(price, 2))

if prediction:
    col2.metric("Prediction", round(prediction, 2))
else:
    col2.warning("Prediction unavailable")

col3.metric("Confidence", f"{confidence*100:.1f}%")

# ==========================
# SIGNAL
# ==========================
signal = data["TradeSignal"].iloc[-1]

if signal == "BUY":
    st.success("📈 BUY SIGNAL")
elif signal == "SELL":
    st.error("📉 SELL SIGNAL")
else:
    st.info("⏳ HOLD")

# ==========================
# BACKTEST
# ==========================
portfolio = backtest(data)
st.metric("Portfolio Value", round(portfolio, 2))

# ==========================
# CANDLESTICK
# ==========================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"]
))

fig.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# RSI
# ==========================
rsi_fig = go.Figure()

rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"))
rsi_fig.add_hline(y=70)
rsi_fig.add_hline(y=30)

rsi_fig.update_layout(template="plotly_dark", title="RSI")

st.plotly_chart(rsi_fig, use_container_width=True)

# ==========================
# MACD
# ==========================
macd_fig = go.Figure()

macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"))
macd_fig.add_trace(go.Scatter(x=data.index, y=data["Signal"], name="Signal"))

macd_fig.update_layout(template="plotly_dark", title="MACD")

st.plotly_chart(macd_fig, use_container_width=True)
