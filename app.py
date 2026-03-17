import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from predictor import fetch_data, predict, add_indicators
from streamlit_autorefresh import st_autorefresh

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="AI Trading Terminal", layout="wide")

# ==========================
# HEADER
# ==========================
st.title("📈 AI Trading Terminal")

st.markdown("### Live Trading Dashboard with AI + Indicators")

# ==========================
# SIDEBAR
# ==========================
stocks = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Tesla": "TSLA"
}

selected = st.sidebar.selectbox("Select Stock", list(stocks.keys()))

refresh = st.sidebar.slider("Refresh (sec)", 1, 5, 2)

st_autorefresh(interval=refresh * 1000, key="refresh")

# ==========================
# FETCH DATA
# ==========================
ticker = stocks[selected]

data = fetch_data(ticker)

if data is None or data.empty:
    st.error("No data available")
    st.stop()

# ==========================
# ADD INDICATORS
# ==========================
data = add_indicators(data)

# ==========================
# PRICE + PREDICTION
# ==========================
price = data["Close"].iloc[-1]
prediction = predict(data)

col1, col2 = st.columns(2)

col1.metric("Current Price", round(price, 2))

if prediction:
    col2.metric("Predicted Price", round(prediction, 2))
else:
    col2.warning("Prediction not available")

# ==========================
# BUY / SELL SIGNAL
# ==========================
if prediction:
    if prediction > price:
        st.success("📈 BUY Signal")
    else:
        st.error("📉 SELL Signal")

# ==========================
# CANDLESTICK CHART
# ==========================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Candlestick"
))

fig.update_layout(
    title="Candlestick Chart",
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# RSI
# ==========================
rsi_fig = go.Figure()

rsi_fig.add_trace(go.Scatter(
    x=data.index,
    y=data["RSI"],
    name="RSI"
))

rsi_fig.add_hline(y=70)
rsi_fig.add_hline(y=30)

rsi_fig.update_layout(title="RSI Indicator", template="plotly_dark")

st.plotly_chart(rsi_fig, use_container_width=True)

# ==========================
# MACD
# ==========================
macd_fig = go.Figure()

macd_fig.add_trace(go.Scatter(
    x=data.index,
    y=data["MACD"],
    name="MACD"
))

macd_fig.add_trace(go.Scatter(
    x=data.index,
    y=data["Signal"],
    name="Signal"
))

macd_fig.update_layout(title="MACD Indicator", template="plotly_dark")

st.plotly_chart(macd_fig, use_container_width=True)
