import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from predictor import (
    fetch_data,
    predict,
    add_indicators,
    generate_signals,
    backtest,
    get_live_price
)

st.set_page_config(page_title="LIVE AI TRADING", layout="wide")
st.title("LIVE AI TRADING TERMINAL")

st_autorefresh(interval=3000, key="live")

stocks = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN"
}

selected = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[selected]

live_price = get_live_price(ticker)

if live_price:
    st.success(f"LIVE PRICE: ${round(live_price, 2)} USD")
else:
    st.warning("Live price unavailable — check FINNHUB_API_KEY env variable")

with st.spinner("Fetching market data..."):
    data = fetch_data(ticker)

if data is None or data.empty:
    st.error("Failed to fetch historical data from yfinance. Try again in a moment.")
    st.stop()

if live_price:
    data.iloc[-1, data.columns.get_loc("Close")] = live_price

data = add_indicators(data)
data = generate_signals(data)
data = data.ffill().bfill()

price = float(data["Close"].iloc[-1])
prediction, confidence = predict(data)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${round(price, 2)}")

if prediction:
    delta = round(prediction - price, 2)
    col2.metric("AI Prediction", f"${round(prediction, 2)}", delta=f"{delta:+.2f}")
else:
    col2.metric("AI Prediction", "N/A")

col3.metric("Confidence", f"{confidence*100:.1f}%" if confidence else "N/A")
col4.metric("Portfolio Value", f"${round(backtest(data), 2)}")

signal = data["TradeSignal"].iloc[-1]
rsi_val = round(float(data["RSI"].iloc[-1]), 1)
macd_val = round(float(data["MACD"].iloc[-1]), 4)

st.markdown("---")
if signal == "BUY":
    st.success(f"BUY SIGNAL  |  RSI: {rsi_val}  |  MACD: {macd_val}")
elif signal == "SELL":
    st.error(f"SELL SIGNAL  |  RSI: {rsi_val}  |  MACD: {macd_val}")
else:
    st.info(f"HOLD  |  RSI: {rsi_val}  |  MACD: {macd_val}")
st.markdown("---")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))
fig.update_layout(
    template="plotly_dark",
    height=500,
    title=f"{selected} — 5-min Candles",
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig, use_container_width=True)

rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(
    x=data.index, y=data["RSI"],
    name="RSI", line=dict(color="#00bfff")
))
rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
rsi_fig.update_layout(template="plotly_dark", height=250, title="RSI (14)")
st.plotly_chart(rsi_fig, use_container_width=True)

macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(
    x=data.index, y=data["MACD"],
    name="MACD", line=dict(color="#ff9900")
))
macd_fig.add_trace(go.Scatter(
    x=data.index, y=data["Signal"],
    name="Signal", line=dict(color="#00ff99")
))
macd_fig.add_bar(
    x=data.index,
    y=data["MACD"] - data["Signal"],
    name="Histogram",
    marker_color="gray"
)
macd_fig.update_layout(template="plotly_dark", height=250, title="MACD")
st.plotly_chart(macd_fig, use_container_width=True)
