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

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="LIVE AI TRADING", layout="wide")

st.title("🚀 LIVE AI TRADING TERMINAL ---- please wait for couple of mins to load the site")

# ==========================
# AUTO REFRESH
# ==========================
st_autorefresh(interval=3000, key="live")

# ==========================
# SIDEBAR
# ==========================
stocks = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Tesla": "TSLA"
}

selected = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[selected]

# ==========================
# LIVE PRICE
# ==========================
live_price = get_live_price(ticker)

if live_price:
    st.success(f"🔴 LIVE PRICE: {round(live_price, 2)} USD")
else:
    st.warning("Live price not available")

# ==========================
# FETCH DATA
# ==========================
data = fetch_data(ticker)

if data is None or data.empty:
    st.error("Failed to fetch data")
    st.stop()

# Replace last price with live price
if live_price:
    data.iloc[-1, data.columns.get_loc("Close")] = live_price

# ==========================
# PROCESS
# ==========================
data = add_indicators(data)
data = generate_signals(data)
data = data.fillna(method="bfill").fillna(method="ffill")

# ==========================
# METRICS
# ==========================
price = data["Close"].iloc[-1]
prediction, confidence = predict(data)

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", round(price, 2))
col2.metric("Prediction", round(prediction, 2))
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
st.metric("Portfolio Value", round(backtest(data), 2))

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
