import streamlit as st
import plotly.graph_objects as go
from predictor import fetch_data, predict, add_indicators, generate_signals, backtest

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="ELITE AI TRADING SYSTEM", layout="wide")

st.title("🚀 ELITE AI TRADING TERMINAL")

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
# FETCH DATA
# ==========================
data = fetch_data(ticker)

if data is None:
    st.error("Data fetch failed")
    st.stop()

if data.empty:
    st.error("Empty data from API")
    st.stop()

# Fix column names (important for stooq)
data.columns = [col.capitalize() for col in data.columns]

st.success("Data loaded successfully")

# ==========================
# PROCESS DATA
# ==========================
data = add_indicators(data)
data = generate_signals(data)

# ==========================
# PRICE + PREDICTION
# ==========================
price = data["Close"].iloc[-1]
prediction = predict(data)

col1, col2 = st.columns(2)

col1.metric("Current Price", round(price, 2))

if prediction:
    col2.metric("AI Prediction", round(prediction, 2))
else:
    col2.warning("Prediction not available")

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
final_value = backtest(data)
st.metric("Portfolio Value (Backtest)", round(final_value, 2))

# ==========================
# CANDLESTICK CHART
# ==========================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"]
))

fig.update_layout(template="plotly_dark", height=600)

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
