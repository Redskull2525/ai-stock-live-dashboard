import streamlit as st
import plotly.graph_objects as go
from predictor import fetch_data, predict, add_indicators, generate_signals, backtest

st.set_page_config(page_title="ELITE AI TRADING SYSTEM", layout="wide")

st.title("🚀 ELITE AI TRADING TERMINAL")

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

if data is None or data.empty:
    st.warning("Using fallback data")
else:
    st.success("Data loaded successfully")

# Fix column names
data.columns = [col.capitalize() for col in data.columns]

# ==========================
# CLEAN DATA (IMPORTANT)
# ==========================
data = data.dropna()

# ==========================
# ADD INDICATORS
# ==========================
data = add_indicators(data)
data = generate_signals(data)

# Remove NaN again after indicators
data = data.dropna()

# ==========================
# CHECK DATA
# ==========================
if len(data) < 10:
    st.error("Not enough data after processing")
    st.stop()

# ==========================
# METRICS
# ==========================
price = data["Close"].iloc[-1]
prediction = predict(data)

col1, col2 = st.columns(2)

col1.metric("Current Price", round(price, 2))

if prediction is not None:
    col2.metric("AI Prediction", round(prediction, 2))
else:
    col2.warning("Prediction unavailable")

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
portfolio_value = backtest(data)
st.metric("Portfolio Value", round(portfolio_value, 2))

# ==========================
# CANDLESTICK
# ==========================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))

fig.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# RSI
# ==========================
if "RSI" in data:
    rsi_fig = go.Figure()

    rsi_fig.add_trace(go.Scatter(
        x=data.index,
        y=data["RSI"],
        name="RSI"
    ))

    rsi_fig.add_hline(y=70)
    rsi_fig.add_hline(y=30)

    rsi_fig.update_layout(template="plotly_dark", title="RSI Indicator")

    st.plotly_chart(rsi_fig, use_container_width=True)

# ==========================
# MACD
# ==========================
if "MACD" in data:
    macd_fig = go.Figure()

    macd_fig.add_trace(go.Scatter(
        x=data.index,
        y=data["MACD"],
        name="MACD"
    ))

    macd_fig.add_trace(go.Scatter(
        x=data.index,
        y=data["Signal"],
        name="Signal Line"
    ))

    macd_fig.update_layout(template="plotly_dark", title="MACD Indicator")

    st.plotly_chart(macd_fig, use_container_width=True)
