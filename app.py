"""
app.py
Clean stock price prediction dashboard.
Shows current price, next-day prediction, and 7-day forecast.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

from predictor import fetch_data, get_live_price, predict_next_days

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

STOCKS = {
    "Apple (AAPL)":      "AAPL",
    "Microsoft (MSFT)":  "MSFT",
    "Google (GOOGL)":    "GOOGL",
    "Amazon (AMZN)":     "AMZN",
    "NVIDIA (NVDA)":     "NVDA",
    "Tesla (TSLA)":      "TSLA",
    "Meta (META)":       "META",
    "Berkshire (BRK-B)": "BRK-B",
    "JPMorgan (JPM)":    "JPM",
    "Visa (V)":          "V",
}

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("Stock Predictor")
selected = st.sidebar.selectbox("Select Stock", list(STOCKS.keys()))
ticker   = STOCKS[selected]

# ==========================
# HEADER
# ==========================
st.title(f"{selected}")
st.caption("AI-powered next-day and 7-day closing price prediction")
st.markdown("---")

# ==========================
# LOAD DATA
# ==========================
with st.spinner("Loading data..."):
    data       = fetch_data(ticker)
    live_price = get_live_price(ticker)

if data is None or data.empty:
    st.error("Failed to load historical data. Check your internet connection.")
    st.stop()

# ==========================
# PREDICTIONS
# ==========================
predictions = predict_next_days(ticker, data, n_days=7)

current_price = live_price if live_price else float(data["Close"].iloc[-1])

if predictions is None:
    st.error("Model not found. Please run model_train.py first and upload models/ and scalers/ folders.")
    st.stop()

next_day_pred = predictions[0]
change        = next_day_pred - current_price
change_pct    = (change / current_price) * 100

# ==========================
# METRICS ROW
# ==========================
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current Price",
    f"${current_price:.2f}",
    help="Live price from Finnhub"
)
col2.metric(
    "Next Day Prediction",
    f"${next_day_pred:.2f}",
    delta=f"{change:+.2f} ({change_pct:+.1f}%)"
)
col3.metric(
    "7-Day High (predicted)",
    f"${max(predictions):.2f}"
)
col4.metric(
    "7-Day Low (predicted)",
    f"${min(predictions):.2f}"
)

st.markdown("---")

# ==========================
# CHARTS — side by side
# ==========================
left, right = st.columns(2)

# --- Historical 60-day price chart ---
with left:
    st.subheader("Historical Price (60 days)")

    hist = data.tail(60)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        name="Close",
        line=dict(color="#4A90D9", width=2)
    ))
    fig_hist.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# --- 7-day forecast chart ---
with right:
    st.subheader("7-Day Price Forecast")

    last_date    = data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(7)]

    fig_pred = go.Figure()

    # Anchor line — last known price to first prediction
    fig_pred.add_trace(go.Scatter(
        x=[last_date, future_dates[0]],
        y=[current_price, predictions[0]],
        mode="lines",
        line=dict(color="#888888", width=1, dash="dot"),
        showlegend=False
    ))

    # Prediction line
    fig_pred.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode="lines+markers",
        name="Predicted Close",
        line=dict(color="#F5A623", width=2),
        marker=dict(size=7)
    ))

    # Today's price reference line
    fig_pred.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="#4A90D9",
        annotation_text=f"Current ${current_price:.2f}",
        annotation_position="top left"
    )

    fig_pred.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        showlegend=False
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# ==========================
# 7-DAY TABLE
# ==========================
st.subheader("7-Day Forecast Table")

rows = []
for i, (date, price) in enumerate(zip(future_dates, predictions)):
    day_change     = price - current_price
    day_change_pct = (day_change / current_price) * 100
    rows.append({
        "Day":        f"Day {i+1}",
        "Date":       date.strftime("%a, %b %d"),
        "Predicted Close": f"${price:.2f}",
        "Change from Today": f"{day_change:+.2f}",
        "Change %":  f"{day_change_pct:+.1f}%"
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Predictions are generated by an LSTM + Transformer model trained on 2 years of daily data. Not financial advice.")
