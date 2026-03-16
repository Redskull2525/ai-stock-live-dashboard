import streamlit as st
import pandas as pd
import time

import plotly.graph_objects as go

from predictor import fetch_live_data, predict_next_minute


st.set_page_config(layout="wide")

st.title("AI Real-Time Stock Prediction Dashboard")


stocks = {

"Apple":"AAPL",
"Google":"GOOGL",
"Meta":"META",
"Oracle":"ORCL",
"Tesla":"TSLA"

}

stock = st.sidebar.selectbox(
    "Select Stock",
    list(stocks.keys())
)

ticker = stocks[stock]

chart_area = st.empty()

table_area = st.empty()

buffer = pd.DataFrame()


while True:

    data = fetch_live_data(ticker)

    prediction = predict_next_minute(data)

    latest_price = data["Close"].iloc[-1]

    timestamp = data.index[-1]

    new_row = pd.DataFrame({

        "Time":[timestamp],
        "Actual":[latest_price],
        "Predicted":[prediction]

    })

    buffer = pd.concat(
        [buffer,new_row]
    ).tail(200)


    fig = go.Figure()

    fig.add_trace(go.Scatter(

        x=buffer["Time"],
        y=buffer["Actual"],
        name="Actual Price"

    ))

    fig.add_trace(go.Scatter(

        x=buffer["Time"],
        y=buffer["Predicted"],
        name="Predicted Price (1m Ahead)"

    ))

    fig.update_layout(

        title="Live Price vs AI Prediction",
        xaxis_title="Time",
        yaxis_title="Price"

    )

    chart_area.plotly_chart(
        fig,
        use_container_width=True
    )

    table_area.dataframe(
        buffer.tail(10)
    )

    time.sleep(0.5)
