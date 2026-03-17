import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from predictor import fetch_data, predict
from streamlit_autorefresh import st_autorefresh

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AI Trading Terminal",
    layout="wide",
    page_icon="📈"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color: white;
}

[data-testid="stSidebar"]{
background: linear-gradient(180deg,#141E30,#243B55);
}

h1,h2,h3{
text-align:center;
color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================

st.title("📈 AI Trading Terminal")

st.markdown("""
### Real-Time Stock Prediction Platform  

👨‍💻 **Abhishek Shelke**  
M.Sc Computer Science  

🔗 GitHub: https://github.com/Redskull2525  
🔗 LinkedIn: https://www.linkedin.com/in/abhishek-s-b98895249  
""")

st.divider()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("⚙️ Controls")

stocks = {
    "Apple":"AAPL",
    "Google":"GOOGL",
    "Meta":"META",
    "Oracle":"ORCL",
    "Tesla":"TSLA"
}

selected = st.sidebar.multiselect(
    "Select Stocks",
    list(stocks.keys()),
    default=["Apple"]
)

refresh_speed = st.sidebar.slider(
    "Refresh Speed (seconds)",
    0.5,
    5.0,
    1.0
)

# ==========================
# AUTO REFRESH
# ==========================

st_autorefresh(
    interval=int(refresh_speed * 1000),
    key="refresh"
)

# ==========================
# SESSION STATE BUFFER
# ==========================

if "data_buffer" not in st.session_state:
    st.session_state.data_buffer = pd.DataFrame()

data_buffer = st.session_state.data_buffer

# ==========================
# MAIN DASHBOARD
# ==========================

st.subheader("📊 Live Price vs AI Prediction")

fig = go.Figure()

for s in selected:

    ticker = stocks[s]

    data = fetch_data(ticker)

    if data.empty:
        continue

    prediction = predict(data)

    price = data["Close"].iloc[-1]

    timestamp = data.index[-1]

    new_row = pd.DataFrame({
        "Time":[timestamp],
        "Stock":[s],
        "Actual":[price],
        "Predicted":[prediction]
    })

    data_buffer = pd.concat(
        [data_buffer,new_row]
    ).tail(300)

    subset = data_buffer[
        data_buffer["Stock"] == s
    ]

    fig.add_trace(go.Scatter(
        x=subset["Time"],
        y=subset["Actual"],
        name=f"{s} Actual"
    ))

    fig.add_trace(go.Scatter(
        x=subset["Time"],
        y=subset["Predicted"],
        name=f"{s} Predicted"
    ))

# ==========================
# PLOT
# ==========================

fig.update_layout(
    title="Live Stock Price vs AI Prediction",
    xaxis_title="Time",
    yaxis_title="Price"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# SAVE BUFFER
# ==========================

st.session_state.data_buffer = data_buffer
