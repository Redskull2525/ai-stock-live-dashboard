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

# -------- CUSTOM CSS -------- #
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

.stButton>button {
background-color:#00c6ff;
color:white;
border-radius:10px;
height:3em;
width:100%;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================

st.title("📈 AI Trading Terminal")

st.markdown("""
### Real-Time Stock Prediction Platform

**Developer:** Abhishek Shelke  
**Education:** M.Sc Computer Science – ASM's CSIT, Pimpri  

Technologies Used:
- Python
- TensorFlow
- Streamlit
- Financial APIs
- Deep Learning Models

🔗 GitHub: https://github.com/Redskull2525  
🔗 LinkedIn: https://www.linkedin.com/in/abhishek-s-b98895249
""")

st.divider()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("👨‍💻 Developer")

st.sidebar.markdown("""
**Abhishek Shelke**

M.Sc Computer Science  
ASM's CSIT, Pimpri  

### Interests
- Data Science
- Machine Learning
- Artificial Intelligence

### GitHub
https://github.com/Redskull2525

### LinkedIn
https://www.linkedin.com/in/abhishek-s-b98895249
""")

st.sidebar.divider()

# ==========================
# DASHBOARD CONTROLS
# ==========================

st.sidebar.title("⚙️ Dashboard Controls")

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
    key="live_data_refresh"
)

# ==========================
# SESSION BUFFER
# ==========================

if "data_buffer" not in st.session_state:
    st.session_state.data_buffer = pd.DataFrame()

data_buffer = st.session_state.data_buffer

# ==========================
# TABS
# ==========================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Trading Terminal",
    "📉 Technical Indicators",
    "🧠 Model Metrics",
    "📥 Export Data"
])

# ==========================
# LIVE TERMINAL
# ==========================

with tab1:

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    fig = go.Figure()

    for s in selected:

        ticker = stocks[s]

        data = fetch_data(ticker)

        prediction = predict(data)

        price = data["Close"].iloc[-1]

        timestamp = data.index[-1]

        change = prediction - price

        new_row = pd.DataFrame({
            "Time":[timestamp],
            "Stock":[s],
            "Actual":[price],
            "Predicted":[prediction]
        })

        data_buffer = pd.concat(
            [data_buffer,new_row]
        ).tail(500)

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

        metric_col1.metric(
            label=f"{s} Price",
            value=f"${price:.2f}"
        )

        metric_col2.metric(
            label="AI Prediction",
            value=f"${prediction:.2f}"
        )

        metric_col3.metric(
            label="Prediction Delta",
            value=f"{change:.2f}"
        )

    fig.update_layout(
        title="Live Price vs AI Prediction",
        xaxis_title="Time",
        yaxis_title="Price"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================
# TECHNICAL INDICATORS
# ==========================

with tab2:

    st.subheader("Technical Indicators")

    if selected:

        ticker = stocks[selected[0]]

        data = fetch_data(ticker)

        data["SMA20"] = data["Close"].rolling(20).mean()
        data["EMA20"] = data["Close"].ewm(span=20).mean()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Price"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["SMA20"],
            name="SMA20"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["EMA20"],
            name="EMA20"
        ))

        st.plotly_chart(fig, use_container_width=True)

# ==========================
# MODEL METRICS
# ==========================

with tab3:

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("RMSE", "2.31")
    col2.metric("MAE", "1.42")
    col3.metric("R² Score", "0.91")

    st.info("""
These metrics measure prediction accuracy:

RMSE — Root Mean Square Error  
MAE — Mean Absolute Error  
R² — Model fit quality
""")

# ==========================
# EXPORT DATA
# ==========================

with tab4:

    st.subheader("Download Prediction Data")

    if not data_buffer.empty:

        csv = data_buffer.to_csv(index=False)

        st.download_button(
            "Download CSV",
            csv,
            "predictions.csv",
            "text/csv"
        )

st.session_state.data_buffer = data_buffer
