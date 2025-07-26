import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import base64
import datetime
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Stock Price Prediction", layout="centered")

# Background setup
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

add_bg_from_local(r"C:\Users\kong1\OneDrive\Pictures\ai-generated-8477673.jpg")

# Title
st.title("📈 Stock Price Prediction")

# Model accuracy dictionary
model_accuracies = {
    "SBI": 0.89,
    "Axis Bank": 0.84,
    "HDFC": 0.87
}

# Load model function
def load_model(bank_name):
    model_paths = {
        "SBI": "sbi_model.pkl",
        "Axis Bank": "axis_model.pkl",
        "HDFC": "hdfc_model.pkl"
    }
    with open(model_paths[bank_name], "rb") as f:
        return pickle.load(f)

# Sidebar to select bank
bank_name = st.sidebar.selectbox("Select Bank", ["SBI", "Axis Bank", "HDFC"])
st.write(f"### Predicting Stock Price for {bank_name}")
st.info(f"🔍 Model Accuracy (R² Score): {model_accuracies[bank_name]*100:.2f}%")

# Input fields
open_price = st.number_input("Open Price", min_value=0.0, format="%.2f")
low_price = st.number_input("Low Price", min_value=0.0, format="%.2f")
price_change = st.number_input("Change %", min_value=-100.0, max_value=100.0, format="%.2f")
volume = st.number_input("Volume", min_value=0.0, format="%.0f")
prev_high = st.number_input("Previous Day High", min_value=0.0, format="%.2f")

selected_date = st.date_input("Select Date", value=datetime.date.today())
day = selected_date.day
month = selected_date.month
year = selected_date.year

# Load model
model = load_model(bank_name)

# Prepare feature input
if bank_name == "SBI":
    features = np.array([[open_price, low_price, open_price, volume, price_change, day, month, year, prev_high]])
elif bank_name == "Axis Bank":
    features = np.array([[open_price, low_price, price_change]])
elif bank_name == "HDFC":
    features = np.array([[open_price, low_price, prev_high]])

if st.button("Predict"):
    prediction = model.predict(features)
    st.success(f"Predicted High Price: ₹{prediction[0]:.2f}")

    # Create stock-style line chart with Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=["Open Price", "Low Price", "Previous High", "Predicted High"],
        y=[open_price, low_price, prev_high, prediction[0]],
        mode='lines+markers',
        name="Price Movement",
        line=dict(color="green", width=2),
        marker=dict(color="black", size=10)
    ))

    fig.update_layout(
        title=f"{bank_name} – Stock Price Line Chart",
        xaxis_title="Price Type",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        xaxis=dict(tickmode='linear'),
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)
