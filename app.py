# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="NEM Price Predictor", layout="wide")

st.title("⚡ NEM Price Predictor (Weather-Based)")
st.markdown("""
Upload your weather data and compare predicted electricity prices against actual NEM forecasts.
""")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload Weather Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display raw data
    st.subheader("🔍 Raw Weather Data")
    st.dataframe(df.head())

    # --- Dummy Prediction Logic ---
    # Make sure your CSV has 'temperature' and 'actual_nem_price' columns
    if "temperature" in df.columns and "actual_nem_price" in df.columns:
        # Example model: price = 10 * temperature + noise
        df['predicted_price'] = df['temperature'] * 10 + 5  # Dummy logic
        
        st.subheader("📈 Predicted vs Actual NEM Prices")

        # Plotting the results
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['predicted_price'], label='Predicted Price', color='green')
        ax.plot(df['actual_nem_price'], label='Actual NEM Price', color='orange')
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Price ($/MWh)")
        ax.set_title("Forecast Comparison")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📊 Prediction Error (MAE)")
        mae = abs(df['predicted_price'] - df['actual_nem_price']).mean()
        st.metric(label="Mean Absolute Error", value=f"${mae:.2f}")
    else:
        st.warning("CSV must contain 'temperature' and 'actual_nem_price' columns.")
else:
    st.info("Please upload a CSV file with weather and NEM price data.")

