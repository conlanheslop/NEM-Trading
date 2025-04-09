import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from matplotlib.colors import LinearSegmentedColormap

# Set page config with custom theme
st.set_page_config(
    page_title="NEM EXTREME Trading Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the flashy trading look
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: #FFD700;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    h1, h2, h3 {
        color: #FFD700 !important;
        text-shadow: 0 0 10px #FF4500, 0 0 20px #FF4500;
        font-weight: bold !important;
    }
    .warning-banner {
        background-color: #FFFF00;
        color: #FF0000;
        padding: 10px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
        border-radius: 5px;
        animation: blinker 1s linear infinite;
    }
    .trade-card {
        background-color: #3D1F1F;
        border: 2px dashed #FF0000;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .trade-suggestion {
        background-color: #2F2F00;
        color: #FFFF00;
        padding: 10px;
        text-align: center;
        font-weight: bold;
        border: 1px solid #FFFF00;
        margin-bottom: 10px;
    }
    .profit-loss {
        background-color: #2F2F00;
        color: #FFFF00;
        padding: 10px;
        text-align: center;
        font-weight: bold;
        border: 1px solid #FFFF00;
    }
    .table-header {
        color: #FFFF00;
        background-color: #8B0000;
        text-align: center !important;
        font-weight: bold;
    }
    .profit {
        color: #00FF00 !important;
    }
    .loss {
        color: #FF0000 !important;
    }
    @keyframes blinker {
        50% {
            opacity: 0.8;
        }
    }
    .stDataFrame {
        background-color: #2D2D2D;
    }
    .css-1ht1j8u {
        overflow-x: auto;
    }
    .css-1ht1j8u table {
        width: 100%;
    }
    /* Customize the file uploader */
    .css-1cpxqw2 {
        background-color: #2D2D2D;
        border: 1px solid #FFD700;
        color: #FFD700;
    }
    /* Buttons */
    .stButton>button {
        background-color: #FF4500;
        color: white;
        font-weight: bold;
        border: 2px solid #FFD700;
    }
    .stButton>button:hover {
        background-color: #FFD700;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Warning banner
st.markdown('<div class="warning-banner">⚠️ EXTREME VOLATILITY DETECTED! TRADE AT YOUR OWN RISK! ⚠️</div>', unsafe_allow_html=True)

# Main title with flame emoji
st.markdown('<h1 style="text-align: center; font-size: 3rem;">🔥 EXTREME TRADING DASHBOARD 🔥</h1>', unsafe_allow_html=True)

# Create a 2-column layout for chart and trading recommendations
col1, col2 = st.columns([7, 3])

# Create a placeholder for charts
with col1:
    chart_placeholder = st.empty()

# Trading recommendations panel
with col2:
    st.markdown('<div class="trade-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">⚠️ TRADE 1 ⚠️</h3>', unsafe_allow_html=True)
    st.markdown('<div class="trade-suggestion">Suggestion: BUY NOW!!!</div>', unsafe_allow_html=True)
    st.markdown('<div class="profit-loss">Profit/Loss: $0</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="trade-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">⚠️ TRADE 2 ⚠️</h3>', unsafe_allow_html=True)
    st.markdown('<div class="trade-suggestion">Suggestion: SELL EVERYTHING!!!</div>', unsafe_allow_html=True)
    st.markdown('<div class="profit-loss">Profit/Loss: $0</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# File uploader for weather data
uploaded_file = st.file_uploader("📁 Upload Weather Dataset (CSV)", type=["csv"])

# Previous trades section
st.markdown('<h2 style="text-align: left;">🔥 PREVIOUS TRADES 🔥</h2>', unsafe_allow_html=True)

# Sample previous trades data
previous_trades = pd.DataFrame({
    'Date/Time': ['2025-04-01 10:00', '2025-03-28 14:30', '2025-03-25 09:15'],
    'Market Price': ['$120', '$115', '$110'],
    'Amount': [50, 30, 40],
    'Profit/Loss': ['+$500', '-$150', '+$400']
})

# Custom table styling with HTML
def highlight_profit_loss(val):
    if val.startswith('+'):
        return f'<span class="profit">{val}</span>'
    elif val.startswith('-'):
        return f'<span class="loss">{val}</span>'
    return val

# Display the previous trades table with custom styling
html_table = '<table style="width:100%; border-collapse: collapse;">'
html_table += '<tr class="table-header">'
for col in previous_trades.columns:
    html_table += f'<th style="padding: 10px; border: 1px solid #FFFF00;">{col}</th>'
html_table += '</tr>'

row_colors = ['#8B0000', '#2D2D2D']  # Alternating row colors
for idx, row in previous_trades.iterrows():
    bg_color = row_colors[idx % 2]
    html_table += f'<tr style="background-color: {bg_color}; text-align: center;">'
    for i, cell in enumerate(row):
        if i == 3:  # Profit/Loss column
            cell_text = highlight_profit_loss(cell)
        else:
            cell_text = cell
        html_table += f'<td style="padding: 10px; border: 1px solid #444444; color: #FFFF00;">{cell_text}</td>'
    html_table += '</tr>'
html_table += '</table>'

st.markdown(html_table, unsafe_allow_html=True)

# Model prediction section
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Display raw data with a flashy title
        st.markdown('<h2>🔍 UPLOADED MARKET DATA</h2>', unsafe_allow_html=True)
        st.dataframe(df.head())
        
        # Transform data for visualization if necessary
        if "temperature" in df.columns:
            # Generate synthetic predictions if actual_nem_price exists
            if "actual_nem_price" in df.columns:
                # Create a more volatile prediction
                volatility = np.random.normal(0, 15, len(df))
                df['predicted_price'] = df['temperature'] * 10 + volatility
            else:
                # Create synthetic data for both
                df['predicted_price'] = df['temperature'] * 10 + np.random.normal(0, 15, len(df))
                df['actual_nem_price'] = df['predicted_price'] + np.random.normal(0, 20, len(df))
            
            # Create a more dramatic visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#1E1E1E')
            ax.set_facecolor('#2D2D2D')
            
            # Plot with neon colors
            ax.plot(df['predicted_price'], label='AI PREDICTION', color='#00FF00', linewidth=2)
            ax.plot(df['actual_nem_price'], label='ACTUAL NEM PRICE', color='#FF4500', linewidth=2)
            
            # Add grid for readability
            ax.grid(True, linestyle='--', alpha=0.7, color='#444444')
            
            # Style the chart
            ax.set_title('EXTREME PRICE FORECAST COMPARISON', color='#FFD700', fontsize=16)
            ax.set_xlabel('Time Index', color='#FFD700')
            ax.set_ylabel('Price ($/MWh)', color='#FFD700')
            ax.tick_params(colors='#FFD700')
            ax.spines['bottom'].set_color('#FFD700')
            ax.spines['top'].set_color('#FFD700') 
            ax.spines['right'].set_color('#FFD700')
            ax.spines['left'].set_color('#FFD700')
            
            # Add legend with custom styling
            legend = ax.legend()
            frame = legend.get_frame()
            frame.set_facecolor('#2D2D2D')
            frame.set_edgecolor('#FFD700')
            for text in legend.get_texts():
                text.set_color('#FFD700')
            
            # Add volatility markers
            for i in range(len(df)):
                if abs(df['predicted_price'].iloc[i] - df['actual_nem_price'].iloc[i]) > 30:
                    ax.axvspan(i-0.5, i+0.5, alpha=0.3, color='red')
            
            # Display the chart
            with chart_placeholder:
                st.pyplot(fig)
            
            # Calculate error metrics with flashy display
            st.markdown('<h2>📊 PREDICTION ACCURACY METRICS</h2>', unsafe_allow_html=True)
            
            # Calculate metrics
            mae = abs(df['predicted_price'] - df['actual_nem_price']).mean()
            max_error = abs(df['predicted_price'] - df['actual_nem_price']).max()
            
            # Create a two-column layout for metrics
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric(
                    label="MEAN ABSOLUTE ERROR",
                    value=f"${mae:.2f}",
                    delta=f"{'-' if mae > 15 else '+'}{abs(15 - mae):.2f}",
                    delta_color="inverse"
                )
            
            with metric_col2:
                st.metric(
                    label="MAXIMUM ERROR",
                    value=f"${max_error:.2f}",
                    delta=f"{'-' if max_error > 50 else '+'}{abs(50 - max_error):.2f}",
                    delta_color="inverse"
                )
            
            # Add some trading action buttons
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if st.button("🚀 EXECUTE TRADE NOW"):
                    st.success("TRADE EXECUTED SUCCESSFULLY!")
            
            with action_col2:
                if st.button("💰 CALCULATE MAX PROFIT"):
                    profit = np.max(np.abs(np.diff(df['actual_nem_price']))) * 10
                    st.success(f"MAXIMUM POTENTIAL PROFIT: ${profit:.2f}")
                    
        else:
            st.warning("⚠️ CSV must contain 'temperature' column for predictions!")
    except Exception as e:
        st.error(f"⚠️ ERROR PROCESSING FILE: {e}")
else:
    # Display a demo chart with random data when no file is uploaded
    with chart_placeholder:
        # Generate some random data for the demo
        x = np.linspace(0, 100, 100)
        y1 = 100 + 30 * np.sin(x/5) + np.random.normal(0, 10, 100)
        y2 = 100 + 30 * np.sin(x/5 + 1) + np.random.normal(0, 10, 100)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1E1E1E')
        ax.set_facecolor('#2D2D2D')
        
        ax.plot(x, y1, color='#00FF00', linewidth=2, label='AI PREDICTION')
        ax.plot(x, y2, color='#FF4500', linewidth=2, label='ACTUAL NEM PRICE')
        
        # Add grid and styling
        ax.grid(True, linestyle='--', alpha=0.7, color='#444444')
        ax.set_title('⚡ SAMPLE NEM PRICE PREDICTION ⚡', color='#FFD700', fontsize=16)
        ax.set_xlabel('Time Index', color='#FFD700')
        ax.set_ylabel('Price ($/MWh)', color='#FFD700')
        ax.tick_params(colors='#FFD700')
        
        # Style the chart borders
        for spine in ax.spines.values():
            spine.set_color('#FFD700')
        
        # Add legend with custom styling
        legend = ax.legend()
        frame = legend.get_frame()
        frame.set_facecolor('#2D2D2D')
        frame.set_edgecolor('#FFD700')
        for text in legend.get_texts():
            text.set_color('#FFD700')
            
        # Display high volatility regions
        for i in range(0, 100, 20):
            if i % 40 == 0:
                ax.axvspan(i, i+10, alpha=0.3, color='red')
        
        st.pyplot(fig)
    
    st.markdown("""
    <div style="background-color: #2D2D2D; padding: 20px; border-radius: 10px; border: 1px solid #FFD700;">
        <h3 style="color: #FFD700;">⚡ UPLOAD YOUR WEATHER DATA TO START EXTREME TRADING ⚡</h3>
        <p style="color: #FFD700;">Upload a CSV containing weather data with at least a 'temperature' column to see predictions.</p>
        <p style="color: #FFD700;">Optional: Include 'actual_nem_price' column to compare with predictions.</p>
    </div>
    """, unsafe_allow_html=True)

# Add a footer with last updated time
st.markdown(f"""
<div style="background-color: #2D2D2D; padding: 10px; position: fixed; bottom: 0; width: 100%; text-align: center; border-top: 1px solid #FFD700;">
    <p style="color: #FFD700; margin: 0;">LAST UPDATED: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | MARKET STATUS: <span style="color:#00FF00">OPEN</span></p>
</div>
""", unsafe_allow_html=True)