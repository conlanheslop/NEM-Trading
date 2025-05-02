import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from matplotlib.colors import LinearSegmentedColormap
import uuid

##################
### PAGE SETUP ###
##################

# Initialize session state variables if they don't exist
if 'total_balance' not in st.session_state:
    st.session_state.total_balance = 10000.0  # Starting available cash balance
if 'trading_balance' not in st.session_state:
    st.session_state.trading_balance = 0.0  # Amount invested in active trades
if 'previous_trades' not in st.session_state:
    st.session_state.previous_trades = pd.DataFrame({
        'Date/Time': ['2025-04-01 10:00', '2025-03-28 14:30', '2025-03-25 09:15'],
        'Market Price': ['$120', '$115', '$110'],
        'Amount': [50, 30, 40],
        'Profit/Loss': ['+$500', '-$150', '+$400'],
        'Status': ['Closed', 'Closed', 'Closed']  # New column to track open/closed trades
    })
if 'active_trades' not in st.session_state:
    st.session_state.active_trades = pd.DataFrame(columns=['Date/Time', 'Market Price', 'Amount', 'Current Value', 'Profit/Loss'])
if 'last_price' not in st.session_state:
    st.session_state.last_price = 120  # Default current price
if 'chart_data' not in st.session_state:
    # Generate initial random data for the demo chart
    x = np.linspace(0, 100, 100)
    y1 = 100 + 30 * np.sin(x/5) + np.random.normal(0, 10, 100)
    y2 = 100 + 30 * np.sin(x/5 + 1) + np.random.normal(0, 10, 100)
    st.session_state.chart_data = {
        'x': x,
        'y1': y1,
        'y2': y2,
        'last_update': datetime.datetime.now()
    }
    st.session_state.last_price = y2[-1]  # Set initial price from chart
if 'chart_id' not in st.session_state:
    st.session_state.chart_id = str(uuid.uuid4())

# Set page config with custom theme
st.set_page_config(
    page_title="NEM EXTREME Trading Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load external CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
#
#
#
###############################################################
# This whole section is just utility functions for setting up #
#           the processes of the application                  #
###############################################################
#
#
#

########################
### UPDATE FUNCTIONS ###
########################

# Function to update chart data every 30 seconds
def update_chart_data():
    current_time = datetime.datetime.now()
    last_update = st.session_state.chart_data['last_update']
    time_diff = (current_time - last_update).total_seconds()
    
    # Update chart data if 30 seconds have passed
    if time_diff >= 30:
        # Generate new data points
        x = st.session_state.chart_data['x']
        
        # Generate new points with some correlation to previous data
        last_y1 = st.session_state.chart_data['y1'][-1]
        last_y2 = st.session_state.chart_data['y2'][-1]
        
        # Create new data with random movement but preserving some trend
        new_y1 = last_y1 + np.random.normal(0, 5)
        new_y2 = last_y2 + np.random.normal(0, 8)
        
        # Ensure prices don't go below 50
        new_y1 = max(50, new_y1)
        new_y2 = max(50, new_y2)
        
        # Update the arrays - remove first element and add new one
        y1 = np.append(st.session_state.chart_data['y1'][1:], new_y1)
        y2 = np.append(st.session_state.chart_data['y2'][1:], new_y2)
        
        # Update session state
        st.session_state.chart_data = {
            'x': x,
            'y1': y1,
            'y2': y2,
            'last_update': current_time
        }
        
        # Update last price
        st.session_state.last_price = new_y2
        
        # Generate new chart ID to force refresh
        st.session_state.chart_id = str(uuid.uuid4())
        
        # Update active trades values based on new price
        update_active_trades(new_y2)
        
        return True
    
    return False

# Function to update active trades values
def update_active_trades(current_price):
    if len(st.session_state.active_trades) > 0:
        # Calculate current values and profit/loss for each active trade
        for i, trade in st.session_state.active_trades.iterrows():
            purchase_price = float(trade['Market Price'].replace('$', ''))
            amount = trade['Amount']
            current_value = amount * current_price
            profit_loss = (current_price - purchase_price) * amount
            
            # Update the DataFrame
            st.session_state.active_trades.at[i, 'Current Value'] = f"${current_value:.2f}"
            if profit_loss >= 0:
                st.session_state.active_trades.at[i, 'Profit/Loss'] = f"+${profit_loss:.2f}"
            else:
                st.session_state.active_trades.at[i, 'Profit/Loss'] = f"-${abs(profit_loss):.2f}"
        
        # Update total trading balance
        st.session_state.trading_balance = sum([amount * current_price for amount in st.session_state.active_trades['Amount']])


#######################
### TRADE EXECUTION ###
#######################

# Function to execute a trade
def execute_trade(action, amount, price):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    trade_value = amount * price
    
    if action == "BUY":
        # Check if user has enough available balance
        if trade_value > st.session_state.total_balance:
            return False, "INSUFFICIENT FUNDS FOR PURCHASE!"
        
        # Deduct from available balance
        st.session_state.total_balance -= trade_value
        
        # Add to trading balance
        st.session_state.trading_balance += trade_value
        
        # Add to active trades
        new_trade = pd.DataFrame({
            'Date/Time': [current_time],
            'Market Price': [f"${price:.2f}"],
            'Amount': [amount],
            'Current Value': [f"${trade_value:.2f}"],
            'Profit/Loss': ["$0.00"]
        })
        
        st.session_state.active_trades = pd.concat([new_trade, st.session_state.active_trades]).reset_index(drop=True)
        
        return True, f"BUY EXECUTED SUCCESSFULLY! ${trade_value:.2f} INVESTED"
    
    elif action == "SELL":
        # Check if there are active trades to sell
        if len(st.session_state.active_trades) == 0:
            return False, "NO ACTIVE TRADES TO SELL!"
        
        # Check if we have enough shares to sell
        total_shares = st.session_state.active_trades['Amount'].sum()
        if amount > total_shares:
            return False, f"INSUFFICIENT SHARES! YOU ONLY HAVE {total_shares} AVAILABLE"
        
        # Calculate the shares to sell from each position (FIFO - First In, First Out)
        shares_to_sell = amount
        trades_to_remove = []
        
        # Create a copy to avoid modifying during iteration
        active_trades_copy = st.session_state.active_trades.copy()
        
        for i, trade in active_trades_copy.iterrows():
            trade_shares = trade['Amount']
            purchase_price = float(trade['Market Price'].replace('$', ''))
            
            if shares_to_sell <= 0:
                break
                
            if shares_to_sell >= trade_shares:
                # Sell entire position
                sold_shares = trade_shares
                trades_to_remove.append(i)
            else:
                # Sell partial position
                sold_shares = shares_to_sell
                st.session_state.active_trades.at[i, 'Amount'] = trade_shares - sold_shares
                
                # Update current value
                new_value = (trade_shares - sold_shares) * price
                st.session_state.active_trades.at[i, 'Current Value'] = f"${new_value:.2f}"
            
            # Calculate profit/loss for this sale
            profit_loss = (price - purchase_price) * sold_shares
            
            # Add to previous trades history
            new_trade = pd.DataFrame({
                'Date/Time': [current_time],
                'Market Price': [f"${price:.2f}"],
                'Amount': [sold_shares],
                'Profit/Loss': [f"+${profit_loss:.2f}" if profit_loss >= 0 else f"-${abs(profit_loss):.2f}"],
                'Status': ['Closed']
            })
            
            st.session_state.previous_trades = pd.concat([new_trade, st.session_state.previous_trades]).reset_index(drop=True)
            
            # Add profit/loss to total balance
            st.session_state.total_balance += (sold_shares * price)
            
            # Reduce remaining shares to sell
            shares_to_sell -= sold_shares
        
        # Remove completely sold positions
        st.session_state.active_trades = st.session_state.active_trades.drop(trades_to_remove).reset_index(drop=True)
        
        # Update trading balance
        st.session_state.trading_balance = sum([amount * price for amount in st.session_state.active_trades['Amount']])
        
        return True, f"SELL EXECUTED SUCCESSFULLY! ${amount * price:.2f} RETURNED TO BALANCE"

# Update chart data - check if it's time to update
chart_updated = update_chart_data()

# Calculate seconds until next update
current_time = datetime.datetime.now()
last_update = st.session_state.chart_data['last_update']
seconds_until_update = max(0, 30 - (current_time - last_update).total_seconds())

#
#
#
#############################################################
# This section is the start of styling and drawing the page #
#############################################################
#
#
#

# Warning banner
st.markdown('<div class="warning-banner">⚠️ EXTREME VOLATILITY DETECTED! TRADE AT YOUR OWN RISK! ⚠️</div>', unsafe_allow_html=True)

# Main title with flame emoji
st.markdown('<h1 style="text-align: center; font-size: 3rem;">🔥 EXTREME TRADING DASHBOARD 🔥</h1>', unsafe_allow_html=True)

# Display total balance and trading balance in two columns
col_balance1, col_balance2 = st.columns(2)
with col_balance1:
    st.markdown(f'<div class="balance-card">AVAILABLE CASH: ${st.session_state.total_balance:.2f}</div>', unsafe_allow_html=True)
with col_balance2:
    st.markdown(f'<div class="balance-card">INVESTED IN TRADES: ${st.session_state.trading_balance:.2f}</div>', unsafe_allow_html=True)

# Create a 2-column layout for chart and trading recommendations
col1, col2 = st.columns([7, 3])

# Create a placeholder for charts
with col1:
    # Display current market price from chart data
    current_price = st.session_state.last_price
    st.markdown(f'<div class="current-price">CURRENT MARKET PRICE: ${current_price:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="update-timer">Chart updates in {int(seconds_until_update)} seconds</div>', unsafe_allow_html=True)
    
    chart_placeholder = st.empty()

#####################
### TRADING PANEL ###
#####################

# Trading recommendations panel with buy/sell functionality
with col2:

    ###############
    ### TRADE 1 ###
    ###############

    st.markdown('<div class="trade-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">⚠️ TRADE 1 ⚠️</h3>', unsafe_allow_html=True)
    
    # Check if price is trending up or down for trade suggestion
    price_trend = "BUY NOW!!!" if st.session_state.chart_data['y2'][-1] > st.session_state.chart_data['y2'][-5] else "SELL NOW!!!"
    st.markdown(f'<div class="trade-suggestion">Suggestion: {price_trend}</div>', unsafe_allow_html=True)
    
    # Current market price for Trade 1 (from chart)
    trade1_price = current_price
    st.markdown(f'<div class="profit-loss">Market Price: ${trade1_price:.2f}</div>', unsafe_allow_html=True)
    
    # Amount to buy/sell
    trade1_amount = st.number_input("Amount:", min_value=1, value=10, key="trade1_amount")
    
    # Calculate cost
    trade1_cost = trade1_price * trade1_amount
    st.markdown(f'<div class="profit-loss">Total Cost: ${trade1_cost:.2f}</div>', unsafe_allow_html=True)
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        if st.button("BUY", key="buy1"):
            success, message = execute_trade("BUY", trade1_amount, trade1_price)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    with col1_2:
        if st.button("SELL", key="sell1"):
            success, message = execute_trade("SELL", trade1_amount, trade1_price)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    st.markdown('</div>', unsafe_allow_html=True)
    

    ###############
    ### TRADE 2 ### 
    ###############

    # st.markdown('<div class="trade-card">', unsafe_allow_html=True)
    # st.markdown('<h3 style="text-align: center;">⚠️ TRADE 2 ⚠️</h3>', unsafe_allow_html=True)

    # Opposite recommendation for Trade 2
    # price_trend2 = "SELL EVERYTHING!!!" if price_trend == "BUY NOW!!!" else "BUY THE DIP!!!"
    # st.markdown(f'<div class="trade-suggestion">Suggestion: {price_trend2}</div>', unsafe_allow_html=True)
    
    # # Current market price for Trade 2 (from chart)
    # trade2_price = current_price
    # st.markdown(f'<div class="profit-loss">Market Price: ${trade2_price:.2f}</div>', unsafe_allow_html=True)
    
    # # Amount to buy/sell
    # trade2_amount = st.number_input("Amount:", min_value=1, value=50, key="trade2_amount")
    
    # # Calculate cost
    # trade2_cost = trade2_price * trade2_amount
    # st.markdown(f'<div class="profit-loss">Total Cost: ${trade2_cost:.2f}</div>', unsafe_allow_html=True)
    
    # col2_1, col2_2 = st.columns(2)
    # with col2_1:
    #     if st.button("BUY", key="buy2"):
    #         success, message = execute_trade("BUY", trade2_amount, trade2_price)
    #         if success:
    #             st.success(message)
    #         else:
    #             st.error(message)
    
    # with col2_2:
    #     if st.button("SELL", key="sell2"):
    #         success, message = execute_trade("SELL", trade2_amount, trade2_price)
    #         if success:
    #             st.success(message)
    #         else:
    #             st.error(message)
    
    # st.markdown('</div>', unsafe_allow_html=True)

#########################
### ACTIVE TRADES BAR ###
#########################

# Active trades section
if len(st.session_state.active_trades) > 0:
    st.markdown('<h2 class="active-trades-title">🔥 ACTIVE TRADES 🔥</h2>', unsafe_allow_html=True)

    def highlight_profit_loss(val):
        if val.startswith('+'):
            return f'<span class="profit">{val}</span>'
        elif val.startswith('-'):
            return f'<span class="loss">{val}</span>'
        return val

    html_table = '<table class="custom-table">'
    html_table += '<tr class="table-header">'
    for col in st.session_state.active_trades.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr>'

    row_classes = ['row-dark', 'row-darker']
    for idx, row in st.session_state.active_trades.iterrows():
        row_class = row_classes[idx % 2]
        html_table += f'<tr class="{row_class}">'
        for i, cell in enumerate(row):
            cell_text = highlight_profit_loss(str(cell)) if i == 4 else cell
            html_table += f'<td>{cell_text}</td>'
        html_table += '</tr>'
    html_table += '</table>'

    st.markdown(html_table, unsafe_allow_html=True)

    if st.button("💰 SELL ALL ACTIVE TRADES"):
        total_amount = st.session_state.active_trades['Amount'].sum()
        success, message = execute_trade("SELL", total_amount, current_price)
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

# Refresh button
if st.button("🔄 REFRESH MARKET DATA"):
    st.session_state.chart_data['last_update'] = datetime.datetime.now() - datetime.timedelta(seconds=30)
    st.rerun()


######################
### CSV ATTACHMENT ###
######################

uploaded_file = st.file_uploader("📁 Upload Weather Dataset (CSV)", type=["csv"])


#######################
### PREVIOUS TRADES ###
#######################

st.markdown('<h2 class="active-trades-title">🔥 PREVIOUS TRADES 🔥</h2>', unsafe_allow_html=True)

def highlight_profit_loss(val):
    if val.startswith('+'):
        return f'<span class="profit">{val}</span>'
    elif val.startswith('-'):
        return f'<span class="loss">{val}</span>'
    return val

html_table = '<table class="custom-table">'
html_table += '<tr class="table-header">'
for col in st.session_state.previous_trades.columns:
    html_table += f'<th>{col}</th>'
html_table += '</tr>'

row_classes = ['row-dark', 'row-darker']
for idx, row in st.session_state.previous_trades.iterrows():
    row_class = row_classes[idx % 2]
    html_table += f'<tr class="{row_class}">'
    for i, cell in enumerate(row):
        cell_text = highlight_profit_loss(str(cell)) if i == 3 else cell
        html_table += f'<td>{cell_text}</td>'
    html_table += '</tr>'
html_table += '</table>'

st.markdown(html_table, unsafe_allow_html=True)

###########################################
### MODEL PREDICTION AFTER CSV ATTACHED ###
###########################################

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
            
            # Update the current price based on the latest actual price
            st.session_state.last_price = df['actual_nem_price'].iloc[-1]
            
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
                    # Use the latest price from the data
                    current_price = df['actual_nem_price'].iloc[-1]
                    trade_amount = 10  # Default amount
                    success, message = execute_trade("BUY", trade_amount, current_price)
                    if success:
                        st.success(f"TRADE EXECUTED SUCCESSFULLY! BOUGHT {trade_amount} AT ${current_price:.2f}")
                    else:
                        st.error(message)
            
            with action_col2:
                if st.button("💰 CALCULATE MAX PROFIT"):
                    profit = np.max(np.abs(np.diff(df['actual_nem_price']))) * 10
                    st.success(f"MAXIMUM POTENTIAL PROFIT: ${profit:.2f}")
                    
        else:
            st.warning("⚠️ CSV must contain 'temperature' column for predictions!")
    except Exception as e:
        st.error(f"⚠️ ERROR PROCESSING FILE: {e}")
else:
    # Display a demo chart with auto-updating random data 
    with chart_placeholder:
        # Use the data from session state
        x = st.session_state.chart_data['x']
        y1 = st.session_state.chart_data['y1']
        y2 = st.session_state.chart_data['y2']
        
        # Create the figure with a unique key to force refresh
        fig, ax = plt.subplots(figsize=(10, 6), num=st.session_state.chart_id)
        fig.patch.set_facecolor('#1E1E1E')
        ax.set_facecolor('#2D2D2D')
        
        ax.plot(x, y1, color='#00FF00', linewidth=2, label='AI PREDICTION')
        ax.plot(x, y2, color='#FF4500', linewidth=2, label='ACTUAL NEM PRICE')
        
        # Add grid and styling
        ax.grid(True, linestyle='--', alpha=0.7, color='#444444')
        ax.set_title('⚡ LIVE NEM PRICE PREDICTION ⚡', color='#FFD700', fontsize=16)
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
        
        # Add current price marker
        ax.axhline(y=y2[-1], color='#FF00FF', linestyle='--', alpha=0.7)
        ax.text(0, y2[-1] + 2, f"CURRENT: ${y2[-1]:.2f}", color='#FF00FF')
        
        st.pyplot(fig)
    
    
    st.markdown("""
    <div class="extreme-card">
        <h3>⚡ UPLOAD YOUR WEATHER DATA TO START EXTREME TRADING ⚡</h3>
        <p>Upload a CSV containing weather data with at least a 'temperature' column to see predictions.</p>
        <p>Optional: Include 'actual_nem_price' column to compare with predictions.</p>
        <p class="italic">Meanwhile, trade with our auto-updating live market data!</p>
    </div>
    """, unsafe_allow_html=True)

#########################
### PORTFOLIO SECTION ###
#########################

# Add a summary of total portfolio value
total_portfolio = st.session_state.total_balance + st.session_state.trading_balance
st.markdown(f"""
<div class="portfolio-summary">
    <h2>PORTFOLIO SUMMARY</h2>
    <p>Available Cash: <span>${st.session_state.total_balance:.2f}</span></p>
    <p>Invested in Trades: <span>${st.session_state.trading_balance:.2f}</span></p>
    <p>TOTAL PORTFOLIO VALUE: <span>${total_portfolio:.2f}</span></p>
</div>
""", unsafe_allow_html=True)

# Add a footer with last updated time
st.markdown(f"""
<div class="footer-bar">
    <p>LAST UPDATED: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
    MARKET STATUS: <span>OPEN</span> |
    PORTFOLIO: <span>${total_portfolio:.2f}</span> |
    CURRENT PRICE: <span class="price">${current_price:.2f}</span></p>
</div>
""", unsafe_allow_html=True)

# Persist session state by saving important variables
if 'first_load' not in st.session_state:
    st.session_state.first_load = False