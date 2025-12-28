import requests
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from config import TELEGRAM_TOKEN, CHAT_ID

# --- CONFIGURATION ---
CHART_PERIOD = "2y"  # Need more history for valid Wave Counts

# --- HELPER: MATH & INDICATORS ---
def calculate_indicators(df):
    """Calculates EMAs, RSI, MACD, and Bollinger Bands."""
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    return df

def get_pivots(df):
    """Finds significant Price Pivots (Highs/Lows)."""
    # Order=10 to find major structural points, not just noise
    df['pivot_high'] = df.iloc[argrelextrema(df['High'].values, np.greater, order=10)[0]]['High']
    df['pivot_low'] = df.iloc[argrelextrema(df['Low'].values, np.less, order=10)[0]]['Low']
    
    pivots = []
    for idx, row in df.iterrows():
        if not np.isnan(row['pivot_high']):
            pivots.append({'idx': idx, 'price': row['pivot_high'], 'type': 'high'})
        if not np.isnan(row['pivot_low']):
            pivots.append({'idx': idx, 'price': row['pivot_low'], 'type': 'low'})
    return pivots

def analyze_wave_setup(pivots, current_price):
    """
    Analyzes the last 4-5 pivots to detect Wave 3 or Wave 5 setups at breakout.
    Returns: (Wave_Status, Target_Price, Stop_Loss)
    """
    if len(pivots) < 5:
        return "Unclear Structure", 0, 0

    # Get last 4 pivots (most recent at end of list)
    p4 = pivots[-1] # Most recent pivot (Should be a Low if we are breaking out Up)
    p3 = pivots[-2] # High
    p2 = pivots[-3] # Low
    p1 = pivots[-4] # High
    
    # --- CHECK FOR WAVE 3 START ---
    # Setup: We finished Wave 2 (p4) and are breaking out above Wave 1 (p3)
    # Pattern: Low(Start) -> High(W1) -> Higher Low(W2) -> BREAKOUT
    if p4['type'] == 'low' and p3['type'] == 'high' and p2['type'] == 'low':
        
        # Rule 1: Wave 2 must NOT retrace 100% of Wave 1
        if p4['price'] > p2['price']: 
            # We have a Higher Low (Valid Wave 2)
            
            # Fibonacci Target for Wave 3 (1.618 extension of Wave 1)
            wave1_height = p3['price'] - p2['price']
            target = p4['price'] + (wave1_height * 1.618)
            stop_loss = p4['price'] # Stop below Wave 2 low
            
            return "🌊 Possible WAVE 3 Start", target, stop_loss

    # --- CHECK FOR WAVE 5 START ---
    # Setup: We finished Wave 4 (p4) and are breaking out above Wave 3 (p3)
    # Pattern: W1_High -> W2_Low -> W3_High -> W4_Low -> BREAKOUT
    if len(pivots) >= 6:
        p5_recent = pivots[-1] # Low (Wave 4)
        p4_high = pivots[-2]   # High (Wave 3)
        p3_low = pivots[-3]    # Low (Wave 2)
        p2_high = pivots[-4]   # High (Wave 1)
        p1_low = pivots[-5]    # Start

        if p5_recent['type'] == 'low' and p4_high['type'] == 'high':
            
            # Rule 1: Wave 4 must NOT overlap Wave 1
            if p5_recent['price'] > p2_high['price']:
                
                # Rule 2: Wave 3 is not the shortest
                w1_len = p2_high['price'] - p1_low['price']
                w3_len = p4_high['price'] - p3_low['price']
                
                if w3_len > w1_len: # Rough check, confirming W3 strength
                    # Target for Wave 5 (Usually equal to Wave 1 or 0.618 of W1+W3)
                    target = p5_recent['price'] + w1_len 
                    stop_loss = p5_recent['price']
                    
                    return "🌊 Possible WAVE 5 Start", target, stop_loss

    return "Correction / Noise", 0, 0

# --- MAIN FUNCTIONS ---
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": message})
    except Exception as e:
        logging.warning(f"⚠️ Telegram send failed: {e}")

def send_chart(symbol: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return

    try:
        # Fetch Data
        ticker = f"{symbol}" if ".NS" in symbol or "=" in symbol else f"{symbol}.NS"
        df = yf.download(ticker, period=CHART_PERIOD, interval="1d", progress=False)
        if df.empty: return
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # Process Data
        df = calculate_indicators(df)
        df_plot = df.reset_index()
        pivots = get_pivots(df)
        
        # --- ELLIOTT WAVE ANALYSIS ---
        current_price = df['Close'].iloc[-1]
        wave_msg, target, stop = analyze_wave_setup(pivots, current_price)

        # Setup Plot
        fig = plt.figure(figsize=(20, 14), facecolor='white') 
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        
        # PANEL 1: PRICE & WAVES
        ax1 = plt.subplot(gs[0])
        x_vals = df_plot.index 
        
        # Candles
        up = df_plot[df_plot.Close >= df_plot.Open]
        down = df_plot[df_plot.Close < df_plot.Open]
        ax1.bar(up.index, up.Close - up.Open, width=0.7, bottom=up.Open, color='#089981', alpha=1.0) 
        ax1.vlines(up.index, up.Low, up.High, color='#089981', linewidth=0.8)
        ax1.bar(down.index, down.Close - down.Open, width=0.7, bottom=down.Open, color='#F23645', alpha=1.0)
        ax1.vlines(down.index, down.Low, down.High, color='#F23645', linewidth=0.8)
        
        # Plot Pivots / Wave Points
        for p in pivots:
            # Match date to index
            match = df_plot[df_plot['Date'] == p['idx']]
            if not match.empty:
                x = match.index[0]
                color = 'green' if p['type'] == 'low' else 'red'
                ax1.plot(x, p['price'], marker='o', markersize=5, color=color)

        # Draw Target Line if Valid Wave
        if target > 0:
            ax1.axhline(target, color='purple', linestyle='--', linewidth=2, label=f'Target: {target:.2f}')
            ax1.axhline(stop, color='red', linestyle=':', linewidth=2, label=f'Invalidation: {stop:.2f}')
            
        # Title
        last_price = df['Close'].iloc[-1]
        pct_change = ((last_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        title_text = f"{symbol}: {wave_msg} | Price: {last_price:.2f}"
        ax1.set_title(title_text, fontsize=18, fontweight='bold', pad=15)
        ax1.grid(True, color='#f0f0f0')
        ax1.legend(loc='upper left')

        # PANEL 2: MACD (Momentum Confirmation)
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(x_vals, df_plot['MACD'], color='#2962FF', label='MACD')
        ax2.plot(x_vals, df_plot['Signal'], color='#FF9800', label='Signal')
        ax2.bar(x_vals, df_plot['Hist'], color=['#26a69a' if v >= 0 else '#ef5350' for v in df_plot['Hist']])
        ax2.set_ylabel("MACD", fontweight='bold')
        ax2.legend()

        # Formatting
        step = max(1, len(df_plot) // 12)
        date_labels = df_plot['Date'].dt.strftime('%b %d')
        ax2.set_xticks(x_vals[::step])
        ax2.set_xticklabels(date_labels[::step], rotation=0)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.04)

        # Send
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)

        caption = f"📊 {symbol} Analysis\nStatus: {wave_msg}\n🎯 Target: {target:.2f}\n🛑 Stop: {stop:.2f}"
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(url, files={'photo': buf}, data={'chat_id': CHAT_ID, 'caption': caption})
        logging.info(f"✅ Wave Chart sent for {symbol}")

    except Exception as e:
        logging.warning(f"❌ Failed to send chart for {symbol}: {e}", exc_info=True)
