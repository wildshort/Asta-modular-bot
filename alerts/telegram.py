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
CHART_PERIOD = "1y" # 1 Year is best for Elliott Wave context

# --- HELPER: MATH & INDICATORS ---
def calculate_indicators(df):
    """Calculates EMA, RSI, MACD, and Bollinger Bands."""
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # BB
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

def find_elliott_waves(df):
    """
    Algorithmic Elliott Wave Detection (Impulse 1-2-3-4-5).
    Returns list of points {'idx', 'price', 'label', 'color'}.
    """
    # 1. Find Pivots (Highs and Lows)
    # Order=5 means a local top/bottom must be higher/lower than 5 candles on each side
    df['pivot_high'] = df.iloc[argrelextrema(df['High'].values, np.greater, order=5)[0]]['High']
    df['pivot_low'] = df.iloc[argrelextrema(df['Low'].values, np.less, order=5)[0]]['Low']
    
    pivots = []
    # Combine into a single sorted list of (Index, Price, Type)
    for idx, row in df.iterrows():
        if not np.isnan(row['pivot_high']):
            pivots.append({'idx': idx, 'price': row['pivot_high'], 'type': 'high', 'i': df.index.get_loc(idx)})
        if not np.isnan(row['pivot_low']):
            pivots.append({'idx': idx, 'price': row['pivot_low'], 'type': 'low', 'i': df.index.get_loc(idx)})
            
    # 2. Iterate to find valid 5-wave sequence
    # Pattern: Low(Start) -> High(1) -> Low(2) -> High(3) -> Low(4) -> High(5)
    waves = []
    
    # We look at the last 10 pivots to find a recent pattern
    if len(pivots) < 6: return []
    
    # Simple Brute Force check of recent pivots for the "Perfect Impulse"
    # We iterate backwards looking for a valid sequence
    for k in range(len(pivots)-1, 4, -1):
        p5 = pivots[k]
        if p5['type'] != 'high': continue # Wave 5 ends on High
        
        # Candidate points (working backwards)
        p4 = pivots[k-1] # Low
        p3 = pivots[k-2] # High
        p2 = pivots[k-3] # Low
        p1 = pivots[k-4] # High
        p0 = pivots[k-5] # Low (Start of Wave 1)
        
        if not (p4['type'] == 'low' and p3['type'] == 'high' and p2['type'] == 'low' and p1['type'] == 'high'):
            continue

        # --- CHECK ELLIOTT RULES ---
        # Rule 1: Wave 2 cannot retrace 100% of Wave 1
        # i.e., Low of Wave 2 must be > Low of Start
        if p2['price'] <= p0['price']: continue
        
        # Rule 2: Wave 3 is not the shortest wave
        len_w1 = p1['price'] - p0['price']
        len_w3 = p3['price'] - p2['price']
        len_w5 = p5['price'] - p4['price']
        if len_w3 < len_w1 and len_w3 < len_w5: continue
        
        # Rule 3: Wave 4 does not overlap Wave 1
        # i.e., Low of Wave 4 must be > High of Wave 1
        if p4['price'] <= p1['price']: continue
        
        # Rule 4 (Basic Definition): Wave 3 must go higher than Wave 1
        if p3['price'] <= p1['price']: continue
        
        # Rule 5 (Basic Definition): Wave 5 must go higher than Wave 3
        if p5['price'] <= p3['price']: continue
        
        # If we pass all rules, we found a valid count!
        waves = [
            {'idx': p1['idx'], 'price': p1['price'], 'label': '❶', 'color': 'blue'},
            {'idx': p2['idx'], 'price': p2['price'], 'label': '❷', 'color': 'orange'},
            {'idx': p3['idx'], 'price': p3['price'], 'label': '❸', 'color': 'blue'},
            {'idx': p4['idx'], 'price': p4['price'], 'label': '❹', 'color': 'orange'},
            {'idx': p5['idx'], 'price': p5['price'], 'label': '❺', 'color': 'purple'},
        ]
        break # Stop after finding the most recent valid pattern
        
    return waves

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
        
        # Get Elliott Waves
        elliott_waves = find_elliott_waves(df)

        # Setup Plot
        fig = plt.figure(figsize=(20, 14), facecolor='white') 
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 0.6, 1, 1])
        
        # --- PANEL 1: PRICE & WAVES ---
        ax1 = plt.subplot(gs[0])
        x_vals = df_plot.index 
        
        # Candles
        up = df_plot[df_plot.Close >= df_plot.Open]
        down = df_plot[df_plot.Close < df_plot.Open]
        ax1.bar(up.index, up.Close - up.Open, width=0.7, bottom=up.Open, color='#089981', alpha=1.0) 
        ax1.vlines(up.index, up.Low, up.High, color='#089981', linewidth=0.8)
        ax1.bar(down.index, down.Close - down.Open, width=0.7, bottom=down.Open, color='#F23645', alpha=1.0)
        ax1.vlines(down.index, down.Low, down.High, color='#F23645', linewidth=0.8)
        
        # EMAs
        ax1.plot(x_vals, df_plot['EMA_5'], color='#2962FF', linewidth=1.2, alpha=0.9, label="EMA 5")
        ax1.plot(x_vals, df_plot['EMA_50'], color='#FF9800', linewidth=1.2, alpha=0.9, label="EMA 50")

        # Plot Elliott Waves
        if elliott_waves:
            # Need to map dates back to integer index
            for w in elliott_waves:
                # Find the integer index for the date
                match = df_plot[df_plot['Date'] == w['idx']]
                if not match.empty:
                    x_idx = match.index[0]
                    # Draw Marker and Label
                    ax1.plot(x_idx, w['price'], marker='o', markersize=8, color=w['color'], markeredgecolor='white', markeredgewidth=1.5, zorder=5)
                    ax1.text(x_idx, w['price']*1.01, w['label'], fontsize=14, fontweight='bold', color=w['color'], ha='center', zorder=6)
            
            # Connect the points with a line to show the wave structure
            w_indices = [df_plot[df_plot['Date'] == w['idx']].index[0] for w in elliott_waves]
            w_prices = [w['price'] for w in elliott_waves]
            ax1.plot(w_indices, w_prices, linestyle='--', color='gray', linewidth=1, alpha=0.7)

        # Title
        last_price = df['Close'].iloc[-1]
        pct_change = ((last_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        color_arrow = "🟢" if pct_change >= 0 else "🔴"
        ax1.set_title(f"{symbol} {color_arrow} ₹{last_price:.2f} ({pct_change:+.2f}%)", fontsize=18, fontweight='bold', pad=15)
        ax1.grid(True, color='#f0f0f0')
        ax1.legend(loc='upper left', frameon=False)

        # --- PANELS 2, 3, 4 (Volume, RSI, MACD) ---
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.bar(x_vals, df_plot['Volume'], color='gray', alpha=0.6)
        ax2.set_ylabel("Vol", fontweight='bold')
        
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(x_vals, df_plot['RSI'], color='#7E57C2')
        ax3.axhline(70, color='#F23645', linestyle='--'); ax3.axhline(30, color='#089981', linestyle='--')
        ax3.set_ylabel("RSI", fontweight='bold')

        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.plot(x_vals, df_plot['MACD'], color='#2962FF'); ax4.plot(x_vals, df_plot['Signal'], color='#FF9800')
        ax4.bar(x_vals, df_plot['Hist'], color=['#26a69a' if v >= 0 else '#ef5350' for v in df_plot['Hist']])
        ax4.set_ylabel("MACD", fontweight='bold')

        # Formatting
        step = max(1, len(df_plot) // 12)
        date_labels = df_plot['Date'].dt.strftime('%b %d')
        ax4.set_xticks(x_vals[::step])
        ax4.set_xticklabels(date_labels[::step], rotation=0)
        plt.setp([ax1.get_xticklabels(), ax2.get_xticklabels(), ax3.get_xticklabels()], visible=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.04)

        # Send
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(url, files={'photo': buf}, data={'chat_id': CHAT_ID, 'caption': f"📊 {symbol} Elliott Wave Analysis"})
        logging.info(f"✅ Elliott Chart sent for {symbol}")

    except Exception as e:
        logging.warning(f"❌ Failed to send chart for {symbol}: {e}", exc_info=True)
