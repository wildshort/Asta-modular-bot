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
CHART_PERIOD = "2y"  # Need history for both trendlines and waves

# --- HELPER: INDICATORS ---
def calculate_indicators(df):
    """Calculates EMAs, RSI, MACD, BB."""
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # BB
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['Close'].rolling(window=20).std())
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['Close'].rolling(window=20).std())
    
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

# --- HELPER: GEOMETRY & STRUCTURE ---
def get_technical_structure(df):
    """ Finds Pivots, Trendlines, and Horizontal Levels based on integer index. """
    df = df.copy()
    df['idx'] = range(len(df)) # Integer index for geometry
    last_idx = df['idx'].iloc[-1]

    # 1. Find Pivots (Order=10 for major swings)
    df['pivot_high'] = df.iloc[argrelextrema(df['High'].values, np.greater, order=10)[0]]['High']
    df['pivot_low'] = df.iloc[argrelextrema(df['Low'].values, np.less, order=10)[0]]['Low']
    
    pivots = []
    pivot_highs_df = []
    pivot_lows_df = []

    for idx, row in df.iterrows():
        i = row['idx']
        if not np.isnan(row['pivot_high']):
            p = {'i': i, 'price': row['pivot_high'], 'type': 'high'}
            pivots.append(p)
            pivot_highs_df.append(p)
        if not np.isnan(row['pivot_low']):
            p = {'i': i, 'price': row['pivot_low'], 'type': 'low'}
            pivots.append(p)
            pivot_lows_df.append(p)

    # 2. Trendlines (Connecting last 2 major peaks/valleys)
    lines = []
    if len(pivot_highs_df) >= 2:
        p1, p2 = pivot_highs_df[-2], pivot_highs_df[-1]
        slope = (p2['price'] - p1['price']) / (p2['i'] - p1['i'])
        y_end = p2['price'] + slope * (last_idx - p2['i'])
        lines.append({'x': [p1['i'], last_idx], 'y': [p1['price'], y_end], 'color': 'red', 'label': 'Resist Line'})

    if len(pivot_lows_df) >= 2:
        p1, p2 = pivot_lows_df[-2], pivot_lows_df[-1]
        slope = (p2['price'] - p1['price']) / (p2['i'] - p1['i'])
        y_end = p2['price'] + slope * (last_idx - p2['i'])
        lines.append({'x': [p1['i'], last_idx], 'y': [p1['price'], y_end], 'color': 'green', 'label': 'Support Line'})

    # 3. Horizontal Levels (6-month High/Low)
    recent_df = df.tail(126) # Approx 6 months
    lines.append({'x': [df['idx'].iloc[0], last_idx], 'y': [recent_df['High'].max(), recent_df['High'].max()], 'color': 'red', 'style': ':', 'label': 'Major Res'})
    lines.append({'x': [df['idx'].iloc[0], last_idx], 'y': [recent_df['Low'].min(), recent_df['Low'].min()], 'color': 'green', 'style': ':', 'label': 'Major Sup'})

    return pivots, lines

# --- HELPER: ELLIOTT WAVE LOGIC ---
def analyze_wave_setup(pivots):
    """ Checks last 4-5 pivots for Wave 3 or 5 setup at current price. """
    if len(pivots) < 5: return "Watching...", 0, 0

    # Most recent pivots
    p4 = pivots[-1]; p3 = pivots[-2]; p2 = pivots[-3]; p1 = pivots[-4]
    
    # Wave 3 Setup: Finished W2 low (p4), breaking out above W1 high (p3)
    if p4['type'] == 'low' and p3['type'] == 'high' and p2['type'] == 'low':
        # Rule: W2 (p4) must be higher than Start (p2)
        if p4['price'] > p2['price']:
            wave1_height = p3['price'] - p2['price']
            target = p4['price'] + (wave1_height * 1.618) # 1.618 ext
            stop = p4['price'] # Below W2
            return "🌊 Possible WAVE 3 Start", target, stop

    # Wave 5 Setup: Finished W4 low (p5), breaking out above W3 high (p4)
    if len(pivots) >= 6:
        p5 = pivots[-1]; p4_h = pivots[-2]; p3_l = pivots[-3]; p2_h = pivots[-4]; p1_l = pivots[-5]
        if p5['type'] == 'low' and p4_h['type'] == 'high':
            # Rule: W4 (p5) no overlap W1 (p2_h) AND W3 not shortest
            w1_len = p2_h['price'] - p1_l['price']
            w3_len = p4_h['price'] - p3_l['price']
            if p5['price'] > p2_h['price'] and w3_len > w1_len:
                target = p5['price'] + w1_len # W5 = W1 often
                stop = p5['price']
                return "🌊 Possible WAVE 5 Start", target, stop

    return "Ranging / Correction", 0, 0

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

        # Process
        df = calculate_indicators(df)
        df_plot = df.reset_index()
        pivots, structure_lines = get_technical_structure(df)
        wave_msg, target, stop = analyze_wave_setup(pivots)
        
        # Setup Plot
        fig = plt.figure(figsize=(20, 16), facecolor='white') 
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 0.6, 1, 1])
        
        # --- PANEL 1: PRICE & STRUCTURE ---
        ax1 = plt.subplot(gs[0])
        x_vals = df_plot.index 
        
        # Candles
        up = df_plot[df_plot.Close >= df_plot.Open]; down = df_plot[df_plot.Close < df_plot.Open]
        ax1.bar(up.index, up.Close - up.Open, width=0.7, bottom=up.Open, color='#089981', alpha=1) 
        ax1.vlines(up.index, up.Low, up.High, color='#089981', linewidth=0.8)
        ax1.bar(down.index, down.Close - down.Open, width=0.7, bottom=down.Open, color='#F23645', alpha=1)
        ax1.vlines(down.index, down.Low, down.High, color='#F23645', linewidth=0.8)
        
        # Overlays (BB, EMA)
        ax1.fill_between(x_vals, df_plot['BB_Upper'], df_plot['BB_Lower'], color='gray', alpha=0.1)
        ax1.plot(x_vals, df_plot['EMA_50'], color='#FF9800', linewidth=1.5, label="EMA 50")

        # Draw Structure (Trendlines & Horizontal)
        for line in structure_lines:
            style = line.get('style', '-')
            ax1.plot(line['x'], line['y'], color=line['color'], linestyle=style, linewidth=1.5, label=line['label'])
            
        # Draw Pivots (Small dots for context)
        for p in pivots:
            color = 'green' if p['type'] == 'low' else 'red'
            ax1.plot(p['i'], p['price'], marker='o', markersize=4, color=color, alpha=0.6)

        # Draw Wave Target/Stop (if valid setup)
        if target > 0:
            ax1.axhline(target, color='#9C27B0', linestyle='--', linewidth=2, label=f'🎯 Target: {target:.2f}')
            ax1.axhline(stop, color='#F23645', linestyle=':', linewidth=2, label=f'🛑 Stop: {stop:.2f}')

        # Title & Grid
        last_price = df['Close'].iloc[-1]
        pct = ((last_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        title = f"{symbol} {'🟢' if pct>=0 else '🔴'} ₹{last_price:.2f} ({pct:+.2f}%) | {wave_msg}"
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax1.grid(True, color='#f0f0f0'); ax1.legend(loc='upper left', frameon=False, fontsize=9)

        # --- PANEL 2, 3, 4 (Vol, RSI, MACD) ---
        ax2 = plt.subplot(gs[1], sharex=ax1)
        vol_colors = ['#089981' if c >= o else '#F23645' for c, o in zip(df_plot['Close'], df_plot['Open'])]
        ax2.bar(x_vals, df_plot['Volume'], color=vol_colors, alpha=0.6); ax2.set_ylabel("Vol", fontweight='bold'); ax2.grid(True, color='#f0f0f0')

        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(x_vals, df_plot['RSI'], color='#7E57C2'); ax3.axhline(70, color='red', ls='--'); ax3.axhline(30, color='green', ls='--')
        ax3.set_ylabel("RSI", fontweight='bold'); ax3.grid(True, color='#f0f0f0'); ax3.set_ylim(0, 100)

        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.plot(x_vals, df_plot['MACD'], color='#2962FF'); ax4.plot(x_vals, df_plot['Signal'], color='#FF9800')
        ax4.bar(x_vals, df_plot['Hist'], color=['#26a69a' if v >= 0 else '#ef5350' for v in df_plot['Hist']], alpha=0.8)
        ax4.set_ylabel("MACD", fontweight='bold'); ax4.grid(True, color='#f0f0f0')

        # Formatting
        step = max(1, len(df_plot) // 12)
        ax4.set_xticks(x_vals[::step]); ax4.set_xticklabels(df_plot['Date'].dt.strftime('%b %d')[::step], rotation=0)
        plt.setp([ax.get_xticklabels() for ax in [ax1, ax2, ax3]], visible=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.04)

        # Send
        buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', dpi=150); buf.seek(0); plt.close(fig)
        caption = f"📊 {symbol} Analysis\nStatus: {wave_msg}\nTarget: {target:.2f} | Stop: {stop:.2f}"
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", files={'photo': buf}, data={'chat_id': CHAT_ID, 'caption': caption})
        logging.info(f"✅ Full Analysis Chart sent for {symbol}")

    except Exception as e:
        logging.warning(f"❌ Failed to send chart for {symbol}: {e}", exc_info=True)
