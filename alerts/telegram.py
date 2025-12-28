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
CHART_PERIOD = "6mo" 

# --- HELPER: MATH & INDICATORS ---

def calculate_indicators(df):
    """Calculates EMA, RSI, MACD, and Bollinger Bands."""
    # EMA
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

def find_divergence(df, order=5):
    """
    Detects Regular and Hidden Divergence between Price and RSI.
    Returns a list of lines to draw on the chart.
    """
    # 1. Find Peaks (Highs) and Troughs (Lows)
    # 'order' defines how many candles on each side must be lower/higher to count as a peak
    df['min_idx'] = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=order)[0]]['Low']
    df['max_idx'] = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=order)[0]]['High']
    
    # Get indices of the last 2 significant peaks/troughs
    lows = df.dropna(subset=['min_idx'])
    highs = df.dropna(subset=['max_idx'])
    
    divergence_lines = []
    
    # Need at least 2 points to compare
    if len(lows) >= 2:
        # Check Last 2 Lows
        p1 = lows.iloc[-2]
        p2 = lows.iloc[-1]
        
        # Price Slope vs RSI Slope
        price_lower_low = p2['Low'] < p1['Low']
        price_higher_low = p2['Low'] > p1['Low']
        rsi_higher_low = p2['RSI'] > p1['RSI']
        rsi_lower_low = p2['RSI'] < p1['RSI']
        
        # Regular Bullish Divergence (Reversal): Price LL + RSI HL
        if price_lower_low and rsi_higher_low:
            divergence_lines.append({
                'type': 'Bullish Div (Reg)', 'color': 'green', 'style': '--',
                'x': [p1.name, p2.name], 'y_price': [p1['Low'], p2['Low']], 'y_rsi': [p1['RSI'], p2['RSI']]
            })
            
        # Hidden Bullish Divergence (Trend Follow): Price HL + RSI LL
        if price_higher_low and rsi_lower_low:
            divergence_lines.append({
                'type': 'Bullish Div (Hid)', 'color': 'lightgreen', 'style': ':',
                'x': [p1.name, p2.name], 'y_price': [p1['Low'], p2['Low']], 'y_rsi': [p1['RSI'], p2['RSI']]
            })

    if len(highs) >= 2:
        # Check Last 2 Highs
        p1 = highs.iloc[-2]
        p2 = highs.iloc[-1]
        
        price_higher_high = p2['High'] > p1['High']
        price_lower_high = p2['High'] < p1['High']
        rsi_lower_high = p2['RSI'] < p1['RSI']
        rsi_higher_high = p2['RSI'] > p1['RSI']

        # Regular Bearish Divergence (Reversal): Price HH + RSI LH
        if price_higher_high and rsi_lower_high:
            divergence_lines.append({
                'type': 'Bearish Div (Reg)', 'color': 'red', 'style': '--',
                'x': [p1.name, p2.name], 'y_price': [p1['High'], p2['High']], 'y_rsi': [p1['RSI'], p2['RSI']]
            })
            
        # Hidden Bearish Divergence (Trend Follow): Price LH + RSI HH
        if price_lower_high and rsi_higher_high:
            divergence_lines.append({
                'type': 'Bearish Div (Hid)', 'color': 'orange', 'style': ':',
                'x': [p1.name, p2.name], 'y_price': [p1['High'], p2['High']], 'y_rsi': [p1['RSI'], p2['RSI']]
            })
            
    return divergence_lines

def get_smart_trendlines(df):
    """
    Generates Angular Trendlines and Horizontal Support/Resistance.
    """
    lines = []
    df = df.copy()
    df['idx'] = range(len(df)) # Integer index for geometry
    
    # 1. ANGULAR TRENDLINES (Connecting Peaks)
    # Find peaks with strictly 5 candles on left and right
    peaks = df.iloc[argrelextrema(df['High'].values, np.greater, order=5)[0]]
    troughs = df.iloc[argrelextrema(df['Low'].values, np.less, order=5)[0]]
    
    if len(peaks) >= 2:
        # Connect last 2 major peaks for Resistance Trendline
        p1, p2 = peaks.iloc[-2], peaks.iloc[-1]
        slope = (p2['High'] - p1['High']) / (p2['idx'] - p1['idx'])
        y_end = p2['High'] + slope * (df['idx'].iloc[-1] - p2['idx'])
        lines.append({'x': [p1['idx'], df['idx'].iloc[-1]], 'y': [p1['High'], y_end], 'color': 'red', 'label': 'Resist Line'})

    if len(troughs) >= 2:
        # Connect last 2 major troughs for Support Trendline
        p1, p2 = troughs.iloc[-2], troughs.iloc[-1]
        slope = (p2['Low'] - p1['Low']) / (p2['idx'] - p1['idx'])
        y_end = p2['Low'] + slope * (df['idx'].iloc[-1] - p2['idx'])
        lines.append({'x': [p1['idx'], df['idx'].iloc[-1]], 'y': [p1['Low'], y_end], 'color': 'green', 'label': 'Support Line'})

    # 2. HORIZONTAL SUPPORT/RESISTANCE (Price Clustering)
    # We round prices to nearest 1% and find levels with max volume/touches
    # Simple Logic: Max and Min of the last 30 days are strong static levels
    recent_high = df['High'].tail(60).max()
    recent_low = df['Low'].tail(60).min()
    
    lines.append({'x': [df['idx'].iloc[0], df['idx'].iloc[-1]], 'y': [recent_high, recent_high], 'color': 'red', 'style': ':', 'label': 'Major Res'})
    lines.append({'x': [df['idx'].iloc[0], df['idx'].iloc[-1]], 'y': [recent_low, recent_low], 'color': 'green', 'style': ':', 'label': 'Major Sup'})

    return lines

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
        df_plot = df.reset_index() # Need integer index for plotting lines accurately
        
        # Get Analysis Lines
        trendlines = get_smart_trendlines(df)
        divergences = find_divergence(df.reset_index(drop=True)) # Reset index for row-based access

        # Setup Plot
        fig = plt.figure(figsize=(20, 14), facecolor='white') 
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 0.6, 1, 1])
        
        # --- PANEL 1: PRICE ---
        ax1 = plt.subplot(gs[0])
        x_vals = df_plot.index 
        
        # Candles
        up = df_plot[df_plot.Close >= df_plot.Open]
        down = df_plot[df_plot.Close < df_plot.Open]
        ax1.bar(up.index, up.Close - up.Open, width=0.7, bottom=up.Open, color='#089981', alpha=1.0, zorder=3) 
        ax1.vlines(up.index, up.Low, up.High, color='#089981', linewidth=0.8, zorder=3)
        ax1.bar(down.index, down.Close - down.Open, width=0.7, bottom=down.Open, color='#F23645', alpha=1.0, zorder=3)
        ax1.vlines(down.index, down.Low, down.High, color='#F23645', linewidth=0.8, zorder=3)
        
        # EMAs & BB
        ax1.plot(x_vals, df_plot['EMA_5'], color='#2962FF', linewidth=1.2, alpha=0.9, label="EMA 5")
        ax1.plot(x_vals, df_plot['EMA_50'], color='#FF9800', linewidth=1.2, alpha=0.9, label="EMA 50")
        ax1.fill_between(x_vals, df_plot['BB_Upper'], df_plot['BB_Lower'], color='#2962FF', alpha=0.03)

        # Draw Trendlines & Horizontal Levels
        for line in trendlines:
            style = line.get('style', '-')
            ax1.plot(line['x'], line['y'], color=line['color'], linestyle=style, linewidth=1.5, zorder=4, label=line.get('label'))
            
        # Draw Divergences on Price
        for div in divergences:
            # We need to map the date index back to integer index for plotting
            # div['x'] contains integers 0..N because we passed reset_index df
            ax1.plot(div['x'], div['y_price'], color=div['color'], linestyle='-', linewidth=2, marker='o', markersize=4, label=div['type'])

        # Title & Grid
        last_price = df['Close'].iloc[-1]
        pct_change = ((last_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        color_arrow = "🟢" if pct_change >= 0 else "🔴"
        ax1.set_title(f"{symbol} {color_arrow} ₹{last_price:.2f} ({pct_change:+.2f}%)", fontsize=18, fontweight='bold', pad=15)
        ax1.grid(True, color='#f0f0f0', linestyle='-', linewidth=0.5)
        ax1.legend(loc='upper left', frameon=False, fontsize=10)

        # --- PANEL 2: VOLUME ---
        ax2 = plt.subplot(gs[1], sharex=ax1)
        vol_colors = ['#089981' if c >= o else '#F23645' for c, o in zip(df_plot['Close'], df_plot['Open'])]
        ax2.bar(x_vals, df_plot['Volume'], color=vol_colors, alpha=0.6, width=0.7)
        ax2.set_ylabel("Vol", fontweight='bold')
        ax2.grid(True, color='#f0f0f0')
        
        # --- PANEL 3: RSI + DIVERGENCE ---
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(x_vals, df_plot['RSI'], color='#7E57C2', linewidth=1.5)
        ax3.axhline(70, linestyle='--', color='#F23645', alpha=0.5)
        ax3.axhline(30, linestyle='--', color='#089981', alpha=0.5)
        ax3.fill_between(x_vals, df_plot['RSI'], 70, where=(df_plot['RSI']>=70), color='#F23645', alpha=0.1)
        ax3.fill_between(x_vals, df_plot['RSI'], 30, where=(df_plot['RSI']<=30), color='#089981', alpha=0.1)
        
        # Draw Divergence Lines on RSI
        for div in divergences:
            ax3.plot(div['x'], div['y_rsi'], color=div['color'], linestyle='-', linewidth=2, marker='o', markersize=4)

        ax3.set_ylabel("RSI", fontweight='bold')
        ax3.grid(True, color='#f0f0f0')
        ax3.set_ylim(0, 100)

        # --- PANEL 4: MACD ---
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.plot(x_vals, df_plot['MACD'], color='#2962FF', linewidth=1.2)
        ax4.plot(x_vals, df_plot['Signal'], color='#FF9800', linewidth=1.2)
        hist_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df_plot['Hist']]
        ax4.bar(x_vals, df_plot['Hist'], color=hist_colors, alpha=0.8, width=0.7)
        ax4.set_ylabel("MACD", fontweight='bold')
        ax4.grid(True, color='#f0f0f0')

        # Formatting
        step = max(1, len(df_plot) // 12)
        date_labels = df_plot['Date'].dt.strftime('%b %d')
        ax4.set_xticks(x_vals[::step])
        ax4.set_xticklabels(date_labels[::step], rotation=0, fontsize=10)
        
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.04)

        # Send
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(url, files={'photo': buf}, data={'chat_id': CHAT_ID, 'caption': f"📊 {symbol} Smart Analysis"})
        logging.info(f"✅ Smart Chart sent for {symbol}")

    except Exception as e:
        logging.warning(f"❌ Failed to send chart for {symbol}: {e}", exc_info=True)
