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
CHART_PERIOD = "1y" 

# --- HELPER: INDICATORS ---
def calculate_indicators(df):
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
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

# --- HELPER: CANDLESTICK PATTERNS (THE SNIPER) ---
def analyze_breakout_candle(df, direction):
    """
    Analyzes the last candle to confirm strength or detect weakness.
    direction: 'bullish' (Breakout) or 'bearish' (Breakdown)
    """
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    body = abs(curr['Close'] - curr['Open'])
    total_range = curr['High'] - curr['Low']
    upper_wick = curr['High'] - max(curr['Close'], curr['Open'])
    lower_wick = min(curr['Close'], curr['Open']) - curr['Low']
    
    # Avoid div by zero
    if total_range == 0: return "Unknown"
    
    body_pct = body / total_range
    
    if direction == 'bullish':
        # 1. TRAP CHECK: Long Upper Wick (Rejection)
        # If the upper wick is bigger than the body, buyers failed to hold.
        if upper_wick > body:
            return "⚠️ FAKEOUT (Wick Rejection)"
            
        # 2. STRENGTH CHECK: Marubozu (Big Body, Small Wicks)
        # Body is > 60% of range and closing near high
        if body_pct > 0.6 and upper_wick < (0.2 * body):
            return "🔥 STRONG MARUBOZU"
            
        # 3. MOMENTUM CHECK: Engulfing
        # Green candle completely eats previous Red candle
        if curr['Close'] > prev['Open'] and curr['Open'] < prev['Close'] and curr['Close'] > curr['Open']:
            return "🔄 ENGULFING CONFIRMATION"
            
        return "✅ Valid Breakout"

    elif direction == 'bearish':
        # 1. TRAP CHECK: Long Lower Wick (Support held)
        if lower_wick > body:
            return "⚠️ FAKEOUT (Support Held)"
            
        # 2. STRENGTH CHECK: Marubozu
        if body_pct > 0.6 and lower_wick < (0.2 * body):
            return "🔥 STRONG DROP (Marubozu)"
            
        # 3. MOMENTUM CHECK: Engulfing
        if curr['Close'] < prev['Open'] and curr['Open'] > prev['Close'] and curr['Close'] < curr['Open']:
            return "🔄 ENGULFING BREAKDOWN"
            
        return "✅ Valid Breakdown"
        
    return "Watching"

# --- HELPER: SMART TRENDLINES ---
def get_breakout_status(df):
    """
    Identifies Trendlines and checks for High-Quality Breakouts.
    """
    df = df.copy()
    df['idx'] = range(len(df))
    
    # Find Pivots (Order=10 for major swings)
    df['pivot_high'] = df.iloc[argrelextrema(df['High'].values, np.greater, order=10)[0]]['High']
    df['pivot_low'] = df.iloc[argrelextrema(df['Low'].values, np.less, order=10)[0]]['Low']
    
    highs = df.dropna(subset=['pivot_high'])
    lows = df.dropna(subset=['pivot_low'])
    
    lines = []
    status = "Watching" # Default
    current_close = df['Close'].iloc[-1]
    last_idx = df['idx'].iloc[-1]

    # 1. RESISTANCE (Bullish Breakout Check)
    if len(highs) >= 2:
        major_high = highs.sort_values('pivot_high', ascending=False).iloc[0]
        subsequent_peaks = highs[highs['idx'] > major_high['idx']]
        
        if not subsequent_peaks.empty:
            recent_peak = subsequent_peaks.iloc[-1]
            slope = (recent_peak['pivot_high'] - major_high['pivot_high']) / (recent_peak['idx'] - major_high['idx'])
            line_val_at_current = recent_peak['pivot_high'] + slope * (last_idx - recent_peak['idx'])
            
            lines.append({'x': [major_high['idx'], last_idx], 'y': [major_high['pivot_high'], line_val_at_current], 'color': 'red', 'label': 'Resist'})
            
            # CHECK BREAKOUT
            if current_close > line_val_at_current and slope < 0:
                confirmation = analyze_breakout_candle(df, 'bullish')
                status = f"🚀 {confirmation}"

    # 2. SUPPORT (Bearish Breakdown Check)
    if len(lows) >= 2:
        major_low = lows.sort_values('pivot_low', ascending=True).iloc[0]
        subsequent_lows = lows[lows['idx'] > major_low['idx']]
        
        if not subsequent_lows.empty:
            recent_low = subsequent_lows.iloc[-1]
            slope = (recent_low['pivot_low'] - major_low['pivot_low']) / (recent_low['idx'] - major_low['idx'])
            line_val_at_current = recent_low['pivot_low'] + slope * (last_idx - recent_low['idx'])
            
            lines.append({'x': [major_low['idx'], last_idx], 'y': [major_low['pivot_low'], line_val_at_current], 'color': 'green', 'label': 'Support'})
            
            # CHECK BREAKDOWN
            if current_close < line_val_at_current and slope > 0:
                confirmation = analyze_breakout_candle(df, 'bearish')
                status = f"🔻 {confirmation}"

    return status, lines

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
        status, lines = get_breakout_status(df)
        
        # --- SNIPER FILTER ---
        # Only send if we have a Breakout/Breakdown (Status is not just 'Watching')
        # AND it's not a 'Fakeout' (optional strictness)
        if status == "Watching": 
            return 

        df_plot = df.reset_index()
        
        # Setup Plot
        fig = plt.figure(figsize=(20, 12), facecolor='white') 
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
        
        # PANEL 1: PRICE
        ax1 = plt.subplot(gs[0])
        x_vals = df_plot.index 
        
        # Candles
        up = df_plot[df_plot.Close >= df_plot.Open]; down = df_plot[df_plot.Close < df_plot.Open]
        ax1.bar(up.index, up.Close - up.Open, width=0.7, bottom=up.Open, color='#089981', alpha=1) 
        ax1.vlines(up.index, up.Low, up.High, color='#089981', linewidth=0.8)
        ax1.bar(down.index, down.Close - down.Open, width=0.7, bottom=down.Open, color='#F23645', alpha=1)
        ax1.vlines(down.index, down.Low, down.High, color='#F23645', linewidth=0.8)
        
        # Overlays
        ax1.plot(x_vals, df_plot['EMA_50'], color='#FF9800', linewidth=1.5, label="EMA 50")
        for line in lines:
            ax1.plot(line['x'], line['y'], color=line['color'], linestyle='-', linewidth=2, label=line['label'])

        # Title
        last_price = df['Close'].iloc[-1]
        pct = ((last_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        # Dynamic Color based on Sniper Status
        title_color = 'green' if 'ROCKET' in status or 'STRONG' in status else 'black'
        
        title = f"{symbol} {'🟢' if pct>=0 else '🔴'} ₹{last_price:.2f} ({pct:+.2f}%) | {status}"
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=15, color=title_color)
        ax1.grid(True, color='#f0f0f0'); ax1.legend(loc='upper left', frameon=False)

        # PANELS 2, 3
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(x_vals, df_plot['RSI'], color='#7E57C2'); ax2.axhline(70, color='red', ls='--'); ax2.axhline(30, color='green', ls='--')
        ax2.set_ylabel("RSI", fontweight='bold'); ax2.grid(True, color='#f0f0f0'); ax2.set_ylim(0, 100)

        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(x_vals, df_plot['MACD'], color='#2962FF'); ax3.plot(x_vals, df_plot['Signal'], color='#FF9800')
        ax3.bar(x_vals, df_plot['Hist'], color=['#26a69a' if v >= 0 else '#ef5350' for v in df_plot['Hist']], alpha=0.8)
        ax3.set_ylabel("MACD", fontweight='bold'); ax3.grid(True, color='#f0f0f0')

        # Formatting
        step = max(1, len(df_plot) // 12)
        ax3.set_xticks(x_vals[::step]); ax3.set_xticklabels(df_plot['Date'].dt.strftime('%b %d')[::step], rotation=0)
        plt.setp([ax.get_xticklabels() for ax in [ax1, ax2]], visible=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.04)

        # Send
        buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', dpi=150); buf.seek(0); plt.close(fig)
        
        caption = f"📊 {symbol} Analysis: {status}"
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", files={'photo': buf}, data={'chat_id': CHAT_ID, 'caption': caption})
        logging.info(f"✅ Chart sent for {symbol}")

    except Exception as e:
        logging.warning(f"❌ Failed to send chart for {symbol}: {e}", exc_info=True)
