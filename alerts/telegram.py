import requests
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import io
import numpy as np
import pandas as pd
import yfinance as yf
from config import TELEGRAM_TOKEN, CHAT_ID

# --- HELPER: Indicator Calculations ---
def calculate_indicators(df):
    """Calculates EMA, RSI, MACD, and Bollinger Bands."""
    # EMA
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands (20, 2)
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

def get_trendline_points(df):
    """
    Identifies trendlines using integer indices (0 to N) instead of dates.
    """
    df = df.copy()
    # Create integer index for calculation
    df['idx'] = range(len(df))
    
    # Identify Pivots
    df['pivot_high'] = df['High'][(df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High'])]
    df['pivot_low'] = df['Low'][(df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low'])]
    
    highs = df.dropna(subset=['pivot_high']).tail(3)
    lows = df.dropna(subset=['pivot_low']).tail(3)

    lines = []
    
    # Resistance
    if len(highs) >= 2:
        x1, y1 = highs['idx'].iloc[-2], highs['High'].iloc[-2]
        x2, y2 = highs['idx'].iloc[-1], highs['High'].iloc[-1]
        
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            # Extend to end of chart
            x_ext = df['idx'].iloc[-1]
            y_ext = y1 + slope * (x_ext - x1)
            lines.append({'x': [x1, x_ext], 'y': [y1, y_ext], 'color': 'purple', 'label': 'Resist'}) # Purple like TradingView

    # Support
    if len(lows) >= 2:
        x1, y1 = lows['idx'].iloc[-2], lows['Low'].iloc[-2]
        x2, y2 = lows['idx'].iloc[-1], lows['Low'].iloc[-1]
        
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            x_ext = df['idx'].iloc[-1]
            y_ext = y1 + slope * (x_ext - x1)
            lines.append({'x': [x1, x_ext], 'y': [y1, y_ext], 'color': 'purple', 'label': 'Support'})
        
    return lines

# --- MAIN FUNCTIONS ---
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      json={"chat_id": CHAT_ID, "text": message})
    except Exception as e:
        logging.warning(f"⚠️ Telegram send failed: {e}")

def send_chart(symbol: str):
    """Generates a TradingView-style chart using Index-Based plotting."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    try:
        # 1. Fetch Data
        ticker = f"{symbol}" if ".NS" in symbol or "=" in symbol else f"{symbol}.NS"
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        if df.empty:
            logging.warning(f"⚠️ No chart data for {symbol}")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Prepare Data
        df = calculate_indicators(df)
        trendlines = get_trendline_points(df)
        
        # Reset index to get 0..N integer range for gapless plotting
        df_plot = df.reset_index()
        
        # Title Stats
        last_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        pct_change = ((last_price - prev_price) / prev_price) * 100
        color_arrow = "🟢" if pct_change >= 0 else "🔴"

        # 3. Setup Layout
        # WIDER Figure (16x12) to reduce congestion
        fig = plt.figure(figsize=(16, 12), facecolor='white') 
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 0.5, 1, 1])
        
        # --- PANEL 1: PRICE (Index Based) ---
        ax1 = plt.subplot(gs[0])
        
        # Use simple integer index for X axis
        x_vals = df_plot.index 
        
        # Candles
        up = df_plot[df_plot.Close >= df_plot.Open]
        down = df_plot[df_plot.Close < df_plot.Open]
        
        # Wider bars (width=0.8) because we have no gaps now
        ax1.bar(up.index, up.Close - up.Open, width=0.8, bottom=up.Open, color='#089981', alpha=0.9) # TradingView Green
        ax1.vlines(up.index, up.Low, up.High, color='#089981', linewidth=1)
        
        ax1.bar(down.index, down.Close - down.Open, width=0.8, bottom=down.Open, color='#F23645', alpha=0.9) # TradingView Red
        ax1.vlines(down.index, down.Low, down.High, color='#F23645', linewidth=1)
        
        # Bollinger Bands
        ax1.plot(x_vals, df_plot['BB_Upper'], color='#2962FF', linewidth=1, alpha=0.3)
        ax1.plot(x_vals, df_plot['BB_Lower'], color='#2962FF', linewidth=1, alpha=0.3)
        ax1.fill_between(x_vals, df_plot['BB_Upper'], df_plot['BB_Lower'], color='#2962FF', alpha=0.05)
        
        # EMAs
        ax1.plot(x_vals, df_plot['EMA_5'], color='#2962FF', linewidth=1, alpha=0.8)   # Blue
        ax1.plot(x_vals, df_plot['EMA_50'], color='#FF9800', linewidth=1, alpha=0.8)  # Orange
        
        # Trendlines
        for line in trendlines:
            ax1.plot(line['x'], line['y'], color='#9C27B0', linestyle='-', linewidth=2) # Purple Trendline

        ax1.set_title(f"{symbol} {color_arrow} ₹{last_price:.2f} ({pct_change:+.2f}%)", fontsize=16, fontweight='bold')
        ax1.grid(True, color='#E0E0E0', linestyle='--', linewidth=0.5)
        ax1.set_ylabel("Price")

        # --- PANEL 2: VOLUME ---
        ax2 = plt.subplot(gs[1], sharex=ax1)
        vol_colors = ['#089981' if c >= o else '#F23645' for c, o in zip(df_plot['Close'], df_plot['Open'])]
        ax2.bar(x_vals, df_plot['Volume'], color=vol_colors, alpha=0.5, width=0.8)
        ax2.grid(True, color='#E0E0E0', linestyle='--', linewidth=0.5)
        ax2.set_ylabel("Vol")
        
        # --- PANEL 3: RSI ---
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(x_vals, df_plot['RSI'], color='#7E57C2', linewidth=1.5) # Purple RSI
        ax3.axhline(70, linestyle='--', color='#F23645', alpha=0.5)
        ax3.axhline(30, linestyle='--', color='#089981', alpha=0.5)
        ax3.fill_between(x_vals, df_plot['RSI'], 70, where=(df_plot['RSI']>=70), color='#F23645', alpha=0.1)
        ax3.fill_between(x_vals, df_plot['RSI'], 30, where=(df_plot['RSI']<=30), color='#089981', alpha=0.1)
        ax3.set_ylabel("RSI")
        ax3.grid(True, color='#E0E0E0', linestyle='--', linewidth=0.5)
        ax3.set_ylim(0, 100)

        # --- PANEL 4: MACD ---
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.plot(x_vals, df_plot['MACD'], color='#2962FF', linewidth=1)
        ax4.plot(x_vals, df_plot['Signal'], color='#FF9800', linewidth=1)
        hist_colors = ['#089981' if v >= 0 else '#F23645' for v in df_plot['Hist']]
        ax4.bar(x_vals, df_plot['Hist'], color=hist_colors, alpha=0.5, width=0.8)
        ax4.set_ylabel("MACD")
        ax4.grid(True, color='#E0E0E0', linestyle='--', linewidth=0.5)

        # --- FORMATTING DATE LABELS ---
        # Since we used integers 0..N, we must map them back to dates for labels
        date_labels = df_plot['Date'].dt.strftime('%b %d') # Format: "Jan 01"
        
        # Show roughly 10 labels across the chart
        step = max(1, len(df_plot) // 10)
        ax1.set_xticks(x_vals[::step])
        ax1.set_xticklabels(date_labels[::step])
        
        # Hide x-labels for upper panels
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        
        # Final Layout Adjustment
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.02) # Zero gap between plots

        # Save & Send
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        plt.close(fig)

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(url, files={'photo': buf}, data={'chat_id': CHAT_ID, 'caption': f"📊 {symbol} Smart Chart"})
        logging.info(f"✅ Smart Chart sent for {symbol}")

    except Exception as e:
        logging.warning(f"❌ Failed to send chart for {symbol}: {e}", exc_info=True)
