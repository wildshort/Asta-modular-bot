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
CHART_PERIOD = "2y" 

# --- HELPER: INDICATORS ---
def calculate_indicators(df):
    df = df.copy()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands (Daily)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    
    # Band Width & Slope (for BKP/Base detection)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    df['BB_Lower_Slope'] = df['BB_Lower'].diff()
    df['BB_Upper_Slope'] = df['BB_Upper'].diff()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (TI)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    return df

def check_asta_setup(df):
    """
    ASTA Swing Logic:
    1. Tide (Weekly) BBNC
    2. Wave (Daily) Patterns (DB/FBD, BKP, TI Uptick, SO BBO)
    """
    # --- 1. PREPARE WEEKLY DATA (TIDE) ---
    # Resample daily to weekly to avoid extra API calls
    df_week = df.resample('W').agg({'Close': 'last'})
    df_week['BB_Mid'] = df_week['Close'].rolling(window=20).mean()
    df_week['BB_Std'] = df_week['Close'].rolling(window=20).std()
    df_week['BB_Upper'] = df_week['BB_Mid'] + (2 * df_week['BB_Std'])
    df_week['BB_Lower'] = df_week['BB_Mid'] - (2 * df_week['BB_Std'])
    
    # Tide Conditions
    # Buy: BBNC on Downside (Lower band not sloping down sharply)
    tide_buy_ok = False
    if len(df_week) > 2:
        # Check slope of weekly lower band. > -0.01 means flat or up (Not Challenged)
        lower_band_slope = df_week['BB_Lower'].diff().iloc[-1]
        tide_buy_ok = lower_band_slope > -5.0 # Loose filter: Just ensure it's not crashing vertically
    
    # Sell: BBNC on Upside (Upper band not sloping up sharply)
    tide_sell_ok = False
    if len(df_week) > 2:
        upper_band_slope = df_week['BB_Upper'].diff().iloc[-1]
        tide_sell_ok = upper_band_slope < 5.0

    # --- 2. PREPARE DAILY DATA (WAVE) ---
    df = calculate_indicators(df)
    df['idx'] = range(len(df))
    
    # Find Pivots for Patterns
    df['pivot_low'] = df.iloc[argrelextrema(df['Low'].values, np.less, order=5)[0]]['Low']
    df['pivot_high'] = df.iloc[argrelextrema(df['High'].values, np.greater, order=5)[0]]['High']
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    signal = "None"
    reasons = []

    # --- LOGIC FOR BUY (Double Bottom / Fake Breakdown) ---
    lows = df.dropna(subset=['pivot_low'])
    if len(lows) >= 2:
        b1 = lows.iloc[-2]; b2 = lows.iloc[-1]
        
        # Pattern: Double Bottom / Shake Out (SO)
        # Check if B2 is near B1 or slightly below (Shake Out)
        # Logic: Price dipped below B1 but recovered?
        dist_pct = abs(b1['pivot_low'] - b2['pivot_low']) / b1['pivot_low']
        is_shakeout = df['Low'].iloc[-5:].min() < b1['pivot_low'] and curr['Close'] > b1['pivot_low']
        
        if (dist_pct < 0.03 or is_shakeout) and tide_buy_ok:
            # CHECK ASTA MANDATORY CONDITIONS
            
            # 1. Candle: Bullish Engulfing or Strong Green
            is_engulfing = curr['Close'] > prev['Open'] and curr['Open'] < prev['Close'] and curr['Close'] > curr['Open']
            
            # 2. BKP (Base/Cup at Lower BB): Band Slope is flat/up
            is_bkp = curr['BB_Lower_Slope'] > -1.0 
            
            # 3. TI Uptick (MACD Hist Increasing)
            is_ti_uptick = curr['Hist'] > prev['Hist']
            
            # 4. RSI > 40
            is_rsi_ok = curr['RSI'] > 40
            
            if is_engulfing and is_ti_uptick and is_rsi_ok:
                signal = "🚀 ASTA BUY"
                if is_shakeout: reasons.append("SO BBO (Shake Out)")
                if is_bkp: reasons.append("BKP (Base Formed)")
                reasons.append("TI Uptick")

    # --- LOGIC FOR SELL (Double Top / Fake Breakout) ---
    highs = df.dropna(subset=['pivot_high'])
    if len(highs) >= 2:
        t1 = highs.iloc[-2]; t2 = highs.iloc[-1]
        
        # Pattern: Double Top / FBO
        is_fbo = df['High'].iloc[-5:].max() > t1['pivot_high'] and curr['Close'] < t1['pivot_high']
        dist_pct = abs(t1['pivot_high'] - t2['pivot_high']) / t1['pivot_high']
        
        if (dist_pct < 0.03 or is_fbo) and tide_sell_ok:
            # 1. Candle: Bearish Engulfing
            is_engulfing = curr['Close'] < prev['Open'] and curr['Open'] > prev['Close'] and curr['Close'] < curr['Open']
            
            # 2. BKT (Top at Upper BB): Band Slope flat/down
            is_bkt = curr['BB_Upper_Slope'] < 1.0
            
            # 3. TI Downtick
            is_ti_downtick = curr['Hist'] < prev['Hist']
            
            # 4. RSI < 60
            is_rsi_ok = curr['RSI'] < 60
            
            if is_engulfing and is_ti_downtick and is_rsi_ok:
                signal = "🔻 ASTA SELL"
                if is_fbo: reasons.append("FBO (Trap)")
                reasons.append("TI Downtick")

    return signal, ", ".join(reasons)

def get_breakout_status(df):
    """ Standard Trendline Logic (Kept as backup) """
    df = df.copy(); df['idx'] = range(len(df))
    df['pivot_high'] = df.iloc[argrelextrema(df['High'].values, np.greater, order=10)[0]]['High']
    df['pivot_low'] = df.iloc[argrelextrema(df['Low'].values, np.less, order=10)[0]]['Low']
    
    lines = []
    status = "Watching"
    last_idx = df['idx'].iloc[-1]; current_close = df['Close'].iloc[-1]
    
    highs = df.dropna(subset=['pivot_high'])
    if len(highs) >= 2:
        major_high = highs.sort_values('pivot_high', ascending=False).iloc[0]
        peaks = highs[highs['idx'] > major_high['idx']]
        if not peaks.empty:
            p2 = peaks.iloc[-1]
            slope = (p2['pivot_high'] - major_high['pivot_high']) / (p2['idx'] - major_high['idx'])
            y_end = p2['pivot_high'] + slope * (last_idx - p2['idx'])
            lines.append({'x': [major_high['idx'], last_idx], 'y': [major_high['pivot_high'], y_end], 'color': 'red', 'label': 'Resist'})
            if current_close > y_end and slope < 0: status = "🚀 TL BREAKOUT"

    lows = df.dropna(subset=['pivot_low'])
    if len(lows) >= 2:
        major_low = lows.sort_values('pivot_low', ascending=True).iloc[0]
        troughs = lows[lows['idx'] > major_low['idx']]
        if not troughs.empty:
            p2 = troughs.iloc[-1]
            slope = (p2['pivot_low'] - major_low['pivot_low']) / (p2['idx'] - major_low['idx'])
            y_end = p2['pivot_low'] + slope * (last_idx - p2['idx'])
            lines.append({'x': [major_low['idx'], last_idx], 'y': [major_low['pivot_low'], y_end], 'color': 'green', 'label': 'Support'})
            if current_close < y_end and slope > 0: status = "🔻 TL BREAKDOWN"

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

        # Process Indicators & Logic
        df = calculate_indicators(df)
        tl_status, lines = get_breakout_status(df)
        asta_signal, asta_reasons = check_asta_setup(df)
        
        # 🚨 FINAL SIGNAL DECISION 🚨
        # ASTA Signals take priority over simple trendlines
        final_status = tl_status
        if asta_signal != "None":
            final_status = f"{asta_signal} | {asta_reasons}"
        
        # Filter: Only send if there is SOME signal (ASTA or Trendline)
        if "Watching" in final_status and asta_signal == "None":
            return # Silence noise

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
        ax1.plot(x_vals, df_plot['EMA_50'], color='#FF9800', lw=1.5, label="EMA 50")
        ax1.plot(x_vals, df_plot['BB_Upper'], color='blue', lw=0.5, alpha=0.3)
        ax1.plot(x_vals, df_plot['BB_Lower'], color='blue', lw=0.5, alpha=0.3)
        ax1.fill_between(x_vals, df_plot['BB_Upper'], df_plot['BB_Lower'], color='blue', alpha=0.05)
        
        for line in lines:
            ax1.plot(line['x'], line['y'], color=line['color'], linestyle='-', lw=2, label=line['label'])

        # Title
        last_price = df['Close'].iloc[-1]
        pct = ((last_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        title_color = 'green' if 'BUY' in final_status or 'ROCKET' in final_status else 'red' if 'SELL' in final_status else 'black'
        
        title = f"{symbol} {'🟢' if pct>=0 else '🔴'} ₹{last_price:.2f} ({pct:+.2f}%) | {final_status}"
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=15, color=title_color)
        ax1.grid(True, color='#f0f0f0'); ax1.legend(loc='upper left', frameon=False)

        # PANELS 2, 3 (RSI, MACD)
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
        caption = f"📊 {symbol} Analysis: {final_status}"
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", files={'photo': buf}, data={'chat_id': CHAT_ID, 'caption': caption})
        logging.info(f"✅ Chart sent for {symbol}")

    except Exception as e:
        logging.warning(f"❌ Failed to send chart for {symbol}: {e}", exc_info=True)
