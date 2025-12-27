import requests
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import io
import pandas as pd
import yfinance as yf
from config import TELEGRAM_TOKEN, CHAT_ID

# --- HELPER: Indicator Calculations ---
def calculate_indicators(df):
    """Calculates EMA, RSI, MACD for plotting."""
    # EMA
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
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

def get_trendline_points(df):
    """
    Identifies simple support/resistance trendlines based on local highs/lows.
    Returns coordinates for drawing lines.
    """
    # Create a copy to avoid SettingWithCopy warnings on the original df
    df = df.copy()

    # 1. Identify Local Highs and Lows (Pivots)
    # We use a window of 1 on each side (shift 1)
    df['pivot_high'] = df['High'][
        (df['High'].shift(1) < df['High']) & 
        (df['High'].shift(-1) < df['High'])
    ]
    df['pivot_low'] = df['Low'][
        (df['Low'].shift(1) > df['Low']) & 
        (df['Low'].shift(-1) > df['Low'])
    ]
    
    # Get the last 3 significant pivots to draw recent trend
    # We check if columns exist to be safe, though assignment above ensures they do
    highs = df.dropna(subset=['pivot_high']).tail(3)
    lows = df.dropna(subset=['pivot_low']).tail(3)

    lines = []
    
    # Resistance Line (Connect last two highs)
    if len(highs) >= 2:
        x1 = mdates.date2num(highs.index[-2])
        y1 = highs['High'].iloc[-2]
        x2 = mdates.date2num(highs.index[-1])
        y2 = highs['High'].iloc[-1]
        
        # Extend line slightly to the right
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            x_ext = mdates.date2num(df.index[-1])
            y_ext = y1 + slope * (x_ext - x1)
            lines.append({'x': [x1, x_ext], 'y': [y1, y_ext], 'color': 'red', 'label': 'Resist'})

    # Support Line (Connect last two lows)
    if len(lows) >= 2:
        x1 = mdates.date2num(lows.index[-2])
        y1 = lows['Low'].iloc[-2]
        x2 = mdates.date2num(lows.index[-1])
        y2 = lows['Low'].iloc[-1]
        
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            x_ext = mdates.date2num(df.index[-1])
            y_ext = y1 + slope * (x_ext - x1)
            lines.append({'x': [x1, x_ext], 'y': [y1, y_ext], 'color': 'green', 'label': 'Support'})
        
    return lines

# --- MAIN FUNCTIONS ---
def send_telegram(message: str):
    """Sends a standard text message."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("🚫 Telegram token or chat ID not set.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}

    try:
        requests.post(url, data=payload)
    except Exception as e:
        logging.warning(f"⚠️ Telegram send failed: {e}")

def send_chart(symbol: str):
    """Generates a 1-Year Candle+Trendline+MACD+RSI chart."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    try:
        # 1. Fetch Data
        ticker = f"{symbol}" if ".NS" in symbol or "=" in symbol else f"{symbol}.NS"
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        if df.empty:
            logging.warning(f"⚠️ No chart data for {symbol}")
            return

        # 🔧 FIX: FLATTEN COLUMNS
        # This fixes the KeyError by ensuring columns are simple strings ('Close') 
        # instead of complex tuples (('Close', 'TATASTEEL.NS'))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Prepare Data
        df = calculate_indicators(df)
        trendlines = get_trendline_points(df)

        # 3. Setup Plot Layout (3 Rows)
        fig = plt.figure(figsize=(12, 12)) 
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
        
        # --- PANEL 1: PRICE, CANDLES & TRENDLINES ---
        ax1 = plt.subplot(gs[0])
        
        # Candlestick Logic
        up = df[df.Close >= df.Open]
        down = df[df.Close < df.Open]
        
        # Width 0.5 works well for 1 year data
        ax1.bar(up.index, up.Close - up.Open, width=0.5, bottom=up.Open, color='green', alpha=0.7)
        ax1.vlines(up.index, up.Low, up.High, color='green', linewidth=0.8)
        
        ax1.bar(down.index, down.Close - down.Open, width=0.5, bottom=down.Open, color='red', alpha=0.7)
        ax1.vlines(down.index, down.Low, down.High, color='red', linewidth=0.8)
        
        # EMAs
        ax1.plot(df.index, df['EMA_5'], label='EMA 5', color='blue', linewidth=1)
        ax1.plot(df.index, df['EMA_50'], label='EMA 50', color='orange', linewidth=1)
        
        # Draw Trendlines
        for line in trendlines:
            ax1.plot(line['x'], line['y'], color=line['color'], linestyle='--', linewidth=1.5, label=line['label'])

        ax1.set_title(f"{symbol} - 1 Year Trend Analysis")
        ax1.grid(True, alpha=0.15)
        ax1.legend(loc='upper left')
        ax1.set_ylabel("Price")
        
        # --- PANEL 2: RSI ---
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(df.index, df['RSI'], color='purple', linewidth=1)
        ax2.axhline(70, linestyle='--', color='red', alpha=0.5)
        ax2.axhline(30, linestyle='--', color='green', alpha=0.5)
        ax2.fill_between(df.index, df['RSI'], 70, where=(df['RSI']>=70), color='red', alpha=0.1)
        ax2.fill_between(df.index, df['RSI'], 30, where=(df['RSI']<=30), color='green', alpha=0.1)
        ax2.set_ylabel("RSI")
        ax2.grid(True, alpha=0.15)
        ax2.set_ylim(0, 100)

        # --- PANEL 3: MACD ---
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1)
        ax3.plot(df.index, df['Signal'], label='Signal', color='orange', linewidth=1)
        
        colors = ['green' if v >= 0 else 'red' for v in df['Hist']]
        ax3.bar(df.index, df['Hist'], color=colors, alpha=0.5, width=0.5)
        
        ax3.set_ylabel("MACD")
        ax3.grid(True, alpha=0.15)
        ax3.legend(loc='lower right', fontsize='small')

        # Formatting
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # Save to Buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)

        # 4. Send to Telegram
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {'photo': buf}
        data = {'chat_id': CHAT_ID, 'caption': f"📊 {symbol} 1-Year Trend + Support/Resistance"}
        
        requests.post(url, files=files, data=data)
        logging.info(f"✅ Trend Chart sent for {symbol}")

    except Exception as e:
        logging.warning(f"❌ Failed to send chart for {symbol}: {e}", exc_info=True)
