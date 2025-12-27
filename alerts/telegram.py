import requests
import logging
import matplotlib.pyplot as plt
import io
import yfinance as yf
from config import TELEGRAM_TOKEN, CHAT_ID

def send_telegram(message: str):
    """Sends a standard text message."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("🚫 Telegram token or chat ID not set.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            logging.info("📤 Telegram message sent successfully.")
        else:
            logging.warning(f"❌ Telegram response error: {response.status_code} - {response.text}")
    except Exception as e:
        logging.warning(f"⚠️ Telegram send failed: {e}")

def send_chart(symbol: str):
    """Generates a chart for the symbol and sends it to Telegram."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return

    try:
        # Download data for chart (separate from scanner to ensure clean frame)
        ticker = f"{symbol}" if ".NS" in symbol or "=" in symbol else f"{symbol}.NS"
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)

        if df.empty:
            logging.warning(f"⚠️ No chart data for {symbol}")
            return

        # --- Plotting ---
        plt.figure(figsize=(10, 6))
        
        # Top panel: Price + SMA
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['Close'], label='Price', color='black', linewidth=1)
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        plt.plot(df.index, df['SMA50'], label='50 SMA', color='orange', linestyle='--', linewidth=1)
        plt.title(f"{symbol} - Daily Trend")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylabel("Price")

        # Bottom panel: RSI
        plt.subplot(2, 1, 2)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        plt.plot(df.index, df['RSI'], label='RSI (14)', color='#8800ff', linewidth=1)
        plt.axhline(70, linestyle='--', color='red', alpha=0.5)
        plt.axhline(30, linestyle='--', color='green', alpha=0.5)
        plt.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), color='red', alpha=0.1)
        plt.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), color='green', alpha=0.1)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylabel("RSI")

        plt.tight_layout()

        # Save to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # --- Send to Telegram ---
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {'photo': buf}
        data = {'chat_id': CHAT_ID, 'caption': f"📊 Chart Analysis: {symbol}"}
        
        requests.post(url, files=files, data=data)
        logging.info(f"✅ Chart sent for {symbol}")

    except Exception as e:
        logging.warning(f"❌ Failed to send chart for {symbol}: {e}")
