import os
import sys
import pandas as pd
import yfinance as yf

# Dynamically locate and add project root to sys.path
current = os.path.abspath(os.path.dirname(__file__))
while not os.path.exists(os.path.join(current, 'watchlist', 'nifty_stocks.py')):
    parent = os.path.dirname(current)
    if parent == current:
        raise RuntimeError("❌ Could not locate project root with watchlist/nifty_stocks.py")
    current = parent

print(f"🔧 Adding to sys.path: {current}")
sys.path.insert(0, current)

from watchlist.nifty_stocks import watchlist

def download_data(symbol, period, interval):
    folder = "scanner/data"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{symbol}_{interval}.csv")
    if os.path.exists(filename):
        print(f"✅ Cached: {symbol} ({interval})")
        return
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if not df.empty:
            df.to_csv(filename)
            print(f"📥 Downloaded: {symbol} ({interval})")
        else:
            print(f"⚠️ No data: {symbol}")
    except Exception as e:
        print(f"❌ Error for {symbol}: {e}")

print("📦 Starting bulk download of stock data...")
for symbol in watchlist:
    download_data(symbol, "6mo", "1d")
    download_data(symbol, "1y", "1wk")
print("✅ Bulk download completed.")
