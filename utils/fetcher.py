# utils/fetcher.py
import yfinance as yf
import time
import logging
import pandas as pd

def retry_download(symbol, period="6mo", interval="1d", retries=3, delay=2) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            if not df.empty:
                return df
        except Exception as e:
            logging.warning(f"⚠️ {symbol} download failed (attempt {attempt+1}): {e}")
        time.sleep(delay)
    logging.warning(f"❌ {symbol} failed after {retries} attempts.")
    return None
