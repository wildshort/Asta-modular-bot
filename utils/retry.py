import time
import logging
import yfinance as yf

def retry_download(symbol: str, period: str, interval: str, retries: int = 3, delay: int = 2):
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            if not df.empty:
                return df
            else:
                logging.warning(f"⚠️ Empty data for {symbol}, attempt {attempt}")
        except Exception as e:
            logging.warning(f"⚠️ Download failed for {symbol} (attempt {attempt}): {e}")
        time.sleep(delay)
    return None
