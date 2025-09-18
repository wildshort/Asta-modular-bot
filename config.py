# config.py
import datetime
import os

# ✅ Telegram Bot Details
TELEGRAM_TOKEN = "7687060477:AAHd9efwSb2oXiZeo-aOGYXviCZVAf1JiEY"
CHAT_ID = -1002737768405
# Watchlist
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "LT.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "AXISBANK.NS",
    "MARUTI.NS", "KOTAKBANK.NS", "WIPRO.NS", "HCLTECH.NS", "BHARTIARTL.NS",
]

# NSE Holidays 2025
NSE_HOLIDAYS = {
    "2025-01-26", "2025-03-29", "2025-04-14", "2025-05-01", "2025-08-15",
    "2025-10-02", "2025-10-24", "2025-11-11", "2025-12-25"
}

def is_market_day():
    today = datetime.datetime.now().date()
    return today.weekday() < 5 and today.isoformat() not in NSE_HOLIDAYS

# Logging directory
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
LOG_FILE = f"{log_dir}/market_scan_{today_str}.log"
