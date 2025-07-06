# scanner/nifty_monitor.py
import time
import threading
import logging
from datetime import datetime, time as dtime

from utils.fetcher import retry_download
from scanner.stock_scanner import find_crossover
from alerts.telegram import send_telegram
from config import is_market_day

def nifty_intraday_check():
    logging.info("🔍 Checking NIFTY 15-min crossover...")
    df_day = retry_download("^NSEI", "6mo", "1d")
    df_15m = retry_download("^NSEI", "5d", "15m")

    if df_day is None or df_15m is None:
        logging.warning("⚠️ Skipping NIFTY check due to data failure")
        return

    # Daily trend check
    ema5_d = df_day['Close'].ewm(span=5).mean()
    ema50_d = df_day['Close'].ewm(span=50).mean()
    daily_crossed = (
        float(ema5_d.iloc[-1]) > float(ema50_d.iloc[-1]) and
        float(ema5_d.iloc[-2]) <= float(ema50_d.iloc[-2])
    )

    if not daily_crossed:
        logging.info("⛔ No daily crossover. Skipping 15m scan.")
        return

    # 15-min crossover check
    ema5_15 = df_15m['Close'].ewm(span=5).mean()
    ema50_15 = df_15m['Close'].ewm(span=50).mean()
    bull, bear = find_crossover(ema5_15, ema50_15)

    if bull:
        msg = "🚨 NIFTY 15-min Bullish Crossover (daily trend confirmed)"
        logging.info(msg)
        send_telegram(msg)
    elif bear:
        msg = "⚠️ NIFTY 15-min Bearish Crossover (daily trend confirmed)"
        logging.info(msg)
        send_telegram(msg)
    else:
        logging.info("✅ NIFTY checked. No 15m crossover right now.")

def run_nifty_monitor():
    def loop():
        while True:
            now = datetime.now()
            if is_market_day() and dtime(9, 15) <= now.time() <= dtime(15, 20):
                nifty_intraday_check()
            time.sleep(300)  # every 5 minutes

    t = threading.Thread(target=loop, daemon=True)
    t.start()
