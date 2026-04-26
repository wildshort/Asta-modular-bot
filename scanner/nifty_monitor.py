"""
Intraday NIFTY monitor. Checks for 15-min EMA crossovers, only firing
when the daily trend confirms.

Kept optional — only runs if explicitly started from somewhere.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, time as dtime

from alerts.telegram import send_telegram
from config import is_market_day
from scanner.stock_scanner import find_crossover
from utils.fetcher import download_single

log = logging.getLogger(__name__)


def nifty_intraday_check() -> None:
    log.info("Checking NIFTY 15-min crossover...")
    df_day = download_single("^NSEI", "6mo", "1d")
    df_15m = download_single("^NSEI", "5d", "15m")

    if df_day is None or df_15m is None:
        log.warning("Skipping NIFTY check due to data failure")
        return

    ema5_d = df_day["Close"].ewm(span=5, adjust=False).mean()
    ema50_d = df_day["Close"].ewm(span=50, adjust=False).mean()
    daily_crossed = (
        float(ema5_d.iloc[-1]) > float(ema50_d.iloc[-1])
        and float(ema5_d.iloc[-2]) <= float(ema50_d.iloc[-2])
    )
    if not daily_crossed:
        log.info("No daily crossover. Skipping 15m scan.")
        return

    ema5_15 = df_15m["Close"].ewm(span=5, adjust=False).mean()
    ema50_15 = df_15m["Close"].ewm(span=50, adjust=False).mean()
    bull, bear = find_crossover(ema5_15, ema50_15)

    if bull:
        msg = "🚨 NIFTY 15-min Bullish Crossover (daily trend confirmed)"
        log.info(msg)
        send_telegram(msg)
    elif bear:
        msg = "⚠️ NIFTY 15-min Bearish Crossover (daily trend confirmed)"
        log.info(msg)
        send_telegram(msg)
    else:
        log.info("NIFTY checked. No 15m crossover right now.")


def run_nifty_monitor() -> None:
    def loop():
        while True:
            now = datetime.now()
            if is_market_day() and dtime(9, 15) <= now.time() <= dtime(15, 20):
                try:
                    nifty_intraday_check()
                except Exception as e:
                    log.exception(f"Nifty monitor error: {e}")
            time.sleep(300)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
