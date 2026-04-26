"""
ChoCH scanner — finds Change-of-Character reversal signals across the watchlist.

Kept intentionally separate from stock_scanner.py so:
  - It can run on its own schedule (or just manually for the trial)
  - A failure here doesn't break the continuation breakout scanner
  - The Telegram message format can be visually distinct
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import pandas as pd

from utils.indicators import atr, rsi
from utils.pivots import detect_choch, ChochResult

log = logging.getLogger(__name__)


# --- Tuning constants. Trial defaults. Easy to bump later. ---
PIVOT_WINDOW = 5             # N bars on each side to confirm a swing pivot
ATR_MULTIPLIER = 0.3         # how far past the pivot the close must be (in ATR)
MIN_BARS = 60                # need enough history to find pivots and classify trend
MIN_TURNOVER_INR_CR = 50.0   # skip illiquid names; ₹50 cr/day is a soft floor


def scan_one(ticker: str, df: pd.DataFrame) -> dict[str, Any] | None:
    """
    Scan a single ticker. Returns a signal dict if ChoCH is detected, else None.

    Expected df columns: Open, High, Low, Close, Volume.
    NOTE: utils.fetcher.download_bulk already drops the unclosed bar — we don't
    repeat that work here.
    """
    if df is None or df.empty or len(df) < MIN_BARS:
        return None

    # Sanity check for required columns
    required = {"High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        log.debug("%s: missing required columns, got %s", ticker, list(df.columns))
        return None

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    atr14 = atr(high, low, close, length=14)
    last_atr = atr14.iloc[-1]
    if pd.isna(last_atr) or last_atr <= 0:
        return None

    # Liquidity filter — skip thinly traded names.
    avg_turnover_cr = float((close * volume).rolling(20).mean().iloc[-1] / 1e7)
    if pd.isna(avg_turnover_cr) or avg_turnover_cr < MIN_TURNOVER_INR_CR:
        return None

    result: ChochResult = detect_choch(
        high=high,
        low=low,
        close=close,
        atr_series=atr14,
        pivot_window=PIVOT_WINDOW,
        atr_multiplier=ATR_MULTIPLIER,
    )

    if result.direction == "None":
        return None

    # Add some context an analyst would want when reading the alert.
    rsi14 = rsi(close, length=14)
    last_price = float(close.iloc[-1])
    pct_change = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0.0

    broken = result.broken_pivot
    return {
        "ticker": ticker,
        "direction": result.direction,           # "Bullish" or "Bearish"
        "prior_trend": result.prior_trend,
        "price": last_price,
        "pct_change": pct_change,
        "break_level": float(broken.price) if broken else None,
        "break_level_date": str(broken.bar_date.date()) if broken else None,
        "break_strength_atr": round(result.break_strength_atr, 2),
        "rsi": round(float(rsi14.iloc[-1]), 1) if pd.notna(rsi14.iloc[-1]) else None,
        "avg_turnover_cr": round(avg_turnover_cr, 1),
        "pivot_chain": [asdict(p) | {"bar_date": str(p.bar_date.date())} for p in result.pivot_chain],
    }


def scan_watchlist(price_data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """
    Scan all tickers. `price_data` is a dict {ticker: ohlcv_dataframe}, the same
    shape returned by utils.fetcher.download_bulk.
    """
    signals: list[dict[str, Any]] = []
    for ticker, df in price_data.items():
        try:
            sig = scan_one(ticker, df)
            if sig:
                signals.append(sig)
                log.info("ChoCH detected: %s %s", ticker, sig["direction"])
        except Exception as e:
            log.exception("ChoCH scan failed for %s: %s", ticker, e)
    return signals


def format_telegram_message(signal: dict[str, Any]) -> str:
    """Visually distinct from the continuation scanner's message format."""
    direction_emoji = "🟢" if signal["direction"] == "Bullish" else "🔴"
    arrow = "↗" if signal["direction"] == "Bullish" else "↘"
    pivot_kind = "high" if signal["direction"] == "Bullish" else "low"
    return (
        f"🔄 ChoCH Alert | Daily\n\n"
        f"{direction_emoji} {signal['ticker']}  |  ₹{signal['price']:.2f} ({signal['pct_change']:+.2f}%)\n"
        f"🎯 Direction      : {signal['direction']} {arrow} "
        f"({signal['prior_trend']} → reversal)\n"
        f"📍 Broke level    : ₹{signal['break_level']:.2f}  "
        f"(swing {pivot_kind} on {signal['break_level_date']})\n"
        f"📏 Break strength : {signal['break_strength_atr']}× ATR past level\n"
        f"📊 RSI Daily      : {signal['rsi']}\n"
        f"💧 Turnover       : ₹{signal['avg_turnover_cr']} cr/day\n"
        f"\nFirst reversal signal after a sustained "
        f"{signal['prior_trend'].lower()}. Watch for follow-through."
    )
