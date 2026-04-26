"""
ChoCH + Fib scanner — finds setups where:
  1. A Change-of-Character has occurred within the last 30 trading days, AND
  2. Price is currently retracing into the 38.2%–61.8% Fibonacci zone
     of the move from the structural low (or high) to the ChoCH break point.

This is significantly more selective than ChoCH alone — most days zero alerts,
occasional days 1-3 high-quality entry setups.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from utils.indicators import atr, rsi
from utils.pivots import (
    find_pivots,
    find_recent_choch,
    find_anchor_low_for_bullish_choch,
    find_anchor_high_for_bearish_choch,
    ChochResult,
    Pivot,
)

log = logging.getLogger(__name__)


# --- Tuning constants. Trial defaults. ---
PIVOT_WINDOW = 5             # bars on each side to confirm a swing pivot
ATR_MULTIPLIER = 0.3         # ATR-scaled threshold for ChoCH break
CHOCH_LOOKBACK_BARS = 30     # only consider ChoCH events from last 30 bars
MIN_BARS = 80                # need enough history (lookback + buffer)
MIN_TURNOVER_INR_CR = 50.0   # liquidity filter

# Fib zone: 38.2% to 61.8% retracement = "golden pocket"
FIB_ZONE_MIN = 0.382
FIB_ZONE_MAX = 0.618
FIB_LEVELS = [0.382, 0.500, 0.618]   # the levels we report in the message


def _fib_price(low: float, high: float, ratio: float, direction: str) -> float:
    """
    Given an anchor low and high, return the price at a given retracement ratio.

    For a Bullish ChoCH (price rallied from low to high, now retracing down):
        50% retracement = midpoint between low and high (price coming down from high)
        ratio=0 means at the high, ratio=1 means at the low.

    For a Bearish ChoCH (price fell from high to low, now retracing up):
        50% retracement = midpoint (price coming up from low)
        ratio=0 means at the low, ratio=1 means at the high.
    """
    if direction == "Bullish":
        return high - (high - low) * ratio
    else:  # Bearish
        return low + (high - low) * ratio


def _current_retracement_pct(price: float, low: float, high: float, direction: str) -> float:
    """How far has price retraced, as a fraction (0.0 to 1.0+)."""
    move = high - low
    if move <= 0:
        return 0.0
    if direction == "Bullish":
        return (high - price) / move
    else:
        return (price - low) / move


def scan_one(ticker: str, df: pd.DataFrame) -> Optional[dict[str, Any]]:
    """
    Scan a single ticker for a ChoCH + Fib entry setup.
    Returns signal dict if conditions met, else None.
    """
    if df is None or df.empty or len(df) < MIN_BARS:
        return None

    required = {"High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        log.debug("%s: missing columns %s", ticker, list(df.columns))
        return None

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    atr14 = atr(high, low, close, length=14)
    last_atr = atr14.iloc[-1]
    if pd.isna(last_atr) or last_atr <= 0:
        return None

    avg_turnover_cr = float((close * volume).rolling(20).mean().iloc[-1] / 1e7)
    if pd.isna(avg_turnover_cr) or avg_turnover_cr < MIN_TURNOVER_INR_CR:
        return None

    # Step 1: find a recent ChoCH event in the last 30 bars
    choch: ChochResult = find_recent_choch(
        high=high,
        low=low,
        close=close,
        atr_series=atr14,
        pivot_window=PIVOT_WINDOW,
        atr_multiplier=ATR_MULTIPLIER,
        lookback_bars=CHOCH_LOOKBACK_BARS,
    )

    if choch.direction == "None" or choch.broken_pivot is None:
        return None

    n = len(close)
    choch_bar_idx = n - 1 - choch.bars_ago

    # Step 2: identify Fib anchors
    all_pivots = find_pivots(high, low, window=PIVOT_WINDOW)

    if choch.direction == "Bullish":
        # Bullish ChoCH = was downtrend, broke up through swing high.
        # Anchor low = lowest BAR (not just pivot) within the prior downleg
        # Anchor high = highest high reached AFTER the ChoCH break (rally peak)
        anchor_low_result = find_anchor_low_for_bullish_choch(
            high=high, low=low, pivots=all_pivots, choch_bar=choch_bar_idx,
        )
        if anchor_low_result is None:
            return None
        anchor_low_price, anchor_low_date = anchor_low_result

        post_choch_highs = high.iloc[choch_bar_idx:]
        anchor_high_price = float(post_choch_highs.max())
        anchor_high_date = post_choch_highs.idxmax()

    else:  # Bearish
        anchor_high_result = find_anchor_high_for_bearish_choch(
            high=high, low=low, pivots=all_pivots, choch_bar=choch_bar_idx,
        )
        if anchor_high_result is None:
            return None
        anchor_high_price, anchor_high_date = anchor_high_result

        post_choch_lows = low.iloc[choch_bar_idx:]
        anchor_low_price = float(post_choch_lows.min())
        anchor_low_date = post_choch_lows.idxmin()

    # Sanity: ensure we have a real move to retrace from
    move_size = anchor_high_price - anchor_low_price
    if move_size <= 0 or move_size < float(last_atr) * 2:
        return None  # Move too small to be meaningful

    # Step 3: failed-ChoCH filter
    last_price = float(close.iloc[-1])
    if choch.direction == "Bullish":
        # If price has fallen back below the anchor low, the reversal failed
        if last_price < anchor_low_price:
            return None
    else:
        # If price has rallied back above the anchor high, the reversal failed
        if last_price > anchor_high_price:
            return None

    # Step 4: is current price in the 38.2%–61.8% retracement zone?
    retracement_pct = _current_retracement_pct(
        last_price, anchor_low_price, anchor_high_price, choch.direction
    )

    if not (FIB_ZONE_MIN <= retracement_pct <= FIB_ZONE_MAX):
        return None

    # We have a setup. Compute Fib levels for the message.
    fib_prices = {
        f"{int(r * 1000) / 10}%": _fib_price(
            anchor_low_price, anchor_high_price, r, choch.direction
        )
        for r in FIB_LEVELS
    }

    # Invalidation level
    if choch.direction == "Bullish":
        invalidation = anchor_low_price
    else:
        invalidation = anchor_high_price

    # Context indicators
    rsi14 = rsi(close, length=14)
    pct_change = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0.0

    choch_break_bar_date = str(close.index[choch_bar_idx].date())

    return {
        "ticker": ticker,
        "direction": choch.direction,
        "prior_trend": choch.prior_trend,
        "price": last_price,
        "pct_change": pct_change,
        "choch_break_level": float(choch.broken_pivot.price),
        "choch_pivot_date": str(choch.broken_pivot.bar_date.date()),  # date the pivot itself was set
        "choch_break_date": choch_break_bar_date,                      # date the break happened
        "choch_bars_ago": choch.bars_ago,
        "anchor_low": round(anchor_low_price, 2),
        "anchor_low_date": str(pd.Timestamp(anchor_low_date).date()),
        "anchor_high": round(anchor_high_price, 2),
        "anchor_high_date": str(pd.Timestamp(anchor_high_date).date()),
        "current_retracement_pct": round(retracement_pct * 100, 1),
        "fib_levels": {k: round(v, 2) for k, v in fib_prices.items()},
        "invalidation": round(invalidation, 2),
        "rsi": round(float(rsi14.iloc[-1]), 1) if pd.notna(rsi14.iloc[-1]) else None,
        "avg_turnover_cr": round(avg_turnover_cr, 1),
    }


def scan_watchlist(price_data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Scan all tickers. Returns list of qualifying setups."""
    signals: list[dict[str, Any]] = []
    for ticker, df in price_data.items():
        try:
            sig = scan_one(ticker, df)
            if sig:
                signals.append(sig)
                log.info(
                    "ChoCH+Fib setup: %s %s @ %.1f%% retracement",
                    ticker, sig["direction"], sig["current_retracement_pct"],
                )
        except Exception as e:
            log.exception("Scan failed for %s: %s", ticker, e)
    return signals


def format_telegram_message(sig: dict[str, Any]) -> str:
    """Detailed entry-setup message with Fib levels and invalidation."""
    direction = sig["direction"]
    emoji = "🟢" if direction == "Bullish" else "🔴"
    arrow = "↗" if direction == "Bullish" else "↘"
    side = "Long" if direction == "Bullish" else "Short"

    fib = sig["fib_levels"]
    fib_keys = sorted(fib.keys(), key=lambda k: float(k.rstrip("%")))

    # Mark which Fib level is closest to current price
    cur_pct = sig["current_retracement_pct"]
    closest_key = min(fib_keys, key=lambda k: abs(float(k.rstrip("%")) - cur_pct))

    fib_lines = []
    for k in fib_keys:
        marker = " ← current" if k == closest_key else ""
        fib_lines.append(f"  • {k} retracement: ₹{fib[k]:.2f}{marker}")
    fib_block = "\n".join(fib_lines)

    return (
        f"🎯 ChoCH + Fib | {sig['ticker']}\n\n"
        f"{emoji} Direction      : {direction} {arrow} ({side} setup)\n"
        f"💰 Current price   : ₹{sig['price']:.2f} ({sig['pct_change']:+.2f}%)\n"
        f"📍 In Fib zone at  : {cur_pct}% retracement\n\n"
        f"🔄 ChoCH event     : {sig['choch_bars_ago']} bars ago ({sig['choch_break_date']})\n"
        f"   Broke ₹{sig['choch_break_level']:.2f} (pivot set {sig['choch_pivot_date']})\n\n"
        f"📐 Move structure:\n"
        f"   Low  ₹{sig['anchor_low']:.2f} ({sig['anchor_low_date']})\n"
        f"   High ₹{sig['anchor_high']:.2f} ({sig['anchor_high_date']})\n\n"
        f"💡 Fib levels:\n"
        f"{fib_block}\n\n"
        f"⛔ Invalidation    : ₹{sig['invalidation']:.2f}\n"
        f"📊 RSI Daily       : {sig['rsi']}\n"
        f"💧 Turnover        : ₹{sig['avg_turnover_cr']} cr/day\n"
        f"\nNot financial advice. Confirm structure on chart."
    )
