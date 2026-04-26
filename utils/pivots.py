"""
Swing pivots, market structure, and Change-of-Character (ChoCH) detection.
Pure functions, no I/O, no side effects — same philosophy as utils/indicators.py.

Definitions used here:
- A swing high at bar i: high[i] is the maximum within [i-N, i+N].
- A swing low at bar i: low[i] is the minimum within [i-N, i+N].
- Trend state is inferred from the last two confirmed swing highs and swing lows:
    Uptrend   = HH and HL  (higher high AND higher low)
    Downtrend = LH and LL  (lower high  AND lower low)
    Range     = anything else
- ChoCH = first close beyond the most recent opposing swing point, by an
  ATR-scaled threshold, AGAINST the prevailing trend.
    Bullish ChoCH: prior trend was Down, latest close > last swing high + k*ATR
    Bearish ChoCH: prior trend was Up,   latest close < last swing low  - k*ATR
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

TrendState = Literal["Uptrend", "Downtrend", "Range"]
ChochDirection = Literal["Bullish", "Bearish", "None"]


@dataclass
class Pivot:
    """A single confirmed swing point."""
    index: int          # positional index into the dataframe
    bar_date: pd.Timestamp
    price: float
    kind: Literal["high", "low"]


@dataclass
class ChochResult:
    """Output of detect_choch — null-safe, easy to serialize."""
    direction: ChochDirection
    prior_trend: TrendState
    broken_pivot: Optional[Pivot]   # the swing point that was breached
    break_price: float              # close price that broke the level
    break_strength_atr: float       # how far past the level, in ATR multiples
    pivot_chain: list[Pivot]        # last few pivots, useful for charting/debug
    bars_ago: int = 0               # how many bars ago the ChoCH closed (0 = today)


def find_pivots(
    high: pd.Series,
    low: pd.Series,
    window: int = 5,
) -> list[Pivot]:
    """
    Find all confirmed swing highs and lows in chronological order.

    A pivot is "confirmed" only if it has `window` bars on BOTH sides — so the
    most recent `window` bars cannot contain a confirmed pivot. This avoids
    look-ahead bias, matching the philosophy already in this codebase.
    """
    if len(high) != len(low):
        raise ValueError("high and low must be the same length")
    if len(high) < 2 * window + 1:
        return []

    h = high.values
    l = low.values
    idx = high.index
    pivots: list[Pivot] = []

    for i in range(window, len(high) - window):
        win_h = h[i - window : i + window + 1]
        win_l = l[i - window : i + window + 1]
        if h[i] == win_h.max():
            pivots.append(Pivot(index=i, bar_date=idx[i], price=float(h[i]), kind="high"))
        if l[i] == win_l.min():
            pivots.append(Pivot(index=i, bar_date=idx[i], price=float(l[i]), kind="low"))

    pivots.sort(key=lambda p: p.index)
    return pivots


def classify_trend_at(pivots: list[Pivot], up_to_bar: int) -> TrendState:
    """
    Classify trend using only pivots that occurred at or before `up_to_bar`.

    Used to determine 'what was the prevailing trend before this ChoCH event'.
    """
    relevant = [p for p in pivots if p.index <= up_to_bar]
    highs = [p for p in relevant if p.kind == "high"]
    lows = [p for p in relevant if p.kind == "low"]

    if len(highs) < 2 or len(lows) < 2:
        return "Range"

    h_prev, h_last = highs[-2], highs[-1]
    l_prev, l_last = lows[-2], lows[-1]

    higher_high = h_last.price > h_prev.price
    higher_low = l_last.price > l_prev.price
    lower_high = h_last.price < h_prev.price
    lower_low = l_last.price < l_prev.price

    if higher_high and higher_low:
        return "Uptrend"
    if lower_high and lower_low:
        return "Downtrend"
    return "Range"


def find_recent_choch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_series: pd.Series,
    pivot_window: int = 5,
    atr_multiplier: float = 0.3,
    lookback_bars: int = 30,
) -> ChochResult:
    """
    Walk backwards through the last `lookback_bars` to find the most recent
    bar where a ChoCH event occurred (a close that broke an opposing swing
    pivot in the prevailing trend's opposite direction).

    Returns the most recent ChoCH if one exists, else direction="None".
    Unlike detect_choch (which only checks the last bar), this gives us the
    structural reversal even if it happened a couple weeks ago — important
    because the Fib retracement entry doesn't usually happen the same day.
    """
    null_result = ChochResult(
        direction="None",
        prior_trend="Range",
        broken_pivot=None,
        break_price=float("nan"),
        break_strength_atr=0.0,
        pivot_chain=[],
        bars_ago=0,
    )

    n = len(close)
    if n < 2 * pivot_window + 5:
        return null_result

    pivots = find_pivots(high, low, window=pivot_window)
    if not pivots:
        return null_result

    # Search from most recent bar backwards.
    start_bar = max(2 * pivot_window, n - lookback_bars)

    for bar_idx in range(n - 1, start_bar - 1, -1):
        # Only consider pivots that were CONFIRMED before this bar.
        # A pivot at index p is confirmed at bar p + pivot_window.
        confirmed_pivots = [p for p in pivots if p.index + pivot_window <= bar_idx]
        if len(confirmed_pivots) < 4:
            continue

        # Trend at this point in time, using only pivots confirmed by then.
        trend = classify_trend_at(confirmed_pivots, up_to_bar=bar_idx - 1)
        if trend == "Range":
            continue

        bar_close = float(close.iloc[bar_idx])
        bar_atr = float(atr_series.iloc[bar_idx]) if pd.notna(atr_series.iloc[bar_idx]) else 0.0
        if bar_atr <= 0:
            continue

        threshold = atr_multiplier * bar_atr

        if trend == "Uptrend":
            # Bearish ChoCH: close below most recent swing low - k*ATR.
            recent_lows = [p for p in confirmed_pivots if p.kind == "low"]
            if recent_lows:
                target = recent_lows[-1]
                if bar_close < (target.price - threshold):
                    # Verify this is the FIRST bar in the sequence to break — avoid
                    # firing repeatedly on every bar after the initial break.
                    prev_close = float(close.iloc[bar_idx - 1]) if bar_idx > 0 else float("inf")
                    if prev_close >= (target.price - threshold):
                        return ChochResult(
                            direction="Bearish",
                            prior_trend=trend,
                            broken_pivot=target,
                            break_price=bar_close,
                            break_strength_atr=(target.price - bar_close) / bar_atr,
                            pivot_chain=confirmed_pivots[-6:],
                            bars_ago=n - 1 - bar_idx,
                        )

        elif trend == "Downtrend":
            # Bullish ChoCH: close above most recent swing high + k*ATR.
            recent_highs = [p for p in confirmed_pivots if p.kind == "high"]
            if recent_highs:
                target = recent_highs[-1]
                if bar_close > (target.price + threshold):
                    prev_close = float(close.iloc[bar_idx - 1]) if bar_idx > 0 else float("-inf")
                    if prev_close <= (target.price + threshold):
                        return ChochResult(
                            direction="Bullish",
                            prior_trend=trend,
                            broken_pivot=target,
                            break_price=bar_close,
                            break_strength_atr=(bar_close - target.price) / bar_atr,
                            pivot_chain=confirmed_pivots[-6:],
                            bars_ago=n - 1 - bar_idx,
                        )

    return null_result


def find_anchor_low_for_bullish_choch(
    high: pd.Series,
    low: pd.Series,
    pivots: list[Pivot],
    choch_bar: int,
    max_lookback: int = 90,
) -> Optional[tuple[float, pd.Timestamp]]:
    """
    Find the structural low that anchors a Bullish ChoCH Fib draw.

    Logic: a Bullish ChoCH happens when price breaks above the most recent
    swing high. The "leg" we want to Fib is the move FROM the bottom of
    the prior downtrend TO the ChoCH break. That bottom is the lowest bar
    between the SECOND-to-last swing high (where the latest leg-down began)
    and the ChoCH break bar.

    Why not just take the lowest swing low? Because that picks deep historical
    lows from many months ago, far outside the current move structure.

    Returns (price, date) or None if no valid anchor found.
    """
    # Find the swing high BEFORE the one that just got broken — this marks
    # the start of the latest downleg.
    highs_before = [
        p for p in pivots
        if p.kind == "high" and p.index < choch_bar
    ]
    if len(highs_before) < 2:
        # Not enough structure; fall back to a simple lookback window
        leg_start = max(0, choch_bar - max_lookback)
    else:
        # Use the second-to-last swing high as the start of the downleg.
        # (The last swing high is the one that just got broken — we want to
        # look for the low that came AFTER the previous high, BEFORE the broken one.)
        prior_high = highs_before[-2]
        leg_start = prior_high.index
        # Cap the lookback in case the structure is weird
        leg_start = max(leg_start, choch_bar - max_lookback)

    # Within [leg_start, choch_bar], find the lowest bar.
    if leg_start >= choch_bar:
        return None

    leg_lows = low.iloc[leg_start : choch_bar + 1]
    if leg_lows.empty:
        return None

    anchor_price = float(leg_lows.min())
    anchor_date = leg_lows.idxmin()
    return anchor_price, anchor_date


def find_anchor_high_for_bearish_choch(
    high: pd.Series,
    low: pd.Series,
    pivots: list[Pivot],
    choch_bar: int,
    max_lookback: int = 90,
) -> Optional[tuple[float, pd.Timestamp]]:
    """
    Mirror of find_anchor_low_for_bullish_choch.

    For a Bearish ChoCH, the relevant high is the top of the upleg that
    just got reversed — i.e., the highest bar between the previous swing
    low and the ChoCH break.
    """
    lows_before = [
        p for p in pivots
        if p.kind == "low" and p.index < choch_bar
    ]
    if len(lows_before) < 2:
        leg_start = max(0, choch_bar - max_lookback)
    else:
        prior_low = lows_before[-2]
        leg_start = prior_low.index
        leg_start = max(leg_start, choch_bar - max_lookback)

    if leg_start >= choch_bar:
        return None

    leg_highs = high.iloc[leg_start : choch_bar + 1]
    if leg_highs.empty:
        return None

    anchor_price = float(leg_highs.max())
    anchor_date = leg_highs.idxmax()
    return anchor_price, anchor_date


# --- Deprecated old functions kept as no-ops to avoid import errors ---
# (The new functions above replace these.)
def find_anchor_low_before(pivots: list[Pivot], up_to_bar: int) -> Optional[Pivot]:
    """DEPRECATED — use find_anchor_low_for_bullish_choch instead."""
    relevant_lows = [p for p in pivots if p.kind == "low" and p.index <= up_to_bar]
    if not relevant_lows:
        return None
    return min(relevant_lows, key=lambda p: p.price)


def find_anchor_high_before(pivots: list[Pivot], up_to_bar: int) -> Optional[Pivot]:
    """DEPRECATED — use find_anchor_high_for_bearish_choch instead."""
    relevant_highs = [p for p in pivots if p.kind == "high" and p.index <= up_to_bar]
    if not relevant_highs:
        return None
    return max(relevant_highs, key=lambda p: p.price)


# Kept for backward compatibility — same as old detect_choch (last-bar only)
def detect_choch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_series: pd.Series,
    pivot_window: int = 5,
    atr_multiplier: float = 0.3,
) -> ChochResult:
    """Detect ChoCH on the most recent bar only. See find_recent_choch for windowed version."""
    return find_recent_choch(
        high, low, close, atr_series,
        pivot_window=pivot_window,
        atr_multiplier=atr_multiplier,
        lookback_bars=1,
    )


def classify_trend(pivots: list[Pivot]) -> TrendState:
    """Trend over the entire pivot history. Convenience wrapper."""
    if not pivots:
        return "Range"
    return classify_trend_at(pivots, up_to_bar=pivots[-1].index)

