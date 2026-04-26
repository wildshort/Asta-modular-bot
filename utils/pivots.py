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
        # Strict equality on the center is fine; ties on flat tops are rare on daily
        # and a duplicate pivot at the same price doesn't break the trend logic.
        if h[i] == win_h.max():
            pivots.append(Pivot(index=i, bar_date=idx[i], price=float(h[i]), kind="high"))
        if l[i] == win_l.min():
            pivots.append(Pivot(index=i, bar_date=idx[i], price=float(l[i]), kind="low"))

    # Sort by bar index (a bar can be both a high pivot and low pivot in odd cases;
    # sort is stable so order among same-index pivots is preserved).
    pivots.sort(key=lambda p: p.index)
    return pivots


def classify_trend(pivots: list[Pivot]) -> TrendState:
    """
    Classify trend from the most recent two highs and two lows.

    Requires at least 2 highs AND 2 lows in the pivot list. If the pivots
    are too sparse, returns 'Range' (we don't guess).
    """
    highs = [p for p in pivots if p.kind == "high"]
    lows = [p for p in pivots if p.kind == "low"]

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


def detect_choch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_series: pd.Series,
    pivot_window: int = 5,
    atr_multiplier: float = 0.3,
) -> ChochResult:
    """
    Detect a Change-of-Character on the most recent bar.

    Args:
        high, low, close: OHLC series for one ticker. Must be aligned.
        atr_series:        ATR series (same length and index as close).
        pivot_window:      Bars on each side required to confirm a pivot.
        atr_multiplier:    How far past the swing the close must be, in ATR units.
                           0.3 = "small but meaningful" — filters tiny wicks.

    Returns ChochResult with direction='None' if no ChoCH is present, or
    'Bullish'/'Bearish' if the most recent bar broke an opposing pivot.
    """
    null_result = ChochResult(
        direction="None",
        prior_trend="Range",
        broken_pivot=None,
        break_price=float("nan"),
        break_strength_atr=0.0,
        pivot_chain=[],
    )

    if len(close) < 2 * pivot_window + 5:
        return null_result

    pivots = find_pivots(high, low, window=pivot_window)
    if not pivots:
        return null_result

    trend = classify_trend(pivots)
    last_close = float(close.iloc[-1])
    last_atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    if last_atr <= 0:
        return null_result

    threshold = atr_multiplier * last_atr

    if trend == "Uptrend":
        # Look for a bearish ChoCH: close below the most recent swing low - k*ATR.
        recent_lows = [p for p in pivots if p.kind == "low"]
        if recent_lows:
            target = recent_lows[-1]
            if last_close < (target.price - threshold):
                return ChochResult(
                    direction="Bearish",
                    prior_trend=trend,
                    broken_pivot=target,
                    break_price=last_close,
                    break_strength_atr=(target.price - last_close) / last_atr,
                    pivot_chain=pivots[-6:],
                )

    elif trend == "Downtrend":
        # Look for a bullish ChoCH: close above the most recent swing high + k*ATR.
        recent_highs = [p for p in pivots if p.kind == "high"]
        if recent_highs:
            target = recent_highs[-1]
            if last_close > (target.price + threshold):
                return ChochResult(
                    direction="Bullish",
                    prior_trend=trend,
                    broken_pivot=target,
                    break_price=last_close,
                    break_strength_atr=(last_close - target.price) / last_atr,
                    pivot_chain=pivots[-6:],
                )

    # Range or no break — return null result with the trend filled in for context.
    return ChochResult(
        direction="None",
        prior_trend=trend,
        broken_pivot=None,
        break_price=last_close,
        break_strength_atr=0.0,
        pivot_chain=pivots[-6:],
    )
