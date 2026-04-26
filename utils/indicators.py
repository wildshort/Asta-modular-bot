"""
Technical indicators. Pure functions, no I/O, no side effects.
Isolating these makes them testable and reusable in the charting module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI using exponential smoothing (standard definition)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Standard MACD. Returns (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def bollinger(close: pd.Series, window: int = 20, mult: float = 2.0):
    """Returns (mid, upper, lower)."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range — measures typical volatility per bar."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder's ADX. Pure implementation (no ta dependency on this path)
    so we avoid 'squeeze' issues with odd-shaped inputs from yfinance.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    atr_series = tr.ewm(alpha=1 / length, min_periods=length).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / length, min_periods=length
    ).mean() / atr_series
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / length, min_periods=length
    ).mean() / atr_series

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, min_periods=length).mean()


def ema_crossover(fast: pd.Series, slow: pd.Series) -> tuple[bool, bool]:
    """Returns (just_crossed_up, just_crossed_down) on the last bar."""
    if len(fast) < 2 or len(slow) < 2:
        return False, False
    f_prev, f_now = fast.iloc[-2], fast.iloc[-1]
    s_prev, s_now = slow.iloc[-2], slow.iloc[-1]
    if pd.isna([f_prev, f_now, s_prev, s_now]).any():
        return False, False
    bull = (f_prev <= s_prev) and (f_now > s_now)
    bear = (f_prev >= s_prev) and (f_now < s_now)
    return bull, bear


def rsi_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 30) -> str:
    """
    Detect classic RSI divergence.

    Bullish divergence: price makes a lower low, RSI makes a higher low.
    Bearish divergence: price makes a higher high, RSI makes a lower high.

    We look at the two most recent swing pivots in the last `lookback` bars.
    This replaces the old stub that always returned "None".
    """
    if len(close) < lookback + 5 or len(rsi_series) < lookback + 5:
        return "None"

    price = close.iloc[-lookback:].values
    r = rsi_series.iloc[-lookback:].values

    # Find simple swing lows/highs: bar that is min/max within a 5-bar window
    def find_pivots(series: np.ndarray, kind: str, window: int = 5):
        pivots = []
        for i in range(window, len(series) - window):
            w = series[i - window : i + window + 1]
            if kind == "low" and series[i] == w.min():
                pivots.append(i)
            elif kind == "high" and series[i] == w.max():
                pivots.append(i)
        return pivots

    low_pivots = find_pivots(price, "low")
    high_pivots = find_pivots(price, "high")

    # Bullish: last two price lows descending, RSI lows ascending
    if len(low_pivots) >= 2:
        p1, p2 = low_pivots[-2], low_pivots[-1]
        if price[p2] < price[p1] and r[p2] > r[p1]:
            return "Bullish"

    # Bearish: last two price highs ascending, RSI highs descending
    if len(high_pivots) >= 2:
        p1, p2 = high_pivots[-2], high_pivots[-1]
        if price[p2] > price[p1] and r[p2] < r[p1]:
            return "Bearish"

    return "None"


def bb_position(close: pd.Series, upper: pd.Series, lower: pd.Series) -> str:
    """Returns 'Upper', 'Lower', or 'Inside'."""
    c = close.iloc[-1]
    u = upper.iloc[-1]
    l = lower.iloc[-1]
    if pd.isna([c, u, l]).any():
        return "Inside"
    if c > u:
        return "Upper"
    if c < l:
        return "Lower"
    return "Inside"


def trend_breakout(close: pd.Series, window: int = 20) -> str:
    """Donchian-style breakout: did price close above/below the prior N-bar range?"""
    if len(close) < window + 2:
        return "None"
    prior_high = close.rolling(window).max().iloc[-2]
    prior_low = close.rolling(window).min().iloc[-2]
    last = close.iloc[-1]
    if pd.isna([prior_high, prior_low, last]).any():
        return "None"
    if last > prior_high:
        return "Bullish"
    if last < prior_low:
        return "Bearish"
    return "None"


def macd_cross(macd_line: pd.Series, signal_line: pd.Series) -> tuple[str, bool]:
    """
    Returns (state, is_above) where:
      state   = "PCO" (positive cross / above) or "NCO" (negative / below)
      is_above = macd > signal on current bar
    """
    if len(macd_line) < 2:
        return "NCO", False
    m_prev, m_now = macd_line.iloc[-2], macd_line.iloc[-1]
    s_prev, s_now = signal_line.iloc[-2], signal_line.iloc[-1]
    above_now = m_now > s_now
    return ("PCO" if above_now else "NCO"), bool(above_now)


def volume_spike(volume: pd.Series, mult: float = 1.8, window: int = 20) -> tuple[bool, float]:
    """Returns (is_spike, ratio_vs_average)."""
    if len(volume) < window:
        return False, 0.0
    avg = volume.rolling(window).mean().iloc[-1]
    cur = volume.iloc[-1]
    if pd.isna(avg) or avg <= 0:
        return False, 0.0
    ratio = float(cur / avg)
    return ratio >= mult, ratio


def avg_turnover_inr(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    """Rough estimate of average daily turnover in INR."""
    if len(close) < window:
        return 0.0
    turnover = (close * volume).rolling(window).mean().iloc[-1]
    return float(turnover) if pd.notna(turnover) else 0.0
