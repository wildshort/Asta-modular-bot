"""
utils/chart_builder.py
======================

Curated chart builder for the Asta modular bot scanner.

Replaces the old chart-drawing code with a curation-first approach:

- Detects trend regime (uptrend / downtrend / range)
- Picks ONE hero trendline aligned with the regime
- Validates the label contextually (support must be below price, etc.)
- Constrains line extension (no infinite projections into empty space)
- Marks breakouts/breakdowns visually with vertical bands and annotations
- Falls back gracefully to "no line drawn" if no qualifying line exists
- Uses volume panel (with breakout bar highlighted) instead of MACD overlay

Hero-line selection rules:

  Regime      | Signal type          | Hero line
  ------------|----------------------|------------------------------------------
  Uptrend     | Bullish breakout     | Horizontal resistance just broken
  Uptrend     | Bullish continuation | Rising support trendline through pivots
  Downtrend   | Bullish reversal     | Descending resistance trendline broken
  Downtrend   | Bearish continuation | Descending resistance still intact
  Range       | Bullish breakout     | Horizontal range top (just broken)
  Range       | Bearish breakdown    | Horizontal range bottom (just broken)
  Any         | (no qualifying line) | Skip; show only EMAs + 20D edge ticks

Color convention:
  - Green bold line: bullish-relevant level (rising support, broken resistance
    in a bullish reversal/breakout)
  - Red bold line: bearish-relevant level (broken support in breakdown,
    intact descending resistance in downtrend)

Usage from main.py:

    from utils.chart_builder import build_chart

    chart_path = build_chart(
        df=ohlcv_dataframe,           # DataFrame with OHLCV columns
        symbol="EXIDEIND.NS",
        last_price=363.45,
        pct_change=2.02,
        score=70,
        direction="bullish",          # "bullish" | "bearish" | "neutral"
        signal_meta={                  # optional: extra info to display
            "rsi_d": 72.6,
            "rsi_w": 69.3,
            "adx": 30.6,
            "vol_ratio": 0.6,
            "tl_breakout": True,        # True if a TL breakout fired
            "twentyD_breakout": True,
        },
        out_dir="/tmp",                # where to save the PNG
    )

Returns the absolute path to the saved PNG file.

Assumptions:
- df has columns: 'Open', 'High', 'Low', 'Close', 'Volume'
- df is sorted oldest -> newest, datetime index
- Latest bar is df.iloc[-1] and is a CLOSED bar (intraday-bar handling
  should be done upstream per existing convention)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyArrowPatch

# -- Indicator math: import from existing utils.indicators where possible.
# We provide local fallbacks so this module is portable if imports change.
try:
    from utils.indicators import ema as _ema_external  # type: ignore
    _HAS_EXTERNAL_EMA = True
except Exception:
    _HAS_EXTERNAL_EMA = False

try:
    from utils.indicators import atr as _atr_external  # type: ignore
    _HAS_EXTERNAL_ATR = True
except Exception:
    _HAS_EXTERNAL_ATR = False


# ============================================================================
# Math helpers (with fallbacks)
# ============================================================================

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    if _HAS_EXTERNAL_EMA:
        try:
            return np.asarray(_ema_external(pd.Series(arr), period))
        except Exception:
            pass
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    if _HAS_EXTERNAL_ATR:
        try:
            df_local = pd.DataFrame({"High": high, "Low": low, "Close": close})
            return np.asarray(_atr_external(df_local, period))
        except Exception:
            pass
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    out = np.empty_like(tr, dtype=float)
    out[: period] = np.nan
    if len(tr) >= period:
        out[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


# ============================================================================
# Pivot detection
# ============================================================================

def _find_pivots(arr: np.ndarray, lookback: int = 5, kind: str = "high") -> list[int]:
    """Return indices of pivot highs (or lows) using a symmetric N-bar window."""
    pivots: list[int] = []
    for i in range(lookback, len(arr) - lookback):
        window = arr[i - lookback : i + lookback + 1]
        if kind == "high" and arr[i] == window.max():
            pivots.append(i)
        elif kind == "low" and arr[i] == window.min():
            pivots.append(i)
    return pivots


# ============================================================================
# Trend regime detection
# ============================================================================

def _detect_regime(closes: np.ndarray, ema50: np.ndarray) -> str:
    """
    Classify the recent market structure as 'uptrend', 'downtrend', or 'range'.

    Heuristic: slope of EMA50 over the last ~30 bars + price position.
    """
    n = len(closes)
    if n < 60:
        return "range"

    recent_ema = ema50[-30:]
    slope = (recent_ema[-1] - recent_ema[0]) / max(abs(recent_ema[0]), 1e-9)

    last_price = closes[-1]
    last_ema = ema50[-1]
    above_ema = last_price > last_ema

    # Slope thresholds: ~3% over 30 bars = clear trend
    if slope > 0.03 and above_ema:
        return "uptrend"
    if slope < -0.03 and not above_ema:
        return "downtrend"
    return "range"


# ============================================================================
# Trendline candidate fitting
# ============================================================================

def _fit_line(p1_idx: int, p1_val: float, p2_idx: int, p2_val: float):
    """Return slope, intercept for a line through two points."""
    if p2_idx == p1_idx:
        return 0.0, p1_val
    slope = (p2_val - p1_val) / (p2_idx - p1_idx)
    intercept = p1_val - slope * p1_idx
    return slope, intercept


def _line_y(x: float, slope: float, intercept: float) -> float:
    return slope * x + intercept


def _count_touches(
    pivots: list[int],
    pivot_vals: np.ndarray,
    slope: float,
    intercept: float,
    tolerance: float,
) -> list[int]:
    """Return indices of pivots that lie within `tolerance` of the line."""
    return [p for p in pivots if abs(pivot_vals[p] - _line_y(p, slope, intercept)) <= tolerance]


def _best_diagonal_line(
    pivots: list[int],
    pivot_vals: np.ndarray,
    n_bars: int,
    atr_recent: float,
    min_span_bars: int = 60,
    min_touches: int = 3,
    last_touch_within: int = 40,
):
    """
    Search for the best diagonal trendline through the given pivots.

    Returns dict with keys: slope, intercept, touches (list of indices),
    span, last_touch, score. Or None if no qualifying line found.

    Scoring prefers:
      1. Longer span (longer-term lines weighted higher per user preference)
      2. More touches
      3. More recent last touch
    """
    if len(pivots) < 2:
        return None

    tolerance = max(atr_recent * 0.6, 1e-6)
    best = None

    # Try every pair as the line definition; check how many other pivots touch
    for i in range(len(pivots)):
        for j in range(i + 1, len(pivots)):
            p1, p2 = pivots[i], pivots[j]
            span = p2 - p1
            if span < min_span_bars:
                continue
            slope, intercept = _fit_line(p1, pivot_vals[p1], p2, pivot_vals[p2])
            touches = _count_touches(pivots, pivot_vals, slope, intercept, tolerance)
            if len(touches) < min_touches:
                continue
            last_touch = touches[-1]
            if (n_bars - 1) - last_touch > last_touch_within:
                continue

            # Score: span dominates (per user preference), then touches, then recency
            score = span * 1.0 + len(touches) * 8.0 - ((n_bars - 1) - last_touch) * 0.5

            if best is None or score > best["score"]:
                best = {
                    "slope": slope,
                    "intercept": intercept,
                    "touches": touches,
                    "span": span,
                    "last_touch": last_touch,
                    "score": score,
                    "first_touch": touches[0],
                }
    return best


def _best_horizontal_level(
    pivots: list[int],
    pivot_vals: np.ndarray,
    n_bars: int,
    atr_recent: float,
    min_span_bars: int = 60,
    min_touches: int = 3,
    last_touch_within: int = 40,
):
    """
    Find the most-respected horizontal level among the given pivots.

    Clusters pivots by price within ATR tolerance, returns the cluster with
    the longest span and most touches.
    """
    if len(pivots) < 2:
        return None

    tolerance = max(atr_recent * 0.6, 1e-6)
    best = None

    for i, p_anchor in enumerate(pivots):
        level = pivot_vals[p_anchor]
        cluster = [p for p in pivots if abs(pivot_vals[p] - level) <= tolerance]
        if len(cluster) < min_touches:
            continue
        span = cluster[-1] - cluster[0]
        if span < min_span_bars:
            continue
        last_touch = cluster[-1]
        if (n_bars - 1) - last_touch > last_touch_within:
            continue

        # Use mean of cluster values for a smoother level
        avg_level = float(np.mean([pivot_vals[p] for p in cluster]))
        score = span * 1.0 + len(cluster) * 8.0 - ((n_bars - 1) - last_touch) * 0.5

        if best is None or score > best["score"]:
            best = {
                "level": avg_level,
                "touches": cluster,
                "span": span,
                "last_touch": last_touch,
                "first_touch": cluster[0],
                "score": score,
            }
    return best


# ============================================================================
# Hero-line selection
# ============================================================================

def _select_hero_line(
    df: pd.DataFrame,
    regime: str,
    direction: str,
    signal_meta: dict,
):
    """
    Decide which line (if any) to draw on this chart, given the regime and signal.

    Returns a dict describing the line to draw, or None.

    Dict shape:
      {
        "kind": "horizontal" | "diagonal",
        "role": "support" | "resistance",
        "broken": bool,                     # True if signal is a breakout/breakdown
        "breakout_bar": int | None,         # index of the breakout bar if any
        "color": "#hex",
        "slope": float, "intercept": float, # for diagonal
        "level": float,                     # for horizontal
        "touches": list[int],
        "first_touch": int, "last_touch": int,
        "span": int,
        "label": str,                        # legend label
      }
    """
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    n = len(df)

    atr_arr = _atr(highs, lows, closes, period=14)
    atr_recent = float(np.nanmean(atr_arr[-20:])) if n >= 20 else float(np.nanmean(atr_arr))
    if not np.isfinite(atr_recent) or atr_recent <= 0:
        atr_recent = float(np.std(closes[-20:])) if n >= 20 else 1.0

    pivot_highs = _find_pivots(highs, lookback=5, kind="high")
    pivot_lows = _find_pivots(lows, lookback=5, kind="low")

    is_bullish = direction == "bullish"
    tl_breakout = bool(signal_meta.get("tl_breakout", False))

    # ---------------- Decision logic ----------------
    # 1. Bullish breakout signal: prefer the level/line that just broke
    if is_bullish and tl_breakout:
        # Try diagonal descending resistance first (downtrend reversal case)
        diag = _best_diagonal_line(pivot_highs, highs, n, atr_recent)
        if diag and diag["slope"] < 0:
            breakout_bar = _find_breakout_bar(closes, diag["slope"], diag["intercept"], "above")
            if breakout_bar is not None:
                return {
                    "kind": "diagonal",
                    "role": "resistance",
                    "broken": True,
                    "breakout_bar": breakout_bar,
                    "color": "#2e7d32",  # green: bullish-broken resistance
                    "slope": diag["slope"],
                    "intercept": diag["intercept"],
                    "touches": diag["touches"],
                    "first_touch": diag["first_touch"],
                    "last_touch": diag["last_touch"],
                    "span": diag["span"],
                    "label": f"Resistance broken ({len(diag['touches'])}t, {diag['span']} bars)",
                }
        # Fallback: horizontal resistance just broken (range-breakout case)
        horiz = _best_horizontal_level(pivot_highs, highs, n, atr_recent)
        if horiz:
            breakout_bar = _find_horizontal_break(closes, horiz["level"], "above")
            if breakout_bar is not None:
                return {
                    "kind": "horizontal",
                    "role": "resistance",
                    "broken": True,
                    "breakout_bar": breakout_bar,
                    "color": "#2e7d32",
                    "level": horiz["level"],
                    "touches": horiz["touches"],
                    "first_touch": horiz["first_touch"],
                    "last_touch": horiz["last_touch"],
                    "span": horiz["span"],
                    "label": f"Resistance broken ({len(horiz['touches'])}t, {horiz['span']} bars)",
                }

    # 2. Bearish breakdown: support that just broke
    if not is_bullish and tl_breakout:
        diag = _best_diagonal_line(pivot_lows, lows, n, atr_recent)
        if diag and diag["slope"] > 0:
            breakout_bar = _find_breakout_bar(closes, diag["slope"], diag["intercept"], "below")
            if breakout_bar is not None:
                return {
                    "kind": "diagonal",
                    "role": "support",
                    "broken": True,
                    "breakout_bar": breakout_bar,
                    "color": "#c62828",  # red: bearish-broken support
                    "slope": diag["slope"],
                    "intercept": diag["intercept"],
                    "touches": diag["touches"],
                    "first_touch": diag["first_touch"],
                    "last_touch": diag["last_touch"],
                    "span": diag["span"],
                    "label": f"Support broken ({len(diag['touches'])}t, {diag['span']} bars)",
                }
        horiz = _best_horizontal_level(pivot_lows, lows, n, atr_recent)
        if horiz:
            breakout_bar = _find_horizontal_break(closes, horiz["level"], "below")
            if breakout_bar is not None:
                return {
                    "kind": "horizontal",
                    "role": "support",
                    "broken": True,
                    "breakout_bar": breakout_bar,
                    "color": "#c62828",
                    "level": horiz["level"],
                    "touches": horiz["touches"],
                    "first_touch": horiz["first_touch"],
                    "last_touch": horiz["last_touch"],
                    "span": horiz["span"],
                    "label": f"Support broken ({len(horiz['touches'])}t, {horiz['span']} bars)",
                }

    # 3. Bullish continuation in uptrend: rising support (intact)
    if is_bullish and regime == "uptrend":
        diag = _best_diagonal_line(pivot_lows, lows, n, atr_recent)
        if diag and diag["slope"] > 0:
            # Validate: line must be currently below price
            current_line_y = _line_y(n - 1, diag["slope"], diag["intercept"])
            if closes[-1] > current_line_y:
                return {
                    "kind": "diagonal",
                    "role": "support",
                    "broken": False,
                    "breakout_bar": None,
                    "color": "#2e7d32",  # green: intact bullish support
                    "slope": diag["slope"],
                    "intercept": diag["intercept"],
                    "touches": diag["touches"],
                    "first_touch": diag["first_touch"],
                    "last_touch": diag["last_touch"],
                    "span": diag["span"],
                    "label": f"Rising support ({len(diag['touches'])}t, {diag['span']} bars)",
                }

    # 4. Bearish continuation in downtrend: descending resistance (intact)
    if not is_bullish and regime == "downtrend":
        diag = _best_diagonal_line(pivot_highs, highs, n, atr_recent)
        if diag and diag["slope"] < 0:
            current_line_y = _line_y(n - 1, diag["slope"], diag["intercept"])
            if closes[-1] < current_line_y:
                return {
                    "kind": "diagonal",
                    "role": "resistance",
                    "broken": False,
                    "breakout_bar": None,
                    "color": "#c62828",  # red: intact bearish resistance
                    "slope": diag["slope"],
                    "intercept": diag["intercept"],
                    "touches": diag["touches"],
                    "first_touch": diag["first_touch"],
                    "last_touch": diag["last_touch"],
                    "span": diag["span"],
                    "label": f"Falling resistance ({len(diag['touches'])}t, {diag['span']} bars)",
                }

    # 5. Nothing qualifies: return None -> chart shows no trendline
    return None


def _find_breakout_bar(closes, slope, intercept, direction: str) -> Optional[int]:
    """First bar in the last 40 bars where close crosses the line in given direction."""
    n = len(closes)
    start = max(1, n - 40)
    for i in range(start, n):
        prev_y = _line_y(i - 1, slope, intercept)
        curr_y = _line_y(i, slope, intercept)
        if direction == "above" and closes[i] > curr_y and closes[i - 1] <= prev_y:
            return i
        if direction == "below" and closes[i] < curr_y and closes[i - 1] >= prev_y:
            return i
    return None


def _find_horizontal_break(closes, level, direction: str) -> Optional[int]:
    n = len(closes)
    start = max(1, n - 40)
    for i in range(start, n):
        if direction == "above" and closes[i] > level and closes[i - 1] <= level:
            return i
        if direction == "below" and closes[i] < level and closes[i - 1] >= level:
            return i
    return None


# ============================================================================
# Drawing
# ============================================================================

def _draw_candles(ax, opens, highs, lows, closes):
    n = len(closes)
    for i in range(n):
        c = "#26a69a" if closes[i] >= opens[i] else "#ef5350"
        ax.plot([i, i], [lows[i], highs[i]], color=c, linewidth=0.6, alpha=0.9)
        bl = min(opens[i], closes[i])
        bh = max(opens[i], closes[i])
        ax.add_patch(Rectangle(
            (i - 0.35, bl), 0.7, bh - bl,
            facecolor=c, edgecolor=c, linewidth=0.5, alpha=0.95
        ))


def _draw_volume(ax, opens, closes, volumes, breakout_bar: Optional[int]):
    n = len(closes)
    avg = pd.Series(volumes).rolling(20).mean()
    cols = ["#26a69a" if closes[i] >= opens[i] else "#ef5350" for i in range(n)]
    ax.bar(range(n), volumes, color=cols, alpha=0.6, width=0.7)
    ax.plot(range(n), avg, color="#424242", linewidth=0.9, alpha=0.7, label="20D avg")
    if breakout_bar is not None and 0 <= breakout_bar < n:
        ax.bar(
            [breakout_bar], [volumes[breakout_bar]],
            color="#1b5e20", alpha=0.95, width=0.7,
            edgecolor="#1b5e20", linewidth=1.2,
        )
        if not np.isnan(avg.iloc[breakout_bar]) and avg.iloc[breakout_bar] > 0:
            ratio = volumes[breakout_bar] / avg.iloc[breakout_bar]
            ax.text(
                breakout_bar, volumes[breakout_bar] * 1.05,
                f"{ratio:.1f}x",
                fontsize=8, ha="center", fontweight="bold", color="#1b5e20",
            )
    ax.set_ylabel("Volume", fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)


def _draw_hero_line(ax, hero, n, closes):
    """Draw the chosen hero line on the price axis."""
    if hero is None:
        return

    color = hero["color"]
    line_start = hero["first_touch"]

    if hero["broken"] and hero["breakout_bar"] is not None:
        line_end = hero["breakout_bar"]
        faded_end = min(n - 1, hero["breakout_bar"] + 8)
    else:
        line_end = n - 1
        faded_end = min(n - 1, n + 5)  # slight forward extension

    if hero["kind"] == "diagonal":
        slope, intercept = hero["slope"], hero["intercept"]
        ax.plot(
            [line_start, line_end],
            [_line_y(line_start, slope, intercept), _line_y(line_end, slope, intercept)],
            color=color, linewidth=2.8, alpha=0.95, label=hero["label"], zorder=5,
        )
        if hero["broken"]:
            ax.plot(
                [line_end, faded_end],
                [_line_y(line_end, slope, intercept), _line_y(faded_end, slope, intercept)],
                color=color, linewidth=1.2, alpha=0.4, linestyle="--", zorder=4,
            )
        # Touch markers
        highs_or_lows = None
    else:  # horizontal
        level = hero["level"]
        ax.plot(
            [line_start, line_end], [level, level],
            color=color, linewidth=2.8, alpha=0.95, label=hero["label"], zorder=5,
        )
        if hero["broken"]:
            ax.plot(
                [line_end, faded_end], [level, level],
                color=color, linewidth=1.2, alpha=0.4, linestyle="--", zorder=4,
            )

    # Touch markers (using actual pivot values)
    for p in hero["touches"]:
        if hero["kind"] == "diagonal":
            y = _line_y(p, hero["slope"], hero["intercept"])
        else:
            y = hero["level"]
        ax.scatter(p, y, s=45, color=color, edgecolor="white", linewidth=1.2, zorder=6)


def _draw_breakout_marker(ax, hero, closes, n):
    if hero is None or not hero["broken"] or hero["breakout_bar"] is None:
        return
    bb = hero["breakout_bar"]
    role = hero["role"]
    is_bullish_break = role == "resistance"  # broken resistance = bullish

    band_color = "#26a69a" if is_bullish_break else "#ef5350"
    band_alpha = 0.15 if is_bullish_break else 0.18
    ax.axvspan(bb - 0.5, bb + 0.5, color=band_color, alpha=band_alpha, zorder=1)

    if is_bullish_break:
        # Arrow from below-left
        text_x = bb - 8
        text_y = closes[bb] - max(closes) * 0.04
        ann_color = "#1b5e20"
        ann_face = "#e8f5e9"
        label = f"Breakout\n@ ₹{closes[bb]:.0f}"
        va = "top"
        arrow_start = (bb - 8, closes[bb] - max(closes) * 0.025)
        arrow_end = (bb - 0.5, closes[bb] - max(closes) * 0.005)
    else:
        # Arrow from above-left
        text_x = bb - 8
        text_y = closes[bb] + max(closes) * 0.04
        ann_color = "#b71c1c"
        ann_face = "#ffebee"
        label = f"Breakdown\n@ ₹{closes[bb]:.0f}"
        va = "bottom"
        arrow_start = (bb - 8, closes[bb] + max(closes) * 0.025)
        arrow_end = (bb - 0.5, closes[bb] + max(closes) * 0.005)

    arrow = FancyArrowPatch(
        arrow_start, arrow_end,
        arrowstyle="->", color=ann_color, linewidth=1.8,
        mutation_scale=18, zorder=7,
    )
    ax.add_patch(arrow)
    ax.text(
        text_x, text_y, label,
        fontsize=9, fontweight="bold", color=ann_color,
        ha="right", va=va,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=ann_face,
                  edgecolor=ann_color, linewidth=0.8),
    )


def _draw_edge_ticks(ax, highs, lows, n, direction):
    """Small 20D high/low markers on the right edge instead of full-width lines."""
    if n >= 20:
        h20 = max(highs[-20:])
        l20 = min(lows[-20:])
        # 20D high
        ax.plot([n - 1, n + 2], [h20, h20], color="#9e9e9e",
                linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(n + 2.5, h20, "20D H", fontsize=7, color="#757575", va="center")
        # 20D low (always show for completeness; especially for bearish)
        ax.plot([n - 1, n + 2], [l20, l20], color="#9e9e9e",
                linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(n + 2.5, l20, "20D L", fontsize=7, color="#757575", va="center")


# ============================================================================
# Main entry point
# ============================================================================

def build_chart(
    df: pd.DataFrame,
    symbol: str,
    last_price: float,
    pct_change: float,
    score: int,
    direction: str,                 # "bullish" | "bearish" | "neutral"
    signal_meta: Optional[dict] = None,
    out_dir: str = "/tmp",
) -> str:
    """
    Build a curated chart and save it as PNG. Returns the absolute path.

    See module docstring for parameter details.
    """
    if signal_meta is None:
        signal_meta = {}

    # Defensive: required columns
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_chart: missing columns {missing}")

    # Reset to integer-indexed for plotting
    df = df.copy()
    if not isinstance(df.index, pd.RangeIndex):
        df_dates = df.index
    else:
        df_dates = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq="B")

    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    volumes = df["Volume"].to_numpy(dtype=float)
    n = len(df)

    if n < 30:
        # Not enough data to do any meaningful curation; bail out with simple chart
        return _build_simple_fallback_chart(
            df_dates, opens, highs, lows, closes, volumes,
            symbol, last_price, pct_change, score, direction, out_dir
        )

    # ---- Indicators ----
    ema5 = _ema(closes, 5)
    ema50 = _ema(closes, 50) if n >= 50 else _ema(closes, max(5, n // 4))

    # ---- Regime + hero line ----
    regime = _detect_regime(closes, ema50)
    hero = _select_hero_line(df, regime, direction, signal_meta)

    # ---- Build figure ----
    fig, (ax_p, ax_v) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={"height_ratios": [4, 1]}, sharex=True,
    )
    fig.patch.set_facecolor("white")

    _draw_candles(ax_p, opens, highs, lows, closes)
    ax_p.plot(range(n), ema5, color="#5c6bc0", linewidth=0.9, alpha=0.5, label="EMA 5")
    ax_p.plot(range(n), ema50, color="#ff9800", linewidth=1.2, alpha=0.6, label="EMA 50")

    _draw_hero_line(ax_p, hero, n, closes)
    _draw_breakout_marker(ax_p, hero, closes, n)
    _draw_edge_ticks(ax_p, highs, lows, n, direction)

    # No-line annotation
    if hero is None:
        ax_p.text(
            0.99, 0.02,
            "No qualifying trendline (60+ bar span, 3+ touches, alive)",
            transform=ax_p.transAxes, fontsize=8.5, color="#757575",
            ha="right", va="bottom", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fafafa",
                      edgecolor="#bdbdbd", linewidth=0.6),
        )

    # Volume panel
    breakout_bar = hero["breakout_bar"] if (hero and hero["broken"]) else None
    _draw_volume(ax_v, opens, closes, volumes, breakout_bar)

    # ---- Title ----
    if direction == "bullish":
        title_color = "#1b5e20"
        dir_emoji = "📈"
    elif direction == "bearish":
        title_color = "#b71c1c"
        dir_emoji = "📉"
    else:
        title_color = "#616161"
        dir_emoji = "➡"

    suffix = ""
    if hero and hero["broken"]:
        if hero["role"] == "resistance":
            suffix = f"  |  TL BREAKOUT ({hero['span']}-bar resistance)"
        else:
            suffix = f"  |  BREAKDOWN ({hero['span']}-bar support)"
    elif hero and not hero["broken"]:
        suffix = f"  |  {hero['label']} intact"
    else:
        suffix = "  |  No clear structure"

    title = (
        f"{symbol}  ₹{last_price:.2f} ({pct_change:+.2f}%)  "
        f"|  {dir_emoji} {direction.upper()} (Score {score}/100){suffix}"
    )
    ax_p.set_title(title, fontsize=12, fontweight="bold", color=title_color, pad=12)

    ax_p.legend(loc="upper left", fontsize=9, frameon=False)
    ax_p.grid(True, alpha=0.2)
    ax_p.set_ylabel("Price (₹)", fontsize=10)
    ax_p.set_xlim(-2, n + 8)

    # x-axis dates
    tick_positions = list(range(0, n, max(1, n // 10)))
    try:
        tick_labels = [pd.Timestamp(df_dates[i]).strftime("%b %d") for i in tick_positions]
    except Exception:
        tick_labels = [str(i) for i in tick_positions]
    ax_v.set_xticks(tick_positions)
    ax_v.set_xticklabels(tick_labels, rotation=0, fontsize=8)

    plt.tight_layout()

    # Save
    safe_symbol = symbol.replace("/", "_").replace(".", "_")
    out_path = os.path.join(out_dir, f"chart_{safe_symbol}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _build_simple_fallback_chart(
    df_dates, opens, highs, lows, closes, volumes,
    symbol, last_price, pct_change, score, direction, out_dir,
) -> str:
    """Minimal chart for stocks with too little data for curation logic."""
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    _draw_candles(ax, opens, highs, lows, closes)
    title_color = "#1b5e20" if direction == "bullish" else "#b71c1c" if direction == "bearish" else "#616161"
    ax.set_title(
        f"{symbol}  ₹{last_price:.2f} ({pct_change:+.2f}%)  |  {direction.upper()} "
        f"(Score {score}/100)  |  Insufficient data for curation",
        fontsize=12, fontweight="bold", color=title_color, pad=12,
    )
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("Price (₹)", fontsize=10)
    plt.tight_layout()
    safe_symbol = symbol.replace("/", "_").replace(".", "_")
    out_path = os.path.join(out_dir, f"chart_{safe_symbol}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
