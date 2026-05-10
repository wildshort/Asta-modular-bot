"""
utils/chart_builder.py
======================

Curated chart builder — v3 (3-panel layout, approach-direction validated touches).

Layout:
  - Price 70%   : candles + Bollinger Bands (filled cyan) + EMAs faded +
                  hero line (bold) + optional secondary line (dotted) +
                  breakout marker + divergence arrow when applicable
  - Volume 15%  : volume bars colored by candle direction, breakout bar highlighted
  - RSI 15%     : RSI(14) with reference lines at 30/50/70, divergence trendlines

Trendline rules:
  - Both horizontal and diagonal lines are evaluated
  - HORIZONTAL is preferred as hero (per user preference)
  - A diagonal can become the hero only if it has noticeably more touches
  - Secondary line (dotted) is drawn only if it's MEANINGFULLY DIFFERENT from
    the hero (slope steep enough, not just a tilted version of the hero)
  - For wick vs body fitting: tries both and picks the cleaner result

Approach-direction validation (the key rule):
  - Resistance touches only count if price approached FROM BELOW
    (the 5 bars before the pivot must all be below the line value)
  - Support touches only count if price approached FROM ABOVE
  - Lines are drawn only from the first VALID touch onwards
  - Touches during a "run-through" (price passing the level) don't count

Selection priority for hero:
  1. Fresh breakout/breakdown (within last 3 bars) — that line wins
  2. Intact line — must be fresh (last touch within 30 bars)
  3. Otherwise: no line drawn

Title: SYMBOL ₹PRICE (±X%) | EVENT(span)   — no score per user request

Public API:
  build_chart_bytes(df, symbol, last_price, pct_change, direction, signal_meta) -> bytes
  diagnose_curation(df, direction, signal_meta) -> dict
  classify_chart(df, direction, signal_meta) -> dict   # used by alerts/telegram.py
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyArrowPatch


# ============================================================================
# Math helpers
# ============================================================================

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _atr(high, low, close, period=14):
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    out = np.empty_like(tr, dtype=float)
    out[: period] = np.nan
    if len(tr) >= period:
        out[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    series = pd.Series(closes)
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()


def _bollinger(closes, period=20, num_std=2):
    s = pd.Series(closes)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid.to_numpy(), upper.to_numpy(), lower.to_numpy()


# ============================================================================
# Pivot detection
# ============================================================================

def _find_pivots(arr: np.ndarray, lookback: int = 5, kind: str = "high") -> list[int]:
    """Symmetric N-bar pivot detection."""
    pivots = []
    for i in range(lookback, len(arr) - lookback):
        window = arr[i - lookback : i + lookback + 1]
        if kind == "high" and arr[i] == window.max():
            pivots.append(i)
        elif kind == "low" and arr[i] == window.min():
            pivots.append(i)
    return pivots


# ============================================================================
# Approach-direction validation
# ============================================================================

def _approach_check(
    pivot_idx: int,
    line_value_at_pivot: float,
    closes: np.ndarray,
    role: str,
    n_check: int = 5,
) -> bool:
    """
    A touch is valid only if price approached the line from the correct side.

    Resistance: 5 bars before pivot must all be BELOW the line value at that bar
    Support:    5 bars before pivot must all be ABOVE the line value at that bar

    This eliminates "run-through" touches where price was passing the level
    rather than respecting it as resistance/support.
    """
    if pivot_idx < n_check:
        return False
    for j in range(1, n_check + 1):
        prior_close = closes[pivot_idx - j]
        if role == "resistance" and prior_close >= line_value_at_pivot:
            return False
        if role == "support" and prior_close <= line_value_at_pivot:
            return False
    return True


def _validate_horizontal_touches(
    candidate_pivots: list[int],
    pivot_vals: np.ndarray,
    level: float,
    closes: np.ndarray,
    role: str,
    tolerance: float,
) -> list[int]:
    """Filter candidate pivots: keep only those that pass approach-direction."""
    valid = []
    for p in candidate_pivots:
        if abs(pivot_vals[p] - level) > tolerance:
            continue
        if _approach_check(p, level, closes, role):
            valid.append(p)
    return valid


def _validate_diagonal_touches(
    candidate_pivots: list[int],
    pivot_vals: np.ndarray,
    slope: float,
    intercept: float,
    closes: np.ndarray,
    role: str,
    tolerance: float,
) -> list[int]:
    """Same as horizontal but the line value varies by bar."""
    valid = []
    for p in candidate_pivots:
        line_y = slope * p + intercept
        if abs(pivot_vals[p] - line_y) > tolerance:
            continue
        if _approach_check(p, line_y, closes, role):
            valid.append(p)
    return valid


# ============================================================================
# Line candidate fitting (with wick + body, pick cleaner)
# ============================================================================

def _best_horizontal_level(
    pivot_highs: list[int],
    pivot_lows: list[int],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    role: str,                # 'resistance' or 'support'
    atr_recent: float,
    n: int,
    min_touches: int = 3,
    min_span_bars: int = 60,
    last_touch_within: int = 40,
) -> Optional[dict]:
    """
    Find the most-respected horizontal level for the given role.

    Tries fitting against:
      - WICKS (high/low values)
      - BODIES (max/min of open/close)
    Picks whichever produces the cleaner (more-touches, longer-span) result.

    Returns dict: { level, touches, first_touch, last_touch, span, score }
    or None if no qualifying level found.
    """
    if role == "resistance":
        pivots = pivot_highs
        wick_vals = highs
        # Body high = max(open, close). We don't have open here directly, but
        # for body fitting we approximate using close (close-only is a valid alt).
    else:
        pivots = pivot_lows
        wick_vals = lows

    if len(pivots) < min_touches:
        return None

    tolerance = max(atr_recent * 0.6, 1e-6)
    best = None

    # Try fitting against wicks AND closes (close ≈ body extreme)
    for vals_label, vals in [("wick", wick_vals), ("close", closes)]:
        for anchor in pivots:
            level = float(vals[anchor])
            # Find candidate pivots within tolerance of this level
            candidates = [p for p in pivots if abs(vals[p] - level) <= tolerance]
            if len(candidates) < min_touches:
                continue
            # Validate approach direction
            valid = _validate_horizontal_touches(
                candidates, vals, level, closes, role, tolerance
            )
            if len(valid) < min_touches:
                continue
            span = valid[-1] - valid[0]
            if span < min_span_bars:
                continue
            last_touch = valid[-1]
            if (n - 1) - last_touch > last_touch_within:
                continue
            # Smooth level using mean of validated touches
            avg_level = float(np.mean([vals[p] for p in valid]))
            # Score
            score = len(valid) * 8 + span * 1.0 - ((n - 1) - last_touch) * 0.5
            candidate = {
                "level": avg_level,
                "touches": valid,
                "first_touch": valid[0],
                "last_touch": last_touch,
                "span": span,
                "score": score,
                "fit_method": vals_label,
            }
            if best is None or score > best["score"]:
                best = candidate
    return best


def _best_diagonal_line(
    pivot_highs: list[int],
    pivot_lows: list[int],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    role: str,                # 'resistance' (descending) or 'support' (rising)
    atr_recent: float,
    n: int,
    min_touches: int = 3,
    min_span_bars: int = 60,
    last_touch_within: int = 40,
) -> Optional[dict]:
    """
    Find the best diagonal line for the given role.
    Resistance must be descending (slope < 0).
    Support must be rising (slope > 0).
    """
    if role == "resistance":
        pivots = pivot_highs
        wick_vals = highs
    else:
        pivots = pivot_lows
        wick_vals = lows

    if len(pivots) < 2:
        return None

    tolerance = max(atr_recent * 0.6, 1e-6)
    best = None

    for vals_label, vals in [("wick", wick_vals), ("close", closes)]:
        for i in range(len(pivots)):
            for j in range(i + 1, len(pivots)):
                p1, p2 = pivots[i], pivots[j]
                if p2 - p1 < min_span_bars:
                    continue
                slope = (vals[p2] - vals[p1]) / (p2 - p1)
                if role == "resistance" and slope >= 0:
                    continue
                if role == "support" and slope <= 0:
                    continue
                intercept = vals[p1] - slope * p1
                # Validate touches
                valid = _validate_diagonal_touches(
                    pivots, vals, slope, intercept, closes, role, tolerance
                )
                if len(valid) < min_touches:
                    continue
                span = valid[-1] - valid[0]
                if span < min_span_bars:
                    continue
                last_touch = valid[-1]
                if (n - 1) - last_touch > last_touch_within:
                    continue
                score = len(valid) * 8 + span * 1.0 - ((n - 1) - last_touch) * 0.5
                candidate = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "touches": valid,
                    "first_touch": valid[0],
                    "last_touch": last_touch,
                    "span": span,
                    "score": score,
                    "fit_method": vals_label,
                }
                if best is None or score > best["score"]:
                    best = candidate
    return best


def _line_y(x: float, slope: float, intercept: float) -> float:
    return slope * x + intercept


# ============================================================================
# Breakout / breakdown detection (freshness)
# ============================================================================

def _find_horizontal_break(closes, level, direction: str, lookback: int = 3) -> Optional[int]:
    """direction: 'above' (bullish breakout) or 'below' (bearish breakdown)"""
    n = len(closes)
    start = max(1, n - lookback)
    for i in range(start, n):
        if direction == "above" and closes[i] > level and closes[i - 1] <= level:
            return i
        if direction == "below" and closes[i] < level and closes[i - 1] >= level:
            return i
    return None


def _find_diagonal_break(closes, slope, intercept, direction: str, lookback: int = 3) -> Optional[int]:
    n = len(closes)
    start = max(1, n - lookback)
    for i in range(start, n):
        prev_y = _line_y(i - 1, slope, intercept)
        curr_y = _line_y(i, slope, intercept)
        if direction == "above" and closes[i] > curr_y and closes[i - 1] <= prev_y:
            return i
        if direction == "below" and closes[i] < curr_y and closes[i - 1] >= prev_y:
            return i
    return None


# ============================================================================
# RSI divergence detection (visual)
# ============================================================================

def _detect_rsi_divergence(
    closes: np.ndarray,
    rsi_vals: np.ndarray,
    pivot_lookback: int = 5,
    window: int = 60,
) -> Optional[dict]:
    """
    Look for divergence in the most recent `window` bars.

    Bullish divergence: price makes a LOWER low, RSI makes a HIGHER low
    Bearish divergence: price makes a HIGHER high, RSI makes a LOWER high

    Returns dict with bar indices and type, or None.
    """
    n = len(closes)
    start = max(0, n - window)

    # Use raw arrays restricted to recent window
    price_lows = _find_pivots(closes[start:], lookback=pivot_lookback, kind="low")
    price_highs = _find_pivots(closes[start:], lookback=pivot_lookback, kind="high")
    rsi_lows = _find_pivots(rsi_vals[start:], lookback=pivot_lookback, kind="low")
    rsi_highs = _find_pivots(rsi_vals[start:], lookback=pivot_lookback, kind="high")

    # Need at least 2 pivots to compare
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        # Take last two price lows
        p1, p2 = price_lows[-2], price_lows[-1]
        # Find nearest RSI lows to the price lows
        # Simpler: just compare last 2 of each list
        r1, r2 = rsi_lows[-2], rsi_lows[-1]
        if closes[start + p2] < closes[start + p1] and rsi_vals[start + r2] > rsi_vals[start + r1]:
            return {
                "type": "bullish",
                "price_pivot1": start + p1,
                "price_pivot2": start + p2,
                "rsi_pivot1": start + r1,
                "rsi_pivot2": start + r2,
            }

    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        p1, p2 = price_highs[-2], price_highs[-1]
        r1, r2 = rsi_highs[-2], rsi_highs[-1]
        if closes[start + p2] > closes[start + p1] and rsi_vals[start + r2] < rsi_vals[start + r1]:
            return {
                "type": "bearish",
                "price_pivot1": start + p1,
                "price_pivot2": start + p2,
                "rsi_pivot1": start + r1,
                "rsi_pivot2": start + r2,
            }

    return None


# ============================================================================
# Hero/secondary line selection
# ============================================================================

def _is_meaningfully_different(
    line_a_kind: str, line_a: dict,
    line_b_kind: str, line_b: dict,
    n_bars: int,
    atr_recent: float,
) -> bool:
    """
    Two lines are 'meaningfully different' if:
      - They differ in kind (one horizontal, one diagonal), AND
      - Their values diverge by more than 3*ATR over the chart span

    Avoids drawing a barely-tilted diagonal alongside a horizontal at the same level.
    """
    if line_a_kind == line_b_kind:
        return False

    # Sample at start, middle, end of overlap
    def value_at(line_kind, line_dict, x):
        if line_kind == "horizontal":
            return line_dict["level"]
        return line_dict["slope"] * x + line_dict["intercept"]

    samples = [n_bars // 4, n_bars // 2, 3 * n_bars // 4, n_bars - 1]
    max_diff = max(
        abs(value_at(line_a_kind, line_a, x) - value_at(line_b_kind, line_b, x))
        for x in samples
    )
    return max_diff > 3 * atr_recent


def _select_lines(
    df: pd.DataFrame,
    direction: str,
    signal_meta: dict,
):
    """
    Returns (hero, secondary) — each is dict-with-metadata or None.

    Hero comes first in priority:
      1. Fresh breakout (line just broken in last 3 bars)
      2. Intact relevant line (must be alive, validated touches)

    Secondary is drawn only if meaningfully different from hero.
    """
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    n = len(df)

    atr_arr = _atr(highs, lows, closes, period=14)
    atr_recent = float(np.nanmean(atr_arr[-20:])) if n >= 20 else float(np.nanmean(atr_arr))
    if not np.isfinite(atr_recent) or atr_recent <= 0:
        atr_recent = float(np.std(closes[-20:])) if n >= 20 else 1.0

    pivot_highs = _find_pivots(highs, lookback=5, kind="high")
    pivot_lows = _find_pivots(lows, lookback=5, kind="low")

    is_bullish = direction == "bullish"
    role = "resistance" if is_bullish else "support"

    # Find candidates for both kinds
    horiz = _best_horizontal_level(
        pivot_highs, pivot_lows, highs, lows, closes,
        role, atr_recent, n,
    )
    diag = _best_diagonal_line(
        pivot_highs, pivot_lows, highs, lows, closes,
        role, atr_recent, n,
    )

    # Determine which is hero (per user: horizontal preferred, diagonal can win on more touches)
    def _wrap(kind, c):
        if c is None:
            return None
        return {"kind": kind, **c}

    horiz_w = _wrap("horizontal", horiz)
    diag_w = _wrap("diagonal", diag)

    # Check freshness — has either just broken in last 3 bars?
    breakout_bar = None
    breakout_kind = None  # which line just broke
    direction_for_break = "above" if is_bullish else "below"

    if horiz_w:
        h_break = _find_horizontal_break(closes, horiz_w["level"], direction_for_break, lookback=3)
        if h_break is not None:
            breakout_bar = h_break
            breakout_kind = "horizontal"
    if diag_w and breakout_bar is None:
        d_break = _find_diagonal_break(
            closes, diag_w["slope"], diag_w["intercept"], direction_for_break, lookback=3
        )
        if d_break is not None:
            breakout_bar = d_break
            breakout_kind = "diagonal"

    # Decide hero
    hero = None
    secondary = None

    if breakout_bar is not None:
        # Fresh breakout — that line is the hero
        if breakout_kind == "horizontal":
            hero = horiz_w
            hero["broken"] = True
            hero["breakout_bar"] = breakout_bar
        else:
            hero = diag_w
            hero["broken"] = True
            hero["breakout_bar"] = breakout_bar
    else:
        # Intact line — pick hero per priority rule
        # Horizontal preferred, but diagonal can override if it has noticeably more touches
        if horiz_w and diag_w:
            diag_touches = len(diag_w["touches"])
            horiz_touches = len(horiz_w["touches"])
            if diag_touches > horiz_touches + 2:  # diagonal must have 3+ more touches to override
                hero = diag_w
            else:
                hero = horiz_w
        elif horiz_w:
            hero = horiz_w
        elif diag_w:
            hero = diag_w
        if hero:
            hero["broken"] = False
            hero["breakout_bar"] = None

    # Pick secondary if meaningfully different
    if hero and horiz_w and diag_w:
        candidate = diag_w if hero["kind"] == "horizontal" else horiz_w
        if _is_meaningfully_different(
            hero["kind"], hero, candidate["kind"], candidate, n, atr_recent
        ):
            secondary = candidate
            secondary["broken"] = False
            secondary["breakout_bar"] = None

    return hero, secondary, atr_recent


# ============================================================================
# Drawing primitives
# ============================================================================

def _draw_candles(ax, opens, highs, lows, closes):
    n = len(closes)
    for i in range(n):
        c = "#26a69a" if closes[i] >= opens[i] else "#ef5350"
        ax.plot([i, i], [lows[i], highs[i]], color=c, linewidth=0.6, alpha=0.95, zorder=3)
        bl = min(opens[i], closes[i])
        bh = max(opens[i], closes[i])
        ax.add_patch(Rectangle(
            (i - 0.35, bl), 0.7, bh - bl,
            facecolor=c, edgecolor=c, linewidth=0.5, alpha=0.95, zorder=3,
        ))


def _draw_line(ax, line, color_hero, color_secondary, n, is_hero=True):
    if line is None:
        return
    color = color_hero if is_hero else color_secondary
    linewidth = 2.8 if is_hero else 1.3
    alpha = 0.95 if is_hero else 0.6
    linestyle = "-" if is_hero else ":"

    line_start = line["first_touch"]
    if line["broken"] and line.get("breakout_bar") is not None:
        line_end = line["breakout_bar"]
        faded_end = min(n - 1, line["breakout_bar"] + 8)
    else:
        line_end = n - 1
        faded_end = None

    if line["kind"] == "horizontal":
        y = line["level"]
        ax.plot([line_start, line_end], [y, y],
                color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                label=_label_for_line(line, is_hero), zorder=5 if is_hero else 4)
        if faded_end is not None and is_hero:
            ax.plot([line_end, faded_end], [y, y],
                    color=color, linewidth=1.2, alpha=0.4, linestyle="--", zorder=4)
    else:  # diagonal
        s, b = line["slope"], line["intercept"]
        ax.plot([line_start, line_end],
                [_line_y(line_start, s, b), _line_y(line_end, s, b)],
                color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                label=_label_for_line(line, is_hero), zorder=5 if is_hero else 4)
        if faded_end is not None and is_hero:
            ax.plot([line_end, faded_end],
                    [_line_y(line_end, s, b), _line_y(faded_end, s, b)],
                    color=color, linewidth=1.2, alpha=0.4, linestyle="--", zorder=4)

    # Touch markers
    if is_hero:
        marker_size = 40
        edge = "white"
        fill = color
    else:
        marker_size = 25
        edge = color
        fill = "none"
    for p in line["touches"]:
        if line["kind"] == "horizontal":
            y = line["level"]
        else:
            y = _line_y(p, line["slope"], line["intercept"])
        ax.scatter(p, y, s=marker_size, facecolor=fill, edgecolor=edge,
                   linewidth=1.0, zorder=6 if is_hero else 5)


def _label_for_line(line: dict, is_hero: bool) -> str:
    role = "Resistance" if line["color"] == "#2e7d32" or is_hero else "Support"
    # Simpler — use what we know
    role_str = "Resistance" if line.get("role_str") == "resistance" else "Support"
    n_t = len(line["touches"])
    span = line["span"]
    if line["broken"]:
        prefix = f"{role_str} broken"
    else:
        prefix = role_str
    if not is_hero:
        prefix = "Diagonal " + prefix.lower() if line["kind"] == "diagonal" else "Horizontal " + prefix.lower()
        return f"{prefix} (secondary, {n_t}t)"
    return f"{prefix} ({n_t}t, {span} bars)"


def _draw_breakout_marker(ax, hero, closes, n):
    if hero is None or not hero.get("broken") or hero.get("breakout_bar") is None:
        return
    bb = hero["breakout_bar"]
    is_bullish_break = (hero["color"] == "#2e7d32")

    band_color = "#26a69a" if is_bullish_break else "#ef5350"
    ax.axvspan(bb - 0.5, bb + 0.5, color=band_color, alpha=0.18, zorder=1)

    if is_bullish_break:
        text_y = closes[bb] - max(closes) * 0.03
        ann_color = "#1b5e20"
        ann_face = "#e8f5e9"
        label = f"Breakout\n@ ₹{closes[bb]:.0f}"
        va = "top"
        arrow_start = (bb - 7, closes[bb] - max(closes) * 0.02)
        arrow_end = (bb - 0.5, closes[bb] - max(closes) * 0.005)
    else:
        text_y = closes[bb] + max(closes) * 0.03
        ann_color = "#b71c1c"
        ann_face = "#ffebee"
        label = f"Breakdown\n@ ₹{closes[bb]:.0f}"
        va = "bottom"
        arrow_start = (bb - 7, closes[bb] + max(closes) * 0.02)
        arrow_end = (bb - 0.5, closes[bb] + max(closes) * 0.005)

    arrow = FancyArrowPatch(arrow_start, arrow_end,
                            arrowstyle="->", color=ann_color, linewidth=1.8,
                            mutation_scale=18, zorder=7)
    ax.add_patch(arrow)
    ax.text(bb - 7.5, text_y, label,
            fontsize=9, fontweight="bold", color=ann_color,
            ha="right", va=va,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=ann_face,
                      edgecolor=ann_color, linewidth=0.8))


def _draw_volume(ax, opens, closes, volumes, breakout_bar):
    n = len(closes)
    avg = pd.Series(volumes).rolling(20).mean()
    cols = ["#26a69a" if closes[i] >= opens[i] else "#ef5350" for i in range(n)]
    ax.bar(range(n), volumes, color=cols, alpha=0.55, width=0.7)
    ax.plot(range(n), avg, color="#424242", linewidth=0.8, alpha=0.7, label="20D avg")
    if breakout_bar is not None and 0 <= breakout_bar < n:
        ax.bar([breakout_bar], [volumes[breakout_bar]], color="#1b5e20",
               alpha=0.95, width=0.7, edgecolor="#1b5e20", linewidth=1.2)
        if not np.isnan(avg.iloc[breakout_bar]) and avg.iloc[breakout_bar] > 0:
            ratio = volumes[breakout_bar] / avg.iloc[breakout_bar]
            ax.text(breakout_bar, volumes[breakout_bar] * 1.05,
                    f"{ratio:.1f}x",
                    fontsize=8, ha="center", fontweight="bold", color="#1b5e20")
    ax.set_ylabel("Volume", fontsize=9)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.grid(True, alpha=0.15)


def _draw_rsi(ax, rsi_vals, divergence: Optional[dict], n: int):
    ax.plot(range(n), rsi_vals, color="#7e57c2", linewidth=1.0, label="RSI(14)")
    ax.axhline(70, color="#d32f2f", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.axhline(50, color="#9e9e9e", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axhline(30, color="#2e7d32", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI", fontsize=9)

    # Divergence trendline on RSI panel
    if divergence is not None:
        r1, r2 = divergence["rsi_pivot1"], divergence["rsi_pivot2"]
        color = "#1b5e20" if divergence["type"] == "bullish" else "#b71c1c"
        ax.plot([r1, r2], [rsi_vals[r1], rsi_vals[r2]],
                color=color, linewidth=1.4, alpha=0.8)
        ax.scatter([r1, r2], [rsi_vals[r1], rsi_vals[r2]],
                   s=25, color=color, edgecolor="white", linewidth=0.8, zorder=5)

    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.grid(True, alpha=0.15)


def _draw_divergence_arrow_on_price(ax, divergence: Optional[dict], closes: np.ndarray):
    if divergence is None:
        return
    p1, p2 = divergence["price_pivot1"], divergence["price_pivot2"]
    color = "#1b5e20" if divergence["type"] == "bullish" else "#b71c1c"
    label = f"{'Bull' if divergence['type'] == 'bullish' else 'Bear'} div"
    # Connect the two price pivots with a thin colored line
    ax.plot([p1, p2], [closes[p1], closes[p2]],
            color=color, linewidth=1.0, alpha=0.7, linestyle="--", zorder=5)
    # Annotation at the second pivot
    ax.text(p2, closes[p2], f" {label}",
            fontsize=8, color=color, fontweight="bold", va="center")


def _draw_bollinger(ax, mid, upper, lower, n):
    ax.fill_between(range(n), lower, upper, color="#4dd0e1", alpha=0.10, zorder=1)
    ax.plot(range(n), upper, color="#26c6da", linewidth=0.5, alpha=0.5, zorder=2)
    ax.plot(range(n), lower, color="#26c6da", linewidth=0.5, alpha=0.5, zorder=2)


def _draw_edge_ticks(ax, highs, lows, n):
    if n >= 20:
        h20 = max(highs[-20:])
        l20 = min(lows[-20:])
        ax.plot([n - 1, n + 2], [h20, h20], color="#9e9e9e",
                linewidth=0.6, linestyle=":", alpha=0.5)
        ax.text(n + 2.5, h20, "20D H", fontsize=7, color="#757575", va="center")
        ax.plot([n - 1, n + 2], [l20, l20], color="#9e9e9e",
                linewidth=0.6, linestyle=":", alpha=0.5)
        ax.text(n + 2.5, l20, "20D L", fontsize=7, color="#757575", va="center")


# ============================================================================
# Public API
# ============================================================================

def _enrich_line_meta(line: Optional[dict], role: str, is_bullish: bool):
    """Add color and role_str fields to a line dict."""
    if line is None:
        return None
    if is_bullish:
        line["color"] = "#2e7d32"  # green for bullish-relevant
    else:
        line["color"] = "#c62828"  # red for bearish-relevant
    line["role_str"] = role
    return line


def _build_figure(
    df: pd.DataFrame,
    symbol: str,
    last_price: float,
    pct_change: float,
    direction: str,
    signal_meta: Optional[dict] = None,
):
    if signal_meta is None:
        signal_meta = {}

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_chart: missing columns {missing}")

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
        return None

    # Indicators
    ema5 = _ema(closes, 5)
    ema50 = _ema(closes, 50) if n >= 50 else _ema(closes, max(5, n // 4))
    mid_bb, up_bb, lo_bb = _bollinger(closes, 20, 2)
    rsi_vals = _rsi(closes)

    # Lines
    hero, secondary, atr_recent = _select_lines(df, direction, signal_meta)
    is_bullish = direction == "bullish"
    role = "resistance" if is_bullish else "support"
    hero = _enrich_line_meta(hero, role, is_bullish)
    secondary = _enrich_line_meta(secondary, role, is_bullish)

    # Divergence
    divergence = _detect_rsi_divergence(closes, rsi_vals)

    # Build figure with 3 panels: 70/15/15
    fig = plt.figure(figsize=(13, 9), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[70, 15, 15], hspace=0.05)
    ax_p = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_p)
    ax_r = fig.add_subplot(gs[2], sharex=ax_p)

    # Price panel
    _draw_bollinger(ax_p, mid_bb, up_bb, lo_bb, n)
    _draw_candles(ax_p, opens, highs, lows, closes)
    ax_p.plot(range(n), ema5, color="#5c6bc0", linewidth=0.9, alpha=0.55, label="EMA 5", zorder=4)
    ax_p.plot(range(n), ema50, color="#ff9800", linewidth=1.2, alpha=0.65, label="EMA 50", zorder=4)
    if hero:
        _draw_line(ax_p, hero, hero["color"], hero["color"], n, is_hero=True)
    if secondary:
        _draw_line(ax_p, secondary, secondary["color"], secondary["color"], n, is_hero=False)
    _draw_breakout_marker(ax_p, hero, closes, n)
    _draw_divergence_arrow_on_price(ax_p, divergence, closes)
    _draw_edge_ticks(ax_p, highs, lows, n)

    if hero is None:
        ax_p.text(
            0.99, 0.02,
            "No qualifying trendline (3+ validated touches required)",
            transform=ax_p.transAxes, fontsize=8.5, color="#757575",
            ha="right", va="bottom", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fafafa",
                      edgecolor="#bdbdbd", linewidth=0.6),
        )

    # Title — no score
    title_color = "#1b5e20" if direction == "bullish" else "#b71c1c" if direction == "bearish" else "#616161"
    if hero and hero.get("broken"):
        if is_bullish:
            event = f"TL BREAKOUT ({hero['span']}-bar resistance)"
        else:
            event = f"BREAKDOWN ({hero['span']}-bar support)"
    elif hero and not hero.get("broken"):
        if hero["kind"] == "horizontal":
            event = f"{role.title()} ({hero['span']}-bar)"
        else:
            event = f"{'Rising support' if is_bullish else 'Falling resistance'} ({hero['span']}-bar)"
    else:
        event = "No clear structure"
    title = f"{symbol}  ₹{last_price:.2f} ({pct_change:+.2f}%)  |  {event}"
    ax_p.set_title(title, fontsize=12, fontweight="bold", color=title_color, pad=10)

    ax_p.set_ylabel("Price (₹)", fontsize=10)
    ax_p.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax_p.grid(True, alpha=0.2)
    ax_p.set_xlim(-2, n + 8)
    plt.setp(ax_p.get_xticklabels(), visible=False)

    # Volume panel
    breakout_bar = hero["breakout_bar"] if (hero and hero.get("broken")) else None
    _draw_volume(ax_v, opens, closes, volumes, breakout_bar)
    plt.setp(ax_v.get_xticklabels(), visible=False)

    # RSI panel
    _draw_rsi(ax_r, rsi_vals, divergence, n)

    # x-axis ticks on bottom only
    tick_positions = list(range(0, n, max(1, n // 10)))
    try:
        tick_labels = [pd.Timestamp(df_dates[i]).strftime("%b %d") for i in tick_positions]
    except Exception:
        tick_labels = [str(i) for i in tick_positions]
    ax_r.set_xticks(tick_positions)
    ax_r.set_xticklabels(tick_labels, rotation=0, fontsize=8)

    plt.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.06, hspace=0.05)
    return fig


def build_chart_bytes(
    df: pd.DataFrame,
    symbol: str,
    last_price: float,
    pct_change: float,
    score: int = 0,                  # accepted but unused (no score in title)
    direction: str = "bullish",
    signal_meta: Optional[dict] = None,
) -> bytes:
    fig = _build_figure(df=df, symbol=symbol, last_price=last_price,
                         pct_change=pct_change, direction=direction,
                         signal_meta=signal_meta)
    if fig is None:
        return _build_simple_fallback_bytes(df, symbol, last_price, pct_change, direction)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_simple_fallback_bytes(df, symbol, last_price, pct_change, direction) -> bytes:
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    _draw_candles(ax, opens, highs, lows, closes)
    title_color = "#1b5e20" if direction == "bullish" else "#b71c1c" if direction == "bearish" else "#616161"
    ax.set_title(
        f"{symbol}  ₹{last_price:.2f} ({pct_change:+.2f}%)  |  Insufficient data",
        fontsize=12, fontweight="bold", color=title_color, pad=12,
    )
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("Price (₹)", fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def diagnose_curation(
    df: pd.DataFrame,
    direction: str,
    signal_meta: Optional[dict] = None,
) -> dict:
    """Return reasoning details for the current chart selection."""
    if signal_meta is None:
        signal_meta = {}
    hero, secondary, atr_recent = _select_lines(df, direction, signal_meta)

    out = {
        "n_bars": len(df),
        "direction": direction,
        "atr_recent": round(atr_recent, 4),
    }
    if hero:
        out["hero"] = {
            "kind": hero["kind"],
            "broken": hero.get("broken", False),
            "breakout_bar": hero.get("breakout_bar"),
            "touches_count": len(hero["touches"]),
            "first_touch": hero["first_touch"],
            "last_touch": hero["last_touch"],
            "span": hero["span"],
            "fit_method": hero.get("fit_method"),
        }
        if hero["kind"] == "horizontal":
            out["hero"]["level"] = round(hero["level"], 2)
        else:
            out["hero"]["slope"] = round(hero["slope"], 4)
    else:
        out["hero"] = None

    if secondary:
        out["secondary"] = {
            "kind": secondary["kind"],
            "touches_count": len(secondary["touches"]),
            "span": secondary["span"],
        }
    else:
        out["secondary"] = None

    return out


def classify_chart(
    df: pd.DataFrame,
    direction: str,
    signal_meta: Optional[dict] = None,
) -> dict:
    """
    Categorize the chart for alert filtering. Used by alerts/telegram.py.

    Returns one of:
      'fresh_breakout' / 'fresh_breakdown'  — line just broke in last 3 bars
      'pullback'                            — intact line, price within 1.5*ATR
      'continuation'                        — intact line, price further away
      'no_structure'                        — no qualifying line
    """
    if signal_meta is None:
        signal_meta = {}
    hero, _, atr_recent = _select_lines(df, direction, signal_meta)
    if hero is None:
        return {"category": "no_structure", "reason": "no qualifying line"}

    if hero.get("broken"):
        if direction == "bullish":
            return {"category": "fresh_breakout", "reason": "resistance broken in last 3 bars"}
        else:
            return {"category": "fresh_breakdown", "reason": "support broken in last 3 bars"}

    # Intact line — pullback or continuation?
    closes = df["Close"].to_numpy(dtype=float)
    last_close = float(closes[-1])
    n = len(df)

    if hero["kind"] == "horizontal":
        line_y_now = hero["level"]
    else:
        line_y_now = hero["slope"] * (n - 1) + hero["intercept"]

    distance = abs(last_close - line_y_now)
    atr_dist = distance / atr_recent if atr_recent > 0 else 999

    if atr_dist <= 1.5:
        return {
            "category": "pullback",
            "reason": f"price within {atr_dist:.2f}xATR of line",
            "distance_atr": round(atr_dist, 2),
        }
    return {
        "category": "continuation",
        "reason": f"price {atr_dist:.2f}xATR away from line",
        "distance_atr": round(atr_dist, 2),
    }
