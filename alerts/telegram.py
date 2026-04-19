"""
Telegram delivery + charting.

v4 changes — proper trendline drawing:
  1. Regression-validated trendlines: fit a line through pivot clusters,
     keep only if the fit is tight (low normalized residuals) AND at least
     3 pivots actually touch the line.
  2. Donchian channel overlay (20-bar recent high/low) — always drawn.
  3. Key horizontal level: the price most "respected" by recent pivots
     (where price bounced the most times within 0.5 ATR).
  4. Breakout state computed from whichever line price is interacting with,
     not from a single arbitrary max/min.
  5. All lines drawn only over their "live" range (not extended back to day 1)
     so charts stay readable.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

try:
    from scipy.signal import argrelextrema
    _HAS_SCIPY = True
except Exception as _e:
    _HAS_SCIPY = False
    _SCIPY_ERR = str(_e)

from config import CHAT_ID, TELEGRAM_TOKEN
from utils.fetcher import download_single
from utils.indicators import bollinger, macd as macd_calc, rsi as rsi_calc, atr as atr_calc

log = logging.getLogger(__name__)

CHART_PERIOD = "1y"
_API = "https://api.telegram.org"

# ------------------ trendline tuning ------------------
_PIVOT_ORDER         = 5      # bars either side for pivot detection
_MIN_TOUCHES         = 3      # a line must touch ≥ N pivots to be kept
_TOUCH_TOLERANCE_ATR = 0.5    # "touch" = within 0.5 ATR of the line
_LOOKBACK_BARS       = 120    # only consider pivots within last N bars
_DONCHIAN_LEN        = 20     # Donchian channel length


# ---------- basic plumbing ----------
def _post_with_retry(url: str, max_attempts: int = 4, **kwargs) -> bool:
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, timeout=20, **kwargs)
            if resp.status_code == 200:
                return True
            if resp.status_code == 429:
                try:
                    wait = int(resp.json().get("parameters", {}).get("retry_after", 5))
                except Exception:
                    wait = 5
                log.warning(f"Telegram 429; sleeping {wait}s (attempt {attempt})")
                time.sleep(wait + 1)
                continue
            log.warning(f"Telegram {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            log.warning(f"Telegram network error (attempt {attempt}): {e}")
        if attempt < max_attempts:
            time.sleep(2 ** attempt)
    return False


def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False
    chunks = [message[i : i + 4000] for i in range(0, len(message), 4000)]
    ok = True
    for chunk in chunks:
        ok = _post_with_retry(
            f"{_API}/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": chunk},
        ) and ok
    return ok


# ---------- pivot helpers ----------
def _find_pivots(values: np.ndarray, kind: str, order: int = _PIVOT_ORDER) -> np.ndarray:
    if _HAS_SCIPY:
        cmp = np.greater if kind == "high" else np.less
        return argrelextrema(values, cmp, order=order)[0]
    # numpy fallback
    out = []
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if (kind == "high" and values[i] == window.max()) or \
           (kind == "low"  and values[i] == window.min()):
            out.append(i)
    return np.array(out, dtype=int)


def _fit_trendline(
    pivot_idx: np.ndarray,
    pivot_val: np.ndarray,
    kind: str,
    atr_val: float,
) -> Optional[dict]:
    """
    Given pivot indices + values, find the best-fit line that:
      - has the expected slope direction (resistance usually flat/down, support flat/up)
      - is touched by >= _MIN_TOUCHES pivots within tolerance
      - has tight fit (low residuals relative to ATR)

    Returns dict with slope, intercept, start_idx, end_idx, touches — or None.
    """
    n = len(pivot_idx)
    if n < _MIN_TOUCHES or atr_val <= 0:
        return None

    best = None
    tol = _TOUCH_TOLERANCE_ATR * atr_val

    # Try every pair of pivots as line anchors, score by touch count + fit
    for a in range(n - 1):
        for b in range(a + 1, n):
            x1, y1 = pivot_idx[a], pivot_val[a]
            x2, y2 = pivot_idx[b], pivot_val[b]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # Prefer shallow slopes for real trendlines
            # Resistance: slope should be <= 0 (ideally flat or descending)
            # Support:    slope should be >= 0 (ideally flat or ascending)
            if kind == "high" and slope > abs(atr_val) * 0.1:
                continue
            if kind == "low" and slope < -abs(atr_val) * 0.1:
                continue

            # Count touches among all pivots
            line_vals = y1 + slope * (pivot_idx - x1)
            dists = np.abs(pivot_val - line_vals)
            touches = int((dists <= tol).sum())
            if touches < _MIN_TOUCHES:
                continue

            # For a resistance line, no pivot high should be significantly ABOVE it
            # (price breaking above would invalidate the line during formation)
            if kind == "high":
                excess = (pivot_val - line_vals) > tol
                if excess.any():
                    continue
            else:
                below = (line_vals - pivot_val) > tol
                if below.any():
                    continue

            # Score: more touches is better; tighter fit (smaller mean dist) is better
            mean_dist = float(np.mean(dists[dists <= tol]))
            score = touches - (mean_dist / tol) * 0.5

            if best is None or score > best["score"]:
                best = {
                    "slope": float(slope),
                    "intercept": float(y1 - slope * x1),
                    "start_idx": int(pivot_idx[a]),
                    "end_idx": int(pivot_idx[-1]),
                    "touches": touches,
                    "score": float(score),
                }

    return best


def _key_horizontal_level(
    pivot_idx: np.ndarray,
    pivot_val: np.ndarray,
    atr_val: float,
) -> Optional[float]:
    """
    Find the price level that the most pivots cluster around.
    Useful for showing 'this is the level price keeps respecting'.
    """
    if len(pivot_val) < 3 or atr_val <= 0:
        return None
    bucket = _TOUCH_TOLERANCE_ATR * atr_val
    # For each pivot, count how many other pivots are within bucket distance
    counts = []
    for v in pivot_val:
        c = int(np.sum(np.abs(pivot_val - v) <= bucket))
        counts.append((c, v))
    counts.sort(reverse=True)
    if counts[0][0] < 3:
        return None
    return float(counts[0][1])


# ---------- main trendline computation ----------
def _compute_trendlines(df: pd.DataFrame, symbol: str) -> tuple[list[dict], str]:
    df = df.copy().reset_index(drop=True)
    n = len(df)
    if n < 30:
        return [], "Watching"

    # Restrict pivot search to the recent lookback window — old trendlines become irrelevant
    lookback_start = max(0, n - _LOOKBACK_BARS)
    slice_high = df["High"].iloc[lookback_start:].values
    slice_low  = df["Low"].iloc[lookback_start:].values

    high_piv_rel = _find_pivots(slice_high, "high")
    low_piv_rel  = _find_pivots(slice_low,  "low")
    # Convert back to absolute indices
    high_piv = (high_piv_rel + lookback_start).astype(int)
    low_piv  = (low_piv_rel  + lookback_start).astype(int)

    # Average True Range for tolerance scaling
    atr_series = atr_calc(df["High"], df["Low"], df["Close"])
    atr_val = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0

    log.info(
        f"[{symbol}] pivots over last {_LOOKBACK_BARS} bars: "
        f"{len(high_piv)} highs / {len(low_piv)} lows, ATR={atr_val:.2f}"
    )

    lines: list[dict] = []
    status = "Watching"
    last_idx = n - 1
    last_close = float(df["Close"].iloc[-1])

    # ---- Resistance ----
    if len(high_piv) >= _MIN_TOUCHES:
        pvals = df["High"].values[high_piv].astype(float)
        res = _fit_trendline(high_piv, pvals, "high", atr_val)
        if res:
            y_now = res["intercept"] + res["slope"] * last_idx
            y_start = res["intercept"] + res["slope"] * res["start_idx"]
            lines.append({
                "x": [res["start_idx"], last_idx],
                "y": [y_start, y_now],
                "color": "#D32F2F",
                "label": f"Resistance ({res['touches']} touches)",
                "style": "-",
                "lw": 2.0,
            })
            if last_close > y_now + _TOUCH_TOLERANCE_ATR * atr_val:
                status = "🚀 TL BREAKOUT"
            log.info(f"[{symbol}] resistance: slope={res['slope']:.3f}, touches={res['touches']}")

    # ---- Support ----
    if len(low_piv) >= _MIN_TOUCHES:
        pvals = df["Low"].values[low_piv].astype(float)
        sup = _fit_trendline(low_piv, pvals, "low", atr_val)
        if sup:
            y_now = sup["intercept"] + sup["slope"] * last_idx
            y_start = sup["intercept"] + sup["slope"] * sup["start_idx"]
            lines.append({
                "x": [sup["start_idx"], last_idx],
                "y": [y_start, y_now],
                "color": "#2E7D32",
                "label": f"Support ({sup['touches']} touches)",
                "style": "-",
                "lw": 2.0,
            })
            if last_close < y_now - _TOUCH_TOLERANCE_ATR * atr_val:
                status = "🔻 TL BREAKDOWN"
            log.info(f"[{symbol}] support: slope={sup['slope']:.3f}, touches={sup['touches']}")

    # ---- Key horizontal level ----
    all_piv_idx = np.concatenate([high_piv, low_piv]) if (len(high_piv) or len(low_piv)) else np.array([])
    all_piv_val = np.concatenate([
        df["High"].values[high_piv] if len(high_piv) else np.array([]),
        df["Low"].values[low_piv] if len(low_piv) else np.array([]),
    ])
    key_level = _key_horizontal_level(all_piv_idx, all_piv_val, atr_val)
    if key_level is not None:
        # Only draw if sufficiently different from existing sloped lines at current bar
        too_close = False
        for ln in lines:
            if abs(ln["y"][-1] - key_level) < atr_val * 0.5:
                too_close = True
                break
        if not too_close:
            start_idx = max(0, n - _LOOKBACK_BARS)
            lines.append({
                "x": [start_idx, last_idx],
                "y": [key_level, key_level],
                "color": "#FFA000",
                "label": f"Key Level ₹{key_level:.0f}",
                "style": ":",
                "lw": 1.8,
            })
            log.info(f"[{symbol}] key horizontal level: {key_level:.2f}")

    # ---- Donchian channel overlay (always drawn) ----
    if n >= _DONCHIAN_LEN + 1:
        recent = df.iloc[-(_DONCHIAN_LEN + 1):-1]   # last N complete bars, excluding current
        dc_high = float(recent["High"].max())
        dc_low  = float(recent["Low"].min())
        start_idx = n - _DONCHIAN_LEN - 1
        lines.append({
            "x": [start_idx, last_idx],
            "y": [dc_high, dc_high],
            "color": "#C62828",
            "label": f"{_DONCHIAN_LEN}D High",
            "style": "--",
            "lw": 1.0,
            "alpha": 0.55,
        })
        lines.append({
            "x": [start_idx, last_idx],
            "y": [dc_low, dc_low],
            "color": "#388E3C",
            "label": f"{_DONCHIAN_LEN}D Low",
            "style": "--",
            "lw": 1.0,
            "alpha": 0.55,
        })
        # If no sloped line triggered status, still detect Donchian breakout
        if status == "Watching":
            if last_close > dc_high:
                status = "🚀 20D BREAKOUT"
            elif last_close < dc_low:
                status = "🔻 20D BREAKDOWN"

    return lines, status


# ---------- chart ----------
def _build_chart(symbol: str, df: pd.DataFrame, title_suffix: str, title_color: str) -> bytes:
    df = df.copy()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_5"]  = df["Close"].ewm(span=5,  adjust=False).mean()
    _, df["BB_Upper"], df["BB_Lower"] = bollinger(df["Close"])
    df["RSI"] = rsi_calc(df["Close"])
    m_line, s_line, h = macd_calc(df["Close"])
    df["MACD"], df["Signal"], df["Hist"] = m_line, s_line, h

    trend_lines, tl_status = _compute_trendlines(df, symbol)
    log.info(f"[{symbol}] drawing {len(trend_lines)} lines, status={tl_status}")

    df_plot = df.reset_index()
    date_col = df_plot.columns[0]

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])

    ax1 = plt.subplot(gs[0])
    x = df_plot.index
    up = df_plot[df_plot.Close >= df_plot.Open]
    down = df_plot[df_plot.Close < df_plot.Open]
    ax1.bar(up.index, up.Close - up.Open, width=0.7, bottom=up.Open, color="#089981")
    ax1.vlines(up.index, up.Low, up.High, color="#089981", linewidth=0.8)
    ax1.bar(down.index, down.Close - down.Open, width=0.7, bottom=down.Open, color="#F23645")
    ax1.vlines(down.index, down.Low, down.High, color="#F23645", linewidth=0.8)

    ax1.plot(x, df_plot["EMA_5"],  color="#2962FF", lw=1.0, label="EMA 5",  alpha=0.75)
    ax1.plot(x, df_plot["EMA_50"], color="#FF9800", lw=1.5, label="EMA 50")
    ax1.plot(x, df_plot["BB_Upper"], color="blue", lw=0.5, alpha=0.4)
    ax1.plot(x, df_plot["BB_Lower"], color="blue", lw=0.5, alpha=0.4)
    ax1.fill_between(x, df_plot["BB_Upper"], df_plot["BB_Lower"], color="blue", alpha=0.05)

    # Draw trendlines on top (zorder=5)
    for line in trend_lines:
        ax1.plot(line["x"], line["y"],
                 color=line["color"],
                 linestyle=line.get("style", "-"),
                 lw=line.get("lw", 2.0),
                 alpha=line.get("alpha", 1.0),
                 label=line["label"],
                 zorder=5)

    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    pct = (last - prev) / prev * 100 if prev else 0.0
    arrow = "🟢" if pct >= 0 else "🔴"
    combined = title_suffix if tl_status == "Watching" else f"{title_suffix}  |  {tl_status}"
    title = f"{symbol}  {arrow} ₹{last:.2f} ({pct:+.2f}%)  |  {combined}"
    ax1.set_title(title, fontsize=14, fontweight="bold", color=title_color, pad=12)
    ax1.grid(True, color="#f0f0f0")
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(x, df_plot["RSI"], color="#7E57C2")
    ax2.axhline(70, color="red",   ls="--", lw=0.8)
    ax2.axhline(30, color="green", ls="--", lw=0.8)
    ax2.axhline(50, color="gray",  ls=":",  lw=0.6)
    ax2.set_ylabel("RSI", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(True, color="#f0f0f0")

    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(x, df_plot["MACD"],   color="#2962FF", label="MACD")
    ax3.plot(x, df_plot["Signal"], color="#FF9800", label="Signal")
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df_plot["Hist"]]
    ax3.bar(x, df_plot["Hist"], color=colors, alpha=0.7)
    ax3.axhline(0, color="gray", lw=0.6)
    ax3.set_ylabel("MACD", fontweight="bold")
    ax3.grid(True, color="#f0f0f0")
    ax3.legend(loc="upper left", frameon=False, fontsize=8)

    step = max(1, len(df_plot) // 10)
    ax3.set_xticks(x[::step])
    ax3.set_xticklabels(df_plot[date_col].dt.strftime("%b %d")[::step], rotation=0)
    plt.setp([ax.get_xticklabels() for ax in [ax1, ax2]], visible=False)
    plt.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.06, hspace=0.08)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def send_chart(symbol: str, direction: str = "", score: float = 0.0) -> bool:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False
    try:
        df = download_single(symbol, period=CHART_PERIOD, interval="1d")
        if df is None or df.empty:
            log.warning(f"No chart data for {symbol}")
            return False

        if direction == "Bullish":
            title_suffix = f"📈 BULLISH (Score {score:.0f}/100)"
            title_color = "green"
        elif direction == "Bearish":
            title_suffix = f"📉 BEARISH (Score {score:.0f}/100)"
            title_color = "red"
        else:
            title_suffix = "Analysis"
            title_color = "black"

        png_bytes = _build_chart(symbol, df, title_suffix, title_color)

        caption = f"📊 {symbol}  |  {title_suffix}"
        ok = _post_with_retry(
            f"{_API}/bot{TELEGRAM_TOKEN}/sendPhoto",
            files={"photo": ("chart.png", png_bytes, "image/png")},
            data={"chat_id": CHAT_ID, "caption": caption},
        )
        if ok:
            log.info(f"Chart sent for {symbol}")
        return ok
    except Exception as e:
        log.warning(f"Chart send failed for {symbol}: {e}", exc_info=True)
        return False
