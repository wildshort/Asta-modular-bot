"""
Telegram delivery + charting.

v6 changes:
  - Chart-building delegated to utils/chart_builder.py (curation-first design).
  - Old v5.2 chart code kept as `_build_chart_v1` for instant rollback.
  - To revert: in send_chart(), change `_build_chart` to `_build_chart_v1`.

v5.2 (kept for backward compat / fallback):
  - FIX: Correctly handle yfinance MultiIndex in BOTH orderings
    (ticker-first from group_by='ticker' AND field-first).
  - Isolated trendline computation in a try/except so a bug there can never
    block a chart being sent.
  - Explicit loud logging of any chart-send failure.
"""
from __future__ import annotations

import io
import logging
import time
import traceback
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

# v6: new curated chart builder
from utils.chart_builder import build_chart_bytes as _curated_build_chart_bytes

log = logging.getLogger(__name__)

CHART_PERIOD = "1y"
_API = "https://api.telegram.org"

# v5.2 trendline tuning (still used by _build_chart_v1 fallback)
_PIVOT_ORDER         = 5
_MIN_TOUCHES         = 3
_TOUCH_TOLERANCE_ATR = 0.5
_LOOKBACK_BARS       = 120
_DONCHIAN_LEN        = 20


# ---------- low-level sender ----------
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


def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can return MultiIndex columns in two formats depending on group_by:
      - ('Close', 'SIEMENS.NS')  — field first, ticker second
      - ('SIEMENS.NS', 'Close')  — ticker first, field second  (group_by='ticker')
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    ohlc_fields = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
    for level in range(df.columns.nlevels):
        level_vals = set(df.columns.get_level_values(level))
        if level_vals & ohlc_fields:
            df = df.copy()
            df.columns = df.columns.get_level_values(level)
            return df
    df = df.copy()
    df.columns = df.columns.get_level_values(0)
    return df


# ============================================================
# v6 chart builder (delegates to utils/chart_builder.py)
# ============================================================
def _build_chart(symbol: str, df: pd.DataFrame, direction: str, score: float) -> bytes:
    """
    v6 chart builder: curation-first.
    Delegates to utils.chart_builder.build_chart_bytes.
    """
    df = _normalize_ohlc_columns(df).copy()

    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
    pct = (last - prev) / prev * 100 if prev else 0.0

    direction_lower = direction.lower() if direction else "neutral"
    if direction_lower not in ("bullish", "bearish", "neutral"):
        direction_lower = "neutral"

    # Detect TL breakout signal heuristically: caller passes direction,
    # we let the curator figure out if a breakout actually occurred.
    # This preserves the existing send_chart signature.
    signal_meta = {
        "tl_breakout": True if direction_lower in ("bullish", "bearish") else False,
    }

    return _curated_build_chart_bytes(
        df=df,
        symbol=symbol,
        last_price=last,
        pct_change=pct,
        score=int(score),
        direction=direction_lower,
        signal_meta=signal_meta,
    )


# ============================================================
# v5.2 chart builder (kept as backup — call instead of _build_chart to revert)
# ============================================================
def _find_pivots(values: np.ndarray, kind: str, order: int = _PIVOT_ORDER) -> np.ndarray:
    if _HAS_SCIPY:
        cmp = np.greater if kind == "high" else np.less
        return argrelextrema(values, cmp, order=order)[0]
    out = []
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if (kind == "high" and values[i] == window.max()) or \
           (kind == "low"  and values[i] == window.min()):
            out.append(i)
    return np.array(out, dtype=int)


def _fit_trendline(pivot_idx, pivot_val, kind, atr_val):
    n = len(pivot_idx)
    if n < _MIN_TOUCHES or atr_val <= 0:
        return None
    best = None
    tol = _TOUCH_TOLERANCE_ATR * atr_val
    for a in range(n - 1):
        for b in range(a + 1, n):
            x1, y1 = float(pivot_idx[a]), float(pivot_val[a])
            x2, y2 = float(pivot_idx[b]), float(pivot_val[b])
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if kind == "high" and slope > atr_val * 0.1:
                continue
            if kind == "low" and slope < -atr_val * 0.1:
                continue
            line_vals = y1 + slope * (pivot_idx.astype(float) - x1)
            dists = np.abs(pivot_val.astype(float) - line_vals)
            touches = int((dists <= tol).sum())
            if touches < _MIN_TOUCHES:
                continue
            if kind == "high":
                if ((pivot_val.astype(float) - line_vals) > tol).any():
                    continue
            else:
                if ((line_vals - pivot_val.astype(float)) > tol).any():
                    continue
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


def _key_horizontal_level(pivot_val, atr_val):
    if len(pivot_val) < 3 or atr_val <= 0:
        return None
    pivot_val = pivot_val.astype(float)
    bucket = _TOUCH_TOLERANCE_ATR * atr_val
    counts = [(int(np.sum(np.abs(pivot_val - v) <= bucket)), float(v)) for v in pivot_val]
    counts.sort(reverse=True)
    if counts[0][0] < 3:
        return None
    return counts[0][1]


def _compute_trendlines_safe(df, symbol):
    try:
        return _compute_trendlines_inner(df, symbol)
    except Exception as e:
        log.warning(f"[{symbol}] v1 trendline calc failed: {e}")
        return [], "Watching"


def _compute_trendlines_inner(df, symbol):
    df = _normalize_ohlc_columns(df).copy().reset_index(drop=True)
    n = len(df)
    if n < 30:
        return [], "Watching"
    lookback_start = max(0, n - _LOOKBACK_BARS)
    high_vals = df["High"].astype(float).values
    low_vals  = df["Low"].astype(float).values
    high_piv_rel = _find_pivots(high_vals[lookback_start:], "high")
    low_piv_rel  = _find_pivots(low_vals[lookback_start:],  "low")
    high_piv = (np.asarray(high_piv_rel) + lookback_start).astype(int)
    low_piv  = (np.asarray(low_piv_rel)  + lookback_start).astype(int)
    atr_series = atr_calc(df["High"], df["Low"], df["Close"])
    atr_val = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    lines, status = [], "Watching"
    last_idx = n - 1
    last_close = float(df["Close"].iloc[-1])
    if len(high_piv) >= _MIN_TOUCHES:
        pvals = high_vals[high_piv]
        res = _fit_trendline(high_piv, pvals, "high", atr_val)
        if res:
            y_now   = res["intercept"] + res["slope"] * last_idx
            y_start = res["intercept"] + res["slope"] * res["start_idx"]
            lines.append({
                "x": [res["start_idx"], last_idx], "y": [y_start, y_now],
                "color": "#D32F2F", "label": f"Resistance ({res['touches']}t)",
                "style": "-", "lw": 2.0,
            })
            if last_close > y_now + _TOUCH_TOLERANCE_ATR * atr_val:
                status = "🚀 TL BREAKOUT"
    if len(low_piv) >= _MIN_TOUCHES:
        pvals = low_vals[low_piv]
        sup = _fit_trendline(low_piv, pvals, "low", atr_val)
        if sup:
            y_now   = sup["intercept"] + sup["slope"] * last_idx
            y_start = sup["intercept"] + sup["slope"] * sup["start_idx"]
            lines.append({
                "x": [sup["start_idx"], last_idx], "y": [y_start, y_now],
                "color": "#2E7D32", "label": f"Support ({sup['touches']}t)",
                "style": "-", "lw": 2.0,
            })
            if last_close < y_now - _TOUCH_TOLERANCE_ATR * atr_val:
                status = "🔻 TL BREAKDOWN"
    if len(high_piv) or len(low_piv):
        all_piv_val = np.concatenate([
            high_vals[high_piv] if len(high_piv) else np.array([]),
            low_vals[low_piv]   if len(low_piv)  else np.array([]),
        ])
        kl = _key_horizontal_level(all_piv_val, atr_val)
        if kl is not None:
            too_close = any(abs(ln["y"][-1] - kl) < atr_val * 0.5 for ln in lines)
            if not too_close:
                start_idx = max(0, n - _LOOKBACK_BARS)
                lines.append({
                    "x": [start_idx, last_idx], "y": [kl, kl],
                    "color": "#FFA000", "label": f"Key ₹{kl:.0f}",
                    "style": ":", "lw": 1.8,
                })
    if n >= _DONCHIAN_LEN + 1:
        recent = df.iloc[-(_DONCHIAN_LEN + 1):-1]
        dc_high = float(recent["High"].max())
        dc_low  = float(recent["Low"].min())
        start_idx = n - _DONCHIAN_LEN - 1
        lines.append({
            "x": [start_idx, last_idx], "y": [dc_high, dc_high],
            "color": "#C62828", "label": f"{_DONCHIAN_LEN}D High",
            "style": "--", "lw": 1.0, "alpha": 0.55,
        })
        lines.append({
            "x": [start_idx, last_idx], "y": [dc_low, dc_low],
            "color": "#388E3C", "label": f"{_DONCHIAN_LEN}D Low",
            "style": "--", "lw": 1.0, "alpha": 0.55,
        })
        if status == "Watching":
            if last_close > dc_high:
                status = "🚀 20D BREAKOUT"
            elif last_close < dc_low:
                status = "🔻 20D BREAKDOWN"
    return lines, status


def _build_chart_v1(symbol: str, df: pd.DataFrame, title_suffix: str, title_color: str) -> bytes:
    """v5.2 chart builder — kept for rollback. Uses 3 panels (price/RSI/MACD)."""
    df = _normalize_ohlc_columns(df).copy()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_5"]  = df["Close"].ewm(span=5,  adjust=False).mean()
    _, df["BB_Upper"], df["BB_Lower"] = bollinger(df["Close"])
    df["RSI"] = rsi_calc(df["Close"])
    m_line, s_line, h = macd_calc(df["Close"])
    df["MACD"], df["Signal"], df["Hist"] = m_line, s_line, h

    trend_lines, tl_status = _compute_trendlines_safe(df, symbol)
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
    for line in trend_lines:
        ax1.plot(line["x"], line["y"], color=line["color"],
                 linestyle=line.get("style", "-"), lw=line.get("lw", 2.0),
                 alpha=line.get("alpha", 1.0), label=line["label"], zorder=5)
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


# ============================================================
# Top-level send_chart
# ============================================================
def classify_chart(df: pd.DataFrame, direction: str) -> dict:
    """
    Classify what kind of setup the chart represents. Returns one of:
      - 'fresh_breakout' / 'fresh_breakdown'  → send to Telegram
      - 'pullback'                            → send to Telegram
      - 'continuation'                        → skip Telegram (still saved to artifact)
      - 'no_structure'                        → skip Telegram

    A pullback = intact line where price is currently within 1.5×ATR of the
    line value. The line is acting as a current floor/ceiling for price.

    A continuation = intact line but price is floating well above/below it
    (>1.5×ATR away). The line is real but not a current decision point.
    """
    from utils.chart_builder import diagnose_curation

    direction_lower = direction.lower() if direction else "neutral"
    if direction_lower not in ("bullish", "bearish", "neutral"):
        direction_lower = "neutral"

    try:
        diag = diagnose_curation(
            df=df,
            direction=direction_lower,
            signal_meta={"tl_breakout": direction_lower in ("bullish", "bearish")},
        )
    except Exception as e:
        return {"category": "no_structure", "reason": f"diag_failed: {e}"}

    chosen = diag.get("chosen")

    if chosen is None:
        return {"category": "no_structure", "reason": "no qualifying line"}

    # Fresh breakout/breakdown — already filtered by chart_builder to last 3 bars
    if chosen.get("broken"):
        if chosen.get("role") == "resistance":
            return {"category": "fresh_breakout", "reason": "resistance broken in last 3 bars"}
        else:
            return {"category": "fresh_breakdown", "reason": "support broken in last 3 bars"}

    # Intact line — distinguish pullback from continuation by proximity
    closes = df["Close"].to_numpy(dtype=float)
    last_close = float(closes[-1])
    atr_recent = diag.get("atr_recent", 1.0) or 1.0

    # Compute current line value
    if chosen.get("kind") == "diagonal":
        # We need slope+intercept; chosen dict from diagnose_curation has them under
        # the candidate keys. Reconstruct from the originating candidate.
        candidate_key = (
            "best_diagonal_support" if chosen.get("role") == "support"
            else "best_diagonal_resistance"
        )
        cand = diag.get(candidate_key, {})
        slope = cand.get("slope")
        # Need intercept — derive from first_touch
        first_touch = chosen.get("first_touch")
        if slope is None or first_touch is None:
            return {"category": "continuation", "reason": "could not compute line value"}
        # We don't have intercept stored, reconstruct via touches
        # Use the last_touch_bar from the candidate to get a known point
        last_touch_bar = cand.get("last_touch_bar")
        if last_touch_bar is None:
            return {"category": "continuation", "reason": "no last_touch reference"}
        # We need actual y-value at last_touch — pull from price data
        if chosen.get("role") == "support":
            y_at_last_touch = float(df["Low"].iloc[last_touch_bar])
        else:
            y_at_last_touch = float(df["High"].iloc[last_touch_bar])
        n = len(df)
        line_y_now = y_at_last_touch + slope * ((n - 1) - last_touch_bar)
    else:
        # Horizontal
        line_y_now = chosen.get("level", last_close)

    distance = abs(last_close - line_y_now)
    atr_dist = distance / atr_recent if atr_recent > 0 else 999

    PULLBACK_THRESHOLD_ATR = 1.5
    if atr_dist <= PULLBACK_THRESHOLD_ATR:
        return {
            "category": "pullback",
            "reason": f"price within {atr_dist:.2f}×ATR of {chosen.get('role')} line",
            "distance_atr": round(atr_dist, 2),
        }
    else:
        return {
            "category": "continuation",
            "reason": f"price {atr_dist:.2f}×ATR away from line — not a current test",
            "distance_atr": round(atr_dist, 2),
        }


def send_chart(symbol: str, direction: str = "", score: float = 0.0) -> bool:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        log.warning(f"[{symbol}] send_chart: missing Telegram creds, skipping")
        return False

    try:
        log.info(f"[{symbol}] send_chart: downloading data...")
        df = download_single(symbol, period=CHART_PERIOD, interval="1d")
        if df is None or df.empty:
            log.warning(f"[{symbol}] send_chart: no data returned")
            return False
        log.info(f"[{symbol}] send_chart: got {len(df)} bars, building chart...")

        if direction == "Bullish":
            title_suffix = f"📈 BULLISH (Score {score:.0f}/100)"
        elif direction == "Bearish":
            title_suffix = f"📉 BEARISH (Score {score:.0f}/100)"
        else:
            title_suffix = "Analysis"

        # Build chart (always — needed for both Telegram and artifact)
        png_bytes = _build_chart(symbol, df, direction, score)
        log.info(f"[{symbol}] send_chart: built {len(png_bytes)} byte PNG")

        # Always save debug artifacts (full record of every signal)
        try:
            _save_debug_artifacts(symbol, df, direction, png_bytes)
        except Exception as e:
            log.warning(f"[{symbol}] debug artifact save failed (non-fatal): {e}")

        # Classify the chart and decide whether to send to Telegram
        df_norm = _normalize_ohlc_columns(df).copy()
        classification = classify_chart(df_norm, direction)
        category = classification.get("category", "unknown")

        log.info(f"[{symbol}] classified as: {category} ({classification.get('reason', '')})")

        # Only send fresh breakouts/breakdowns and pullbacks to Telegram
        SEND_CATEGORIES = {"fresh_breakout", "fresh_breakdown", "pullback"}
        if category not in SEND_CATEGORIES:
            log.info(f"[{symbol}] ⏸  skipping Telegram send (category={category})")
            return True  # not an error — intentional skip

        # Build caption that reflects the category
        if category == "fresh_breakout":
            cat_label = "🚀 FRESH BREAKOUT"
        elif category == "fresh_breakdown":
            cat_label = "🔻 FRESH BREAKDOWN"
        elif category == "pullback":
            cat_label = "🎯 PULLBACK"
        else:
            cat_label = ""

        caption = f"📊 {symbol}  |  {title_suffix}"
        if cat_label:
            caption = f"{cat_label}\n{caption}"

        log.info(f"[{symbol}] send_chart: uploading to Telegram...")
        ok = _post_with_retry(
            f"{_API}/bot{TELEGRAM_TOKEN}/sendPhoto",
            files={"photo": ("chart.png", png_bytes, "image/png")},
            data={"chat_id": CHAT_ID, "caption": caption},
        )
        if ok:
            log.info(f"[{symbol}] ✅ Chart sent successfully ({category})")
        else:
            log.error(f"[{symbol}] ❌ Telegram sendPhoto failed after retries")
        return ok

    except Exception as e:
        log.error(f"[{symbol}] ❌ send_chart FAILED: {type(e).__name__}: {e}")
        log.error(traceback.format_exc())
        return False


def _save_debug_artifacts(symbol: str, df: pd.DataFrame, direction: str, png_bytes: bytes) -> None:
    """
    Save chart PNG and diagnostic JSON to scanner/output/ so they get picked up
    by the GitHub Actions artifact upload step.
    """
    import os
    import json
    from utils.chart_builder import diagnose_curation

    safe_symbol = symbol.replace("/", "_").replace(".", "_")

    charts_dir = "scanner/output/charts"
    diag_dir = "scanner/output/diag"
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    # Save chart PNG
    chart_path = os.path.join(charts_dir, f"{safe_symbol}.png")
    with open(chart_path, "wb") as f:
        f.write(png_bytes)

    # Compute and save diagnostic JSON
    df_norm = _normalize_ohlc_columns(df).copy()
    direction_lower = direction.lower() if direction else "neutral"
    if direction_lower not in ("bullish", "bearish", "neutral"):
        direction_lower = "neutral"

    try:
        diag = diagnose_curation(
            df=df_norm,
            direction=direction_lower,
            signal_meta={"tl_breakout": direction_lower in ("bullish", "bearish")},
        )
    except Exception as e:
        diag = {"error": f"{type(e).__name__}: {e}"}

    diag["symbol"] = symbol
    diag["direction_input"] = direction
    diag["last_price"] = float(df_norm["Close"].iloc[-1]) if len(df_norm) else None

    # Add classification (what category this chart fell into)
    try:
        classification = classify_chart(df_norm, direction)
        diag["classification"] = classification
        diag["sent_to_telegram"] = classification.get("category") in {
            "fresh_breakout", "fresh_breakdown", "pullback"
        }
    except Exception as e:
        diag["classification"] = {"category": "unknown", "error": str(e)}
        diag["sent_to_telegram"] = False

    diag_path = os.path.join(diag_dir, f"{safe_symbol}.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=str)

    log.info(f"[{symbol}] debug artifacts saved: {chart_path}, {diag_path}")
