"""
Telegram delivery + charting.

Improvements over old code:
  - Retries with exponential backoff on network errors.
  - Honors Telegram's 429 Retry-After header instead of silently failing.
  - Returns success/failure instead of swallowing errors.
  - Chart uses the same indicator module as the scanner (single source of truth).
  - Chart title shows the composite score and direction, matching the text alert.
  - Trendline breakout overlay restored from original version (red resistance /
    green support lines drawn from pivot points).
"""
from __future__ import annotations

import io
import logging
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.signal import argrelextrema

from config import CHAT_ID, TELEGRAM_TOKEN
from utils.fetcher import download_single
from utils.indicators import bollinger, macd as macd_calc, rsi as rsi_calc

log = logging.getLogger(__name__)

CHART_PERIOD = "1y"
_API = "https://api.telegram.org"


# ------------------ LOW-LEVEL SENDERS ------------------
def _post_with_retry(url: str, max_attempts: int = 4, **kwargs) -> bool:
    """POST with exponential backoff; respects Telegram 429 Retry-After."""
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
        log.debug("Telegram credentials missing; skipping send.")
        return False

    chunks = [message[i : i + 4000] for i in range(0, len(message), 4000)]
    ok = True
    for chunk in chunks:
        success = _post_with_retry(
            f"{_API}/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": chunk},
        )
        ok = ok and success
    return ok


# ------------------ TRENDLINE DETECTION (restored from old code) ------------------
def _compute_trendlines(df: pd.DataFrame) -> tuple[list[dict], str]:
    """
    Detects major pivot-based trendlines and whether price has broken them.

    Returns (lines_to_draw, status) where lines_to_draw is a list of
    {x: [start_idx, end_idx], y: [start_price, end_price], color, label}
    and status is "🚀 TL BREAKOUT" / "🔻 TL BREAKDOWN" / "Watching".
    """
    df = df.copy()
    df["idx"] = range(len(df))

    try:
        df["pivot_high"] = df.iloc[
            argrelextrema(df["High"].values, np.greater, order=10)[0]
        ]["High"]
        df["pivot_low"] = df.iloc[
            argrelextrema(df["Low"].values, np.less, order=10)[0]
        ]["Low"]
    except Exception as e:
        log.debug(f"Pivot detection failed: {e}")
        return [], "Watching"

    lines: list[dict] = []
    status = "Watching"
    last_idx = df["idx"].iloc[-1]
    current_close = float(df["Close"].iloc[-1])

    # Resistance trendline connecting major high -> recent high
    highs = df.dropna(subset=["pivot_high"])
    if len(highs) >= 2:
        major_high = highs.sort_values("pivot_high", ascending=False).iloc[0]
        peaks = highs[highs["idx"] > major_high["idx"]]
        if not peaks.empty:
            p2 = peaks.iloc[-1]
            try:
                slope = (p2["pivot_high"] - major_high["pivot_high"]) / (
                    p2["idx"] - major_high["idx"]
                )
                y_end = p2["pivot_high"] + slope * (last_idx - p2["idx"])
                lines.append(
                    {
                        "x": [int(major_high["idx"]), int(last_idx)],
                        "y": [float(major_high["pivot_high"]), float(y_end)],
                        "color": "#D32F2F",
                        "label": "Resistance",
                    }
                )
                if current_close > y_end and slope < 0:
                    status = "🚀 TL BREAKOUT"
            except Exception as e:
                log.debug(f"Resistance line calc failed: {e}")

    # Support trendline connecting major low -> recent low
    lows = df.dropna(subset=["pivot_low"])
    if len(lows) >= 2:
        major_low = lows.sort_values("pivot_low", ascending=True).iloc[0]
        troughs = lows[lows["idx"] > major_low["idx"]]
        if not troughs.empty:
            p2 = troughs.iloc[-1]
            try:
                slope = (p2["pivot_low"] - major_low["pivot_low"]) / (
                    p2["idx"] - major_low["idx"]
                )
                y_end = p2["pivot_low"] + slope * (last_idx - p2["idx"])
                lines.append(
                    {
                        "x": [int(major_low["idx"]), int(last_idx)],
                        "y": [float(major_low["pivot_low"]), float(y_end)],
                        "color": "#2E7D32",
                        "label": "Support",
                    }
                )
                if current_close < y_end and slope > 0:
                    status = "🔻 TL BREAKDOWN"
            except Exception as e:
                log.debug(f"Support line calc failed: {e}")

    return lines, status


# ------------------ CHART ------------------
def _build_chart(symbol: str, df: pd.DataFrame, title_suffix: str, title_color: str) -> bytes:
    df = df.copy()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    _, df["BB_Upper"], df["BB_Lower"] = bollinger(df["Close"])
    df["RSI"] = rsi_calc(df["Close"])
    m_line, s_line, h = macd_calc(df["Close"])
    df["MACD"], df["Signal"], df["Hist"] = m_line, s_line, h

    # Compute trendlines + breakout status (restored from old version)
    trend_lines, tl_status = _compute_trendlines(df)

    df_plot = df.reset_index()
    date_col = df_plot.columns[0]  # 'Date' or 'Datetime'

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])

    # ---- Panel 1: Price ----
    ax1 = plt.subplot(gs[0])
    x = df_plot.index
    up = df_plot[df_plot.Close >= df_plot.Open]
    down = df_plot[df_plot.Close < df_plot.Open]
    ax1.bar(up.index, up.Close - up.Open, width=0.7, bottom=up.Open, color="#089981")
    ax1.vlines(up.index, up.Low, up.High, color="#089981", linewidth=0.8)
    ax1.bar(down.index, down.Close - down.Open, width=0.7, bottom=down.Open, color="#F23645")
    ax1.vlines(down.index, down.Low, down.High, color="#F23645", linewidth=0.8)

    ax1.plot(x, df_plot["EMA_5"], color="#2962FF", lw=1.0, label="EMA 5", alpha=0.8)
    ax1.plot(x, df_plot["EMA_50"], color="#FF9800", lw=1.5, label="EMA 50")
    ax1.plot(x, df_plot["BB_Upper"], color="blue", lw=0.5, alpha=0.4)
    ax1.plot(x, df_plot["BB_Lower"], color="blue", lw=0.5, alpha=0.4)
    ax1.fill_between(x, df_plot["BB_Upper"], df_plot["BB_Lower"], color="blue", alpha=0.05)

    # ---- Draw trendlines (the restored old-chart feature) ----
    for line in trend_lines:
        ax1.plot(line["x"], line["y"], color=line["color"], linestyle="-",
                 lw=2, label=line["label"])

    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    pct = (last - prev) / prev * 100 if prev else 0.0
    arrow = "🟢" if pct >= 0 else "🔴"

    # Merge score-based status with trendline status in the title
    combined_suffix = title_suffix
    if tl_status != "Watching":
        combined_suffix = f"{title_suffix}  |  {tl_status}"

    title = f"{symbol}  {arrow} ₹{last:.2f} ({pct:+.2f}%)  |  {combined_suffix}"
    ax1.set_title(title, fontsize=14, fontweight="bold", color=title_color, pad=12)
    ax1.grid(True, color="#f0f0f0")
    ax1.legend(loc="upper left", frameon=False)

    # ---- Panel 2: RSI ----
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(x, df_plot["RSI"], color="#7E57C2")
    ax2.axhline(70, color="red", ls="--", lw=0.8)
    ax2.axhline(30, color="green", ls="--", lw=0.8)
    ax2.axhline(50, color="gray", ls=":", lw=0.6)
    ax2.set_ylabel("RSI", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(True, color="#f0f0f0")

    # ---- Panel 3: MACD ----
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(x, df_plot["MACD"], color="#2962FF", label="MACD")
    ax3.plot(x, df_plot["Signal"], color="#FF9800", label="Signal")
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df_plot["Hist"]]
    ax3.bar(x, df_plot["Hist"], color=colors, alpha=0.7)
    ax3.axhline(0, color="gray", lw=0.6)
    ax3.set_ylabel("MACD", fontweight="bold")
    ax3.grid(True, color="#f0f0f0")
    ax3.legend(loc="upper left", frameon=False, fontsize=8)

    # X-axis labels
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
    """Send an annotated chart to Telegram. Gracefully no-ops if creds missing."""
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
