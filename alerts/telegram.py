"""
Telegram delivery + charting.

Improvements over old code:
  - Retries with exponential backoff on network errors.
  - Honors Telegram's 429 Retry-After header instead of silently failing.
  - Returns success/failure instead of swallowing errors.
  - Chart uses the same indicator module as the scanner (single source of truth).
  - Chart title shows the composite score and direction, matching the text alert.
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
import pandas as pd
import requests

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
                # Rate limited — honor the retry-after hint
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
            time.sleep(2 ** attempt)  # 2, 4, 8, 16...
    return False


def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        log.debug("Telegram credentials missing; skipping send.")
        return False

    # Telegram max message length is 4096 chars; split long messages safely.
    chunks = [message[i : i + 4000] for i in range(0, len(message), 4000)]
    ok = True
    for chunk in chunks:
        success = _post_with_retry(
            f"{_API}/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": chunk},
        )
        ok = ok and success
    return ok


# ------------------ CHART ------------------
def _build_chart(symbol: str, df: pd.DataFrame, title_suffix: str, title_color: str) -> bytes:
    df = df.copy()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    _, df["BB_Upper"], df["BB_Lower"] = bollinger(df["Close"])
    df["RSI"] = rsi_calc(df["Close"])
    m_line, s_line, h = macd_calc(df["Close"])
    df["MACD"], df["Signal"], df["Hist"] = m_line, s_line, h

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

    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    pct = (last - prev) / prev * 100 if prev else 0.0
    arrow = "🟢" if pct >= 0 else "🔴"
    title = f"{symbol}  {arrow} ₹{last:.2f} ({pct:+.2f}%)  |  {title_suffix}"
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
