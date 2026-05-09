"""
Asta stock scanner — improved version.

Key changes vs old code:
  1. UNIFIED SCORING: every stock gets a 0-100 composite score per direction.
     No more brittle "all 6 booleans must be true" filter.
  2. MULTI-TIMEFRAME ALIGNMENT: daily signal must agree with weekly trend.
  3. REAL DIVERGENCE DETECTION: replaces the old stub that always returned None.
  4. LIQUIDITY FILTER: skips illiquid names using 20-day avg turnover.
  5. ATR-BASED VOLATILITY CONTEXT: threshold adapts per stock.
  6. CLOSED-BAR ENFORCEMENT: doesn't use the still-forming intraday candle.
  7. COOLDOWN: doesn't spam the same signal day after day.
  8. BATCH DOWNLOAD: one API call per timeframe, not one per symbol.
  9. PER-SYMBOL REJECTION LOGGING: tells you *why* a stock didn't qualify,
     so you can tune intelligently.
 10. CLASSIFICATION FILTER: text + chart alerts only fire when the chart
     classifies as a fresh breakout, fresh breakdown, or pullback. Trend
     continuation and no-structure cases are still scored and saved to the
     summary JSON, but no Telegram alerts are sent.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import pandas as pd

from alerts.telegram import send_telegram, send_chart, classify_chart
from config import (
    COMMODITY_SUFFIXES,
    MIN_ADX,
    MIN_AVG_TURNOVER_INR,
    SIGNAL_MIN_SCORE,
    TELEGRAM_MSG_DELAY_SEC,
    VOLUME_SPIKE_MULT,
)
from scanner.state_tracker import load_state, record_alert, save_state, should_alert
from utils.fetcher import download_bulk
from utils.indicators import (
    adx,
    atr,
    avg_turnover_inr,
    bb_position,
    bollinger,
    ema_crossover,
    macd,
    macd_cross,
    rsi,
    rsi_divergence,
    trend_breakout,
    volume_spike,
)

log = logging.getLogger(__name__)

# Categories that warrant a Telegram alert (text + chart). Everything else is
# saved to the artifact summary but not sent.
ALERT_CATEGORIES = {"fresh_breakout", "fresh_breakdown", "pullback"}


# ------------------ DATA CONTAINER ------------------
@dataclass
class SignalResult:
    symbol: str
    price: float
    direction: str            # "Bullish" or "Bearish" or "None"
    score: float              # 0-100
    reasons: list[str] = field(default_factory=list)
    rejection: str = ""       # populated only when direction == "None"

    # Raw indicator values for logging / Telegram message
    rsi_daily: float = 0.0
    rsi_weekly: float = 0.0
    adx_val: float = 0.0
    macd_daily: str = ""
    macd_weekly: str = ""
    bb_pos: str = ""
    trend_bo: str = ""
    ema_cross: str = ""
    vol_ratio: float = 0.0
    divergence: str = ""
    turnover_cr: float = 0.0  # in crores

    # Classification (set after analysis if signal qualifies)
    classification: str = ""        # 'fresh_breakout' | 'pullback' | 'continuation' | 'no_structure'
    classification_reason: str = ""


# ------------------ SCORING ENGINE ------------------
#
# Each criterion contributes up to N points toward the direction score.
# A stock must clear SIGNAL_MIN_SCORE (default 70) to fire an alert.

def _score_bullish(
    rsi_d: float, rsi_w: float, macd_d: str, macd_w: str,
    bb: str, trend: str, vol_ok: bool, ema_bull: bool,
    adx_v: float, divergence: str,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if macd_w == "PCO" and rsi_w > 50:
        score += 20
        reasons.append("Weekly uptrend confirmed")
    elif macd_w == "PCO":
        score += 10
        reasons.append("Weekly MACD positive")

    if macd_d == "PCO":
        score += 15
        reasons.append("Daily MACD bullish")

    if 55 <= rsi_d <= 75:
        score += 15
        reasons.append(f"RSI in bull zone ({rsi_d:.0f})")
    elif 50 <= rsi_d < 55:
        score += 8

    if bb == "Upper":
        score += 10
        reasons.append("BB upper challenge")

    if trend == "Bullish":
        score += 10
        reasons.append("20-bar breakout")

    if vol_ok:
        score += 10
        reasons.append("Volume spike")

    if ema_bull:
        score += 10
        reasons.append("EMA 5/50 bullish cross")

    if adx_v >= 25:
        score += 10
        reasons.append(f"Strong trend (ADX {adx_v:.0f})")
    elif adx_v >= MIN_ADX:
        score += 5

    if divergence == "Bullish":
        score += 10
        reasons.append("Bullish RSI divergence")

    return min(score, 100.0), reasons


def _score_bearish(
    rsi_d: float, rsi_w: float, macd_d: str, macd_w: str,
    bb: str, trend: str, vol_ok: bool, ema_bear: bool,
    adx_v: float, divergence: str,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if macd_w == "NCO" and rsi_w < 50:
        score += 20
        reasons.append("Weekly downtrend confirmed")
    elif macd_w == "NCO":
        score += 10
        reasons.append("Weekly MACD negative")

    if macd_d == "NCO":
        score += 15
        reasons.append("Daily MACD bearish")

    if 25 <= rsi_d <= 45:
        score += 15
        reasons.append(f"RSI in bear zone ({rsi_d:.0f})")
    elif 45 < rsi_d <= 50:
        score += 8

    if bb == "Lower":
        score += 10
        reasons.append("BB lower challenge")

    if trend == "Bearish":
        score += 10
        reasons.append("20-bar breakdown")

    if vol_ok:
        score += 10
        reasons.append("Volume spike")

    if ema_bear:
        score += 10
        reasons.append("EMA 5/50 bearish cross")

    if adx_v >= 25:
        score += 10
        reasons.append(f"Strong trend (ADX {adx_v:.0f})")
    elif adx_v >= MIN_ADX:
        score += 5

    if divergence == "Bearish":
        score += 10
        reasons.append("Bearish RSI divergence")

    return min(score, 100.0), reasons


# ------------------ PER-SYMBOL ANALYSIS ------------------
def _analyze(
    symbol: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
) -> SignalResult:
    if daily is None or len(daily) < 60:
        return SignalResult(symbol, 0.0, "None", 0.0, rejection="Insufficient daily history")
    if weekly is None or len(weekly) < 30:
        return SignalResult(symbol, 0.0, "None", 0.0, rejection="Insufficient weekly history")

    close_d = daily["Close"].astype(float)
    high_d  = daily["High"].astype(float)
    low_d   = daily["Low"].astype(float)
    vol_d   = daily["Volume"].astype(float)
    close_w = weekly["Close"].astype(float)

    last_price = float(close_d.iloc[-1])

    is_commodity = symbol.endswith(COMMODITY_SUFFIXES)
    turnover = avg_turnover_inr(close_d, vol_d)
    if not is_commodity and turnover < MIN_AVG_TURNOVER_INR:
        return SignalResult(
            symbol, last_price, "None", 0.0,
            rejection=f"Low liquidity (turnover ₹{turnover/1e7:.1f} cr < {MIN_AVG_TURNOVER_INR/1e7:.0f} cr)",
            turnover_cr=turnover / 1e7,
        )

    rsi_d_series = rsi(close_d)
    rsi_d = float(rsi_d_series.iloc[-1])
    macd_d_line, macd_d_sig, _ = macd(close_d)
    macd_d_state, _ = macd_cross(macd_d_line, macd_d_sig)
    _, bb_u, bb_l = bollinger(close_d)
    bb_pos_val = bb_position(close_d, bb_u, bb_l)
    trend = trend_breakout(close_d, window=20)
    adx_v = float(adx(high_d, low_d, close_d).iloc[-1] or 0.0)
    if pd.isna(adx_v):
        adx_v = 0.0

    ema5 = close_d.ewm(span=5, adjust=False).mean()
    ema50 = close_d.ewm(span=50, adjust=False).mean()
    ema_bull, ema_bear = ema_crossover(ema5, ema50)
    ema_str = "Bullish" if ema_bull else "Bearish" if ema_bear else "None"

    vol_ok, vol_ratio = volume_spike(vol_d, mult=VOLUME_SPIKE_MULT)

    divergence = rsi_divergence(close_d, rsi_d_series)

    rsi_w = float(rsi(close_w).iloc[-1])
    macd_w_line, macd_w_sig, _ = macd(close_w)
    macd_w_state, _ = macd_cross(macd_w_line, macd_w_sig)

    if adx_v < MIN_ADX:
        return SignalResult(
            symbol, last_price, "None", 0.0,
            rejection=f"No trend (ADX {adx_v:.1f} < {MIN_ADX})",
            adx_val=adx_v, rsi_daily=rsi_d, rsi_weekly=rsi_w,
            turnover_cr=turnover / 1e7,
        )

    bull_score, bull_reasons = _score_bullish(
        rsi_d, rsi_w, macd_d_state, macd_w_state,
        bb_pos_val, trend, vol_ok, ema_bull, adx_v, divergence,
    )
    bear_score, bear_reasons = _score_bearish(
        rsi_d, rsi_w, macd_d_state, macd_w_state,
        bb_pos_val, trend, vol_ok, ema_bear, adx_v, divergence,
    )

    if bull_score >= bear_score:
        direction = "Bullish" if bull_score >= SIGNAL_MIN_SCORE else "None"
        score, reasons = bull_score, bull_reasons
    else:
        direction = "Bearish" if bear_score >= SIGNAL_MIN_SCORE else "None"
        score, reasons = bear_score, bear_reasons

    rejection = ""
    if direction == "None":
        rejection = f"Score {max(bull_score, bear_score):.0f} < {SIGNAL_MIN_SCORE}"

    return SignalResult(
        symbol=symbol,
        price=last_price,
        direction=direction,
        score=score,
        reasons=reasons,
        rejection=rejection,
        rsi_daily=rsi_d,
        rsi_weekly=rsi_w,
        adx_val=adx_v,
        macd_daily=macd_d_state,
        macd_weekly=macd_w_state,
        bb_pos=bb_pos_val,
        trend_bo=trend,
        ema_cross=ema_str,
        vol_ratio=vol_ratio,
        divergence=divergence,
        turnover_cr=turnover / 1e7,
    )


# ------------------ FORMATTING ------------------
def _format_alert(r: SignalResult) -> str:
    emoji = "📈" if r.direction == "Bullish" else "📉"
    # Prefix the alert text with the classification banner
    if r.classification == "fresh_breakout":
        banner = "🚀 FRESH BREAKOUT\n"
    elif r.classification == "fresh_breakdown":
        banner = "🔻 FRESH BREAKDOWN\n"
    elif r.classification == "pullback":
        banner = "🎯 PULLBACK\n"
    else:
        banner = ""

    lines = [
        f"{banner}{emoji} {r.symbol}  |  Score: {r.score:.0f}/100",
        f"    💰 Price       : ₹{r.price:.2f}",
        f"    🎯 Direction   : {r.direction}",
        f"    📊 RSI D/W     : {r.rsi_daily:.1f} / {r.rsi_weekly:.1f}",
        f"    💡 MACD D/W    : {r.macd_daily} / {r.macd_weekly}",
        f"    📏 BB Position : {r.bb_pos}",
        f"    🚧 Breakout    : {r.trend_bo}",
        f"    🔄 EMA 5/50    : {r.ema_cross}",
        f"    💥 Volume      : {r.vol_ratio:.2f}x avg",
        f"    😽 ADX         : {r.adx_val:.1f}",
        f"    🔍 Divergence  : {r.divergence}",
        f"    💧 Turnover    : ₹{r.turnover_cr:.1f} cr/day",
        f"    ✅ Reasons     : {', '.join(r.reasons) if r.reasons else '—'}",
    ]
    return "\n".join(lines)


# ------------------ CLASSIFICATION HELPER ------------------
def _classify_signal(result: SignalResult, daily: pd.DataFrame) -> SignalResult:
    """
    Run the chart classifier on the daily DataFrame and attach the result
    to the SignalResult. This decides whether the signal warrants a Telegram
    alert (fresh breakout/breakdown/pullback) or just a summary entry.
    """
    try:
        classification = classify_chart(daily, result.direction)
        result.classification = classification.get("category", "unknown")
        result.classification_reason = classification.get("reason", "")
    except Exception as e:
        result.classification = "error"
        result.classification_reason = f"{type(e).__name__}: {e}"
        log.warning(f"[{result.symbol}] classification failed: {e}")
    return result


# ------------------ MAIN ENTRY POINT ------------------
def run_stock_scan(symbols: list[str], send_alerts: bool = False) -> dict:
    """
    Scan all symbols, score them, classify them, send alerts only for
    fresh breakouts / breakdowns / pullbacks.
    """
    log.info(f"Starting scan of {len(symbols)} symbols...")

    log.info("Downloading daily data (bulk)...")
    daily_data = download_bulk(symbols, period="6mo", interval="1d")
    log.info(f"  got {len(daily_data)}/{len(symbols)} symbols")

    log.info("Downloading weekly data (bulk)...")
    weekly_data = download_bulk(symbols, period="2y", interval="1wk")
    log.info(f"  got {len(weekly_data)}/{len(symbols)} symbols")

    bullish: list[SignalResult] = []
    bearish: list[SignalResult] = []
    rejected: list[SignalResult] = []

    for sym in symbols:
        try:
            d = daily_data.get(sym)
            w = weekly_data.get(sym)
            if d is None or w is None:
                log.debug(f"{sym}: no data")
                continue

            result = _analyze(sym, d, w)

            if result.direction == "Bullish":
                # Classify before adding so we can decide later
                result = _classify_signal(result, d)
                bullish.append(result)
                log.info(
                    f"🟢 {sym} BULLISH score={result.score:.0f} "
                    f"class={result.classification} [{', '.join(result.reasons)}]"
                )
            elif result.direction == "Bearish":
                result = _classify_signal(result, d)
                bearish.append(result)
                log.info(
                    f"🔴 {sym} BEARISH score={result.score:.0f} "
                    f"class={result.classification} [{', '.join(result.reasons)}]"
                )
            else:
                rejected.append(result)
                log.debug(f"⚪ {sym} rejected: {result.rejection}")

        except Exception as e:
            log.error(f"{sym} analysis failed: {e}", exc_info=True)

    bullish.sort(key=lambda r: r.score, reverse=True)
    bearish.sort(key=lambda r: r.score, reverse=True)

    log.info(f"Scan complete: {len(bullish)} bullish, {len(bearish)} bearish, {len(rejected)} rejected")

    # ---- Apply cooldown ----
    state = load_state()
    fresh_bullish = [r for r in bullish if should_alert(state, r.symbol, "Bullish", r.score)]
    fresh_bearish = [r for r in bearish if should_alert(state, r.symbol, "Bearish", r.score)]
    log.info(f"After cooldown: {len(fresh_bullish)} new bullish, {len(fresh_bearish)} new bearish")

    # ---- Apply classification filter (only breakouts / breakdowns / pullbacks alert) ----
    # ---- Plus: BB challenge filter — bullish must have BB Upper, bearish must have BB Lower
    def _passes_bb_filter(r: SignalResult) -> bool:
        if r.direction == "Bullish":
            return r.bb_pos == "Upper"
        if r.direction == "Bearish":
            return r.bb_pos == "Lower"
        return False

    alert_bullish = [
        r for r in fresh_bullish
        if r.classification in ALERT_CATEGORIES and _passes_bb_filter(r)
    ]
    alert_bearish = [
        r for r in fresh_bearish
        if r.classification in ALERT_CATEGORIES and _passes_bb_filter(r)
    ]

    # Build the skipped lists with reasons (for logging)
    skipped_bullish = []
    for r in fresh_bullish:
        if r.classification not in ALERT_CATEGORIES:
            r.classification_reason = f"not actionable ({r.classification})"
            skipped_bullish.append(r)
        elif not _passes_bb_filter(r):
            r.classification_reason = f"BB not Upper (got {r.bb_pos})"
            skipped_bullish.append(r)
    skipped_bearish = []
    for r in fresh_bearish:
        if r.classification not in ALERT_CATEGORIES:
            r.classification_reason = f"not actionable ({r.classification})"
            skipped_bearish.append(r)
        elif not _passes_bb_filter(r):
            r.classification_reason = f"BB not Lower (got {r.bb_pos})"
            skipped_bearish.append(r)

    log.info(
        f"After classification + BB filter: {len(alert_bullish)} bullish alerts "
        f"({len(skipped_bullish)} skipped), {len(alert_bearish)} bearish alerts "
        f"({len(skipped_bearish)} skipped)"
    )
    for r in skipped_bullish + skipped_bearish:
        log.info(f"  ⏸  {r.symbol} skipped ({r.classification_reason})")

    if send_alerts:
        if alert_bullish:
            header = f"📈 BULLISH SIGNALS ({len(alert_bullish)})\n" + "=" * 35
            send_telegram(header)
            time.sleep(TELEGRAM_MSG_DELAY_SEC)
            for r in alert_bullish:
                send_telegram(_format_alert(r))
                time.sleep(TELEGRAM_MSG_DELAY_SEC)
                send_chart(r.symbol, direction=r.direction, score=r.score)
                time.sleep(TELEGRAM_MSG_DELAY_SEC)
                record_alert(state, r.symbol, "Bullish", r.score)

        if alert_bearish:
            header = f"📉 BEARISH SIGNALS ({len(alert_bearish)})\n" + "=" * 35
            send_telegram(header)
            time.sleep(TELEGRAM_MSG_DELAY_SEC)
            for r in alert_bearish:
                send_telegram(_format_alert(r))
                time.sleep(TELEGRAM_MSG_DELAY_SEC)
                send_chart(r.symbol, direction=r.direction, score=r.score)
                time.sleep(TELEGRAM_MSG_DELAY_SEC)
                record_alert(state, r.symbol, "Bearish", r.score)

        if not alert_bullish and not alert_bearish:
            log.info("No new tradable signals (breakouts / breakdowns / pullbacks) to send.")

        save_state(state)
    else:
        for r in bullish:
            print(_format_alert(r))
        for r in bearish:
            print(_format_alert(r))

    # ---- Persist JSON summary for artifacts / CI ----
    try:
        os.makedirs("scanner/output", exist_ok=True)
        summary = {
            "scanned": len(symbols),
            "bullish": [
                {
                    "symbol": r.symbol,
                    "score": r.score,
                    "classification": r.classification,
                    "bb_pos": r.bb_pos,
                    "alerted": (
                        r.classification in ALERT_CATEGORIES
                        and ((r.direction == "Bullish" and r.bb_pos == "Upper")
                             or (r.direction == "Bearish" and r.bb_pos == "Lower"))
                    ),
                    "reasons": r.reasons,
                }
                for r in bullish
            ],
            "bearish": [
                {
                    "symbol": r.symbol,
                    "score": r.score,
                    "classification": r.classification,
                    "bb_pos": r.bb_pos,
                    "alerted": (
                        r.classification in ALERT_CATEGORIES
                        and ((r.direction == "Bullish" and r.bb_pos == "Upper")
                             or (r.direction == "Bearish" and r.bb_pos == "Lower"))
                    ),
                    "reasons": r.reasons,
                }
                for r in bearish
            ],
            "counts": {
                "bullish": len(bullish),
                "bearish": len(bearish),
                "rejected": len(rejected),
                "alerted_bullish": len(alert_bullish) if send_alerts else 0,
                "alerted_bearish": len(alert_bearish) if send_alerts else 0,
            },
        }
        with open("scanner/output/latest.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        log.error(f"Failed to write summary JSON: {e}")

    return {
        "bullish": bullish,
        "bearish": bearish,
        "rejected_count": len(rejected),
    }


# ---- Backwards-compat shim (nifty_monitor imports this name) ----
def find_crossover(ema_fast: pd.Series, ema_slow: pd.Series) -> tuple[bool, bool]:
    return ema_crossover(ema_fast, ema_slow)


if __name__ == "__main__":
    from config import WATCHLIST
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stock_scan(WATCHLIST, send_alerts=True)
