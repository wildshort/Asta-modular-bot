import os
import json
import logging
import pandas as pd
import yfinance as yf
from alerts.telegram import send_telegram
from utils.retry import retry_download
from ta.trend import ADXIndicator

# ------------------ LOGGING SETUP ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# --- Helper Functions ---

def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def check_bb_challenge(series: pd.Series) -> str:
    rm = series.rolling(window=20).mean()
    rs = series.rolling(window=20).std()
    upper = rm + 2*rs
    lower = rm - 2*rs
    if series.iloc[-1] > upper.iloc[-1]:
        return "Upper BB Challenge"
    elif series.iloc[-1] < lower.iloc[-1]:
        return "Lower BB Challenge"
    else:
        return "None"

def check_macd_status(macd: pd.Series, signal: pd.Series):
    if macd.iloc[-2] <= signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        return "PCO", True
    elif macd.iloc[-2] >= signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        return "NCO", False
    else:
        return ("PCO" if macd.iloc[-1] > signal.iloc[-1] else "NCO"), macd.iloc[-1] > signal.iloc[-1]

def find_ema_crossover(ema5: pd.Series, ema50: pd.Series):
    try:
        e5_prev, e5_now = ema5.iloc[-2], ema5.iloc[-1]
        e50_prev, e50_now = ema50.iloc[-2], ema50.iloc[-1]
        bullish = (e5_prev <= e50_prev) and (e5_now > e50_now)
        bearish = (e5_prev >= e50_prev) and (e5_now < e50_now)
        return bullish, bearish
    except Exception:
        return False, False

def check_trend_line_breakout(series: pd.Series) -> str:
    try:
        if series.iloc[-1] > series.rolling(window=20).max().iloc[-2]:
            return "Bullish Trend BO"
        elif series.iloc[-1] < series.rolling(window=20).min().iloc[-2]:
            return "Bearish Trend BD"
        else:
            return "None"
    except Exception:
        return "None"

def check_volume_spike(volume: pd.Series) -> bool:
    return len(volume) >= 20 and volume.iloc[-1] > 1.5 * volume.rolling(window=20).mean().iloc[-1]

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series):
    if len(high) < 14:
        return 0.0
    try:
        adx = ADXIndicator(high.squeeze(), low.squeeze(), close.squeeze(), window=14)
        val = adx.adx().iloc[-1]
        return float(val) if pd.notna(val) else 0.0
    except Exception:
        return 0.0

def detect_rsi_divergence(price: pd.Series, rsi: pd.Series) -> str:
    return "None"


# --- Result Formatting & Scanning Logic ---

def format_result_block(results: list[dict], signal_type: str) -> str:
    lines = []
    for r in results:
        lines.append(
            f"{r['Symbol']} ({r['Bias']}):\n"
            f"    📉 Price            : ₹{r['Price']:.2f}\n"
            f"    📊 RSI (Wave/Daily) : {r['RSI Wave']:.2f}\n"
            f"    📊 RSI (Tide/Weekly): {r['RSI Tide']:.2f}\n"
            f"    🎯 BB Challenge      : {r['BB Challenge']}\n"
            f"    💡 MACD (Wave)       : {r['MACD Wave']} ({'>0' if r['MACD Wave > 0'] else '<0'})\n"
            f"    🌊 MACD (Tide)       : {r['MACD Tide']} ({'>0' if r['MACD Tide > 0'] else '<0'})\n"
            f"    🔄 EMA 5/50 Xover   : {r['Crossover']}\n"
            f"    📈 Trend Line        : {r['Trend']}\n"
            f"    💥 Volume Spike      : {'Yes' if r['Volume Spike'] else 'No'}\n"
            f"    😽 ADX Strength      : {r['ADX']:.2f}\n"
            f"    🔍 RSI Divergence    : {r['RSI Divergence']}\n"
        )
    return f"\n\n{signal_type} Signals:\n" + "\n".join(lines)


def run_stock_scan(symbols: list[str], send_alerts: bool = False):
    all_signals = []

    for sym in symbols:
        try:
            # Daily data
            df = retry_download(sym, "6mo", "1d")
            if df.empty or len(df) < 60:
                continue
            close = df['Close'].squeeze()
            high  = df['High'].squeeze()
            low   = df['Low'].squeeze()
            vol   = df['Volume'].squeeze()

            # Compute daily (wave) indicators
            ema5        = close.ewm(span=5, adjust=False).mean()
            ema50       = close.ewm(span=50, adjust=False).mean()
            rsi_wave    = compute_rsi(close).iloc[-1]
            macd_wave, sig_wave, _ = compute_macd(close)
            bb_status   = check_bb_challenge(close)
            macd_wave_status, macd_wave_pos = check_macd_status(macd_wave, sig_wave)
            crossover_bull, crossover_bear = find_ema_crossover(ema5, ema50)
            crossover   = "Bullish" if crossover_bull else "Bearish" if crossover_bear else "None"
            trend_stat  = check_trend_line_breakout(close)
            vol_spike   = check_volume_spike(vol)
            adx_val     = compute_adx(high, low, close)
            rsi_div     = detect_rsi_divergence(close, compute_rsi(close))

            # Weekly data for tide RSI & MACD
            df_week      = retry_download(sym, "1y", "1wk")
            if df_week is None or df_week.empty or len(df_week) < 26:
                continue
            weekly_close = df_week['Close'].squeeze()
            rsi_tide     = compute_rsi(weekly_close).iloc[-1]
            macd_tide, sig_tide, _ = compute_macd(weekly_close)
            macd_tide_status, macd_tide_pos = check_macd_status(macd_tide, sig_tide)

            all_signals.append({
                "Symbol": sym,
                "Price": close.iloc[-1],
                "RSI Wave": rsi_wave,
                "RSI Tide": rsi_tide,
                "BB Challenge": bb_status,
                "MACD Wave": macd_wave_status,
                "MACD Wave > 0": macd_wave_pos,
                "MACD Tide": macd_tide_status,
                "MACD Tide > 0": macd_tide_pos,
                "Crossover": crossover,
                "Trend": trend_stat,
                "Volume Spike": vol_spike,
                "ADX": adx_val,
                "RSI Divergence": rsi_div,
                "Bias": "Neutral"
            })

        except Exception as e:
            logging.error(f"{sym} → error: {e}", exc_info=True)
            continue

    bullish, bearish = [], []

    for r in all_signals:
        # ADX filter
        if r['ADX'] < 12:
            continue

        # Bullish: weekly RSI > 50 AND daily RSI > 60
        if (r['RSI Tide'] > 50 and r['RSI Wave'] > 60
            and r['BB Challenge'] == "Upper BB Challenge"
            and r['Trend'] == "Bullish Trend BO"
            and r['MACD Wave'] == "PCO"
            and r['MACD Tide'] == "PCO"):
            r['Bias'] = 'PCO'
            bullish.append(r)
            logging.info(f"🔔 Bullish → {r['Symbol']}")

        # Bearish: weekly RSI < 50 AND daily RSI < 40
        elif (r['RSI Tide'] < 50 and r['RSI Wave'] < 40
              and r['BB Challenge'] == "Lower BB Challenge"
              and r['Trend'] == "Bearish Trend BD"
              and r['MACD Wave'] == "NCO"
              and r['MACD Tide'] == "NCO"):
            r['Bias'] = 'NCO'
            bearish.append(r)
            logging.info(f"🔕 Bearish → {r['Symbol']}")

        # Output results (console)
    if bullish:
        msg = format_result_block(bullish, "📈 Bullish")
        print(msg)
        # Telegram disabled during testing
        # if send_alerts:
        #     send_telegram(msg)

    if bearish:
        msg = format_result_block(bearish, "📉 Bearish")
        print(msg)
        # Telegram disabled during testing
        # if send_alerts:
        #     send_telegram(msg)

    # --- Always write artifact-friendly outputs ---
    try:
        os.makedirs("scanner/output", exist_ok=True)

        # Compact machine-readable summary
        summary = {
            "bullish": [r["Symbol"] for r in bullish],
            "bearish": [r["Symbol"] for r in bearish],
            "counts": {"bullish": len(bullish), "bearish": len(bearish)}
        }
        with open("scanner/output/latest.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Optional: simple text list (handy to open in Actions UI)
        with open("scanner/output/latest.txt", "w") as f:
            f.write("Bullish:\n")
            for s in summary["bullish"]:
                f.write(f"- {s}\n")
            f.write("\nBearish:\n")
            for s in summary["bearish"]:
                f.write(f"- {s}\n")

    except Exception as e:
        logging.error(f"Failed to write artifacts: {e}", exc_info=True)
if __name__ == "__main__":
    from watchlist.nifty_stocks import watchlist
    logging.info("⏳ Market scanner starting now...")
    run_stock_scan(watchlist, send_alerts=False)
    logging.info("✅ One-time scan complete. Check logs and Telegram.")
