import logging
import pandas as pd
import yfinance as yf
from ta.trend import ADXIndicator
import os
import sys

# --- Dynamic Path Setup to Access watchlist ---
current = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current, '..'))  # go one level up from 'scanner'
sys.path.insert(0, project_root)

from watchlist.nifty_stocks import watchlist

# --- Remove symbols that repeatedly fail to load ---
watchlist = [sym for sym in watchlist if sym not in ['ADANITRANS.NS', 'GMRINFRA.NS', 'LAXMIMACH.NS', 'MAHINDCIE.NS']]

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

def check_bb_challenge(series: pd.Series) -> bool:
    rolling_mean = series.rolling(window=20).mean()
    return series.iloc[-1] > rolling_mean.iloc[-1]

def check_macd_status(macd: pd.Series, signal: pd.Series):
    latest = macd.iloc[-1] - signal.iloc[-1]
    return ("PCO" if latest > 0 else "NCO"), latest > 0

def find_ema_crossover(ema5: pd.Series, ema50: pd.Series):
    try:
        e5_prev = ema5.iloc[-2]
        e5_now = ema5.iloc[-1]
        e50_prev = ema50.iloc[-2]
        e50_now = ema50.iloc[-1]
        bullish = (e5_prev <= e50_prev) and (e5_now > e50_now)
        bearish = (e5_prev >= e50_prev) and (e5_now < e50_now)
        return bullish, bearish
    except Exception:
        return False, False

def check_trend_line_breakout(series: pd.Series) -> str:
    try:
        return "Trend BO" if series.iloc[-1] > series.rolling(window=20).max().iloc[-2] else "None"
    except Exception:
        return "None"

def check_volume_spike(volume: pd.Series) -> bool:
    return volume.iloc[-1] > volume.rolling(window=20).mean().iloc[-1] * 1.5

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series):
    adx = ADXIndicator(high.squeeze(), low.squeeze(), close.squeeze(), window=14)
    return adx.adx().iloc[-1]

def detect_rsi_divergence(price: pd.Series, rsi: pd.Series) -> str:
    return "None"

def check_stochastic_pco(close: pd.Series) -> bool:
    low_14 = close.rolling(window=14).min()
    high_14 = close.rolling(window=14).max()
    k_percent = 100 * ((close - low_14) / (high_14 - low_14))
    return k_percent.iloc[-1] > k_percent.iloc[-2] and k_percent.iloc[-2] < 20

def format_result_block(results: list[dict], signal_type: str) -> str:
    lines = []
    for r in results:
        lines.append(
            f"{r['Symbol']} ({r['Bias']}):\n"
            f"   📉 Price         : ₹{r['Price']:.2f}\n"
            f"   🎯 BB Challenge  : {'Yes' if r['BB Challenge'] else 'No'}\n"
            f"   💡 MACD (Wave)   : {r['MACD Wave']} ({'>0' if r['MACD Wave > 0'] else '<0'})\n"
            f"   🌊 MACD (Tide)   : {r['MACD Tide']} ({'>0' if r['MACD Tide > 0'] else '<0'})\n"
            f"   🔄 EMA 5/50 Xover: {r['Crossover']}\n"
            f"   📈 Trend Line    : {r['Trend']}\n"
            f"   💥 Volume Spike  : {'Yes' if r['Volume Spike'] else 'No'}\n"
            f"   😽 ADX Strength  : {r['ADX']:.2f}\n"
            f"   🔍 RSI Divergence: {r['RSI Divergence']}\n"
            f"   🔮 SOBBO         : {r['SOBBO']}\n"
            f"   📉 Stoch. PCO    : {'Yes' if r['Stochastic PCO'] else 'No'}\n"
        )
    return f"\n\n{signal_type} Signals:\n" + "\n".join(lines)

def retry_download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    for _ in range(3):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()

def run_stock_scan(symbols: list[str], send_alerts: bool = False):
    all_results = []
    skipped_symbols = []

    for sym in symbols:
        try:
            df = retry_download(sym, "6mo", "1d")
            if df.empty or df.shape[0] < 60:
                skipped_symbols.append(sym)
                continue

            if df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
                skipped_symbols.append(sym)
                continue

            open_ = df['Open'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            close = df['Close'].squeeze()
            volume = df['Volume'].squeeze()

            ema5 = close.ewm(span=5).mean()
            ema50 = close.ewm(span=50).mean()
            rsi = compute_rsi(close)
            macd_wave, signal_wave, _ = compute_macd(close)

            df_week = retry_download(sym, "1y", "1wk")
            if df_week is None or df_week.shape[0] < 26:
                skipped_symbols.append(sym)
                continue
            macd_tide, signal_tide, _ = compute_macd(df_week["Close"].squeeze())

            bb = check_bb_challenge(close)
            macd_wave_status, macd_wave_positive = check_macd_status(macd_wave, signal_wave)
            macd_tide_status, macd_tide_positive = check_macd_status(macd_tide, signal_tide)
            crossover_bullish, crossover_bearish = find_ema_crossover(ema5, ema50)
            crossover = "Bullish" if crossover_bullish else "Bearish" if crossover_bearish else "None"
            trend = check_trend_line_breakout(close)
            vol_spike = check_volume_spike(volume)
            adx_val = compute_adx(high, low, close)
            rsi_div = detect_rsi_divergence(close, rsi)
            stochastic_pco = check_stochastic_pco(close)

            sobbo = "Yes" if close.iloc[-1] > close.iloc[-5:-1].min() * 1.02 else "No"

            if bb and trend.startswith("Trend") and macd_wave_status == macd_tide_status:
                all_results.append({
                    "Symbol": sym,
                    "Price": close.iloc[-1],
                    "BB Challenge": bb,
                    "MACD Wave": macd_wave_status,
                    "MACD Wave > 0": macd_wave_positive,
                    "MACD Tide": macd_tide_status,
                    "MACD Tide > 0": macd_tide_positive,
                    "Crossover": crossover,
                    "Trend": trend,
                    "Volume Spike": vol_spike,
                    "ADX": adx_val,
                    "RSI Divergence": rsi_div,
                    "Bias": macd_wave_status if macd_wave_status == macd_tide_status else "Neutral",
                    "SOBBO": sobbo,
                    "Stochastic PCO": stochastic_pco
                })

        except Exception:
            skipped_symbols.append(sym)
            continue

    if all_results:
        bullish = [r for r in all_results if r['Bias'] == 'PCO']
        bearish = [r for r in all_results if r['Bias'] == 'NCO']

        if bullish:
            msg = format_result_block(bullish, "📈 Bullish")
            print(msg)

        if bearish:
            msg = format_result_block(bearish, "📉 Bearish")
            print(msg)

    if skipped_symbols:
        print(f"\n❌ Skipped {len(skipped_symbols)} symbols (no data or delisted): {skipped_symbols}\n")

# --- Trigger the scan if needed ---

if __name__ == "__main__":
    run_stock_scan(watchlist, send_alerts=False)
