import logging
import pandas as pd
import yfinance as yf
from alerts.telegram import send_telegram
from utils.retry import retry_download
from ta.trend import ADXIndicator

# --- Helper Functions (No changes in this section) ---

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
    rolling_mean = series.rolling(window=20).mean()
    rolling_std = series.rolling(window=20).std()
    upper = rolling_mean + (2 * rolling_std)
    lower = rolling_mean - (2 * rolling_std)
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
        if series.iloc[-1] > series.rolling(window=20).max().iloc[-2]:
            return "Bullish Trend BO"
        elif series.iloc[-1] < series.rolling(window=20).min().iloc[-2]:
            return "Bearish Trend BD"
        else:
            return "None"
    except Exception:
        return "None"

def check_volume_spike(volume: pd.Series) -> bool:
    if len(volume) < 20:
        return False
    return volume.iloc[-1] > volume.rolling(window=20).mean().iloc[-1] * 1.5

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series):
    if high.empty or low.empty or close.empty or len(high) < 14:
        return 0.0
    try:
        adx = ADXIndicator(high.squeeze(), low.squeeze(), close.squeeze(), window=14)
        adx_value = adx.adx().iloc[-1]
        return adx_value if pd.notna(adx_value) else 0.0 # Ensure it's not NaN
    except Exception:
        return 0.0

def detect_rsi_divergence(price: pd.Series, rsi: pd.Series) -> str:
    return "None"

# --- Result Formatting and Scanning Logic (Section with changes) ---

def format_result_block(results: list[dict], signal_type: str) -> str:
    lines = []
    for r in results:
        lines.append(
            f"{r['Symbol']} ({r['Bias']}):\n"
            f"    📉 PriceRaj         : ₹{r['Price']:.2f}\n"
            f"    🎯 BB Challenge   : {r['BB Challenge']}\n"
            f"    💡 MACD (Wave)    : {r['MACD Wave']} ({'>0' if r['MACD Wave > 0'] else '<0'})\n"
            f"    🌊 MACD (Tide)    : {r['MACD Tide']} ({'>0' if r['MACD Tide > 0'] else '<0'})\n"
            f"    🔄 EMA 5/50 Xover: {r['Crossover']}\n"
            f"    📈 Trend Line     : {r['Trend']}\n"
            f"    💥 Volume Spike   : {'Yes' if r['Volume Spike'] else 'No'}\n"
            f"    😽 ADX Strength   : {r['ADX']:.2f}\n"
            f"    🔍 RSI Divergence: {r['RSI Divergence']}\n"
        )
    return f"\n\n{signal_type} Signals:\n" + "\n".join(lines)

def run_stock_scan(symbols: list[str], send_alerts: bool = False):
    all_potential_signals = [] 

    for sym in symbols:
        try:
            df = retry_download(sym, "6mo", "1d")
            if df.empty or df.shape[0] < 60:
                continue

            if df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
                continue

            open_ = df['Open'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            close = df['Close'].squeeze()
            volume = df['Volume'].squeeze()

            ema5 = close.ewm(span=5, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            rsi = compute_rsi(close)
            macd_wave, signal_wave, _ = compute_macd(close)

            df_week = retry_download(sym, "1y", "1wk")
            if df_week is None or df_week.empty or df_week.shape[0] < 26:
                continue
            macd_tide, signal_tide, _ = compute_macd(df_week["Close"].squeeze())

            bb_status = check_bb_challenge(close)
            macd_wave_status, macd_wave_positive = check_macd_status(macd_wave, signal_wave)
            macd_tide_status, macd_tide_positive = check_macd_status(macd_tide, signal_tide)
            crossover_bullish, crossover_bearish = find_ema_crossover(ema5, ema50)
            crossover = "Bullish" if crossover_bullish else "Bearish" if crossover_bearish else "None"
            trend_status = check_trend_line_breakout(close)
            vol_spike = check_volume_spike(volume)
            adx_val = compute_adx(high, low, close)
            rsi_div = detect_rsi_divergence(close, rsi)

            all_potential_signals.append({
                "Symbol": sym,
                "Price": close.iloc[-1],
                "BB Challenge": bb_status,
                "MACD Wave": macd_wave_status,
                "MACD Wave > 0": macd_wave_positive,
                "MACD Tide": macd_tide_status,
                "MACD Tide > 0": macd_tide_positive,
                "Crossover": crossover,
                "Trend": trend_status,
                "Volume Spike": vol_spike,
                "ADX": adx_val,
                "RSI Divergence": rsi_div,
                "Bias": "Neutral"
            })

        except Exception as e:
            continue

    bullish_final_signals = []
    bearish_final_signals = []

    for r in all_potential_signals:
        # ------------------- START OF MODIFICATION -------------------
        # First, check ADX. If it's below 12, skip this stock entirely.
        if r['ADX'] < 12:
            continue
        # -------------------- END OF MODIFICATION --------------------

        is_bullish = False
        if (r['BB Challenge'] == "Upper BB Challenge" and
            r['Trend'] == "Bullish Trend BO" and
            r['MACD Wave'] == "PCO" and r['MACD Tide'] == "PCO"):
            is_bullish = True
        elif (r['Crossover'] == "Bullish" and
              r['MACD Wave'] == "PCO" and r['MACD Tide'] == "PCO"):
            is_bullish = True
        elif (r['BB Challenge'] == "Upper BB Challenge" and
              r['Crossover'] == "Bullish"):
            is_bullish = True
        
        is_bearish = False
        if (r['BB Challenge'] == "Lower BB Challenge" and
            r['Trend'] == "Bearish Trend BD" and
            r['MACD Wave'] == "NCO" and r['MACD Tide'] == "NCO"):
            is_bearish = True
        elif (r['Crossover'] == "Bearish" and
              r['MACD Wave'] == "NCO" and r['MACD Tide'] == "NCO"):
            is_bearish = True
        elif (r['BB Challenge'] == "Lower BB Challenge" and
              r['Crossover'] == "Bearish"):
            is_bearish = True

        # Assign bias based on the determined conditions
        if is_bullish and not is_bearish:
            r['Bias'] = 'PCO'
            bullish_final_signals.append(r)
        elif is_bearish and not is_bullish:
            r['Bias'] = 'NCO'
            bearish_final_signals.append(r)

    if bullish_final_signals:
        msg = format_result_block(bullish_final_signals, "📈 Bullish")
        print(msg)
        if send_alerts:
            send_telegram(msg)

    if bearish_final_signals:
        msg = format_result_block(bearish_final_signals, "📉 Bearish")
        print(msg)
        if send_alerts:
            send_telegram(msg)

if __name__ == "__main__":
    from watchlist.nifty_stocks import watchlist
    run_stock_scan(watchlist, send_alerts=True)
