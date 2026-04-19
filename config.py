"""
Central configuration for the Asta scanner.

SECURITY:
  Telegram credentials are read ONLY from environment variables.
  Never commit real tokens to the repo. Set via:
    - GitHub Actions: Settings -> Secrets -> Actions
    - Local:          export TELEGRAM_BOT_TOKEN=...
                      export TELEGRAM_CHAT_ID=...
"""
import datetime
import os

# ------------------ TELEGRAM (env-only, no hardcoded fallback) ------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID        = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# ------------------ SIGNAL TUNING KNOBS ------------------
# Master threshold: minimum composite score (0-100) required to fire a signal.
#   80 = Strict   (2-5 signals/day; fewer but higher conviction)
#   70 = Balanced (5-15/day) <- RECOMMENDED DEFAULT
#   60 = Loose    (15-30/day; more noise)
SIGNAL_MIN_SCORE = 70

# Trend strength filter. ADX < 20 = no clear trend (choppy). Industry standard.
MIN_ADX = 20.0

# Volume spike multiplier (current volume vs 20-day average)
VOLUME_SPIKE_MULT = 1.8

# Minimum average daily turnover in INR (filters illiquid stocks).
# 5 crore = 50,000,000. Raise for more conservative filtering.
MIN_AVG_TURNOVER_INR = 5_00_00_000

# Cooldown: don't re-alert same symbol+direction within N calendar days
# unless score improves by at least RE_ALERT_SCORE_DELTA.
ALERT_COOLDOWN_DAYS  = 3
RE_ALERT_SCORE_DELTA = 10

# Drop the last bar if it's still forming (prevents look-ahead bias during market hours).
DROP_UNCLOSED_BAR = True

# Telegram rate limit protection (Telegram allows ~1 msg/sec per chat).
TELEGRAM_MSG_DELAY_SEC = 1.2

# ------------------ WATCHLIST ------------------
WATCHLIST = [
    # Private Banks
    "AUBANK.NS", "AXISBANK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "IDFCFIRSTB.NS", "INDUSINDBK.NS",
    "KOTAKBANK.NS", "RBLBANK.NS", "YESBANK.NS",
    # PSU Banks
    "BANKBARODA.NS", "BANKINDIA.NS", "MAHABANK.NS", "CANBK.NS",
    "CENTRALBK.NS", "INDIANB.NS", "IOB.NS", "PSB.NS",
    "PNB.NS", "SBIN.NS", "UCOBANK.NS", "UNIONBANK.NS",
    # Financial Services
    "BAJFINANCE.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BAJAJFINSV.NS",
    "ICICIGI.NS", "ICICIPRULI.NS", "IRFC.NS", "MUTHOOTFIN.NS",
    "MANAPPURAM.NS", "RECLTD.NS", "SUNDARMFIN.NS", "IIFL.NS", "LICI.NS",
    # IT
    "TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS",
    "COFORGE.NS", "LTIM.NS", "MPHASIS.NS",
    # Pharma
    "ABBOTINDIA.NS", "AJANTPHARM.NS", "ALKEM.NS", "AUROPHARMA.NS",
    "BIOCON.NS", "CIPLA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "GLAND.NS", "GLENMARK.NS", "GRANULES.NS", "IPCALAB.NS",
    "JBCHEPHARM.NS", "LAURUSLABS.NS", "LUPIN.NS", "MANKIND.NS",
    "NATCOPHARM.NS", "TORNTPHARM.NS",
    # FMCG
    "BRITANNIA.NS", "COLPAL.NS", "DABUR.NS", "EMAMILTD.NS",
    "GODREJCP.NS", "HINDUNILVR.NS", "ITC.NS", "MARICO.NS",
    "NESTLEIND.NS", "PATANJALI.NS", "RADICO.NS", "TATACONSUM.NS",
    "UBL.NS", "VBL.NS", "PIDILITIND.NS",
    # Auto
    "ASHOKLEY.NS", "BAJAJ-AUTO.NS", "BALKRISIND.NS", "BHARATFORG.NS",
    "BOSCHLTD.NS", "EICHERMOT.NS", "EXIDEIND.NS", "HEROMOTOCO.NS",
    "MRF.NS", "M&M.NS", "MARUTI.NS", "TVSMOTOR.NS", "TATAMOTORS.NS",
    "TIINDIA.NS", "ESCORTS.NS", "DMART.NS",
    # Metals
    "APLAPOLLO.NS", "ADANIENT.NS", "HINDALCO.NS", "HINDCOPPER.NS",
    "HINDZINC.NS", "JSWSTEEL.NS", "JSL.NS", "JINDALSTEL.NS",
    "NMDC.NS", "NATIONALUM.NS", "SAIL.NS",
    "TATASTEEL.NS", "VEDL.NS", "WELCORP.NS",
    # Energy/Power
    "NTPC.NS", "POWERGRID.NS", "NHPC.NS", "RELIANCE.NS", "ONGC.NS",
    "IOC.NS", "GAIL.NS", "BPCL.NS", "COALINDIA.NS",
    # Capital Goods
    "LT.NS", "SIEMENS.NS", "ABB.NS", "BEL.NS", "BHEL.NS",
    "CUMMINSIND.NS", "HAL.NS",
    # Consumer Durables
    "TITAN.NS", "VOLTAS.NS", "HAVELLS.NS", "WHIRLPOOL.NS",
    "BATAINDIA.NS", "PAGEIND.NS", "PEL.NS",
    # Telecom/Media
    "BHARTIARTL.NS", "IDEA.NS", "DBCORP.NS", "DISHTV.NS", "HATHWAY.NS",
    "NAZARA.NS", "NETWORK18.NS", "PVRINOX.NS", "SAREGAMA.NS",
    "SUNTV.NS", "TIPSMUSIC.NS", "ZEEL.NS",
    # Realty/Infra
    "ANANTRAJ.NS", "BRIGADE.NS", "DLF.NS", "GODREJPROP.NS",
    "LODHA.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "PRESTIGE.NS",
    "RAYMOND.NS", "SOBHA.NS", "ADANIGREEN.NS", "ADANIPORTS.NS",
    "AMBUJACEM.NS", "APOLLOHOSP.NS", "CGPOWER.NS", "GRASIM.NS",
    "HINDPETRO.NS", "INDHOTEL.NS",
    # Utilities
    "ADANIPOWER.NS", "ATGL.NS", "AEGISLOG.NS", "CESC.NS",
    "CASTROLIND.NS", "GUJGASLTD.NS", "GSPL.NS", "POWERINDIA.NS",
    # Exchanges/Others
    "BSE.NS", "MCX.NS", "CDSL.NS", "ADANIENSOL.NS", "INDIGO.NS",
    "NAUKRI.NS", "SHREECEM.NS", "TRENT.NS",
    # Commodities
    "GC=F", "SI=F", "NG=F", "HG=F", "MGC=F", "SIL=F", "QG=F",
]

# Commodity futures don't have INR turnover — exempt from turnover filter
COMMODITY_SUFFIXES = ("=F",)

# ------------------ NSE HOLIDAYS 2025 ------------------
NSE_HOLIDAYS = {
    "2025-01-26", "2025-03-29", "2025-04-14", "2025-05-01", "2025-08-15",
    "2025-10-02", "2025-10-24", "2025-11-11", "2025-12-25",
}


def is_market_day() -> bool:
    today = datetime.datetime.now().date()
    return today.weekday() < 5 and today.isoformat() not in NSE_HOLIDAYS


def is_market_open_now() -> bool:
    """Used to decide whether to drop the unclosed bar."""
    now = datetime.datetime.now()
    if not is_market_day():
        return False
    t = now.time()
    return datetime.time(9, 15) <= t <= datetime.time(15, 30)


# ------------------ PATHS ------------------
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
LOG_FILE = f"{log_dir}/market_scan_{today_str}.log"

STATE_DIR = os.path.join(os.getcwd(), "scanner", "state")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "alert_state.json")
