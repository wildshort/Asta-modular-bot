import datetime
import os

# ✅ Telegram Bot Details
# Reads from GitHub Secrets first, but defaults to your new keys for local testing
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8511631522:AAFkcZcvfLJd8aaBq01JEXJca-SegB-9wxM")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "707246649")

# ✅ Full Watchlist (Centralized here)
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
    "GC=F","SI=F","NG=F","HG=F","MGC=F","SIL=F","QG=F"
]

# NSE Holidays 2025
NSE_HOLIDAYS = {
    "2025-01-26", "2025-03-29", "2025-04-14", "2025-05-01", "2025-08-15",
    "2025-10-02", "2025-10-24", "2025-11-11", "2025-12-25"
}

def is_market_day():
    today = datetime.datetime.now().date()
    # 0=Mon, 4=Fri. So <5 is Weekday.
    return today.weekday() < 5 and today.isoformat() not in NSE_HOLIDAYS

# Logging directory
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
LOG_FILE = f"{log_dir}/market_scan_{today_str}.log"
