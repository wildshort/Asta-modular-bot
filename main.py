import logging
from datetime import datetime
from scanner.stock_scanner import run_stock_scan
# ✅ Import the centralized watchlist from config
from config import WATCHLIST 

# 🧾 Setup logging
log_file = f"logs/market_scan_{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

print("⏳ Market scanner starting now...")

# 🔁 Run scan with ALERTS ENABLED
run_stock_scan(WATCHLIST, send_alerts=True)

print("✅ One-time scan complete. Check logs and Telegram.")
