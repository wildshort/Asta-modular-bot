"""
Entry point for the Asta scanner.

Env vars required:
  TELEGRAM_BOT_TOKEN  (from BotFather)
  TELEGRAM_CHAT_ID    (your Telegram chat/channel ID)

Run:
  python main.py
"""
import logging
import sys
from datetime import datetime

from config import (
    CHAT_ID,
    LOG_FILE,
    SIGNAL_MIN_SCORE,
    TELEGRAM_TOKEN,
    WATCHLIST,
    is_market_day,
)
from scanner.stock_scanner import run_stock_scan


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Quiet down noisy libraries
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main() -> int:
    setup_logging()
    log = logging.getLogger("main")

    log.info("=" * 60)
    log.info(f"Asta scanner starting @ {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Watchlist: {len(WATCHLIST)} symbols")
    log.info(f"Min score: {SIGNAL_MIN_SCORE}/100 | Market day: {is_market_day()}")
    log.info("=" * 60)

    # Credential sanity check
    creds_ok = bool(TELEGRAM_TOKEN and CHAT_ID)
    if not creds_ok:
        log.warning(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in environment. "
            "Running in DRY-RUN mode (no Telegram alerts will be sent)."
        )

    try:
        result = run_stock_scan(WATCHLIST, send_alerts=creds_ok)
    except Exception as e:
        log.exception(f"Scanner crashed: {e}")
        return 1

    log.info("=" * 60)
    log.info(
        f"Scan done: {len(result['bullish'])} bullish, "
        f"{len(result['bearish'])} bearish, "
        f"{result['rejected_count']} rejected."
    )
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
