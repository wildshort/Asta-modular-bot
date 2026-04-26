"""
Entry point for the ChoCH trial scanner.

Run via:
    python choch_main.py

Reads the same TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars as main.py.
For the trial, this is meant to be triggered manually via GitHub Actions
(workflow_dispatch). Cron schedule is intentionally NOT set yet.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Local imports — assume the same module layout as the rest of the repo.
# If fetcher exposes a different function name, change this line and ONLY this line.
from utils.fetcher import fetch_batch  # noqa: F401  -- assumed signature, see notes
from alerts.telegram import send_message  # noqa: F401  -- assumed function name
from watchlist.nifty_stocks import NIFTY_STOCKS  # noqa: F401  -- assumed list name

from scanner.choch_scanner import scan_watchlist, format_telegram_message


# --- Logging setup mirrors main.py style ---
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("scanner/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / f"choch_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("choch_main")


def check_credentials() -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars.")
        return False
    log.info("Credentials present. chat_id ends with ...%s", chat_id[-4:])
    return True


def main() -> int:
    log.info("=" * 60)
    log.info("ChoCH trial scanner starting")
    log.info("=" * 60)

    if not check_credentials():
        return 1

    tickers = list(NIFTY_STOCKS)
    log.info("Fetching daily data for %d tickers...", len(tickers))

    # ---- ASSUMPTION: fetch_batch(tickers, period, interval) -> dict[str, DataFrame]
    # If your fetcher's signature differs, edit this single call.
    try:
        price_data = fetch_batch(tickers, period="1y", interval="1d")
    except TypeError:
        # Fallback in case fetcher uses different kwargs
        log.warning("fetch_batch signature mismatch — trying positional only.")
        price_data = fetch_batch(tickers)

    log.info("Got data for %d/%d tickers.", len(price_data), len(tickers))

    signals = scan_watchlist(price_data)
    log.info("Scan complete. %d ChoCH signals detected.", len(signals))

    # Persist results for the GitHub Actions Run Summary.
    output_path = OUTPUT_DIR / "choch_latest.json"
    with output_path.open("w") as f:
        json.dump(
            {
                "run_at": datetime.now().isoformat(),
                "signal_count": len(signals),
                "signals": signals,
            },
            f,
            indent=2,
            default=str,
        )
    log.info("Wrote results to %s", output_path)

    if not signals:
        # Send a heartbeat so we know the run actually executed.
        send_message("🔄 ChoCH trial run: 0 signals detected.")
        return 0

    # Send each signal as its own Telegram message — easier to mute/forward individually.
    for sig in signals:
        msg = format_telegram_message(sig)
        try:
            send_message(msg)
            log.info("Sent: %s %s", sig["ticker"], sig["direction"])
        except Exception:
            log.exception("Failed to send Telegram message for %s", sig["ticker"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
