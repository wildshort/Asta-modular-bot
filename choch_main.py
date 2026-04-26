"""
Entry point for the ChoCH trial scanner.

Run via:
    python choch_main.py

Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars (consumed indirectly
through alerts.telegram which reads them via config.py).

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

# --- Imports matched to the actual repo ---
from utils.fetcher import download_bulk
from alerts.telegram import send_telegram
from watchlist.nifty_stocks import watchlist

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

    # The watchlist mixes NSE equities (.NS) and commodity futures (=F).
    # ChoCH on commodities is meaningful too, but for a clean trial keep
    # only equities — commodity volume/turnover semantics differ.
    all_symbols = list(watchlist)
    equity_symbols = [s for s in all_symbols if s.endswith(".NS")]
    log.info(
        "Watchlist: %d total, %d equities (filtered for trial)",
        len(all_symbols),
        len(equity_symbols),
    )

    log.info("Fetching daily data for %d tickers...", len(equity_symbols))
    price_data = download_bulk(equity_symbols, period="1y", interval="1d")
    log.info("Got data for %d/%d tickers.", len(price_data), len(equity_symbols))

    if not price_data:
        log.error("No price data returned — aborting.")
        send_telegram("⚠️ ChoCH trial: no price data returned from fetcher.")
        return 2

    signals = scan_watchlist(price_data)
    log.info("Scan complete. %d ChoCH signals detected.", len(signals))

    # Persist results for the GitHub Actions Run Summary.
    output_path = OUTPUT_DIR / "choch_latest.json"
    with output_path.open("w") as f:
        json.dump(
            {
                "run_at": datetime.now().isoformat(),
                "scanned_count": len(price_data),
                "signal_count": len(signals),
                "signals": signals,
            },
            f,
            indent=2,
            default=str,
        )
    log.info("Wrote results to %s", output_path)

    if not signals:
        # Heartbeat so we know the run actually executed.
        send_telegram(
            f"🔄 ChoCH trial run: 0 signals across {len(price_data)} tickers."
        )
        return 0

    # Send each signal as its own Telegram message.
    sent = 0
    for sig in signals:
        msg = format_telegram_message(sig)
        try:
            ok = send_telegram(msg)
            if ok:
                sent += 1
                log.info("Sent: %s %s", sig["ticker"], sig["direction"])
            else:
                log.warning("send_telegram returned False for %s", sig["ticker"])
        except Exception:
            log.exception("Failed to send Telegram message for %s", sig["ticker"])

    log.info("Sent %d/%d signal messages.", sent, len(signals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
