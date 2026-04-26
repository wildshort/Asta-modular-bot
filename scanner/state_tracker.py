"""
Persistent state for alert cooldown / de-duplication.

Uses JSON (not pickle) for safety, readability, and Git-diffability.
Records the last alert per (symbol, direction) with timestamp and score,
so we can suppress repeat alerts unless the signal has meaningfully strengthened.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict

from config import ALERT_COOLDOWN_DAYS, RE_ALERT_SCORE_DELTA, STATE_FILE

log = logging.getLogger(__name__)


def load_state() -> Dict[str, dict]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"State file unreadable, starting fresh: {e}")
        return {}


def save_state(state: Dict[str, dict]) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, STATE_FILE)  # atomic write
    except Exception as e:
        log.error(f"Failed to save state: {e}")


def _key(symbol: str, direction: str) -> str:
    return f"{symbol}::{direction}"


def should_alert(state: Dict[str, dict], symbol: str, direction: str, score: float) -> bool:
    """
    Return True if we should fire an alert for this (symbol, direction).
    Suppresses alerts within ALERT_COOLDOWN_DAYS unless the score has improved
    by at least RE_ALERT_SCORE_DELTA.
    """
    entry = state.get(_key(symbol, direction))
    if not entry:
        return True

    try:
        last_time = datetime.fromisoformat(entry["ts"])
    except Exception:
        return True

    if datetime.now() - last_time > timedelta(days=ALERT_COOLDOWN_DAYS):
        return True

    last_score = float(entry.get("score", 0))
    if score >= last_score + RE_ALERT_SCORE_DELTA:
        log.info(
            f"{symbol} [{direction}]: re-alerting, score improved "
            f"{last_score:.0f} -> {score:.0f}"
        )
        return True

    return False


def record_alert(state: Dict[str, dict], symbol: str, direction: str, score: float) -> None:
    state[_key(symbol, direction)] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "score": round(score, 2),
    }
