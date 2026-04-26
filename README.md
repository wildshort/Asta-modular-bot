# Asta Scanner — Improved Version

A Python stock/crypto scanner that pushes high-quality signals to Telegram.
This version replaces the original with substantially fewer false signals.

---

## 🔴 FIRST: Revoke the old Telegram bot token

The previous `config.py` had a hardcoded token and chat ID. Anyone with
access to the old zip can use that bot. **Revoke it immediately:**

1. Open Telegram, message `@BotFather`
2. Send `/revoke` → pick your bot → you'll get a new token
3. Update your GitHub Actions secret and local `.env` with the new token

---

## What changed (and why)

| Area | Old behavior | New behavior |
|---|---|---|
| Signal logic | All-or-nothing boolean AND of 6 criteria | Weighted 0-100 composite score |
| Filter threshold | ADX ≥ 12 (too loose) | ADX ≥ 20 (real trend) |
| Unclosed-bar bias | Used partially-formed intraday bar | Drops last bar during market hours |
| Volume spike | 1.5× (weak) | 1.8× with liquidity floor |
| Liquidity filter | None | Skips illiquid names (< ₹5 cr turnover) |
| RSI divergence | Stub — always returned "None" | Real pivot-based detection |
| Weekly confirmation | Implicit, brittle | Explicit weighted contribution |
| De-dup / cooldown | None — re-alerted every run | 3-day cooldown per symbol+direction |
| Telegram retries | None — silent failures | Exponential backoff, honors 429 |
| yfinance calls | 1 per symbol × 2 timeframes (400+ calls) | 2 bulk calls total |
| Secrets | Hardcoded in source | Env vars only |
| Dead code | `detect_rsi_divergence` stub, dup `retry_download` | Removed |

---

## Setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set credentials (bash/zsh)
export TELEGRAM_BOT_TOKEN="<your new token>"
export TELEGRAM_CHAT_ID="<your chat id>"

# 3. Run
python main.py
```

### GitHub Actions
Add these as repo secrets at Settings → Secrets → Actions:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

---

## Tuning

All tuning knobs live at the top of `config.py`:

```python
SIGNAL_MIN_SCORE     = 70    # 80 = strict (2-5/day), 70 = balanced, 60 = loose
MIN_ADX              = 20.0  # trend strength gate
VOLUME_SPIKE_MULT    = 1.8   # volume vs 20-day avg
MIN_AVG_TURNOVER_INR = 5_00_00_000   # 5 cr/day liquidity floor
ALERT_COOLDOWN_DAYS  = 3     # don't re-alert same symbol within N days
```

**Start with defaults.** After a week or two of watching which signals played
out and which didn't, tune `SIGNAL_MIN_SCORE` up or down.

---

## How the scoring works

Each symbol gets a 0-100 score for each direction. Points come from:

| Criterion | Max points |
|---|---|
| Weekly trend alignment (MACD + RSI > 50) | 20 |
| Daily MACD state | 15 |
| RSI in healthy zone (55-75 bull / 25-45 bear) | 15 |
| Bollinger Band challenge | 10 |
| 20-bar Donchian breakout | 10 |
| Volume spike (≥ 1.8× avg) | 10 |
| EMA 5/50 fresh crossover | 10 |
| ADX strength | 10 |
| RSI divergence (bonus) | 10 |

A signal fires when the total clears `SIGNAL_MIN_SCORE`. Stocks below
`MIN_ADX` or liquidity floor are rejected before scoring.

---

## Why this reduces false signals

1. **Weekly trend gate.** The single biggest source of false signals was
   daily patterns firing against the weekly trend. Now weekly MACD+RSI
   contributes 20 of the 100 points — you can't score high fighting the
   higher timeframe.

2. **Closed-bar enforcement.** The old scanner evaluated indicators on
   the still-forming intraday bar. RSI, MACD, and EMA crossovers would
   flip-flop through the day. New scanner drops the unclosed bar.

3. **Liquidity floor.** Illiquid names produce erratic indicators. Filtering
   by turnover removes most of the worst offenders.

4. **ATR is computed but not yet used for thresholds — future knob.**

5. **Cooldown.** Even a good signal shouldn't re-alert daily. 3-day
   cooldown with score-improvement override is the right default.

---

## File layout

```
main.py                  Entry point with logging + creds check
config.py                All tuning knobs + watchlist + env var reads
alerts/telegram.py       Message/chart sending with retries + rate limiting
scanner/stock_scanner.py Main scoring engine
scanner/state_tracker.py Cooldown persistence (JSON)
scanner/nifty_monitor.py Optional intraday NIFTY monitor
utils/fetcher.py         Batched yfinance downloader
utils/indicators.py      Pure indicator math (RSI, MACD, ADX, ATR, divergence, etc.)
utils/retry.py           Back-compat shim only
```

---

## Debugging / tuning workflow

1. Run once. Check the log file in `logs/`.
2. Every rejected symbol logs *why* it was rejected at DEBUG level.
   Run with `python -c "import logging; logging.basicConfig(level=logging.DEBUG)"`
   or tweak `main.py`.
3. `scanner/output/latest.json` persists every scan's result — diff it
   across days to see what your scanner actually caught.
4. If you get 0 signals for multiple days, lower `SIGNAL_MIN_SCORE` to 65.
   If you get too many, raise to 75.
