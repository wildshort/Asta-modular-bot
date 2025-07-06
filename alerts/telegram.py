# alerts/telegram.py
import requests
import logging
from config import TELEGRAM_TOKEN, CHAT_ID

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("🚫 Telegram token or chat ID not set.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            logging.info("📤 Telegram message sent successfully.")
        else:
            logging.warning(f"❌ Telegram response error: {response.status_code} - {response.text}")
    except Exception as e:
        logging.warning(f"⚠️ Telegram send failed: {e}")
