# scanner/state_tracker.py
import pickle
import os
import logging

STATE_FILE = "crossover_state.pkl"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"⚠️ Failed to load state: {e}")
    return {}

def save_state(state: dict):
    try:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(state, f)
    except Exception as e:
        logging.warning(f"❌ Failed to save state: {e}")
