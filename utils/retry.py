"""
Backwards-compatibility shim. The old code had two copies of retry_download.
All new code should import from utils.fetcher instead.
"""
from utils.fetcher import download_single


def retry_download(symbol: str, period: str, interval: str, retries: int = 3, delay: int = 2):
    """Deprecated: use utils.fetcher.download_single or download_bulk instead."""
    return download_single(symbol, period, interval)
