import re
from typing import Literal


TickerType = Literal["etf", "stock", "us_stock"]


def is_chinese_ticker(ticker: str) -> bool:
    """Return True when ticker is a 6-digit mainland China symbol."""
    return bool(re.fullmatch(r"\d{6}", ticker.strip()))


def get_ticker_type(ticker: str) -> TickerType:
    """
    Classify ticker type for routing.

    - ETF: 51/15/56 prefix
    - A-share stock: 60/00/30/68 prefix
    - Otherwise: us_stock
    """
    symbol = ticker.strip()
    if not is_chinese_ticker(symbol):
        return "us_stock"

    if symbol.startswith(("51", "15", "56")):
        return "etf"
    if symbol.startswith(("60", "00", "30", "68")):
        return "stock"
    return "stock"
