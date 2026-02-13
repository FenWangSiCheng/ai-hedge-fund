import pandas as pd

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)
from src.tools import akshare_api, api_us
from src.utils.ticker import is_chinese_ticker


def _is_china_market_ticker(ticker: str) -> bool:
    return is_chinese_ticker(ticker)


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    if _is_china_market_ticker(ticker):
        return akshare_api.get_prices(ticker, start_date, end_date, api_key=api_key)
    return api_us.get_prices(ticker, start_date, end_date, api_key=api_key)


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    if _is_china_market_ticker(ticker):
        return akshare_api.get_financial_metrics(ticker, end_date, period=period, limit=limit, api_key=api_key)
    return api_us.get_financial_metrics(ticker, end_date, period=period, limit=limit, api_key=api_key)


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    if _is_china_market_ticker(ticker):
        return akshare_api.search_line_items(
            ticker,
            line_items,
            end_date,
            period=period,
            limit=limit,
            api_key=api_key,
        )
    return api_us.search_line_items(
        ticker,
        line_items,
        end_date,
        period=period,
        limit=limit,
        api_key=api_key,
    )


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    if _is_china_market_ticker(ticker):
        return akshare_api.get_insider_trades(
            ticker,
            end_date,
            start_date=start_date,
            limit=limit,
            api_key=api_key,
        )
    return api_us.get_insider_trades(
        ticker,
        end_date,
        start_date=start_date,
        limit=limit,
        api_key=api_key,
    )


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    if _is_china_market_ticker(ticker):
        return akshare_api.get_company_news(
            ticker,
            end_date,
            start_date=start_date,
            limit=limit,
            api_key=api_key,
        )
    return api_us.get_company_news(
        ticker,
        end_date,
        start_date=start_date,
        limit=limit,
        api_key=api_key,
    )


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    if _is_china_market_ticker(ticker):
        return akshare_api.get_market_cap(ticker, end_date, api_key=api_key)
    return api_us.get_market_cap(ticker, end_date, api_key=api_key)


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    return api_us.prices_to_df(prices)


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    if _is_china_market_ticker(ticker):
        return akshare_api.get_price_data(ticker, start_date, end_date, api_key=api_key)
    return api_us.get_price_data(ticker, start_date, end_date, api_key=api_key)
