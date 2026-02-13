import contextlib
import datetime
import io
import time

import akshare as ak
import pandas as pd

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)
from src.utils.ticker import get_ticker_type

# Global cache instance
_cache = get_cache()

# Financial frames cache to avoid repeated heavy requests per ticker
_financial_frames_cache: dict[str, dict[str, pd.DataFrame]] = {}


def _retry(func, max_retries: int = 2, base_sleep: float = 0.8):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(base_sleep * (attempt + 1))
    raise last_error


def _quiet_call(func, *args, **kwargs):
    """Suppress noisy tqdm/progress output from AkShare internals."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return func(*args, **kwargs)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text or text in {"--", "nan", "None", "False"}:
            return None
        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100.0
            except ValueError:
                return None
        unit_scale = 1.0
        if text.endswith("亿"):
            unit_scale = 1e8
            text = text[:-1]
        elif text.endswith("万"):
            unit_scale = 1e4
            text = text[:-1]
        try:
            return float(text) * unit_scale
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _normalize_growth(value: float | None) -> float | None:
    if value is None:
        return None
    # Some sources use percentage points (6.5) while others already use ratios (0.065).
    if abs(value) > 2:
        return value / 100.0
    return value


def _normalize_margin_pct(value: float | None) -> float | None:
    if value is None:
        return None
    if abs(value) > 1:
        return value / 100.0
    return value


def _to_report_date(value) -> str | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _to_ak_date(date_text: str) -> str:
    return date_text.replace("-", "")


def _to_cn_symbol(ticker: str) -> str:
    # Eastmoney report APIs expect sh/sz prefix.
    if ticker.startswith(("5", "6", "9")):
        return f"sh{ticker}"
    return f"sz{ticker}"


def _filter_date_range(df: pd.DataFrame, date_column: str, start_date: str, end_date: str) -> pd.DataFrame:
    if df is None or df.empty or date_column not in df.columns:
        return pd.DataFrame() if df is None else df

    start_dt = pd.to_datetime(start_date, errors="coerce")
    end_dt = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt):
        return df

    out = df.copy()
    out["_date"] = pd.to_datetime(out[date_column], errors="coerce")
    out = out[out["_date"].notna()]
    out = out[(out["_date"] >= start_dt) & (out["_date"] <= end_dt)]
    out = out.drop(columns=["_date"])
    return out


def _first_non_null(record: dict, keys: list[str]) -> float | None:
    for key in keys:
        if key in record:
            value = _safe_float(record.get(key))
            if value is not None:
                return value
    return None


def _sum_non_null(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(sum(valid))


def _normalize_report_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    if "REPORT_DATE" in normalized.columns:
        normalized["report_period"] = normalized["REPORT_DATE"].apply(_to_report_date)
    elif "report_date" in normalized.columns:
        normalized["report_period"] = normalized["report_date"].apply(_to_report_date)
    else:
        return pd.DataFrame()

    normalized = normalized[normalized["report_period"].notna()]
    normalized = normalized.sort_values("report_period", ascending=False)
    return normalized


def _get_abstract_lookup(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    if df is None or df.empty:
        return lookup

    normalized = _normalize_report_df(df)
    if normalized.empty or "metric_name" not in normalized.columns:
        return lookup

    for _, row in normalized.iterrows():
        report_period = row.get("report_period")
        metric_name = row.get("metric_name")
        if not report_period or not metric_name:
            continue

        metric_value = _safe_float(row.get("value"))
        metric_yoy = _safe_float(row.get("yoy"))
        if report_period not in lookup:
            lookup[report_period] = {}

        if metric_value is not None:
            lookup[report_period][metric_name] = metric_value
        if metric_yoy is not None:
            lookup[report_period][f"{metric_name}__yoy"] = metric_yoy

    return lookup


def _get_financial_frames(ticker: str) -> dict[str, pd.DataFrame]:
    if ticker in _financial_frames_cache:
        return _financial_frames_cache[ticker]

    symbol = _to_cn_symbol(ticker)

    frames: dict[str, pd.DataFrame] = {
        "profit": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "abstract": pd.DataFrame(),
    }

    try:
        frames["profit"] = _retry(lambda: _quiet_call(ak.stock_profit_sheet_by_report_em, symbol=symbol))
    except Exception:
        pass

    try:
        frames["balance"] = _retry(lambda: _quiet_call(ak.stock_balance_sheet_by_report_em, symbol=symbol))
    except Exception:
        pass

    try:
        frames["cashflow"] = _retry(lambda: _quiet_call(ak.stock_cash_flow_sheet_by_report_em, symbol=symbol))
    except Exception:
        pass

    try:
        frames["abstract"] = _retry(lambda: _quiet_call(ak.stock_financial_abstract_new_ths, symbol=ticker))
    except Exception:
        pass

    for key in ("profit", "balance", "cashflow", "abstract"):
        frames[key] = _normalize_report_df(frames[key])

    _financial_frames_cache[ticker] = frames
    return frames


def _build_financial_records(ticker: str) -> list[dict]:
    frames = _get_financial_frames(ticker)
    abstract_lookup = _get_abstract_lookup(frames["abstract"])

    maps: dict[str, dict[str, dict]] = {}
    all_dates: set[str] = set()

    for name in ("profit", "balance", "cashflow"):
        df = frames[name]
        if df.empty:
            maps[name] = {}
            continue
        map_by_date = {
            row["report_period"]: row.to_dict()
            for _, row in df.iterrows()
            if row.get("report_period")
        }
        maps[name] = map_by_date
        all_dates.update(map_by_date.keys())

    all_dates.update(abstract_lookup.keys())

    records: list[dict] = []
    for report_period in sorted(all_dates, reverse=True):
        record = {"report_period": report_period, "currency": "CNY"}
        for name in ("profit", "balance", "cashflow"):
            record.update(maps.get(name, {}).get(report_period, {}))
        record["_abstract"] = abstract_lookup.get(report_period, {})
        records.append(record)

    return records


def _compute_base_values(record: dict) -> dict[str, float | None]:
    abstract = record.get("_abstract", {})

    revenue = _first_non_null(record, ["TOTAL_OPERATE_INCOME", "OPERATE_INCOME"])
    net_income = _first_non_null(record, ["PARENT_NETPROFIT", "NETPROFIT", "parent_holder_net_profit"])
    operating_income = _first_non_null(record, ["OPERATE_PROFIT"])

    total_assets = _first_non_null(record, ["TOTAL_ASSETS"])
    total_liabilities = _first_non_null(record, ["TOTAL_LIABILITIES"])
    current_assets = _first_non_null(record, ["TOTAL_CURRENT_ASSETS"])
    current_liabilities = _first_non_null(record, ["TOTAL_CURRENT_LIAB"])
    shareholders_equity = _first_non_null(record, ["TOTAL_PARENT_EQUITY", "TOTAL_EQUITY"])
    outstanding_shares = _first_non_null(record, ["SHARE_CAPITAL"])
    cash_and_equivalents = _first_non_null(record, ["MONETARYFUNDS", "END_CCE"])

    interest_expense = _first_non_null(record, ["INTEREST_EXPENSE"])
    income_tax = _first_non_null(record, ["INCOME_TAX"])
    total_profit = _first_non_null(record, ["TOTAL_PROFIT"])

    debt_components = [
        _first_non_null(record, ["SHORT_LOAN"]),
        _first_non_null(record, ["LONG_LOAN"]),
        _first_non_null(record, ["BOND_PAYABLE"]),
        _first_non_null(record, ["NONCURRENT_LIAB_1YEAR"]),
        _first_non_null(record, ["LOAN_ADVANCE"]),
    ]
    total_debt = _sum_non_null(debt_components)
    if total_debt is None:
        total_debt = total_liabilities

    gross_profit = _first_non_null(record, ["GROSS_PROFIT"])
    if gross_profit is None:
        total_cost = _first_non_null(record, ["TOTAL_OPERATE_COST"])
        if revenue is not None and total_cost is not None:
            gross_profit = revenue - total_cost

    operating_margin = None
    if operating_income is not None and revenue not in (None, 0):
        operating_margin = operating_income / revenue

    gross_margin = None
    abstract_gross_margin = _safe_float(abstract.get("sale_gross_margin"))
    if abstract_gross_margin is not None:
        gross_margin = _normalize_margin_pct(abstract_gross_margin)
    elif gross_profit is not None and revenue not in (None, 0):
        gross_margin = gross_profit / revenue

    net_margin = None
    abstract_net_margin = _safe_float(abstract.get("sale_net_interest_ratio"))
    if abstract_net_margin is not None:
        net_margin = _normalize_margin_pct(abstract_net_margin)
    elif net_income is not None and revenue not in (None, 0):
        net_margin = net_income / revenue

    capex_raw = _first_non_null(record, ["CONSTRUCT_LONG_ASSET"])
    capital_expenditure = -abs(capex_raw) if capex_raw is not None else None

    depreciation_and_amortization = _sum_non_null(
        [
            _first_non_null(record, ["FA_IR_DEPR"]),
            _first_non_null(record, ["IA_AMORTIZE"]),
            _first_non_null(record, ["LPE_AMORTIZE"]),
            _first_non_null(record, ["USERIGHT_ASSET_AMORTIZE"]),
        ]
    )

    operating_cash_flow = _first_non_null(record, ["NETCASH_OPERATE"])
    free_cash_flow = None
    if operating_cash_flow is not None:
        free_cash_flow = operating_cash_flow - abs(capex_raw or 0)

    tax_rate = None
    if income_tax is not None and total_profit not in (None, 0):
        tax_rate = max(0.0, min(0.5, income_tax / total_profit))

    ebit = None
    if operating_income is not None:
        ebit = operating_income + (interest_expense or 0)
    elif net_income is not None:
        ebit = net_income + (income_tax or 0) + (interest_expense or 0)

    ebitda = None
    if ebit is not None:
        ebitda = ebit + (depreciation_and_amortization or 0)

    return_on_invested_capital = None
    if operating_income is not None:
        effective_tax = tax_rate if tax_rate is not None else 0.25
        invested_capital = (shareholders_equity or 0) + (total_debt or 0) - (cash_and_equivalents or 0)
        if invested_capital > 0:
            return_on_invested_capital = operating_income * (1 - effective_tax) / invested_capital

    operating_expense = _sum_non_null(
        [
            _first_non_null(record, ["SALE_EXPENSE"]),
            _first_non_null(record, ["MANAGE_EXPENSE"]),
            _first_non_null(record, ["FINANCE_EXPENSE"]),
            _first_non_null(record, ["RESEARCH_EXPENSE"]),
        ]
    )

    research_and_development = _first_non_null(record, ["RESEARCH_EXPENSE"])
    goodwill_and_intangible_assets = _sum_non_null(
        [
            _first_non_null(record, ["GOODWILL"]),
            _first_non_null(record, ["INTANGIBLE_ASSET"]),
        ]
    )

    earnings_per_share = _first_non_null(record, ["BASIC_EPS"])
    if earnings_per_share is None:
        earnings_per_share = _safe_float(abstract.get("basic_eps"))

    book_value_per_share = _safe_float(abstract.get("calc_per_net_assets"))
    if book_value_per_share is None and shareholders_equity not in (None, 0) and outstanding_shares not in (None, 0):
        book_value_per_share = shareholders_equity / outstanding_shares

    dividends = _first_non_null(record, ["ASSIGN_DIVIDEND_PORFIT"])

    issuance_or_purchase_of_equity_shares = _first_non_null(record, ["ACCEPT_INVEST_CASH"])

    return {
        "revenue": revenue,
        "net_income": net_income,
        "operating_income": operating_income,
        "gross_profit": gross_profit,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "shareholders_equity": shareholders_equity,
        "outstanding_shares": outstanding_shares,
        "cash_and_equivalents": cash_and_equivalents,
        "interest_expense": interest_expense,
        "income_tax": income_tax,
        "total_profit": total_profit,
        "total_debt": total_debt,
        "capital_expenditure": capital_expenditure,
        "depreciation_and_amortization": depreciation_and_amortization,
        "operating_cash_flow": operating_cash_flow,
        "free_cash_flow": free_cash_flow,
        "ebit": ebit,
        "ebitda": ebitda,
        "operating_expense": operating_expense,
        "research_and_development": research_and_development,
        "goodwill_and_intangible_assets": goodwill_and_intangible_assets,
        "return_on_invested_capital": return_on_invested_capital,
        "earnings_per_share": earnings_per_share,
        "book_value_per_share": book_value_per_share,
        "dividends_and_other_cash_distributions": dividends,
        "issuance_or_purchase_of_equity_shares": issuance_or_purchase_of_equity_shares,
    }


def _build_line_item(record: dict, requested_items: list[str], ticker: str, period: str) -> LineItem:
    values = _compute_base_values(record)
    payload = {
        "ticker": ticker,
        "report_period": record["report_period"],
        "period": period,
        "currency": record.get("currency", "CNY"),
    }
    for item in requested_items:
        payload[item] = values.get(item)
    return LineItem(**payload)


def _empty_financial_metrics_payload(ticker: str, report_period: str, period: str, currency: str) -> dict:
    return {
        "ticker": ticker,
        "report_period": report_period,
        "period": period,
        "currency": currency,
        "market_cap": None,
        "enterprise_value": None,
        "price_to_earnings_ratio": None,
        "price_to_book_ratio": None,
        "price_to_sales_ratio": None,
        "enterprise_value_to_ebitda_ratio": None,
        "enterprise_value_to_revenue_ratio": None,
        "free_cash_flow_yield": None,
        "peg_ratio": None,
        "gross_margin": None,
        "operating_margin": None,
        "net_margin": None,
        "return_on_equity": None,
        "return_on_assets": None,
        "return_on_invested_capital": None,
        "asset_turnover": None,
        "inventory_turnover": None,
        "receivables_turnover": None,
        "days_sales_outstanding": None,
        "operating_cycle": None,
        "working_capital_turnover": None,
        "current_ratio": None,
        "quick_ratio": None,
        "cash_ratio": None,
        "operating_cash_flow_ratio": None,
        "debt_to_equity": None,
        "debt_to_assets": None,
        "interest_coverage": None,
        "revenue_growth": None,
        "earnings_growth": None,
        "book_value_growth": None,
        "earnings_per_share_growth": None,
        "free_cash_flow_growth": None,
        "operating_income_growth": None,
        "ebitda_growth": None,
        "payout_ratio": None,
        "earnings_per_share": None,
        "book_value_per_share": None,
        "free_cash_flow_per_share": None,
    }


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch CN stock/ETF prices from AkShare."""
    _ = api_key
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    ticker_type = get_ticker_type(ticker)
    start = _to_ak_date(start_date)
    end = _to_ak_date(end_date)

    df = pd.DataFrame()
    try:
        if ticker_type == "etf":
            df = _retry(
                lambda: ak.fund_etf_hist_em(
                    symbol=ticker,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust="qfq",
                )
            )
        else:
            df = _retry(
                lambda: ak.stock_zh_a_hist(
                    symbol=ticker,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust="qfq",
                )
            )
    except Exception:
        df = pd.DataFrame()

    # Fallback sources when Eastmoney blocks requests intermittently.
    if df is None or df.empty:
        try:
            if ticker_type == "etf":
                df = _retry(lambda: ak.fund_etf_hist_sina(symbol=_to_cn_symbol(ticker)))
                df = _filter_date_range(df, "date", start_date, end_date)
            else:
                df = _retry(lambda: ak.stock_zh_a_daily(symbol=_to_cn_symbol(ticker), adjust="qfq"))
                df = _filter_date_range(df, "date", start_date, end_date)
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return []

    prices: list[Price] = []
    for _, row in df.iterrows():
        date_text = _to_report_date(row.get("日期") or row.get("date"))
        if not date_text:
            continue
        open_price = _safe_float(row.get("开盘") if "开盘" in row else row.get("open"))
        close_price = _safe_float(row.get("收盘") if "收盘" in row else row.get("close"))
        high_price = _safe_float(row.get("最高") if "最高" in row else row.get("high"))
        low_price = _safe_float(row.get("最低") if "最低" in row else row.get("low"))
        volume = _safe_int(row.get("成交量") if "成交量" in row else row.get("volume"))
        if None in (open_price, close_price, high_price, low_price, volume):
            continue
        prices.append(
            Price(
                open=open_price,
                close=close_price,
                high=high_price,
                low=low_price,
                volume=volume,
                time=date_text,
            )
        )

    prices.sort(key=lambda x: x.time)
    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Build financial metrics from AkShare financial statements."""
    _ = api_key
    if get_ticker_type(ticker) == "etf":
        return []

    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    records = _build_financial_records(ticker)
    if not records:
        return []

    records = [r for r in records if str(r.get("report_period", "")) <= end_date]
    if not records:
        return []

    if period == "annual":
        annual_records = [r for r in records if str(r["report_period"]).endswith("12-31")]
        if annual_records:
            records = annual_records

    records = records[: max(limit, 1)]
    market_cap = get_market_cap(ticker, end_date)

    metrics_list: list[FinancialMetrics] = []
    for idx, record in enumerate(records):
        base = _compute_base_values(record)
        older_base = _compute_base_values(records[idx + 1]) if idx + 1 < len(records) else {}
        abstract = record.get("_abstract", {})

        payload = _empty_financial_metrics_payload(
            ticker=ticker,
            report_period=record["report_period"],
            period=period,
            currency=record.get("currency", "CNY"),
        )

        revenue = base.get("revenue")
        net_income = base.get("net_income")
        total_assets = base.get("total_assets")
        total_liabilities = base.get("total_liabilities")
        shareholders_equity = base.get("shareholders_equity")
        cash_and_equivalents = base.get("cash_and_equivalents")
        total_debt = base.get("total_debt")
        free_cash_flow = base.get("free_cash_flow")
        ebit = base.get("ebit")
        ebitda = base.get("ebitda")
        current_assets = base.get("current_assets")
        current_liabilities = base.get("current_liabilities")
        interest_expense = base.get("interest_expense")
        operating_cash_flow = base.get("operating_cash_flow")
        outstanding_shares = base.get("outstanding_shares")

        enterprise_value = None
        if market_cap is not None:
            enterprise_value = market_cap + (total_debt or 0) - (cash_and_equivalents or 0)

        payload["market_cap"] = market_cap
        payload["enterprise_value"] = enterprise_value
        payload["price_to_earnings_ratio"] = (
            market_cap / net_income if market_cap and net_income not in (None, 0) else None
        )
        payload["price_to_book_ratio"] = (
            market_cap / shareholders_equity if market_cap and shareholders_equity not in (None, 0) else None
        )
        payload["price_to_sales_ratio"] = (
            market_cap / revenue if market_cap and revenue not in (None, 0) else None
        )
        payload["enterprise_value_to_ebitda_ratio"] = (
            enterprise_value / ebitda if enterprise_value and ebitda not in (None, 0) else None
        )
        payload["enterprise_value_to_revenue_ratio"] = (
            enterprise_value / revenue if enterprise_value and revenue not in (None, 0) else None
        )
        payload["free_cash_flow_yield"] = (
            free_cash_flow / market_cap if free_cash_flow is not None and market_cap not in (None, 0) else None
        )
        earnings_growth = _normalize_growth(_safe_float(abstract.get("calculate_parent_holder_net_profit_yoy_growth_ratio")))
        payload["peg_ratio"] = (
            payload["price_to_earnings_ratio"] / (earnings_growth * 100)
            if payload["price_to_earnings_ratio"] is not None and earnings_growth not in (None, 0)
            else None
        )
        payload["gross_margin"] = base.get("gross_margin")
        payload["operating_margin"] = base.get("operating_margin")
        payload["net_margin"] = base.get("net_margin")
        roe = _safe_float(abstract.get("index_weighted_avg_roe")) or _safe_float(abstract.get("index_full_diluted_roe"))
        payload["return_on_equity"] = _normalize_margin_pct(roe) if roe is not None else (
            net_income / shareholders_equity if net_income is not None and shareholders_equity not in (None, 0) else None
        )
        payload["return_on_assets"] = (
            net_income / total_assets if net_income is not None and total_assets not in (None, 0) else None
        )
        payload["return_on_invested_capital"] = base.get("return_on_invested_capital")
        payload["asset_turnover"] = revenue / total_assets if revenue and total_assets not in (None, 0) else None
        payload["inventory_turnover"] = _safe_float(abstract.get("inventory_turnover_ratio"))
        dso = _safe_float(abstract.get("receive_accounts_turnover_days"))
        payload["days_sales_outstanding"] = dso
        payload["receivables_turnover"] = 365.0 / dso if dso not in (None, 0) else None
        inv_days = _safe_float(abstract.get("inventory_turnover_days"))
        payload["operating_cycle"] = (
            (inv_days or 0) + (dso or 0) if inv_days is not None or dso is not None else None
        )
        working_capital = (
            current_assets - current_liabilities
            if current_assets is not None and current_liabilities is not None
            else None
        )
        payload["working_capital_turnover"] = (
            revenue / working_capital if revenue and working_capital not in (None, 0) else None
        )
        payload["current_ratio"] = _safe_float(abstract.get("current_ratio")) or (
            current_assets / current_liabilities if current_assets is not None and current_liabilities not in (None, 0) else None
        )
        payload["quick_ratio"] = _safe_float(abstract.get("quick_ratio"))
        payload["cash_ratio"] = (
            cash_and_equivalents / current_liabilities
            if cash_and_equivalents is not None and current_liabilities not in (None, 0)
            else None
        )
        payload["operating_cash_flow_ratio"] = (
            operating_cash_flow / current_liabilities
            if operating_cash_flow is not None and current_liabilities not in (None, 0)
            else None
        )
        payload["debt_to_equity"] = (
            total_liabilities / shareholders_equity
            if total_liabilities is not None and shareholders_equity not in (None, 0)
            else None
        )
        payload["debt_to_assets"] = (
            total_liabilities / total_assets if total_liabilities is not None and total_assets not in (None, 0) else None
        )
        payload["interest_coverage"] = (
            ebit / interest_expense if ebit is not None and interest_expense not in (None, 0) else None
        )
        payload["revenue_growth"] = _normalize_growth(
            _safe_float(abstract.get("calculate_operating_income_total_yoy_growth_ratio"))
            or _first_non_null(record, ["TOTAL_OPERATE_INCOME_YOY", "OPERATE_INCOME_YOY"])
        )
        payload["earnings_growth"] = earnings_growth or _normalize_growth(
            _first_non_null(record, ["PARENT_NETPROFIT_YOY", "NETPROFIT_YOY"])
        )
        current_bvps = base.get("book_value_per_share")
        older_bvps = older_base.get("book_value_per_share")
        payload["book_value_growth"] = (
            (current_bvps - older_bvps) / abs(older_bvps)
            if current_bvps is not None and older_bvps not in (None, 0)
            else None
        )
        payload["earnings_per_share_growth"] = _normalize_growth(_safe_float(abstract.get("basic_eps__yoy")))
        older_fcf = older_base.get("free_cash_flow")
        payload["free_cash_flow_growth"] = (
            (free_cash_flow - older_fcf) / abs(older_fcf)
            if free_cash_flow is not None and older_fcf not in (None, 0)
            else None
        )
        payload["operating_income_growth"] = _normalize_growth(_first_non_null(record, ["OPERATE_PROFIT_YOY"]))
        older_ebitda = older_base.get("ebitda")
        payload["ebitda_growth"] = (
            (ebitda - older_ebitda) / abs(older_ebitda)
            if ebitda is not None and older_ebitda not in (None, 0)
            else None
        )
        payouts = base.get("dividends_and_other_cash_distributions")
        payload["payout_ratio"] = (
            payouts / net_income if payouts is not None and net_income not in (None, 0) else None
        )
        payload["earnings_per_share"] = base.get("earnings_per_share")
        payload["book_value_per_share"] = base.get("book_value_per_share")
        payload["free_cash_flow_per_share"] = (
            free_cash_flow / outstanding_shares
            if free_cash_flow is not None and outstanding_shares not in (None, 0)
            else None
        )

        metrics_list.append(FinancialMetrics(**payload))

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics_list])
    return metrics_list


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Search financial line items from AkShare merged financial statements."""
    _ = (end_date, api_key)
    if get_ticker_type(ticker) == "etf":
        return []

    records = _build_financial_records(ticker)
    if not records:
        return []

    records = [r for r in records if str(r.get("report_period", "")) <= end_date]
    if not records:
        return []

    if period == "annual":
        annual_records = [r for r in records if str(r["report_period"]).endswith("12-31")]
        if annual_records:
            records = annual_records

    records = records[: max(limit, 1)]
    return [_build_line_item(record, line_items, ticker, period) for record in records]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """No reliable public insider-trade source in AkShare for A-shares yet."""
    _ = (ticker, end_date, start_date, limit, api_key)
    return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch CN stock news from Eastmoney via AkShare."""
    _ = api_key
    if get_ticker_type(ticker) == "etf":
        return []

    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    try:
        df = _retry(lambda: ak.stock_news_em(symbol=ticker))
    except Exception:
        return []

    if df is None or df.empty:
        return []

    start_dt = pd.to_datetime(start_date, errors="coerce") if start_date else None
    end_dt = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(end_dt):
        end_dt = None

    news_items: list[CompanyNews] = []
    for _, row in df.iterrows():
        published = pd.to_datetime(row.get("发布时间"), errors="coerce")
        if pd.isna(published):
            continue
        if start_dt is not None and published < start_dt:
            continue
        if end_dt is not None and published > end_dt + pd.Timedelta(days=1):
            continue

        news_items.append(
            CompanyNews(
                ticker=ticker,
                title=str(row.get("新闻标题") or ""),
                author=str(row.get("文章来源") or "Unknown"),
                source=str(row.get("文章来源") or "Eastmoney"),
                date=published.strftime("%Y-%m-%d"),
                url=str(row.get("新闻链接") or ""),
                sentiment=None,
            )
        )

    news_items.sort(key=lambda x: x.date, reverse=True)
    news_items = news_items[:limit]
    _cache.set_company_news(cache_key, [n.model_dump() for n in news_items])
    return news_items


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from AkShare individual stock info."""
    _ = (end_date, api_key)
    if get_ticker_type(ticker) == "etf":
        return None

    # For historical dates, estimate market cap from nearest price * latest known shares before end_date.
    try:
        end_dt = pd.to_datetime(end_date, errors="coerce")
    except Exception:
        end_dt = None
    if end_dt is not None and not pd.isna(end_dt):
        today = pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d"))
        if end_dt < today:
            start_dt = (end_dt - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            historical_prices = get_prices(ticker, start_dt, end_date)
            last_close = historical_prices[-1].close if historical_prices else None
            records = [r for r in _build_financial_records(ticker) if str(r.get("report_period", "")) <= end_date]
            if records:
                shares = _compute_base_values(records[0]).get("outstanding_shares")
                if last_close is not None and shares not in (None, 0):
                    return float(last_close * shares)

    try:
        info_df = _retry(lambda: ak.stock_individual_info_em(symbol=ticker))
    except Exception:
        return None

    if info_df is None or info_df.empty:
        return None

    market_cap_row = info_df[info_df["item"] == "总市值"]
    if market_cap_row.empty:
        return None

    return _safe_float(market_cap_row.iloc[0]["value"])


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
