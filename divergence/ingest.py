"""Ingestion: fetch public, free data for a Track A company.

Primary source  : Screener.in public company pages (quarterly results incl.
                  GNPA/NNPA for banks, annual balance sheet). Two parses are
                  taken -- standalone and consolidated -- as a first
                  consistency cross-check.
Cross-check     : yfinance (Yahoo Finance) monthly price history, used for
                  market context and quarter-date sanity. Guarded by a hard
                  timeout: if Yahoo is slow the run continues without it and
                  the gap is logged, never silently hidden.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests

UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

SCREENER_BASE = "https://www.screener.in/company/{scrip}/"
SCREENER_CONS = "https://www.screener.in/company/{scrip}/consolidated/"


def _clean_label(x: object) -> str:
    return (
        str(x)
        .replace("\xa0", " ")
        .strip()
        .rstrip("+")
        .strip()
        .lower()
    )


def _num(v: object) -> float:
    if v is None:
        return float("nan")
    s = str(v).replace(",", "").replace("%", "").strip()
    if s in ("", "-", "nan", "NaN"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _quarter_end(label: str) -> pd.Timestamp:
    ts = pd.to_datetime(label.strip(), format="%b %Y")
    return ts + pd.offsets.MonthEnd(0)


def _parse_screener_table(table: pd.DataFrame, is_annual: bool) -> pd.DataFrame:
    """Screener tables are metric-rows x period-columns. Return periods x metrics."""
    table = table.copy()
    table.columns = [str(c).strip() for c in table.columns]
    label_col = table.columns[0]
    table = table.rename(columns={label_col: "metric"})
    table["metric"] = table["metric"].map(_clean_label)
    table = table.set_index("metric")
    out: dict[pd.Timestamp, dict[str, float]] = {}
    for col in table.columns:
        if col.lower() in ("ttm",):
            continue
        try:
            if is_annual:
                ts = pd.to_datetime(col.strip(), format="%b %Y") + pd.offsets.MonthEnd(0)
            else:
                ts = _quarter_end(col)
        except Exception:
            continue
        out[ts] = {m: _num(v) for m, v in table[col].items()}
    df = pd.DataFrame(out).T.sort_index()
    df.index.name = "period_end"
    return df


@dataclass
class ScreenerData:
    url: str
    quarterly: pd.DataFrame
    annual_bs: pd.DataFrame
    fetched_at: str


def fetch_screener(scrip: str, consolidated: bool = False) -> ScreenerData:
    url = (SCREENER_CONS if consolidated else SCREENER_BASE).format(scrip=scrip)
    resp = requests.get(url, headers=UA, timeout=40)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    quarterly = _parse_screener_table(tables[0], is_annual=False)
    annual_bs = pd.DataFrame()
    for t in tables:
        labels = [_clean_label(x) for x in t.iloc[:, 0].astype(str)]
        if "deposits" in labels and "borrowing" in labels:
            annual_bs = _parse_screener_table(t, is_annual=True)
            break
    return ScreenerData(
        url=url,
        quarterly=quarterly,
        annual_bs=annual_bs,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


@dataclass
class YahooData:
    ticker: str
    monthly_close: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    note: str = ""


def _yahoo_chart_api(ticker: str) -> pd.Series:
    """Direct Yahoo Finance chart API (the same endpoint yfinance wraps)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    resp = requests.get(url, params={"range": "3y", "interval": "1mo"}, headers=UA, timeout=20)
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    ts = pd.to_datetime(result["timestamp"], unit="s")
    closes = result["indicators"]["quote"][0]["close"]
    s = pd.Series(closes, index=ts, dtype=float).dropna()
    s.index = s.index.tz_localize(None)
    return s


def fetch_yahoo_prices(ticker: str, timeout_s: int = 25) -> YahooData:
    """Monthly closes: yfinance first (hard timeout), Yahoo chart API fallback.

    Yahoo's crumb/session endpoints can stall from some networks while the
    chart API stays reachable; either way the data is Yahoo Finance's. The
    route used is recorded in `note` so the evidence trail stays honest.
    """
    result: dict[str, object] = {}

    def _work() -> None:
        try:
            import yfinance as yf

            hist = yf.Ticker(ticker).history(period="3y", interval="1mo")
            if hist is not None and not hist.empty:
                s = hist["Close"].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                result["series"] = s
            else:
                result["note"] = "yfinance returned an empty price history"
        except Exception as exc:  # pragma: no cover - network dependent
            result["note"] = f"yfinance error: {exc!r}"

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(timeout_s)
    if not th.is_alive() and "series" in result:
        return YahooData(ticker=ticker, monthly_close=result["series"], note="via yfinance")  # type: ignore[arg-type]
    yf_note = "yfinance timed out" if th.is_alive() else str(result.get("note", "no data"))
    try:
        s = _yahoo_chart_api(ticker)
        if not s.empty:
            return YahooData(
                ticker=ticker,
                monthly_close=s,
                note=f"via Yahoo chart API ({yf_note}; fell back to the API yfinance wraps)",
            )
    except Exception as exc:  # pragma: no cover - network dependent
        return YahooData(ticker=ticker, note=f"{yf_note}; chart API also failed: {exc!r}")
    return YahooData(ticker=ticker, note=yf_note)
