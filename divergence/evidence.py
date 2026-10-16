"""Evidence store: every claim in the note traces to a row here.

Source classes follow the Round 1 report's taxonomy:
  (a) statutory filing / annual report      (b) exchange announcement / results
  (c) investor presentation / earnings call (d) credible news
  (e) competitor / industry / regulator data (calc) deterministic computation

Design rule (anti-hard-coding): this module contains NO company facts.
The class-(b) rows are generated from the live parse itself. Independent
press corroboration for a company lives as DATA in
``corroboration/<SCRIP>.json`` next to the code -- each entry carries the
outlet, the URL and the reported figures, and the pipeline machine-checks
those figures against the parsed panel's latest quarter, printing MATCH /
MISMATCH into the note. A company with no corroboration file simply earns
no class-(d) rows, and the strength grade falls accordingly -- the grade
is never propped up by another company's coverage.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

CORROBORATION_DIR = Path(__file__).resolve().parent.parent / "corroboration"


@dataclass
class Evidence:
    claim: str
    source_class: str
    source: str
    url: str
    detail: str


def _fmt(metric: str, v: float) -> str:
    if metric.endswith("_pct"):
        return f"{v:.2f}%"
    return f"Rs {v:,.0f} cr"


def _checks_text(panel: pd.DataFrame, checks: list[dict]) -> str:
    """Machine-verify reported figures against the panel's latest quarter."""
    last = panel.iloc[-1]
    parts = []
    for chk in checks:
        metric = chk["metric"]
        expected = float(chk["expected"])
        tol = float(chk.get("tol", 0.0))
        actual = float(last[metric])
        verdict = "MATCH" if abs(actual - expected) <= tol else "MISMATCH"
        label = chk.get("label", metric)
        parts.append(
            f"{label}: reported {_fmt(metric, expected)} vs panel "
            f"{_fmt(metric, actual)} -- {verdict}"
        )
    return " Pipeline check on the latest quarter -- " + "; ".join(parts) + "."


def load_corroboration(scrip: str) -> list[dict]:
    path = CORROBORATION_DIR / f"{scrip}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def build_evidence(
    company: str,
    scrip: str,
    ticker: str,
    panel: pd.DataFrame,
    standalone_quarterly: pd.DataFrame,
    cons_quarterly: pd.DataFrame,
    yahoo_note: str,
    yahoo_last_close: float | None,
) -> list[Evidence]:
    last = panel.iloc[-1]
    overlap = standalone_quarterly.index.intersection(cons_quarterly.index)
    same_labels = len(overlap) == len(standalone_quarterly) == len(cons_quarterly)
    rows = [
        Evidence(
            claim="Quarterly panel: revenue, interest, expenses, other income, net profit, GNPA %, NNPA %",
            source_class="b",
            source=f"Screener.in -- {company} standalone quarterly results table (public rendering of statutory results)",
            url=f"https://www.screener.in/company/{scrip}/",
            detail=(
                f"Parsed live: {len(standalone_quarterly)} quarters on the standalone page; "
                f"the panel uses the most recent {len(panel)}. Latest quarter in panel: "
                f"net profit Rs {last['net_profit']:,.0f} cr, GNPA {last['gnpa_pct']:.2f}%, "
                f"NNPA {last['nnpa_pct']:.2f}%."
            ),
        ),
        Evidence(
            claim="Basis consistency: standalone and consolidated parses cover the same quarters",
            source_class="b",
            source=f"Screener.in -- {company} consolidated quarterly results table",
            url=f"https://www.screener.in/company/{scrip}/consolidated/",
            detail=(
                f"Consolidated parse returned {len(cons_quarterly)} quarters x "
                f"{cons_quarterly.shape[1]} metrics; quarter labels agree with the "
                f"standalone parse on all {len(overlap)} overlapping quarters "
                f"({'full agreement' if same_labels else 'partial overlap -- see run log'}). "
                "The panel itself is built on the standalone basis throughout."
            ),
        ),
    ]
    for c in load_corroboration(scrip):
        detail = c["detail"]
        if c.get("checks"):
            detail += _checks_text(panel, c["checks"])
        rows.append(
            Evidence(
                claim=c["claim"],
                source_class=c.get("source_class", "d"),
                source=c["source"],
                url=c["url"],
                detail=detail,
            )
        )
    rows.append(
        Evidence(
            claim="Implied provision coverage = (GNPA% - NNPA%) / GNPA%, computed per quarter",
            source_class="calc",
            source="DIVERGENCE metric engine (pandas) over the parsed panel",
            url="https://github.com/millerthehacker22307/divergence-financial-research-agent",
            detail=(
                "Deterministic identity applied to reported ratios; reproducible from "
                f"examples/panel_{scrip}.csv committed with this note's run."
            ),
        )
    )
    if yahoo_last_close is not None:
        rows.append(
            Evidence(
                claim="Market context: share price over the panel window",
                source_class="ctx",
                source=f"Yahoo Finance -- {ticker} monthly closes",
                url=f"https://finance.yahoo.com/quote/{ticker}",
                detail=f"Fetched {yahoo_note}; latest monthly close Rs {yahoo_last_close:,.0f}. Context only -- class 'ctx' never counts toward the evidence-strength grade.",
            )
        )
    elif yahoo_note:
        rows.append(
            Evidence(
                claim="Market context (attempted)",
                source_class="ctx",
                source=f"Yahoo Finance -- {ticker} monthly closes",
                url=f"https://finance.yahoo.com/quote/{ticker}",
                detail=f"Price cross-check unavailable this run: {yahoo_note}. No claim in the discovery rests on price data.",
            )
        )
    return rows


def evidence_classes(rows: list[Evidence]) -> set[str]:
    """Classes that count toward the strength rubric (a-e only; 'ctx' and
    'calc' are recorded for audit but never inflate the grade)."""
    return {r.source_class for r in rows if r.source_class in {"a", "b", "c", "d", "e"}}


def to_dicts(rows: list[Evidence]) -> list[dict]:
    return [asdict(r) for r in rows]
