"""Panel builder: turn raw Screener parses into a clean, derived-metric panel.

Conventions
-----------
* All money figures are Rs crore, exactly as Screener reports them.
* The panel window for the MVP is the post-merger entity: HDFC Ltd merged
  into HDFC Bank on 1 Jul 2023, so the Jun-2023 quarter (old standalone
  bank) is a structural break and is excluded; the panel is the 12 quarters
  Sep-2023 .. Jun-2026.
* Derived metrics are computed here, deterministically -- nothing downstream
  (and no language model) ever computes a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POST_MERGER_START = pd.Timestamp("2023-09-30")

COLMAP = {
    "revenue": "revenue",
    "interest": "interest_expense",
    "expenses": "expenses",
    "other income": "other_income",
    "profit before tax": "pbt",
    "net profit": "net_profit",
    "eps in rs": "eps",
    "gross npa %": "gnpa_pct",
    "net npa %": "nnpa_pct",
}


def build_panel(standalone_quarterly: pd.DataFrame) -> pd.DataFrame:
    q = standalone_quarterly.rename(columns=COLMAP)
    keep = [c for c in COLMAP.values() if c in q.columns]
    panel = q[keep].copy()
    panel = panel[panel.index >= POST_MERGER_START].sort_index()

    panel["nii"] = panel["revenue"] - panel["interest_expense"]
    panel["interest_burden_pct"] = 100.0 * panel["interest_expense"] / panel["revenue"]
    panel["gap_pp"] = panel["gnpa_pct"] - panel["nnpa_pct"]
    # Provision coverage implied purely by the reported GNPA/NNPA pair:
    # PCR = (GNPA - NNPA) / GNPA. This is the standard identity; technical
    # write-offs make it an approximation of the disclosed PCR, which the
    # note states openly.
    panel["implied_pcr_pct"] = 100.0 * panel["gap_pp"] / panel["gnpa_pct"]
    for col in ("revenue", "net_profit", "nii"):
        panel[f"{col}_yoy_pct"] = 100.0 * panel[col].pct_change(4)

    panel["one_off_flag"] = _flag_one_offs(panel)
    return panel


def _flag_one_offs(panel: pd.DataFrame) -> pd.Series:
    """Flag quarters whose expense / other-income lines are extreme outliers.

    Mar-2024 (floating-provision build) and Jun-2025 (stake-sale gain in
    other income + elevated provisions) are the known cases in this window;
    the rule is mechanical so it generalises: a line > 1.4x the panel median
    of that line marks the quarter as one-off contaminated.
    """
    flags = pd.Series(False, index=panel.index)
    for col in ("expenses", "other_income"):
        if col in panel:
            med = panel[col].median()
            flags |= panel[col] > 1.4 * med
    return flags


def funding_mix(annual_bs: pd.DataFrame) -> pd.DataFrame:
    """Annual funding mix from the Screener balance-sheet table (supporting)."""
    if annual_bs.empty or "deposits" not in annual_bs.columns:
        return pd.DataFrame()
    bs = annual_bs[["deposits", "borrowing"]].copy()
    bs["borrowings_per_deposit_pct"] = 100.0 * bs["borrowing"] / bs["deposits"]
    return bs
