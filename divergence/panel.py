"""Panel builder: turn raw Screener parses into a clean, derived-metric panel.

Conventions
-----------
* All money figures are Rs crore, exactly as Screener reports them.
* The panel window is generic: the most recent PANEL_QUARTERS quarters of
  the standalone results, whatever company is being run. (For HDFC Bank
  this window happens to start at Sep-2023, the first full quarter of the
  merged entity -- the Jun-2023 pre-merger quarter drops out of the window
  on its own, which is what the Round 1 note documents.) Structural breaks
  inside the window are handled by the mechanical one-off flags below,
  never by company-specific date edits.
* Derived metrics are computed here, deterministically -- nothing downstream
  (and no language model) ever computes a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PANEL_QUARTERS = 12

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
    panel = panel.sort_index().tail(PANEL_QUARTERS)

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

    The rule is purely mechanical so it generalises to any company: a line
    > 1.4x the panel median of that line marks the quarter as one-off
    contaminated (provision builds, stake-sale gains and similar items all
    surface this way in the expense / other-income lines).
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
