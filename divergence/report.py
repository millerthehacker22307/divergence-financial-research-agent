"""Report generator: render the fixed 8-part note as a professional PDF.

Jinja2 -> WeasyPrint, reusing the Round 1 report's navy/gold equity-research
look. The chart is drawn by matplotlib from the same computed panel, so the
picture and the table can never disagree.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .engine import Divergence, qlabel

NAVY = "#0d2440"
GOLD = "#b07d10"
RED = "#9e2b25"

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _band_for(div: Divergence) -> tuple[float, str]:
    """The break band of the winning pair, mirroring the detectors in
    engine.py: coverage / burden pairs break *below* early-mean minus a
    margin; the growth-gap pair breaks *above* +3pp."""
    gap = div.gap
    if div.pair_id == "growth_gap":
        return 3.0, "above"
    if div.pair_id == "spread_relief":
        return float(gap.iloc[:4].mean()) - 1.0, "below"
    return float(gap.iloc[:4].mean()) - 3.0, "below"


def make_chart(panel: pd.DataFrame, div: Divergence, out_png: Path) -> Path:
    labels = [qlabel(ts) for ts in panel.index]
    x = list(range(len(panel)))
    pos = {ts: i for i, ts in enumerate(panel.index)}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 3.9), sharex=True,
                                   gridspec_kw={"hspace": 0.12, "height_ratios": [1, 1]})
    ax1.plot(x, panel["gnpa_pct"], marker="o", ms=3.5, lw=1.8, color=NAVY, label="Gross NPA %")
    ax1.plot(x, panel["nnpa_pct"], marker="s", ms=3.5, lw=1.8, color=RED, label="Net NPA %")
    ax1.set_ylabel("NPA ratio (%)", fontsize=8)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax1.grid(alpha=0.25, lw=0.5)
    ax1.tick_params(labelsize=7.5)
    ax1.set_title(f"Winning divergence: {div.title}",
                  fontsize=9.5, color=NAVY, loc="left", weight="bold")

    gap = div.gap
    gx = [pos[ts] for ts in gap.index]
    band, direction = _band_for(div)
    ax2.plot(gx, gap.values, marker="o", ms=3.5, lw=1.8, color=GOLD, label=div.gap_name)
    ax2.axhline(band, color=RED, lw=1.1, ls="--", label=f"Break band ({band:.1f})")
    beyond = (gap > band) if direction == "above" else (gap < band)
    ax2.fill_between(gx, gap.values, band, where=beyond.values, color=RED, alpha=0.13)
    if div.change_point is not None and div.change_point in pos:
        cp = pos[div.change_point]
        ax2.axvline(cp, color=RED, lw=0.9, ls=":")
        ax2.annotate("change-point", (cp, float(gap.min())), fontsize=7.5, color=RED,
                     xytext=(cp + 0.15, float(gap.min()) + 0.06 * (float(gap.max()) - float(gap.min()) + 1e-9)))
    ax2.set_ylabel(div.gap_name, fontsize=7.5)
    ax2.legend(fontsize=7.5, frameon=False, loc="best")
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.tick_params(labelsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=38, ha="left", fontsize=7)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def render_note(
    *,
    company: str,
    scrip: str,
    ticker: str,
    track: str,
    run_stamp: str,
    panel: pd.DataFrame,
    div: Divergence,
    runners_up: list[Divergence],
    strength: str,
    strength_reason: str,
    evidence_rows: list[dict],
    funding_rows: list[dict],
    chart_png_name: str,
    out_pdf: Path,
) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    tpl = env.get_template("note.html.j2")
    num_rows = []
    for ts, r in panel.iterrows():
        num_rows.append(
            {
                "q": qlabel(ts),
                "gnpa": f"{r['gnpa_pct']:.2f}",
                "nnpa": f"{r['nnpa_pct']:.2f}",
                "pcr": f"{r['implied_pcr_pct']:.1f}",
                "nii": f"{r['nii']:,.0f}",
                "pat": f"{r['net_profit']:,.0f}",
                "one_off": bool(r["one_off_flag"]),
            }
        )
    html = tpl.render(
        company=company,
        scrip=scrip,
        ticker=ticker,
        track=track,
        run_stamp=run_stamp,
        window=f"{qlabel(panel.index[0])} to {qlabel(panel.index[-1])} ({len(panel)} quarters)",
        discovery=div.discovery,
        why=div.why_it_matters,
        counterargument=div.counterargument,
        falsification=div.falsification,
        timeline=div.timeline,
        strength=strength,
        strength_reason=strength_reason,
        strength_class={"Strong": "bs", "Moderate": "bm", "Weak": "bw"}[strength],
        num_rows=num_rows,
        band_note=div.band_note,
        pair_title=div.title,
        persistence=div.persistence,
        score=f"{div.score:.1f}",
        runners_up=[
            {"title": d.title, "persistence": d.persistence, "score": f"{d.score:.1f}"}
            for d in runners_up
        ],
        evidence=evidence_rows,
        funding=funding_rows,
        chart=chart_png_name,
        one_off_quarters=[qlabel(ts) for ts in panel.index[panel["one_off_flag"]]],
    )
    from weasyprint import HTML

    HTML(string=html, base_url=str(out_pdf.parent)).write_pdf(str(out_pdf))
    return out_pdf
