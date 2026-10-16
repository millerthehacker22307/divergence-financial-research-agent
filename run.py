#!/usr/bin/env python3
"""DIVERGENCE — MVP pipeline entry point.

Usage:
    python run.py --scrip HDFCBANK --name "HDFC Bank" --ticker HDFCBANK.NS

Stages: ingest (Screener standalone + consolidated, Yahoo context) ->
panel build -> divergence scan -> hypothesis narrative -> evidence store ->
graded 8-part PDF note. Every number in the note is computed by pandas
from the fetched panel; the run log and panel CSV are written alongside
the PDF so the whole run is auditable and reproducible.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from divergence import engine, evidence as ev, ingest, panel as panel_mod, report

TRACK = "Track A — Banking, NBFCs & Financial Services"

LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="DIVERGENCE MVP pipeline")
    ap.add_argument("--scrip", default="HDFCBANK", help="Screener/NSE scrip")
    ap.add_argument("--name", default="HDFC Bank")
    ap.add_argument("--ticker", default="HDFCBANK.NS", help="Yahoo Finance ticker")
    ap.add_argument("--out", default="output", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %H:%M IST")

    log(f"DIVERGENCE MVP run — {args.name} ({args.scrip}) — {TRACK}")
    log("Stage 1/6 INGEST — Screener.in standalone + consolidated pages")
    sa = ingest.fetch_screener(args.scrip, consolidated=False)
    cons = ingest.fetch_screener(args.scrip, consolidated=True)
    log(f"  standalone quarterly table: {sa.quarterly.shape[1]} metrics x {sa.quarterly.shape[0]} quarters  ({sa.url})")
    log(f"  consolidated quarterly table: {cons.quarterly.shape[1]} metrics x {cons.quarterly.shape[0]} quarters  ({cons.url})")
    log("Stage 1b    INGEST — Yahoo Finance price history (context cross-check)")
    yahoo = ingest.fetch_yahoo_prices(args.ticker)
    last_close = None
    if not yahoo.monthly_close.empty:
        last_close = float(yahoo.monthly_close.iloc[-1])
        log(f"  monthly closes: {len(yahoo.monthly_close)} months, latest Rs {last_close:,.0f} ({yahoo.note})")
    else:
        log(f"  price cross-check unavailable: {yahoo.note}")

    log("Stage 2/6 PANEL — most recent 12 quarters, derived metrics, one-off flags")
    panel = panel_mod.build_panel(sa.quarterly)
    log(f"  panel: {len(panel)} quarters, {engine.qlabel(panel.index[0])} .. {engine.qlabel(panel.index[-1])}")
    flagged = [engine.qlabel(ts) for ts in panel.index[panel['one_off_flag']]]
    log(f"  one-off-flagged quarters: {', '.join(flagged) if flagged else 'none'}")
    panel_path = out / f"panel_{args.scrip}.csv"
    panel.to_csv(panel_path)
    log(f"  panel written: {panel_path}")

    log("Stage 3/6 SCAN — paired-series divergence detectors")
    results = engine.scan(panel)
    for d in results:
        log(f"  {d.title}: persistence {d.persistence} qtrs, drift {d.drift:+.2f}, score {d.score:.1f}")
    div = results[0]
    log(f"  strongest divergence: {div.title} (score {div.score:.1f})")
    log(f"  change-point: {engine.qlabel(div.change_point) if div.change_point is not None else 'n/a'}")

    log("Stage 4/6 HYPOTHESIS — narrative composed from computed values only")
    div = engine.narrate(panel, div, args.name)

    log("Stage 5/6 EVIDENCE — claim-level trail + strength grading")
    rows = ev.build_evidence(
        args.name, args.scrip, args.ticker, panel,
        sa.quarterly, cons.quarterly, yahoo.note, last_close,
    )
    classes = ev.evidence_classes(rows)
    strength, strength_reason = engine.grade(div.persistence, classes)
    log(f"  evidence rows: {len(rows)}; independent source classes: {sorted(classes)}")
    log(f"  evidence strength: {strength}")
    (out / f"evidence_{args.scrip}.json").write_text(
        json.dumps(ev.to_dicts(rows), indent=2), encoding="utf-8"
    )

    log("Stage 6/6 REPORT — 8-part note (Jinja2 -> WeasyPrint)")
    funding = panel_mod.funding_mix(sa.annual_bs)
    funding_rows = [
        {"year": ts.strftime("Mar %Y"), "ratio": f"{r['borrowings_per_deposit_pct']:.1f}"}
        for ts, r in funding.tail(4).iterrows()
    ]
    chart_name = f"chart_{args.scrip}.png"
    report.make_chart(panel, div, out / chart_name)
    pdf_path = out / f"divergence_{args.scrip}_note.pdf"
    report.render_note(
        company=args.name,
        scrip=args.scrip,
        ticker=args.ticker,
        track=TRACK,
        run_stamp=stamp,
        panel=panel,
        div=div,
        runners_up=results[1:],
        strength=strength,
        strength_reason=strength_reason,
        evidence_rows=ev.to_dicts(rows),
        funding_rows=funding_rows,
        chart_png_name=chart_name,
        out_pdf=pdf_path,
    )
    log(f"  PDF written: {pdf_path}")

    findings = {
        "company": args.name,
        "scrip": args.scrip,
        "run_stamp": stamp,
        "discovery": div.discovery,
        "evidence_strength": strength,
        "strength_reason": strength_reason,
        "winning_pair": div.pair_id,
        "persistence_quarters": div.persistence,
        "score": round(div.score, 2),
        "all_pairs": [
            {"pair": d.pair_id, "title": d.title, "persistence": d.persistence, "score": round(d.score, 2)}
            for d in results
        ],
    }
    (out / f"findings_{args.scrip}.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print(f"DISCOVERY ({args.name}):")
    print(div.discovery)
    print(f"EVIDENCE STRENGTH: {strength} — {strength_reason}")
    print(f"PDF: {pdf_path}")
    print("=" * 78)

    (out / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
