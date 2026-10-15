# DIVERGENCE — an autonomous financial research agent

**Finding the Next Signal · ATMOS 2026 (BITS Pilani, Hyderabad) · Track A — Banking, NBFCs & Financial Services**
Participant: Mrityunjay Mishra (solo)

Most AI finance tools summarise. DIVERGENCE investigates. It never starts
from headlines: it builds a quarterly numeric panel from public sources,
computes **paired series that should move together** — and investigates only
where the pair **diverges persistently**. The discovery lives between the
documents, in a broken numerical relationship, and every finding must
survive a devil's-advocate pass and carry a falsification condition before
it is written up in the brief's fixed 8-part format.

> Compute first, read second. The LLM (in the full build) cites computed
> numbers; it never computes them. In this MVP every figure is produced by
> deterministic pandas code from the fetched panel — clone the repo, run
> it, and you get the same numbers.

## MVP scope (this repository)

One end-to-end pipeline run on one Track A company — **HDFC Bank**
(NSE: HDFCBANK) — exactly the Round 1 "strong signal" cut:

| Stage | What happens |
|---|---|
| 1 · Ingest | Screener.in public pages parsed live (standalone + consolidated quarterly results; annual balance sheet). Yahoo Finance price history fetched as market context (yfinance, chart-API fallback). |
| 2 · Panel | 12 post-merger quarters (Sep-2023 → Jun-2026); the Jun-2023 pre-merger quarter is excluded as a structural break. Derived metrics: NII, interest burden, GNPA–NNPA gap, **implied provision coverage**. Mechanical one-off flags (Mar-2024, Jun-2025). |
| 3 · Scan | Three paired-series detectors (asset-quality gap · growth gap · spread relief), one transparent score: `persistence × |drift| / σ`. |
| 4 · Hypothesis | The winning divergence is written up from templates whose every number is read out of the panel — nothing is invented or hard-coded. |
| 5 · Evidence | Claim-level trail with source classes and live links; strength graded by the fixed rubric from the Round 1 report. |
| 6 · Report | The 8-part note as a professional PDF (Jinja2 → WeasyPrint), with the computed table and chart. |

## The sample discovery (HDFC Bank, run 15 Oct 2026)

> **HDFC Bank's headline asset quality looks healed — gross NPA has fallen
> from its 1.42% peak (Dec-2024) to 1.17% (Jun-2026) — but the cushion
> underneath it has thinned for 8 straight quarters: the provision coverage
> implied by the bank's own GNPA/NNPA pair has slid from 73.9% (Sep-2023)
> to 65.0% (Jun-2026), while net NPA, at 0.41%, is still higher than in
> every quarter of the first panel year. Over the full panel the gross
> ratio is down 0.17pp while the net ratio has not fallen at all.**

**Evidence strength: MODERATE** — two independent source classes (results
data parsed from Screener's render of the statutory results; independent
filing coverage in the financial press corroborating the parsed figures to
the crore), 8 quarters of persistence. Not Strong, honestly: the primary
numeric trail is a single aggregator parse, and the strongest
counterargument — a secured, mortgage-heavy merged book rationally needs
less coverage, and technical write-offs mechanically compress the implied
ratio — cannot be fully separated with public quarterly data alone. The
full note (all 8 parts, counterargument and falsification condition
included) is in [`examples/divergence_HDFCBANK_note.pdf`](examples/divergence_HDFCBANK_note.pdf);
the panel it was computed from is [`examples/panel_HDFCBANK.csv`](examples/panel_HDFCBANK.csv).

Also flagged by the same scan, not headlined: funding-cost relief
(interest burden, persistence 3, score 6.5) and the revenue-vs-profit
growth gap (persistence 3, score 5.5).

## Run it

```bash
pip install -r requirements.txt
python run.py --scrip HDFCBANK --name "HDFC Bank" --ticker HDFCBANK.NS
# -> output/divergence_HDFCBANK_note.pdf, panel CSV, evidence JSON, run log
```

Any other bank/NBFC with a Screener page can be run the same way
(`--scrip SBIN`, `--scrip ICICIBANK`, …) — no company-specific code exists
in the pipeline; the finals build extends the metric library and source
map to the whole of Track A.

## Layout

```
run.py                  CLI entry point (6 logged stages)
divergence/
  ingest.py             Screener parser + Yahoo context (timeout-guarded)
  panel.py              Panel builder, derived metrics, one-off flags
  engine.py             Divergence detectors, ranking, grading, narratives
  evidence.py           Claim-level evidence store (source classes a–e)
  report.py             Chart (matplotlib) + note renderer (WeasyPrint)
  templates/note.html.j2  The 8-part research-note template
examples/               Sample note PDF + the panel CSV it came from
tools/make_demo_video.py  Renders the ≤2-min demo video from a real run log
```

## Honesty notes

- **Implied PCR** = (GNPA − NNPA) / GNPA is an identity applied to the
  reported ratios; it approximates the disclosed PCR (which excludes
  technical write-offs). For Jun-2026 it gives 65.0%; the same identity on
  the filing's absolute figures (GNPA Rs 35,846 cr, NNPA Rs 12,357 cr)
  gives 65.5% — the two agree within ratio rounding, which is stated in
  the note.
- One-off quarters (expense / other-income outliers) are flagged
  mechanically and can never serve as persistence evidence for flow pairs.
- Public sources only; no paid or proprietary data; no investment advice —
  every output is a research hypothesis with a falsification condition.

## Roadmap to the finals (25 Oct 2026)

Filing-PDF parsing (BSE/NSE) as the primary source class · peer panel for
sector-vs-company separation · hypothesis engine + devil's-advocate pass
as separate reasoning steps · dry runs on every Track A NIFTY 50 company ·
live run on the judges' company with the same generator that made the
sample note.
