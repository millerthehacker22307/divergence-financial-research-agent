#!/usr/bin/env python3
"""Render the <=2-minute MVP demo video from a REAL pipeline run.

    python tools/make_demo_video.py --out demo.mp4

Scenes: title card -> terminal session replaying the actual run log
(output/run_log.txt, produced by run.py) -> the discovery card (from
output/findings_*.json) -> page 1 and 2 of the actual generated PDF ->
end card. Frames are drawn with PIL and encoded with ffmpeg. Nothing in
the terminal is invented: the log lines are the run's own.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
NAVY = (13, 36, 64)
GOLD = (240, 192, 64)
DARK = (10, 16, 26)
TERM_BG = (13, 20, 33)
TEXT = (214, 222, 235)
DIM = (140, 152, 168)
GREEN = (126, 231, 135)
WHITE = (245, 247, 250)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
MONO = str(FONT_DIR / "DejaVuSansMono.ttf")
MONO_B = str(FONT_DIR / "DejaVuSansMono-Bold.ttf")
SANS = str(FONT_DIR / "DejaVuSans.ttf")
SANS_B = str(FONT_DIR / "DejaVuSans-Bold.ttf")


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


class Reel:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        workdir.mkdir(parents=True, exist_ok=True)
        self.frames: list[tuple[Path, float]] = []
        self.i = 0

    def add(self, img: Image.Image, seconds: float) -> None:
        p = self.workdir / f"f{self.i:04d}.png"
        img.save(p)
        self.frames.append((p, seconds))
        self.i += 1

    def encode(self, out: Path) -> None:
        lst = self.workdir / "frames.txt"
        lines = []
        for p, s in self.frames:
            lines.append(f"file '{p.name}'")
            lines.append(f"duration {s:.3f}")
        lines.append(f"file '{self.frames[-1][0].name}'")
        lst.write_text("\n".join(lines) + "\n")
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst.name,
                "-vf", "fps=30,format=yuv420p", "-c:v", "libx264",
                "-preset", "medium", "-crf", "20", str(out),
            ],
            cwd=self.workdir, check=True, capture_output=True,
        )


def base(color=DARK) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), color)
    return img, ImageDraw.Draw(img)


def title_card(reel: Reel) -> None:
    img, d = base(NAVY)
    d.rectangle([0, 0, W, 10], fill=GOLD)
    d.text((70, 180), "DIVERGENCE", font=font(SANS_B, 84), fill=WHITE)
    d.text((72, 285), "An autonomous financial research agent", font=font(SANS, 34), fill=GOLD)
    d.text((72, 350), "Compute first, read second — discoveries from divergences,\nnot headlines.", font=font(SANS, 25), fill=TEXT)
    d.text((72, 470), "MVP demo · Finding the Next Signal — ATMOS 2026", font=font(SANS_B, 24), fill=WHITE)
    d.text((72, 512), "Track A — Banking, NBFCs & Financial Services", font=font(SANS, 24), fill=TEXT)
    d.text((72, 580), "Live run on:  HDFC Bank  (NSE: HDFCBANK)", font=font(MONO_B, 26), fill=GREEN)
    reel.add(img, 4.0)


def terminal_scene(reel: Reel, log_lines: list[str]) -> None:
    cmd = 'python run.py --scrip HDFCBANK --name "HDFC Bank" --ticker HDFCBANK.NS'
    shown: list[str] = []

    def draw() -> Image.Image:
        img, d = base()
        d.rectangle([30, 26, W - 30, H - 26], fill=TERM_BG)
        d.rectangle([30, 26, W - 30, 66], fill=(24, 34, 52))
        for j, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([52 + j * 26, 38, 68 + j * 26, 54], fill=c)
        d.text((W // 2 - 150, 36), "divergence — pipeline run", font=font(MONO, 16), fill=DIM)
        y = 84
        for ln in shown[-24:]:
            col = GOLD if "Stage" in ln or "DIVERGENCE MVP" in ln else TEXT
            if ln.startswith("$"):
                col = GREEN
            d.text((52, y), ln, font=font(MONO, 17), fill=col)
            y += 26
        d.rectangle([52, y + 2, 62, y + 20], fill=TEXT)  # cursor block
        return img

    # type the command
    for k in range(0, len(cmd) + 1, 6):
        shown = [f"$ {cmd[:k]}"]
        reel.add(draw(), 0.09)
    shown = [f"$ {cmd}"]
    reel.add(draw(), 0.8)

    body: list[str] = []
    for raw in log_lines:
        for piece in wrap(raw, 96):
            body.append(piece)
    cur: list[str] = []
    for ln in body:
        cur.append(ln)
        shown = [f"$ {cmd}"] + cur
        extra = 1.15 if ("Stage" in ln or "strongest divergence" in ln) else 0.0
        reel.add(draw(), 0.92 + extra)
    # closing ls
    shown = shown + ["", "$ ls output/", "chart_HDFCBANK.png   divergence_HDFCBANK_note.pdf   evidence_HDFCBANK.json",
                     "findings_HDFCBANK.json   panel_HDFCBANK.csv   run_log.txt"]
    reel.add(draw(), 3.2)


def discovery_card(reel: Reel, findings: dict) -> None:
    img, d = base(NAVY)
    d.rectangle([0, 0, W, 10], fill=GOLD)
    d.text((70, 60), "THE DISCOVERY", font=font(SANS_B, 30), fill=GOLD)
    y = 120
    for ln in wrap(findings["discovery"], 62):
        d.text((70, y), ln, font=font(SANS, 26), fill=WHITE)
        y += 38
    y += 26
    d.rectangle([70, y, 470, y + 52], fill=(176, 125, 16))
    d.text((88, y + 10), f"EVIDENCE STRENGTH: {findings['evidence_strength'].upper()}", font=font(SANS_B, 23), fill=WHITE)
    d.text((70, y + 78), f"Persistence: {findings['persistence_quarters']} quarters · Engine score: {findings['score']}",
           font=font(MONO, 21), fill=TEXT)
    reel.add(img, 11.0)


def pdf_scene(reel: Reel, pdf: Path, workdir: Path) -> None:
    subprocess.run(["pdftoppm", "-png", "-r", "100", "-f", "1", "-l", "2", str(pdf), str(workdir / "page")],
                   check=True, capture_output=True)
    for page, caption in [(1, "The generated 8-part note — page 1: discovery, strength, why it matters"),
                          (2, "Page 2: the numbers — the pair that broke, computed from the panel")]:
        p = workdir / f"page-{page}.png"
        if not p.exists():
            continue
        sheet = Image.open(p).convert("RGB")
        img, d = base((38, 42, 50))
        scale = min((W - 80) / sheet.width, (H - 110) / sheet.height)
        sheet = sheet.resize((int(sheet.width * scale), int(sheet.height * scale)))
        img.paste(sheet, ((W - sheet.width) // 2, 16))
        d.rectangle([0, H - 62, W, H], fill=NAVY)
        d.text((40, H - 44), caption, font=font(SANS_B, 21), fill=WHITE)
        reel.add(img, 9.0)


def end_card(reel: Reel) -> None:
    img, d = base(NAVY)
    d.rectangle([0, 0, W, 10], fill=GOLD)
    d.text((70, 210), "Every number in this note is computed", font=font(SANS_B, 36), fill=WHITE)
    d.text((70, 262), "by code from public data — and reproducible.", font=font(SANS_B, 36), fill=WHITE)
    d.text((70, 350), "github.com/millerthehacker22307/divergence-financial-research-agent", font=font(MONO, 22), fill=GOLD)
    d.text((70, 402), "Sample note + panel:  examples/   ·   Run it yourself:  python run.py", font=font(SANS, 23), fill=TEXT)
    d.text((70, 500), "DIVERGENCE — Finding the Next Signal · ATMOS 2026", font=font(SANS, 24), fill=DIM)
    reel.add(img, 6.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/demo.mp4")
    ap.add_argument("--rundir", default="output")
    ap.add_argument("--workdir", default="/tmp/divergence_demo_frames")
    args = ap.parse_args()

    rundir = Path(args.rundir)
    log_lines = (rundir / "run_log.txt").read_text(encoding="utf-8").splitlines()
    findings = json.loads((rundir / "findings_HDFCBANK.json").read_text(encoding="utf-8"))
    pdf = rundir / "divergence_HDFCBANK_note.pdf"

    reel = Reel(Path(args.workdir))
    title_card(reel)
    terminal_scene(reel, log_lines)
    discovery_card(reel, findings)
    pdf_scene(reel, pdf, Path(args.workdir))
    end_card(reel)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    reel.encode(out)
    print(f"demo video written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
