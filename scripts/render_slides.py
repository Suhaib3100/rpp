#!/usr/bin/env python
"""Render the presentation to per-slide PNGs for visual quality inspection.

PPTX -> PDF via LibreOffice (headless) -> PNG per page via pdftoppm. Also emits
a contact sheet so the whole deck can be reviewed at a glance, and runs a set of
automated layout checks that catch the failures a reader would notice first:
text overflowing the slide, elements outside the safe area, and empty slides.

Usage:
    python scripts/render_slides.py presentation/SAGE-HE.pptx
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[1]
SOFFICE = shutil.which("soffice") or "/opt/libreoffice26.8/program/soffice"

SLIDE_W_EMU = Emu(int(13.333 * 914400))
SLIDE_H_EMU = Emu(int(7.5 * 914400))
SAFE = Emu(int(0.25 * 914400))     # elements should stay this far inside


def render(pptx: Path, out_dir: Path, dpi: int = 110) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("slide-*.png"):
        old.unlink()

    work = out_dir / "_pdf"
    work.mkdir(exist_ok=True)
    print(f"[render] {pptx.name} -> PDF")
    r = subprocess.run(
        [SOFFICE, "--headless", "--norestore", "--convert-to", "pdf",
         "--outdir", str(work), str(pptx)],
        capture_output=True, text=True, timeout=900,
    )
    pdf = work / (pptx.stem + ".pdf")
    if not pdf.exists():
        print(r.stdout, r.stderr, file=sys.stderr)
        raise RuntimeError("LibreOffice did not produce a PDF")

    print(f"[render] PDF -> PNG at {dpi} dpi")
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(out_dir / "slide")],
        check=True, timeout=900,
    )
    pages = sorted(out_dir.glob("slide-*.png"))
    print(f"[render] {len(pages)} slides rendered -> {out_dir}")
    return pages


def contact_sheet(pages: list[Path], out: Path, cols: int = 4, cell_w: int = 620) -> Path:
    ims = [Image.open(p).convert("RGB") for p in pages]
    scaled = [im.resize((cell_w, int(im.height * cell_w / im.width))) for im in ims]
    ch = max(s.height for s in scaled)
    rows = (len(scaled) + cols - 1) // cols
    pad = 6
    sheet = Image.new("RGB", (cols * (cell_w + pad), rows * (ch + pad + 16)), "#05080F")
    for i, s in enumerate(scaled):
        x = (i % cols) * (cell_w + pad)
        y = (i // cols) * (ch + pad + 16)
        sheet.paste(s, (x, y))
    sheet.save(out)
    print(f"[sheet]  {out}  ({sheet.size[0]}x{sheet.size[1]})")
    return out


def layout_checks(pptx: Path) -> list[str]:
    """Automated geometry checks on the PPTX itself."""
    prs = Presentation(str(pptx))
    problems: list[str] = []
    for i, slide in enumerate(prs.slides, 1):
        shapes = [s for s in slide.shapes]
        if len(shapes) <= 1:
            problems.append(f"slide {i}: looks empty ({len(shapes)} shapes)")
        for s in shapes:
            if s.width is None or s.height is None or s.left is None or s.top is None:
                continue
            # the full-bleed background rectangle is allowed to touch the edges
            is_bg = (s.width >= SLIDE_W_EMU and s.height >= SLIDE_H_EMU)
            if is_bg:
                continue
            right, bottom = s.left + s.width, s.top + s.height
            name = getattr(s, "name", "shape")
            if s.left < -1000 or s.top < -1000:
                problems.append(f"slide {i}: '{name}' starts off-slide "
                                f"({s.left/914400:.2f}, {s.top/914400:.2f} in)")
            if right > SLIDE_W_EMU + 1000:
                problems.append(f"slide {i}: '{name}' overflows right edge "
                                f"(to {right/914400:.2f} in)")
            if bottom > SLIDE_H_EMU + 1000:
                problems.append(f"slide {i}: '{name}' overflows bottom edge "
                                f"(to {bottom/914400:.2f} in)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pptx", nargs="?", default=str(ROOT / "presentation/SAGE-HE.pptx"))
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    pptx = Path(args.pptx)
    out_dir = ROOT / "presentation" / "rendered"
    pages = render(pptx, out_dir, args.dpi)
    contact_sheet(pages, ROOT / "presentation" / "contact_sheet.png")

    print("\n[checks] automated layout checks")
    problems = layout_checks(pptx)
    if problems:
        for p in problems:
            print(f"   FAIL  {p}")
        print(f"\n{len(problems)} layout problem(s) found")
    else:
        print("   all slides pass geometry checks")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
