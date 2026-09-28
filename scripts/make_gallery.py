#!/usr/bin/env python3
"""
Rescue Pulse - Build the documentation gallery.

The README photos are phone snapshots at wildly different aspect ratios
(0.95 to 3.0), so in a row they sit ragged and each one carries 1.4-2.5 MB of
pixels to display at 300 px. This crops every source to the ratio its row needs,
resizes it to the size it is actually displayed at, and frames it in a titled
card that matches the diagram palette.

    python3 scripts/make_gallery.py           # rebuild assets/gallery/
    python3 scripts/make_gallery.py --check   # report only, write nothing

Source photos are never modified. Re-run after replacing any source image.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUT = ASSETS / "gallery"

# Palette, matching the animated SVGs.
BG = "#0d1117"
SURFACE = "#161b22"
EDGE = "#21262d"
FG = "#e6edf3"
DIM = "#7d8590"
CYAN = "#39c5cf"
AMBER = "#d29922"
GREEN = "#2ea043"
RED = "#ff2d2d"
PURPLE = "#bc8cff"
LIME = "#7ee787"
PINK = "#ff6ec7"

FONT = "DejaVu-Sans-Mono-Bold"
FONT_REG = "DejaVu-Sans-Mono"

# Each entry: source, output stem, card width, crop ratio (w/h), vertical bias
# (0 = keep the top, 0.5 = centre, 1 = keep the bottom), title, accent.
# `2*width` is the source resolution, i.e. 2x for a retina README.
CARDS = [
    # --- Phase 1: the ST7735S panel ---
    ("RescuePulse-Display-Boot.jpg", "display-boot", 320, 4 / 3, 0.42,
     "BOOT SCREEN", CYAN),
    ("Traffic_Demo.jpg", "traffic-demo", 320, 4 / 3, 0.40,
     "LIVE DETECTION", LIME),
    # --- Phase 1: direction of arrival ---
    ("Siren-Left.jpg", "doa-left", 250, 4 / 3, 0.38,
     "SOURCE LEFT", AMBER),
    ("Siren-Right.jpg", "doa-right", 250, 4 / 3, 0.42,
     "SOURCE RIGHT", RED),
    ("Centre-Siren.jpg", "doa-centre", 250, 4 / 3, 0.40,
     "SOURCE CENTRE", GREEN),
    # --- Phase 2: bench verification, the 3x3 GPIO lamp array ---
    ("traffic_init.jpg", "bench-normal", 430, 4 / 3, 0.45,
     "MODE_NORMAL", GREEN),
    ("Left_detection.jpg", "bench-left", 430, 4 / 3, 0.45,
     "SIREN ON LEFT", AMBER),
    ("Center_detection.jpg", "bench-centre", 430, 4 / 3, 0.45,
     "SIREN ON CENTRE", GREEN),
    ("Right_detect.jpg", "bench-right", 430, 4 / 3, 0.45,
     "SIREN ON RIGHT", RED),
    # --- model ---
    ("../models/confusion_matrix.png", "confusion-matrix", 330, 1 / 1, 0.5,
     "CONFUSION MATRIX", PURPLE),
    ("../models/training_curves.png", "training-curves", 900, 3 / 1, 0.5,
     "TRAINING CURVES", PURPLE),
    ("Train-RescuePulse.png", "model-summary", 430, 16 / 9, 0.0,
     "MODEL SUMMARY", PINK),
]

BORDER = 2
PAD = 8


def have_magick():
    return subprocess.run(["which", "magick"], capture_output=True).returncode == 0


def text_metrics(text, font, pointsize):
    """Rendered (width, height) of `text` in pixels."""
    r = subprocess.run(
        ["magick", "-font", font, "-pointsize", str(pointsize),
         "-format", "%w %h", f"label:{text}", "info:"],
        capture_output=True, text=True)
    try:
        w, h = (int(x) for x in r.stdout.split())
        return w, h
    except ValueError:
        return 0, 0


def build(src, stem, width, ratio, bias, title, accent, check=False):
    sp = (ASSETS / src).resolve()
    if not sp.exists():
        return f"MISSING SOURCE  {src}", None
    iw, ih = (int(x) for x in
              subprocess.run(["identify", "-format", "%w %h", str(sp)],
                             capture_output=True, text=True).stdout.split())
    target_w = width * 2
    target_h = int(round(target_w / ratio))

    # Centre-weighted crop to the requested ratio.
    if iw / ih > ratio:                       # too wide -> trim the sides
        cw = int(round(ih * ratio))
        cx = (iw - cw) // 2
        crop = f"{cw}x{ih}+{cx}+0"
        resize = f"{target_w}x{target_h}!"
    else:                                      # too tall -> trim top/bottom
        ch = int(round(iw / ratio))
        cy = int(round((ih - ch) * bias))
        cy = max(0, min(cy, ih - ch))
        crop = f"{iw}x{ch}+0+{cy}"
        resize = f"{target_w}x{target_h}!"

    title_pt = max(11, width // 20)
    meta_pt = max(9, width // 26)

    # The title bar is sized from the measured text boxes, not a fixed height:
    # the point sizes scale with card width, so a constant BAR clipped the
    # provenance line on every card wider than ~300px.
    rule = 3
    gap = 3
    _, title_h = text_metrics(title, FONT, title_pt)
    brand_w, brand_h = text_metrics("ON-DEVICE", FONT_REG, meta_pt)
    bar = rule + PAD + title_h + gap + brand_h + PAD
    card_w = target_w + BORDER * 2
    card_h = target_h + bar + BORDER * 2
    # In --check mode the card is rendered to a scratch file so the reported
    # dimensions and byte size are real, then discarded without touching OUT.
    tmp_dir = None
    if check:
        tmp_dir = tempfile.TemporaryDirectory()
        dst = Path(tmp_dir.name) / f"{stem}.jpg"
    else:
        dst = OUT / f"{stem}.jpg"
    photo_bottom = target_h + BORDER          # y where the title bar starts
    title_y = photo_bottom + rule + PAD
    brand_y = title_y + title_h + gap
    # Right-hand provenance is placed at an explicit x derived from the measured
    # text width. `-gravity northeast` + a negative X offset is NOT reliable for
    # right-insetting (the offset is applied in the opposite direction and the
    # glyphs run off the canvas), so measure and place instead of anchoring.
    brand_x = card_w - BORDER - PAD - brand_w

    # Compose: crop -> resize -> 2px border -> pad a title bar underneath.
    # `-extent` grows the canvas downward in SURFACE, so the bar is painted by
    # the background rather than by a -draw that could cover the photo.
    cmd = [
        "magick", str(sp),
        "-crop", crop, "+repage",
        "-resize", resize,
        "-bordercolor", SURFACE, "-border", str(BORDER),
        "-background", SURFACE, "-gravity", "north",
        "-extent", f"{card_w}x{card_h}",
        # accent rule separating photo from caption
        "-fill", accent, "-stroke", "none",
        "-draw", f"rectangle 0,{photo_bottom} {card_w},{photo_bottom + rule}",
        # title, bottom-left of the bar
        "-gravity", "northwest",
        "-font", FONT, "-pointsize", str(title_pt), "-fill", accent,
        "-annotate", f"+{PAD}+{title_y}", title,
        # provenance, the two ends of the bar
        "-font", FONT_REG, "-pointsize", str(meta_pt), "-fill", DIM,
        "-annotate", f"+{PAD}+{brand_y}", "RESCUEPULSE",
        "-annotate", f"+{brand_x}+{brand_y}", "ON-DEVICE",
        "-quality", "88", "-interlace", "Plane", "-strip",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if tmp_dir:
            tmp_dir.cleanup()
        return f"FAILED  {src}: {r.stderr.strip().splitlines()[-1][:90]}", None
    kb = dst.stat().st_size / 1024
    src_kb = sp.stat().st_size / 1024
    if tmp_dir:
        tmp_dir.cleanup()
    return (f"  {stem:18} {iw}x{ih} -> {card_w}x{card_h}  "
            f"{src_kb:7.0f} KB -> {kb:5.0f} KB"), kb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    if not have_magick():
        print("  ImageMagick 'magick' not found - cannot build the gallery")
        return 1

    if not a.check:
        OUT.mkdir(parents=True, exist_ok=True)
    print("Rescue Pulse - gallery")
    print("=" * 74)
    total = 0.0
    for src, stem, width, ratio, bias, title, accent in CARDS:
        if a.check and not (OUT / f"{stem}.jpg").exists():
            print(f"  MISSING OUTPUT  {stem}.jpg")
            continue
        line, kb = build(src, stem, width, ratio, bias, title, accent, check=a.check)
        print(line)
        if kb:
            total += kb
    print("=" * 74)
    print(f"  {len(CARDS)} cards, {total / 1024:.1f} MB total in assets/gallery/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
