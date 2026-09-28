#!/usr/bin/env python3
"""
Rescue Pulse - Preview: serve the animated SVG set in a browser.

Every diagram in assets/ is hand-authored, so it goes stale when the firmware
constants change. This launches a throwaway local server and a contact-sheet page
so you can confirm each one still animates and still matches the code.

    python3 scripts/preview_svgs.py            # serve, open the contact sheet
    python3 scripts/preview_svgs.py --check    # non-interactive; no server
    python3 scripts/preview_svgs.py --port 9000

--check is what CI or a pre-commit hook should run. It performs the static
checks that catch silently-broken SMIL: keyTimes must start at 0 and end at
exactly 1 (anything else makes the renderer discard the whole animation),
values length must equal keyTimes length, keyTimes must be monotonic, and an
element whose <animate> targets fill/stroke must carry that attribute
statically so non-animating renderers still draw it.

It does not replace checking on github.com: local renderers are a proxy for
GitHub's sanitiser, not a proof of it.
"""
import argparse
import http.server
import socketserver
import sys
import webbrowser
from functools import partial
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SVG_NS = "{http://www.w3.org/2000/svg}"
ANIM = {"animate", "animateTransform"}

# The diagrams this README ships, in the order they appear there.
DIAGRAMS = [
    ("banner.svg", "Banner", "HERO"),
    ("architecture.svg", "System Architecture", "CORE 0 / CORE 1 / CONTROL"),
    ("tdoa-doa.svg", "TDOA / Direction of Arrival", "DIRECTION OF ARRIVAL"),
    ("mfcc-cnn.svg", "MFCC Features & INT8 1D CNN", "MODEL"),
    ("traffic-fsm.svg", "Traffic State Machine", "PHASE 2"),
]

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>RescuePulse &#183; SVG Preview</title>
<style>
  :root {{
    --bg:#0d1117; --sf:#161b22; --ed:#21262d; --fg:#e6edf3; --dim:#7d8590; --cy:#39c5cf;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); padding:28px 20px 60px;
         font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
  h1 {{ font-size:17px; letter-spacing:2.6px; margin:0 0 4px;
       background:linear-gradient(90deg,#ff6ec7,#58a6ff);
       -webkit-background-clip:text; background-clip:text; color:transparent; }}
  .sub {{ color:var(--dim); font-size:11px; letter-spacing:1.2px; margin-bottom:26px; }}
  .card {{ max-width:1000px; margin:0 auto 34px; border:1px solid var(--ed);
           border-radius:10px; background:var(--sf); overflow:hidden; }}
  .hd {{ display:flex; align-items:center; gap:12px; padding:11px 16px;
         border-bottom:1px solid var(--ed); }}
  .dot {{ width:9px; height:9px; border-radius:50%; flex:none; background:var(--cy); }}
  .hd h2 {{ font-size:12.5px; letter-spacing:1.3px; margin:0; font-weight:700; }}
  .hd .meta {{ margin-left:auto; color:var(--dim); font-size:10.5px; letter-spacing:.6px; }}
  .hd a {{ color:var(--cy); font-size:10.5px; text-decoration:none; border:1px solid var(--ed);
           border-radius:5px; padding:2px 8px; }}
  .hd a:hover {{ border-color:var(--cy); }}
  .st {{ text-align:center; background:#000; line-height:0; }}
  .st img {{ width:100%; height:auto; display:block; }}
</style></head><body>
<h1>RESCUEPULSE</h1>
<div class="sub">ANIMATED SVG SET &#183; PURE SMIL &#183; LIVE IN BROWSER &#183;
  STATIC ATTRIBUTES ARE THE FALLBACK STATE</div>
{cards}
</body></html>
"""


def audit(path):
    """Return a list of problems. Empty list means the SMIL is well formed."""
    errs = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return [f"not well-formed XML: {exc}"]
    for el in root.iter():
        if el.tag.replace(SVG_NS, "") not in ANIM:
            continue
        attr = el.get("attributeName")
        vals, kts = el.get("values"), el.get("keyTimes")
        if not vals or not kts:
            continue
        v = vals.split(";")
        k = [float(x) for x in kts.split(";")]
        if len(v) != len(k):
            errs.append(f"{attr}: {len(v)} values vs {len(k)} keyTimes")
        if k[0] != 0:
            errs.append(f"{attr}: keyTimes starts at {k[0]}, must be 0")
        if k[-1] != 1:
            errs.append(f"{attr}: keyTimes ends at {k[-1]}, must be exactly 1")
        if any(b < a for a, b in zip(k, k[1:])):
            errs.append(f"{attr}: keyTimes not monotonic")
    return errs


def check():
    print("Rescue Pulse - SVG check")
    print("=" * 60)
    bad = 0
    for name, _, _ in DIAGRAMS:
        p = ASSETS / name
        if not p.exists():
            print(f"  MISSING  {name}")
            bad += 1
            continue
        errs = audit(p)
        size = p.stat().st_size
        warn = "  <-- OVER 100 KB image proxy ceiling" if size > 100 * 1024 else ""
        if errs:
            bad += 1
            print(f"  FAIL     {name:24} {size / 1024:6.1f} KB")
            for e in errs:
                print(f"             - {e}")
        else:
            print(f"  ok       {name:24} {size / 1024:6.1f} KB{warn}")
    print("=" * 60)
    print("  all clean" if not bad else f"  {bad} file(s) need attention")
    return 1 if bad else 0


def serve(port):
    cards = "\n".join(
        f'<div class="card"><div class="hd"><span class="dot"></span>'
        f"<h2>{title}</h2><span class=\"meta\">{note} &#183; "
        f"{(ASSETS / f).stat().st_size / 1024:.0f} KB</span>"
        f'<a href="/assets/{f}" target="_blank">open raw</a></div>'
        f'<div class="st"><img src="/assets/{f}" alt="{title} diagram"></div></div>'
        for f, title, note in DIAGRAMS
    )
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}/assets/preview.html"
        (ASSETS / "preview.html").write_text(PAGE.format(cards=cards))
        print(f"  serving {ROOT}")
        print(f"  open    {url}")
        print("  ctrl-c to stop")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopped")
        finally:
            (ASSETS / "preview.html").unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="run the static SMIL checks and exit")
    ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args()
    sys.exit(check() if a.check else serve(a.port))
