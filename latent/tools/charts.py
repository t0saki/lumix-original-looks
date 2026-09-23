"""Synthetic test charts C1..C7 (TOOLS_SPEC §T3).

These charts exist to expose what a photograph hides: banding, posterisation,
per-step tint, hue twist inside a gradient, gamut-boundary creases and shadow
noise colouring.  Therefore:

* everything is computed in float and stored as **float32 sRGB code values**
  in ``[0, 1]`` (shape ``(H, W, 3)``) — the ``.npz`` is the chart, the ``.png``
  next to it is only a preview;
* **no dithering anywhere** — a gradient is the exact analytic ramp, so a LUT
  that introduces contouring shows contouring and nothing masks it;
* every pixel is inside the sRGB gamut (asserted before saving).  C5 is the one
  chart that deliberately *shows* the gamut boundary: out-of-gamut (L, C, h)
  cells are painted neutral grey rather than clipped;
* deterministic: the only randomness (C7 shadow noise) comes from a pinned
  seed, so two builds are bit-identical;
* every chart is at most 1200 x 800 px, sized for the sheet budget in T4.

Text labels are burnt into the chart itself (neutral grey on the neutral
surround) so a chart stays self-describing after it has been through a LUT.

CLI
---
    $R/py tools/charts.py build            # write all charts to $W/charts/
    $R/py tools/charts.py build --only C1,C7
    $R/py tools/charts.py info             # list names / sizes / descriptions

API
---
    build_chart(name) -> np.ndarray (float32, H x W x 3, sRGB code)
    load_chart(name)  -> np.ndarray (float32) read back from $W/charts/<name>.npz
    chart_path(name)  -> Path
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine.color import (
    lch_to_oklab,
    oklab_to_linear_srgb,
    srgb_decode,
    srgb_encode,
)
from engine.color import _max_chroma as max_chroma  # vectorised bisection

# ---------------------------------------------------------------------------
# paths / constants
# ---------------------------------------------------------------------------

ROOT = Path(os.environ.get("LATENT_ROOT", Path(__file__).resolve().parent.parent))
WORK = Path(os.environ.get("LATENT_WORK", ROOT / "work.nosync"))
CHARTS_DIR = WORK / "charts"

CHART_NAMES = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")

MAX_W, MAX_H = 1200, 800

BG = 0.18  # neutral surround (sRGB code)
FG = 0.88  # label ink
FG_DIM = 0.62

NOISE_SEED = 20260921  # pinned: C7 must be reproducible bit for bit

_GAMUT_EPS = 5e-4  # on linear values, same order as T1's in-gamut eps


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _font(size: int):
    for cand in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _text(arr, x, y, s, size=13, color=FG, anchor="la"):
    """Alpha-blend ASCII text into a float code-value array (no quantisation)."""
    s = str(s)
    if not s:
        return
    font = _font(size)
    truetype = isinstance(font, ImageFont.FreeTypeFont)
    h, w = arr.shape[:2]
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    if truetype:
        d.text((x, y), s, fill=255, font=font, anchor=anchor)
    else:  # bitmap fallback: no anchor support, approximate
        tw = d.textlength(s, font=font)
        if anchor[0] == "m":
            x -= tw / 2.0
        elif anchor[0] == "r":
            x -= tw
        d.text((x, y), s, fill=255, font=font)
    a = (np.asarray(mask, dtype=np.float64) / 255.0)[..., None]
    col = np.asarray(color, dtype=np.float64)
    if col.ndim == 0:
        col = np.repeat(col[None], 3, axis=0)
    arr *= 1.0 - a
    arr += a * col


def _canvas(w, h, bg=BG):
    return np.full((h, w, 3), float(bg), dtype=np.float64)


def _code_from_oklch(L, C, h_deg):
    """OKLCh -> sRGB code values + a boolean in-gamut mask (linear, eps 5e-4)."""
    L = np.asarray(L, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    h_deg = np.asarray(h_deg, dtype=np.float64)
    lin = oklab_to_linear_srgb(lch_to_oklab(L, C, h_deg))
    ok = np.all((lin >= -_GAMUT_EPS) & (lin <= 1.0 + _GAMUT_EPS), axis=-1)
    return srgb_encode(np.clip(lin, 0.0, 1.0)), ok


def _safe_chroma(L, h_deg, C, headroom=0.98):
    """Chroma clamped to `headroom` x the in-gamut maximum at (L, h)."""
    cm = max_chroma(np.asarray(L, float), np.asarray(h_deg, float), iters=30)
    return np.minimum(np.asarray(C, float), headroom * cm)


def _fill(arr, y0, y1, x0, x1, value):
    arr[y0:y1, x0:x1] = value


def _hue_lerp(h0, h1, t):
    """Shortest-arc hue interpolation, degrees."""
    d = (h1 - h0 + 180.0) % 360.0 - 180.0
    return (h0 + d * t) % 360.0


def _assert_ok(name, arr):
    assert arr.ndim == 3 and arr.shape[2] == 3, f"{name}: bad shape {arr.shape}"
    h, w = arr.shape[:2]
    assert w <= MAX_W and h <= MAX_H, f"{name}: {w}x{h} exceeds {MAX_W}x{MAX_H}"
    assert np.isfinite(arr).all(), f"{name}: non-finite values"
    lo, hi = float(arr.min()), float(arr.max())
    assert lo >= -1e-7 and hi <= 1.0 + 1e-7, f"{name}: out of [0,1] ({lo}, {hi})"


# ---------------------------------------------------------------------------
# C1 — grey ramp (continuous + 21 steps) + 21 colour patches
# ---------------------------------------------------------------------------

C1_PATCH_L = 0.62
C1_PATCH_C = 0.10


def build_C1():
    W, H = 1200, 400
    a = _canvas(W, H)
    x0, x1 = 20, 1180
    span = x1 - x0

    _text(a, 12, 5, "C1  grey ramp 0-1 (continuous, undithered) | 21 grey steps | "
                    f"21 hue patches  L={C1_PATCH_L:.2f} C={C1_PATCH_C:.2f}", 14)

    # continuous ramp, exact linear code ramp across the band
    t = np.linspace(0.0, 1.0, span)
    a[28:168, x0:x1] = t[None, :, None]

    # 21 grey steps 0,5,...,100 %
    edges = x0 + np.round(np.linspace(0, span, 22)).astype(int)
    steps = np.linspace(0.0, 1.0, 21)
    for i, v in enumerate(steps):
        _fill(a, 172, 272, edges[i], edges[i + 1], v)
        cx = 0.5 * (edges[i] + edges[i + 1])
        _text(a, cx, 275, f"{int(round(v * 255))}", 11, FG_DIM, anchor="ma")

    # 21 hue patches on one L/C circle (chroma clamped to the gamut)
    hues = np.linspace(0.0, 360.0, 22)[:-1]
    Cs = _safe_chroma(np.full_like(hues, C1_PATCH_L), hues, C1_PATCH_C)
    codes, ok = _code_from_oklch(np.full_like(hues, C1_PATCH_L), Cs, hues)
    assert ok.all(), "C1: hue patches out of gamut"
    for i, h in enumerate(hues):
        _fill(a, 296, 376, edges[i], edges[i + 1], codes[i])
        cx = 0.5 * (edges[i] + edges[i + 1])
        _text(a, cx, 379, f"h{int(round(h))}", 11, FG_DIM, anchor="ma")
    return a


# ---------------------------------------------------------------------------
# C2 — skin ladder: 6 hues (25..75 deg) x 9 lightness x 2 chroma
# ---------------------------------------------------------------------------

C2_HUES = np.array([25.0, 35.0, 45.0, 55.0, 65.0, 75.0])
C2_LS = np.linspace(0.30, 0.87, 9)
C2_CS = (0.075, 0.12)


def build_C2():
    W, H = 1200, 712
    a = _canvas(W, H)
    _text(a, 12, 5, "C2  skin ladder   hue 25-75 deg x L 0.30-0.87 x chroma "
                    "(chroma clamped to gamut where needed)", 14)

    pw, ph = 90, 70
    y_top = 62
    blocks = [(56, C2_CS[0]), (620, C2_CS[1])]
    for bx, C in blocks:
        _text(a, bx, 26, f"C = {C:.3f}", 13, FG)
        for j, h in enumerate(C2_HUES):
            _text(a, bx + j * pw + pw / 2, 44, f"{h:.0f}", 11, FG_DIM, anchor="ma")
        Lg, Hg = np.meshgrid(C2_LS, C2_HUES, indexing="ij")
        Cg = _safe_chroma(Lg, Hg, np.full_like(Lg, C))
        codes, ok = _code_from_oklch(Lg, Cg, Hg)
        assert ok.all(), "C2: patch out of gamut"
        for i in range(len(C2_LS)):
            for j in range(len(C2_HUES)):
                y = y_top + i * ph
                x = bx + j * pw
                _fill(a, y, y + ph, x, x + pw, codes[i, j])
    for i, L in enumerate(C2_LS):
        _text(a, 50, y_top + i * ph + ph / 2, f"{L:.2f}", 11, FG_DIM, anchor="rm")
    _text(a, 12, 694, "rows: OKLCh L   columns: OKLCh hue (deg)   left block low "
                      "chroma / right block high chroma", 11, FG_DIM)
    return a


# ---------------------------------------------------------------------------
# C3 — four sky gradients, zenith (left) -> horizon (right)
# ---------------------------------------------------------------------------

# (label, (L,C,h) zenith, (L,C,h) horizon)
C3_GRADIENTS = (
    ("clear noon: zenith blue -> pale horizon", (0.72, 0.085, 255.0), (0.90, 0.018, 240.0)),
    ("hazy: soft blue-grey -> warm white", (0.78, 0.035, 232.0), (0.93, 0.014, 75.0)),
    ("dusk: orange -> blue", (0.70, 0.130, 55.0), (0.40, 0.100, 265.0)),
    ("deep twilight: indigo -> violet glow", (0.18, 0.050, 275.0), (0.45, 0.035, 300.0)),
)


def build_C3():
    W, H = 1200, 650
    a = _canvas(W, H)
    _text(a, 12, 5, "C3  sky gradients, zenith (left) -> horizon (right), undithered "
                    "OKLCh interpolation", 14)
    x0, x1 = 20, 1180
    span = x1 - x0
    t = np.linspace(0.0, 1.0, span)
    for i, (label, za, ho) in enumerate(C3_GRADIENTS):
        y = 30 + i * 152
        _text(a, 20, y, label, 12, FG)
        L = za[0] + (ho[0] - za[0]) * t
        C = za[1] + (ho[1] - za[1]) * t
        h = _hue_lerp(za[2], ho[2], t)
        codes, ok = _code_from_oklch(L, C, h)
        assert ok.all(), f"C3: gradient {i} leaves the gamut"
        a[y + 18 : y + 148, x0:x1] = codes[None, :, :]
    return a


# ---------------------------------------------------------------------------
# C4 — Granger rainbow: hue x chroma, plus a hue x lightness wedge
# ---------------------------------------------------------------------------

C4_L = 0.65
C4_WEDGE_C = 0.10


def build_C4():
    W, H = 1200, 556
    a = _canvas(W, H)
    _text(a, 12, 5, "C4  Granger rainbow", 14)
    x0, x1 = 20, 1180
    span = x1 - x0
    hue = np.linspace(0.0, 360.0, span)

    # block 1: chroma 0 (top) -> gamut boundary (bottom) at fixed L
    _text(a, 20, 26, f"hue x chroma   L = {C4_L:.2f}   (bottom row = sRGB gamut hull)",
          12, FG)
    h1 = 300
    cmax = max_chroma(np.full_like(hue, C4_L), hue, iters=30)
    frac = np.linspace(0.0, 0.98, h1)[:, None]
    Cg = frac * cmax[None, :]
    Hg = np.broadcast_to(hue[None, :], Cg.shape)
    codes, ok = _code_from_oklch(np.full_like(Cg, C4_L), Cg, Hg)
    assert ok.all(), "C4: chroma block out of gamut"
    a[44 : 44 + h1, x0:x1] = codes

    # block 2: lightness wedge at (almost) constant chroma
    _text(a, 20, 350, f"hue x lightness   C = min({C4_WEDGE_C:.2f}, 0.95 x Cmax)   "
                      "L 0.95 (top) -> 0.08 (bottom)", 12, FG)
    h2 = 160
    Lcol = np.linspace(0.95, 0.08, h2)[:, None]
    Lg = np.broadcast_to(Lcol, (h2, span))
    Hg2 = np.broadcast_to(hue[None, :], (h2, span))
    Cg2 = _safe_chroma(Lg, Hg2, np.full((h2, span), C4_WEDGE_C), headroom=0.95)
    codes2, ok2 = _code_from_oklch(Lg, Cg2, Hg2)
    assert ok2.all(), "C4: lightness wedge out of gamut"
    a[368 : 368 + h2, x0:x1] = codes2

    for hv in range(0, 361, 60):
        cx = x0 + (hv / 360.0) * (span - 1)
        _text(a, cx, 532, f"{hv}", 11, FG_DIM, anchor="ma")
    return a


# ---------------------------------------------------------------------------
# C5 — hue x chroma polar discs at four lightnesses
# ---------------------------------------------------------------------------

C5_LS = (0.35, 0.50, 0.65, 0.80)
# Chroma at the disc rim, shared by all four discs so they stay comparable.
# 0.22 is the top of the fingerprint probe's chroma sweep (T2 chroma_vs_c) and
# keeps 30-65 % of each disc in gamut; a larger rim leaves mostly grey.
C5_RADIUS_C = 0.22
C5_OOG = 0.50  # neutral grey for out-of-gamut / outside the disc


def build_C5():
    W, H = 1200, 352
    a = _canvas(W, H)
    _text(a, 12, 5, "C5  hue x chroma discs   angle = hue (0 deg right, CCW), "
                    f"radius = chroma 0 -> {C5_RADIUS_C:.2f}", 14)
    size = 280
    yy, xx = np.mgrid[0:size, 0:size]
    u = (xx + 0.5) / size * 2.0 - 1.0
    v = 1.0 - (yy + 0.5) / size * 2.0
    r = np.hypot(u, v)
    theta = np.degrees(np.arctan2(v, u)) % 360.0
    inside = r <= 1.0
    for i, L in enumerate(C5_LS):
        x = 16 + i * 296
        _text(a, x + size / 2, 26, f"L = {L:.2f}", 13, FG, anchor="ma")
        C = r * C5_RADIUS_C
        codes, ok = _code_from_oklch(np.full_like(C, L), C, theta)
        keep = inside & ok
        tile = np.where(keep[..., None], codes, C5_OOG)
        a[44 : 44 + size, x : x + size] = tile
    _text(a, 12, 330, "grey = outside the sRGB gamut at that (L, hue, chroma)",
          11, FG_DIM)
    return a


# ---------------------------------------------------------------------------
# C6 — memory colours
# ---------------------------------------------------------------------------

# name, sRGB 8-bit code
C6_COLOURS = (
    ("foliage sunlit", (112, 148, 62)),
    ("foliage shade", (52, 76, 44)),
    ("grass", (126, 152, 70)),
    ("autumn leaf", (186, 96, 40)),
    ("sky blue", (105, 150, 205)),
    ("deep sky", (52, 96, 176)),
    ("sand", (206, 180, 144)),
    ("brick", (150, 78, 60)),
    ("skin light", (238, 203, 179)),
    ("skin mid", (198, 150, 120)),
    ("skin deep", (104, 70, 54)),
    ("denim", (72, 98, 136)),
    ("coke red", (227, 29, 37)),
    ("taxi yellow", (245, 188, 20)),
    ("neon magenta", (238, 42, 190)),
    ("neon cyan", (34, 224, 234)),
    ("white shirt", (240, 240, 238)),
    ("warm wall", (231, 214, 192)),
    ("grey card 18%", (119, 119, 119)),
    ("black hair", (28, 26, 28)),
)


def build_C6():
    W, H = 1200, 762
    a = _canvas(W, H)
    _text(a, 12, 5, "C6  memory colours (sRGB 8-bit values burnt in)", 14)
    cols, pw, ph = 5, 216, 150
    pitch_x, pitch_y = 236, 180
    for k, (name, rgb8) in enumerate(C6_COLOURS):
        r, c = divmod(k, cols)
        x = 20 + c * pitch_x
        y = 34 + r * pitch_y
        val = np.asarray(rgb8, dtype=np.float64) / 255.0
        _fill(a, y, y + ph, x, x + pw, val)
        _text(a, x, y + ph + 3, name, 12, FG)
        _text(a, x, y + ph + 18, "%d,%d,%d" % rgb8, 10, FG_DIM)
    return a


# ---------------------------------------------------------------------------
# C7 — shadow noise wedge
# ---------------------------------------------------------------------------

C7_CODE_MAX = 0.12  # dark ramp 0 -> 12 % code
C7_NOISE_A = 0.030  # shot-noise coefficient, linear light
C7_NOISE_Y0 = 1.0e-4  # read-noise floor, linear light
C7_BANDS = ("clean (reference)", "luma noise", "chroma noise (per-channel)")


def build_C7():
    W, H = 1200, 476
    a = _canvas(W, H)
    _text(a, 12, 5, f"C7  shadow wedge, code 0 -> {C7_CODE_MAX:.2f}, signal-dependent "
                    f"noise in linear light (seed {NOISE_SEED})", 14)
    x0, x1 = 20, 1180
    span = x1 - x0
    bh = 120
    code = np.linspace(0.0, C7_CODE_MAX, span)
    y_lin = srgb_decode(code)
    sigma = C7_NOISE_A * np.sqrt(y_lin + C7_NOISE_Y0)  # (span,)
    rng = np.random.default_rng(NOISE_SEED)

    for i, label in enumerate(C7_BANDS):
        y = 30 + i * 142
        _text(a, 20, y, label, 12, FG)
        base = np.broadcast_to(y_lin[None, :, None], (bh, span, 3)).copy()
        if i == 1:
            n = rng.standard_normal((bh, span))[:, :, None] * sigma[None, :, None]
            base = base + n
        elif i == 2:
            n = rng.standard_normal((bh, span, 3)) * sigma[None, :, None]
            luma = n.mean(axis=2, keepdims=True)
            base = base + 0.4 * luma + 1.3 * (n - luma)
        a[y + 18 : y + 18 + bh, x0:x1] = srgb_encode(np.clip(base, 0.0, 1.0))

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        cv = frac * C7_CODE_MAX
        cx = x0 + frac * (span - 1)
        anchor = "la" if frac == 0.0 else ("ra" if frac == 1.0 else "ma")
        _text(a, cx, 456, f"{cv:.3f} ({cv * 255:.0f})", 11, FG_DIM, anchor=anchor)
    return a


# ---------------------------------------------------------------------------
# registry / io
# ---------------------------------------------------------------------------

_BUILDERS = {
    "C1": build_C1,
    "C2": build_C2,
    "C3": build_C3,
    "C4": build_C4,
    "C5": build_C5,
    "C6": build_C6,
    "C7": build_C7,
}

DESCRIPTIONS = {
    "C1": "grey ramp (continuous + 21 steps) + 21 hue patches",
    "C2": "skin ladder: 6 hues 25-75 deg x 9 lightness x 2 chroma",
    "C3": "four sky gradients (clear noon, hazy, dusk, deep twilight)",
    "C4": "Granger rainbow: hue x chroma and hue x lightness",
    "C5": "hue x chroma polar discs at L = .35/.50/.65/.80",
    "C6": "memory colours (20 patches incl. deep skin)",
    "C7": "shadow noise wedge, 0-12 % code, luma + chroma noise",
}


def _norm(name: str) -> str:
    n = str(name).strip().upper()
    if n not in _BUILDERS:
        raise KeyError(f"unknown chart {name!r}; known: {', '.join(CHART_NAMES)}")
    return n


def build_chart(name: str) -> np.ndarray:
    """Build one chart in memory. float32 sRGB code values, (H, W, 3)."""
    n = _norm(name)
    arr = _BUILDERS[n]()
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    _assert_ok(n, arr)
    return arr


def chart_path(name: str, charts_dir: Path | None = None) -> Path:
    return Path(charts_dir or CHARTS_DIR) / f"{_norm(name)}.npz"


def save_chart(name: str, arr: np.ndarray, charts_dir: Path | None = None) -> Path:
    n = _norm(name)
    d = Path(charts_dir or CHARTS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    npz = d / f"{n}.npz"
    np.savez_compressed(
        npz,
        chart=arr.astype(np.float32),
        name=n,
        description=DESCRIPTIONS[n],
    )
    png = d / f"{n}.png"
    Image.fromarray(
        np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
    ).save(png, optimize=True)
    return npz


def load_chart(name: str, charts_dir: Path | None = None) -> np.ndarray:
    """Read a built chart back as float32 sRGB code values."""
    p = chart_path(name, charts_dir)
    if not p.exists():
        raise FileNotFoundError(f"{p} — run `py tools/charts.py build` first")
    with np.load(p) as z:
        return np.asarray(z["chart"], dtype=np.float32)


def build_all(charts_dir: Path | None = None, only=None, verbose=True) -> dict:
    d = Path(charts_dir or CHARTS_DIR)
    names = [_norm(x) for x in only] if only else list(CHART_NAMES)
    out = {}
    for n in names:
        arr = build_chart(n)
        npz = save_chart(n, arr, d)
        png = npz.with_suffix(".png")
        out[n] = {
            "shape": [int(arr.shape[1]), int(arr.shape[0])],
            "npz_kb": round(npz.stat().st_size / 1024, 1),
            "png_kb": round(png.stat().st_size / 1024, 1),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        if verbose:
            i = out[n]
            print(
                f"{n}  {i['shape'][0]:>4}x{i['shape'][1]:<4} "
                f"npz {i['npz_kb']:>7.1f} kB  png {i['png_kb']:>7.1f} kB  "
                f"range [{i['min']:.3f}, {i['max']:.3f}]  {DESCRIPTIONS[n]}"
            )
    (d / "charts.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="synthetic charts C1..C7")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build charts into $W/charts/")
    b.add_argument("--out", default=None, help="output directory")
    b.add_argument("--only", default=None, help="comma list, e.g. C1,C7")
    sub.add_parser("info", help="list chart names and descriptions")
    args = ap.parse_args(argv)

    if args.cmd == "info":
        for n in CHART_NAMES:
            p = chart_path(n)
            state = "built" if p.exists() else "-"
            print(f"{n}  {state:<5} {DESCRIPTIONS[n]}")
        return 0

    only = args.only.split(",") if args.only else None
    build_all(Path(args.out) if args.out else None, only=only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
