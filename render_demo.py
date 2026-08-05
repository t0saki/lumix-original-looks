#!/usr/bin/env python3
"""Render demo comparison images for the LUMIX Original Looks suite.

The demo .RW2 files are not part of the repository; point --raw-dir at your
own LUMIX raws.

Per RW2 demo file:
  1. Develop the RAW with LibRaw (rawpy) to a neutral display-referred sRGB
     base.  NOTE: this is a *LibRaw neutral development approximating* the
     LUMIX in-camera "Standard" photo style -- it is NOT identical to the
     in-camera Standard engine (different demosaic, tone and color rendering).
     The demo RW2 files were shot with an in-camera LUT active, so their
     embedded JPEGs are unusable as a base; the base must be developed here.
  2. Resize to a long edge of 2000 px (Lanczos, done in float to avoid an
     extra 8-bit quantisation before the LUT).
  3. Apply each 33-point .cube look at 100% via vectorized tetrahedral
     interpolation and write JPEGs.

Also builds a labeled contact grid per scene, a larger "hero" figure for one
scene, and blog-optimized web copies.

Exposure choice (measured, see --no-auto-bright / --auto-bright-thr):
LibRaw auto-brighten is left ENABLED (no_auto_bright=False) with the LibRaw
default threshold.  With it disabled the night scene and the hazy telephoto
cityscape develop visibly under-exposed and flat; enabled, all four scenes land
where a normal correctly-exposed camera JPEG would.  Tightening the threshold
below ~0.005 makes LibRaw refuse to brighten the night frame at all (its neon
highlights already exceed the tighter clip budget), so the default is kept.

Everything is deterministic: fixed look order, fixed resampling filter, fixed
JPEG quality, no per-scene hand tuning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from looks.cubeio import read_lut, tetrahedral_interpolation  # noqa: E402

# Fixed presentation order of the suite (also the grid order).
LOOK_ORDER = [
    "Fieldnote",
    "Heartland",
    "Meridian",
    "Matinee",
    "Postcard",
    "Skylight",
    "NightMarket",
    "DuskTide",
    "Canopy",
    "Lowsun",
]

BASE_LABEL = "Base (neutral RAW development)"

# Hero scene: late-afternoon street frame -- skin tones, saturated blues and
# greens, red awnings, sunlit grass and open sky, so warmth / density / hue
# differences all read at once.  Looks: one neutral, one warm, one dense, one
# scene specialist (golden hour).
HERO_SCENE_DEFAULT = "P1035574"
HERO_LOOKS_DEFAULT = ["Meridian", "Heartland", "Matinee", "Lowsun"]

# Measured mean dE00 vs identity over a photographic color cloud, for captions.
LOOK_STRENGTH = {
    "Meridian": 3.45,
    "Skylight": 3.48,
    "Lowsun": 3.81,
    "Postcard": 4.00,
    "Canopy": 4.30,
    "Matinee": 4.47,
    "Fieldnote": 5.15,
    "NightMarket": 5.68,
    "Heartland": 5.80,
    "DuskTide": 6.45,
}

# Reference LUTs in common use, same measurement, for the strength chart.
REFERENCE_STRENGTH = {
    "RealaAce": 5.04,
    "FujiClassicNeg-CN": 8.65,
    "LeicaNatural": 8.77,
    "Kodak2383": 10.74,
}

BG = (16, 16, 16)
STRIP = (32, 32, 32)
TEXT = (238, 238, 238)
SUBTEXT = (168, 168, 168)

# Chart colors: categorical slots 1 and 2 of the validated default data-viz
# palette (checked with the palette validator: adjacent CVD ΔE 24.7, normal
# vision ΔE 33.6, both >= 3:1 contrast on a white surface).
CHART_OURS = "#2a78d6"
CHART_REF = "#eb6834"
CHART_INK = "#0b0b0b"
CHART_INK_SOFT = "#52514e"
CHART_GRID = "#e4e4e1"

FONT_CANDIDATES_BOLD = [
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 2),
    ("/System/Library/Fonts/Helvetica.ttc", 1),
]
FONT_CANDIDATES_REGULAR = [
    ("/System/Library/Fonts/Helvetica.ttc", 0),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
]


# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #
def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR
    for path, index in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=index)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def fit_font(text: str, max_width: int, start_size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Largest font (<= start_size) whose rendering of `text` fits max_width."""
    size = max(int(start_size), 9)
    while size > 9:
        font = load_font(size, bold)
        if font.getlength(text) <= max_width:
            return font
        size -= 1
    return load_font(9, bold)


# --------------------------------------------------------------------------- #
# raw development / image maths
# --------------------------------------------------------------------------- #
def develop_raw(path: Path, no_auto_bright: bool, auto_bright_thr: float | None, bps: int = 16) -> np.ndarray:
    """LibRaw neutral development -> float32 display-referred sRGB in [0, 1].

    Approximates (does NOT reproduce) the in-camera LUMIX Standard photo style.
    Orientation: rawpy/LibRaw applies the camera orientation flag by default
    (user_flip=-1), so portrait frames come out upright.
    """
    kwargs = dict(
        use_camera_wb=True,
        output_color=rawpy.ColorSpace.sRGB,
        gamma=(2.4, 12.92),  # the sRGB transfer curve
        output_bps=bps,
        no_auto_bright=no_auto_bright,
    )
    if auto_bright_thr is not None:
        kwargs["auto_bright_thr"] = auto_bright_thr
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(**kwargs)
    scale = 65535.0 if bps == 16 else 255.0
    return (rgb.astype(np.float32) / scale).clip(0.0, 1.0)


def resize_float(img: np.ndarray, long_edge: int) -> np.ndarray:
    """Lanczos resize in float (per channel) so no 8-bit step precedes the LUT."""
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    scale = long_edge / float(max(h, w))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    chans = [
        np.asarray(
            Image.fromarray(np.ascontiguousarray(img[..., c], dtype=np.float32), mode="F").resize(
                (nw, nh), Image.LANCZOS
            ),
            dtype=np.float32,
        )
        for c in range(img.shape[2])
    ]
    return np.clip(np.stack(chans, axis=-1), 0.0, 1.0)


def apply_look(lut, img: np.ndarray) -> np.ndarray:
    out = tetrahedral_interpolation(lut, img.reshape(-1, 3).astype(np.float64))
    return np.clip(out.reshape(img.shape), 0.0, 1.0).astype(np.float32)


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def stats(arr: np.ndarray) -> dict[str, float]:
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return {
        "mean": float(lum.mean()),
        "p99": float(np.percentile(lum, 99)),
        "clip_pct": float((arr >= 254.0 / 255.0).mean() * 100.0),
        "black_pct": float((lum <= 1.0 / 255.0).mean() * 100.0),
    }


# --------------------------------------------------------------------------- #
# layout helpers
# --------------------------------------------------------------------------- #
def thumb(img: Image.Image, long_edge: int) -> Image.Image:
    out = img.copy()
    out.thumbnail((long_edge, long_edge), Image.LANCZOS)  # aspect preserved, no crop
    return out


def make_tile(img: Image.Image, label: str, sublabel: str = "") -> Image.Image:
    """Tile = image with a dark label strip underneath (white text)."""
    w, h = img.size
    base_size = max(11, int(round(w / 21.0)))  # font size proportional to tile width
    font = fit_font(label, w - 16, base_size, bold=True)
    sub_font = load_font(max(9, int(round(base_size * 0.78))), bold=False) if sublabel else None
    line_h = int(round(base_size * 1.32))
    strip_h = line_h + (int(round(base_size * 1.02)) if sublabel else 0) + int(round(base_size * 0.5))

    canvas = Image.new("RGB", (w, h + strip_h), STRIP)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = h + int(round(base_size * 0.22))
    draw.text((w / 2.0, y), label, font=font, fill=TEXT, anchor="ma")
    if sublabel and sub_font is not None:
        draw.text((w / 2.0, y + line_h), sublabel, font=sub_font, fill=SUBTEXT, anchor="ma")
    return canvas


def assemble_grid(tiles: list[Image.Image], cols: int, title: str, subtitle: str = "", pad: int = 12) -> Image.Image:
    tw = max(t.size[0] for t in tiles)
    th = max(t.size[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols

    title_size = max(16, int(round(tw / 15.0)))
    sub_size = max(12, int(round(title_size * 0.62)))
    header_h = int(round(title_size * 1.5)) + (int(round(sub_size * 1.9)) if subtitle else 0) + pad

    width = cols * tw + (cols + 1) * pad
    height = header_h + rows * th + (rows + 1) * pad - pad
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    tfont = fit_font(title, width - 2 * pad - 8, title_size, bold=True)
    draw.text((pad + 2, pad), title, font=tfont, fill=TEXT)
    if subtitle:
        sfont = fit_font(subtitle, width - 2 * pad - 8, sub_size, bold=False)
        draw.text((pad + 2, pad + int(round(title_size * 1.45))), subtitle, font=sfont, fill=SUBTEXT)

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = header_h + r * (th + pad)
        canvas.paste(tile, (x, y))
    return canvas


def caption_panel(size: tuple[int, int], lines: list[tuple[str, bool]]) -> Image.Image:
    """Text panel used to fill the spare cell of the hero figure."""
    w, h = size
    panel = Image.new("RGB", (w, h), STRIP)
    draw = ImageDraw.Draw(panel)
    base_size = max(13, int(round(w / 26.0)))
    x = int(round(w * 0.07))
    y = int(round(h * 0.10))
    for text, strong in lines:
        if not text:
            y += int(round(base_size * 0.85))
            continue
        font = fit_font(text, w - 2 * x, base_size if strong else int(round(base_size * 0.9)), bold=strong)
        draw.text((x, y), text, font=font, fill=TEXT if strong else SUBTEXT)
        y += int(round((base_size if strong else base_size * 0.9) * 1.62))
    return panel


def save_jpeg(img: Image.Image, path: Path, quality: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=quality, subsampling=0 if quality >= 90 else 2, optimize=True)
    return path


def web_copy(src: Path, dst_dir: Path, long_edge: int, quality: int, max_bytes: int) -> Path:
    """Blog-optimized copy; JPEG quality steps down until it fits the budget."""
    img = thumb(Image.open(src).convert("RGB"), long_edge)
    dst = dst_dir / src.name
    if src.suffix.lower() == ".png":
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "PNG", optimize=True)
        if dst.stat().st_size > max_bytes:  # flat chart art quantizes losslessly enough
            img.convert("P", palette=Image.ADAPTIVE, colors=256).save(dst, "PNG", optimize=True)
        return dst
    q = quality
    while True:
        save_jpeg(img, dst, q)
        if dst.stat().st_size <= max_bytes or q <= 60:
            return dst
        q -= 4


# --------------------------------------------------------------------------- #
# strength chart
# --------------------------------------------------------------------------- #
def build_strength_chart(path: Path, width_px: int = 1600, dpi: int = 200) -> Path:
    """Horizontal bar chart of measured look strength (mean dE00 vs identity)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch

    rows = [(n, v, False) for n, v in LOOK_STRENGTH.items()]
    rows += [(n, v, True) for n, v in REFERENCE_STRENGTH.items()]
    rows.sort(key=lambda r: (r[1], r[0]))  # ascending, deterministic ties
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    is_ref = [r[2] for r in rows]

    fig_w = width_px / dpi
    fig_h = 0.40 * len(rows) + 1.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.175, right=0.985, top=0.828, bottom=0.095)

    x_max = max(values) * 1.13
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(len(rows) - 0.5, -0.5)  # ascending downward
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9.5, color=CHART_INK)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=0, colors=CHART_INK_SOFT, labelsize=9)
    ax.xaxis.grid(True, color=CHART_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("mean ΔE00 (CIEDE2000)", fontsize=9.5, color=CHART_INK_SOFT, labelpad=7)

    # Rounded 4 px data-ends: convert the radius from pixels to data units and
    # correct the y axis with mutation_aspect so the corners stay circular.
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    x_per_px = x_max / bbox.width
    y_per_px = len(rows) / bbox.height
    radius = 4.0 * x_per_px
    aspect = y_per_px / x_per_px
    bar_h = 0.62

    for i, (value, ref) in enumerate(zip(values, is_ref)):
        ax.add_patch(
            FancyBboxPatch(
                (0.0, i - bar_h / 2.0),
                max(value, radius * 2.2),
                bar_h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                mutation_aspect=aspect,
                linewidth=0,
                facecolor=CHART_REF if ref else CHART_OURS,
            )
        )
        ax.text(
            value + 6.0 * x_per_px,
            i,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color=CHART_INK_SOFT,
        )

    fig.text(0.012, 0.968, "Look strength: mean ΔE00 vs identity (photographic color cloud)",
             fontsize=13.5, fontweight="bold", color=CHART_INK, va="top")
    fig.text(0.012, 0.918,
             "Higher = further from the untouched image.\n"
             "33-point .cube looks, display sRGB in/out, measured over the same color cloud.",
             fontsize=9.5, color=CHART_INK_SOFT, va="top", linespacing=1.55)
    ax.legend(
        handles=[
            Patch(facecolor=CHART_OURS, label="LUMIX Original Looks (this suite)"),
            Patch(facecolor=CHART_REF, label="Reference LUTs in common use"),
        ],
        loc="upper right",  # top rows hold the shortest bars, so this corner is free
        frameon=False,
        fontsize=9.5,
        labelcolor=CHART_INK_SOFT,
        handlelength=1.1,
        handleheight=1.1,
        borderaxespad=1.2,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Render demo comparison images for the LUMIX Original Looks suite.")
    ap.add_argument("--raw-dir", type=Path, required=True, help="directory of .RW2 demo files (not part of the repo)")
    ap.add_argument("--luts", type=Path, default=script_dir / "luts", help="directory of .cube looks")
    ap.add_argument(
        "--output",
        type=Path,
        default=script_dir / "previews/demo",
        help="output directory (web/ subfolder is created inside)",
    )
    ap.add_argument("--long-edge", type=int, default=2000, help="long edge of the full-size renders")
    ap.add_argument("--tile-long-edge", type=int, default=640, help="long edge of contact-grid tiles")
    ap.add_argument("--hero-long-edge", type=int, default=940, help="long edge of hero figure tiles")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality for full-size renders")
    ap.add_argument("--web-long-edge", type=int, default=1600)
    ap.add_argument("--web-quality", type=int, default=82)
    ap.add_argument("--web-max-kb", type=int, default=400, help="size budget per web copy")
    ap.add_argument("--chart", type=Path, default=None, help="strength chart path (default: <output>/../strength_chart.png)")
    ap.add_argument("--chart-width", type=int, default=1600)
    ap.add_argument("--skip-chart", action="store_true")
    ap.add_argument("--hero-scene", default=HERO_SCENE_DEFAULT, help="stem of the scene used for hero_*.jpg")
    ap.add_argument("--hero-looks", nargs="+", default=HERO_LOOKS_DEFAULT, help="4 look names for the hero figure")
    ap.add_argument(
        "--no-auto-bright",
        action="store_true",
        help="disable LibRaw auto-brighten (default: enabled, see module docstring)",
    )
    ap.add_argument("--auto-bright-thr", type=float, default=None, help="LibRaw auto-bright clip budget (default: LibRaw default)")
    ap.add_argument("--bps", type=int, default=16, choices=(8, 16), help="LibRaw output bit depth")
    ap.add_argument("--grid-cols-landscape", type=int, default=4)
    ap.add_argument("--grid-cols-portrait", type=int, default=6)
    args = ap.parse_args()

    out_dir: Path = args.output
    web_dir = out_dir / "web"
    out_dir.mkdir(parents=True, exist_ok=True)
    web_dir.mkdir(parents=True, exist_ok=True)

    luts = {}
    for name in LOOK_ORDER:
        path = args.luts / f"{name}.cube"
        if not path.exists():
            print(f"ERROR missing look: {path}", file=sys.stderr)
            return 1
        luts[name] = read_lut(path)
    print(f"loaded {len(luts)} looks: {', '.join(luts)}", flush=True)

    raws = sorted(args.raw_dir.glob("*.RW2"))
    if not raws:
        print(f"ERROR no .RW2 found in {args.raw_dir}", file=sys.stderr)
        return 1

    written: list[Path] = []
    web_sources: list[Path] = []

    for raw_path in raws:
        stem = raw_path.stem
        full = develop_raw(raw_path, args.no_auto_bright, args.auto_bright_thr, args.bps)
        base = resize_float(full, args.long_edge)
        del full
        bh, bw = base.shape[:2]
        orient = "portrait" if bh > bw else "landscape"
        s = stats(base)
        print(
            f"\n{stem}: {bw}x{bh} {orient}  base mean={s['mean']:.3f} p99={s['p99']:.3f} "
            f"clip={s['clip_pct']:.3f}% black={s['black_pct']:.3f}%",
            flush=True,
        )

        base_pil = to_pil(base)
        base_file = save_jpeg(base_pil, out_dir / f"base_{stem}.jpg", args.quality)
        written.append(base_file)
        web_sources.append(base_file)

        tiles = [make_tile(thumb(base_pil, args.tile_long_edge), BASE_LABEL)]
        hero_cells: dict[str, Image.Image] = {}
        if stem == args.hero_scene:
            hero_cells[BASE_LABEL] = thumb(base_pil, args.hero_long_edge)

        for name, lut in luts.items():
            look = apply_look(lut, base)
            ls = stats(look)
            look_pil = to_pil(look)
            look_file = save_jpeg(look_pil, out_dir / f"look_{stem}_{name}.jpg", args.quality)
            written.append(look_file)
            delta = float(np.abs(look.astype(np.float64) - base.astype(np.float64)).mean() * 255.0)
            flag = ""
            if ls["clip_pct"] > s["clip_pct"] + 1.0:
                flag += "  [!] highlight clipping grew"
            if ls["black_pct"] > s["black_pct"] + 1.0:
                flag += "  [!] shadow crush grew"
            print(
                f"  {name:<12} mean={ls['mean']:.3f} p99={ls['p99']:.3f} clip={ls['clip_pct']:.3f}% "
                f"black={ls['black_pct']:.3f}% meanΔ8bit={delta:.2f}{flag}",
                flush=True,
            )
            tiles.append(make_tile(thumb(look_pil, args.tile_long_edge), name))
            if stem == args.hero_scene:
                if name in args.hero_looks:
                    hero_cells[name] = thumb(look_pil, args.hero_long_edge)
                web_sources.append(look_file)
            del look, look_pil

        cols = args.grid_cols_portrait if orient == "portrait" else args.grid_cols_landscape
        grid = assemble_grid(
            tiles,
            cols,
            f"{stem} — LUMIX Original Looks",
            "Neutral RAW development (base) and all 10 looks applied at 100%",
        )
        grid_file = save_jpeg(grid, out_dir / f"grid_{stem}.jpg", args.quality)
        written.append(grid_file)
        web_sources.append(grid_file)
        print(f"  grid {grid.size[0]}x{grid.size[1]} ({cols} cols) -> {grid_file.name}", flush=True)

        if hero_cells:
            missing = [n for n in args.hero_looks if n not in hero_cells]
            if missing:
                print(f"WARN hero looks not found: {missing}", file=sys.stderr)
            hero_tiles = [make_tile(hero_cells[BASE_LABEL], BASE_LABEL, f"{stem} — LUMIX S9, RW2")]
            for name in args.hero_looks:
                if name in hero_cells:
                    strength = LOOK_STRENGTH.get(name)
                    sub = f"100%  ·  mean ΔE00 {strength:.2f}" if strength else "100%"
                    hero_tiles.append(make_tile(hero_cells[name], name, sub))
            cell_size = hero_tiles[0].size
            if len(hero_tiles) % 2 == 1:
                hero_tiles.append(
                    caption_panel(
                        cell_size,
                        [
                            ("LUMIX Original Looks", True),
                            ("Base: LibRaw neutral development of the RW2", False),
                            ("(approximates, does not reproduce, the", False),
                            ("in-camera Standard photo style).", False),
                            ("", False),
                            ("Each look applied at 100% in display sRGB.", False),
                            ("", False),
                            ("  ·  ".join(args.hero_looks[:2]), True),
                            ("  ·  ".join(args.hero_looks[2:]), True),
                            ("", False),
                            ("Full suite: see grid_" + stem + ".jpg", False),
                        ],
                    )
                )
            hero = assemble_grid(
                hero_tiles,
                2,
                f"{stem} — base vs four contrasting looks",
                "LUMIX Original Looks, each applied at 100% to the neutral RAW development",
            )
            hero_file = save_jpeg(hero, out_dir / f"hero_{stem}.jpg", args.quality)
            written.append(hero_file)
            web_sources.append(hero_file)
            print(f"  hero {hero.size[0]}x{hero.size[1]} -> {hero_file.name}", flush=True)

        del base, base_pil, tiles, hero_cells

    if not args.skip_chart:
        chart_path = args.chart or (out_dir.parent / "strength_chart.png")
        chart = build_strength_chart(chart_path, args.chart_width)
        written.append(chart)
        web_sources.append(chart)
        print(f"\nstrength chart -> {chart} ({chart.stat().st_size / 1024:.0f} KB)", flush=True)

    max_bytes = args.web_max_kb * 1024
    print("\nweb copies:", flush=True)
    for src in web_sources:
        dst = web_copy(src, web_dir, args.web_long_edge, args.web_quality, max_bytes)
        written.append(dst)
        print(f"  {dst.name}  {dst.stat().st_size / 1024:.0f} KB", flush=True)

    oversize = [p for p in written if p.parent == web_dir and p.stat().st_size > max_bytes]
    if oversize:
        print("WARN web copies over 400 KB: " + ", ".join(p.name for p in oversize), file=sys.stderr)

    print(f"\ndone: {len(written)} files in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
