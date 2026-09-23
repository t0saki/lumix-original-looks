"""Review sheets for vision models (T4) — strict, asserted image budget.

A judgement image that exceeds the viewer's budget is silently downsampled
before any model sees it, so the tiles arrive smaller than intended.  Every
saving path in this module therefore *asserts* the limits instead of trusting
the caller:

* saved image  <= 1,150,000 px total and <= 1536 px on the long edge
* tile long edge <= SOLO 980 / PAIR 660 / TRIAD 480 / QUAD 460 / SIX 360 px
* never more than 6 tiles
* JPEG quality 95, ``subsampling=0`` (4:4:4), sRGB ICC profile embedded
* labels burnt into a strip under each tile (ASCII)
* a ``<sheet>.json`` sidecar listing every tile (base, look, strength, crop)

The requested tile size is an upper bound: when the tiles' aspect ratio would
push the canvas past the budget, the layout shrinks the tiles (never the other
way round) and records the size it actually used in the sidecar.

Sheet kinds: ``solo`` (1), ``pair`` (2), ``triad`` (3), ``quad`` (4),
``six`` (6), plus ``chart`` (three full-width rows: original / through LUT /
x8 amplified difference).

CLI::

    py tools/sheets.py solo  --base PANA9997 --look A.cube --out sheet.jpg
    py tools/sheets.py pair  --base PANA9997 --a base --b A.cube [--patches face]
    py tools/sheets.py triad --base B --a base --b A.cube --c B.cube --out ...
    py tools/sheets.py quad  --base B --look A.cube --look B.cube --look C.cube
    py tools/sheets.py six   --base B --look ... (or --bases id,id,... --look L)
    py tools/sheets.py chart --chart C1 --look A.cube --out ...
    py tools/sheets.py strip --base B --look A.cube --look B.cube ... --out-dir D
    py tools/sheets.py selftest [--out-dir $W/review]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont

from tools import render
from tools.render import WORK

__all__ = [
    "MAX_PIXELS",
    "MAX_EDGE",
    "MAX_TILES",
    "TILE_LONG",
    "Tile",
    "PatchStrip",
    "sheet",
    "solo",
    "pair",
    "triad",
    "quad",
    "six",
    "chart_sheet",
    "patch_strip",
    "look_strip",
    "selftest",
]

# --------------------------------------------------------------------------- #
# hard limits (assertions, not conventions)
# --------------------------------------------------------------------------- #
if not __debug__:  # the budget is enforced with assert; -O would silence it
    raise RuntimeError("tools.sheets must not run under python -O: its hard limits are assertions")

MAX_PIXELS = 1_150_000
MAX_EDGE = 1536
MAX_TILES = 6

TILE_LONG = {"solo": 980, "pair": 660, "triad": 480, "quad": 460, "six": 360}
TILE_COUNT = {"solo": 1, "pair": 2, "triad": 3, "quad": 4, "six": 6}
TILE_COLS = {"solo": 1, "pair": 2, "triad": 3, "quad": 2, "six": 3}
CHART_LONG = 1400  # chart rows are full width, not square tiles

# --------------------------------------------------------------------------- #
# style (R05 section 2.2)
# --------------------------------------------------------------------------- #
BG = (28, 28, 30)
LABEL_BG = (46, 46, 50)
INK = (238, 238, 236)
MUTED = (150, 150, 148)
PAD = 22
GUTTER = 24
LABEL_H = 46
HEADER_H = 64
STRIP_H = 124

FONT_TITLE = 30
FONT_LABEL = 26
FONT_NUM = 20

JPEG_KW = dict(quality=95, subsampling=0, optimize=True)

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}
_ICC: bytes | None = None


def _font(size: int) -> ImageFont.FreeTypeFont:
    size = int(size)
    if size not in _font_cache:
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                try:
                    _font_cache[size] = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
        else:  # pragma: no cover - PIL always ships a default
            _font_cache[size] = ImageFont.load_default(size=size)
    return _font_cache[size]


def _fit_font(text: str, width: int, size: int, minimum: int = 12) -> ImageFont.FreeTypeFont:
    """Largest font <= *size* whose rendering of *text* fits in *width* px."""
    while size > minimum:
        font = _font(size)
        if font.getbbox(text)[2] <= width:
            return font
        size -= 1
    return _font(max(minimum, size))


def _srgb_icc() -> bytes:
    global _ICC
    if _ICC is None:
        _ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return _ICC


def _ascii(text: str) -> str:
    if not str(text).isascii():
        raise ValueError(f"labels must be ASCII (burnt-in text): {text!r}")
    return str(text)


# --------------------------------------------------------------------------- #
# tiles
# --------------------------------------------------------------------------- #
@dataclass
class Tile:
    """One picture in a sheet, plus the provenance that lands in the sidecar."""

    image: np.ndarray
    label: str
    base: str | None = None
    look: str | None = None
    strength: float | None = None
    crop: str | None = None
    source: str | None = None
    extra: dict = field(default_factory=dict)

    def meta(self, tile_px: tuple[int, int]) -> dict:
        info = {
            "label": self.label,
            "base": self.base,
            "look": self.look,
            "strength": self.strength,
            "crop": self.crop,
            "source": self.source,
            "tile_px": [int(tile_px[0]), int(tile_px[1])],
        }
        info.update(self.extra)
        return info


def _as_tile(item: Tile | tuple | np.ndarray) -> Tile:
    if isinstance(item, Tile):
        return item
    if isinstance(item, tuple) and len(item) == 2:
        return Tile(image=np.asarray(item[0]), label=str(item[1]))
    return Tile(image=np.asarray(item), label="")


# --------------------------------------------------------------------------- #
# patch strip
# --------------------------------------------------------------------------- #
def _norm_boxes(boxes) -> list[tuple[str, tuple[float, float, float, float] | None]]:
    """Accept ``{name: box}``, ``[(name, box), ...]``, ``[box, ...]`` or ``[dict]``."""
    if isinstance(boxes, dict):
        return [(str(k), tuple(v) if v is not None else None) for k, v in boxes.items()]
    out = []
    for i, item in enumerate(boxes):
        if isinstance(item, dict):
            name = str(item.get("name", f"patch{i + 1}"))
            box = item.get("box", item.get("rect"))
            out.append((name, tuple(box) if box is not None else None))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            out.append((str(item[0]), tuple(item[1]) if item[1] is not None else None))
        else:
            out.append((f"patch{i + 1}", tuple(item)))
    return out


@dataclass
class PatchStrip:
    """Measured swatch pairs, rendered at whatever width the sheet needs."""

    measurements: list[dict]
    height: int = STRIP_H

    def render(self, width: int) -> Image.Image:
        n = max(1, len(self.measurements))
        img = Image.new("RGB", (int(width), int(self.height)), LABEL_BG)
        draw = ImageDraw.Draw(img)
        cell_w = int(width) // n
        sw = min(66, max(40, cell_w // 6), self.height - 56)
        for i, m in enumerate(self.measurements):
            x0 = i * cell_w + 12
            if i:
                draw.line([(i * cell_w, 6), (i * cell_w, self.height - 6)], fill=(80, 80, 84), width=1)
            name = _ascii(m["name"])
            draw.text((x0, 8), name, font=_fit_font(name, cell_w - 24, 20), fill=INK, anchor="la")
            top = 34
            b_rgb = tuple(int(round(255 * v)) for v in np.clip(m["before"]["rgb"], 0, 1))
            a_rgb = tuple(int(round(255 * v)) for v in np.clip(m["after"]["rgb"], 0, 1))
            draw.rectangle([x0, top, x0 + sw, top + sw], fill=b_rgb, outline=(90, 90, 94))
            draw.rectangle([x0 + sw + 8, top, x0 + 2 * sw + 8, top + sw], fill=a_rgb, outline=(90, 90, 94))
            draw.text((x0, top + sw + 4), "before", font=_font(15), fill=MUTED, anchor="la")
            draw.text((x0 + sw + 8, top + sw + 4), "after", font=_font(15), fill=MUTED, anchor="la")
            tx = x0 + 2 * sw + 22
            avail = cell_w - (tx - i * cell_w) - 12
            cr = m["cr"]
            line1 = f"dL {m['dL']:+.3f}  C x{cr:.2f}" if np.isfinite(cr) else f"dL {m['dL']:+.3f}"
            line2 = f"dh {m['dh']:+.1f} deg" if np.isfinite(m["dh"]) else f"dC {m['dC']:+.4f} (neutral)"
            line3 = f"L {m['before']['L']:.3f}>{m['after']['L']:.3f}"
            f_num = _fit_font(line1, avail, FONT_NUM, minimum=13)
            draw.text((tx, 34), line1, font=f_num, fill=INK, anchor="la")
            draw.text((tx, 34 + 26), line2, font=f_num, fill=INK, anchor="la")
            draw.text((tx, 34 + 52), line3, font=_fit_font(line3, avail, 17, 12), fill=MUTED, anchor="la")
        return img


def patch_strip(img_before: np.ndarray, img_after: np.ndarray, boxes) -> PatchStrip:
    """Measure OKLab means in *boxes* before/after a LUT and draw them.

    *boxes* may be ``{name: (x0, y0, x1, y1)}``, a list of ``(name, box)``, a
    list of boxes, or a list of ``{"name":…, "box":…}`` dicts.  Boxes whose
    values are all <= 1 are normalised fractions of the image.  The result is
    attachable under any sheet via ``pair(..., strip=…)``.
    """
    items = _norm_boxes(boxes)
    if not 1 <= len(items) <= 4:
        raise ValueError(f"a patch strip takes 1..4 boxes, got {len(items)}")
    measurements = []
    for name, box in items:
        m = render.patch_measure(img_before, img_after, box)
        m["name"] = _ascii(name)
        m["box"] = list(box) if box is not None else None
        measurements.append(m)
    return PatchStrip(measurements=measurements)


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def _layout(shapes: list[tuple[int, int]], cols: int, has_header: bool, strip_h: int) -> tuple[int, int, int, int]:
    cell_h = max(s[0] for s in shapes)
    cell_w = max(s[1] for s in shapes)
    rows = math.ceil(len(shapes) / cols)
    canvas_w = 2 * PAD + cols * cell_w + (cols - 1) * GUTTER
    canvas_h = (
        PAD
        + (HEADER_H if has_header else 0)
        + rows * (cell_h + LABEL_H)
        + (rows - 1) * GUTTER
        + (strip_h + GUTTER if strip_h else 0)
        + PAD
    )
    return cell_w, cell_h, canvas_w, canvas_h


def _fit_long_edge(images: list[np.ndarray], long_edge: int, cols: int, has_header: bool, strip_h: int) -> int:
    """Largest tile long edge <= *long_edge* whose canvas obeys the budget."""
    edge = int(long_edge)
    while edge >= 120:
        shapes = []
        for img in images:
            h, w = img.shape[0], img.shape[1]
            scale = edge / float(max(h, w))
            shapes.append((max(1, int(round(h * scale))), max(1, int(round(w * scale)))))
        _, _, cw, ch = _layout(shapes, cols, has_header, strip_h)
        if cw * ch <= MAX_PIXELS and max(cw, ch) <= MAX_EDGE:
            return edge
        edge -= 4
    raise ValueError("cannot fit these tiles inside the image budget")


def _draw_label(canvas: Image.Image, x: int, y: int, w: int, text: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x, y, x + w - 1, y + LABEL_H - 1], fill=LABEL_BG)
    text = _ascii(text)
    if text:
        font = _fit_font(text, w - 20, FONT_LABEL, minimum=13)
        draw.text((x + 10, y + LABEL_H // 2), text, font=font, fill=INK, anchor="lm")


def compose(
    tiles: Sequence[Tile | tuple | np.ndarray],
    *,
    kind: str,
    title: str | None = None,
    subtitle: str | None = None,
    strip: PatchStrip | Image.Image | None = None,
    cols: int | None = None,
    tile_long: int | None = None,
    allow_partial: bool = False,
) -> tuple[Image.Image, dict]:
    """Assemble a sheet canvas plus its sidecar metadata (does not save).

    *allow_partial* permits fewer tiles than the kind nominally holds (a short
    last page of a look strip) — the kind's tile-size limit still applies.
    """
    items = [_as_tile(t) for t in tiles]
    if not items:
        raise ValueError("a sheet needs at least one tile")
    if len(items) > MAX_TILES:
        raise AssertionError(f"never more than {MAX_TILES} tiles per sheet, got {len(items)}")
    if kind in TILE_COUNT and len(items) != TILE_COUNT[kind] and not allow_partial:
        raise ValueError(f"{kind} takes exactly {TILE_COUNT[kind]} tiles, got {len(items)}")
    if kind in TILE_COUNT and len(items) > TILE_COUNT[kind]:
        raise ValueError(f"{kind} holds at most {TILE_COUNT[kind]} tiles, got {len(items)}")

    cols = cols if cols is not None else TILE_COLS.get(kind, len(items))
    limit = tile_long if tile_long is not None else TILE_LONG.get(kind, CHART_LONG)
    has_header = bool(title)
    strip_h = 0
    if strip is not None:
        strip_h = strip.height if isinstance(strip, PatchStrip) else strip.size[1]

    images = [np.asarray(t.image, dtype=np.float64) for t in items]
    edge = _fit_long_edge(images, limit, cols, has_header, strip_h)
    scaled = [render.resize_long_edge(img, edge) for img in images]
    shapes = [(s.shape[0], s.shape[1]) for s in scaled]
    cell_w, cell_h, canvas_w, canvas_h = _layout(shapes, cols, has_header, strip_h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)
    top = PAD
    if has_header:
        t_text = _ascii(title or "")
        draw.text((PAD, PAD + 2), t_text, font=_fit_font(t_text, canvas_w - 2 * PAD, FONT_TITLE), fill=INK)
        if subtitle:
            s_text = _ascii(subtitle)
            draw.text(
                (PAD, PAD + FONT_TITLE + 8),
                s_text,
                font=_fit_font(s_text, canvas_w - 2 * PAD, FONT_NUM),
                fill=MUTED,
            )
        top += HEADER_H

    tile_meta = []
    for i, (tile, img) in enumerate(zip(items, scaled)):
        row, col = divmod(i, cols)
        cx = PAD + col * (cell_w + GUTTER)
        cy = top + row * (cell_h + LABEL_H + GUTTER)
        h, w = img.shape[0], img.shape[1]
        ox = cx + (cell_w - w) // 2
        oy = cy + (cell_h - h) // 2
        canvas.paste(render.to_pil(img), (ox, oy))
        _draw_label(canvas, cx, cy + cell_h, cell_w, tile.label)
        tile_meta.append(tile.meta((w, h)))

    if strip is not None:
        strip_img = strip.render(canvas_w - 2 * PAD) if isinstance(strip, PatchStrip) else strip
        canvas.paste(strip_img, (PAD, canvas_h - PAD - strip_h))

    meta = {
        "kind": kind,
        "title": title,
        "subtitle": subtitle,
        "canvas_px": [canvas_w, canvas_h],
        "megapixels": round(canvas_w * canvas_h / 1e6, 4),
        "tile_long_edge": edge,
        "tile_long_edge_limit": limit,
        "cols": cols,
        "tiles": tile_meta,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if strip is not None and isinstance(strip, PatchStrip):
        meta["patch_strip"] = [
            {
                "name": m["name"],
                "box": m["box"],
                "before": {k: m["before"][k] for k in ("L", "a", "b", "C", "h")},
                "after": {k: m["after"][k] for k in ("L", "a", "b", "C", "h")},
                "dL": m["dL"],
                "dC": m["dC"],
                "cr": m["cr"],
                "dh": m["dh"],
            }
            for m in strip.measurements
        ]
    return canvas, meta


_sha_cache: dict[str, str] = {}


def _source_sha(source: str | None) -> str | None:
    """sha256 of a tile's source .cube, so a sheet can always be traced back."""
    if not source:
        return None
    if source not in _sha_cache:
        p = Path(source)
        if not p.is_file():
            return None
        _sha_cache[source] = render.sha256_file(p)
    return _sha_cache[source]


def save_sheet(canvas: Image.Image, path: str | Path, meta: dict) -> Path:
    """Assert every hard limit, then write the JPEG and its ``.json`` sidecar."""
    path = Path(path)
    w, h = canvas.size
    kind = meta.get("kind", "?")
    n_tiles = len(meta.get("tiles", []))
    assert n_tiles <= MAX_TILES, f"{path.name}: {n_tiles} tiles > {MAX_TILES}"
    assert w * h <= MAX_PIXELS, f"{path.name}: {w}x{h} = {w * h} px > {MAX_PIXELS}"
    assert max(w, h) <= MAX_EDGE, f"{path.name}: long edge {max(w, h)} > {MAX_EDGE}"
    limit = meta.get("tile_long_edge_limit", TILE_LONG.get(kind, CHART_LONG))
    for t in meta.get("tiles", []):
        tw, th = t["tile_px"]
        assert max(tw, th) <= limit, f"{path.name}: tile {max(tw, th)} px > {limit} for kind {kind}"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "JPEG", icc_profile=_srgb_icc(), **JPEG_KW)
    meta = dict(meta)
    meta["tiles"] = [dict(t, sha256=_source_sha(t.get("source"))) for t in meta.get("tiles", [])]
    meta["file"] = str(path)
    meta["bytes"] = path.stat().st_size
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def sheet(tiles, *, kind: str, out: str | Path, **kwargs) -> Path:
    """Compose and save in one call (see :func:`compose`)."""
    canvas, meta = compose(tiles, kind=kind, **kwargs)
    return save_sheet(canvas, out, meta)


# --------------------------------------------------------------------------- #
# sheet kinds
# --------------------------------------------------------------------------- #
def solo(tile, *, out, title=None, subtitle=None, strip=None) -> Path:
    return sheet([tile], kind="solo", out=out, title=title, subtitle=subtitle, strip=strip)


def pair(tile_a, tile_b, *, out, title=None, subtitle=None, strip=None) -> Path:
    return sheet([tile_a, tile_b], kind="pair", out=out, title=title, subtitle=subtitle, strip=strip)


def triad(a, b, c, *, out, title=None, subtitle=None, strip=None) -> Path:
    return sheet([a, b, c], kind="triad", out=out, title=title, subtitle=subtitle, strip=strip)


def quad(tiles, *, out, title=None, subtitle=None, strip=None) -> Path:
    return sheet(list(tiles), kind="quad", out=out, title=title, subtitle=subtitle, strip=strip)


def six(tiles, *, out, title=None, subtitle=None, strip=None) -> Path:
    return sheet(list(tiles), kind="six", out=out, title=title, subtitle=subtitle, strip=strip)


def chart_sheet(
    chart: np.ndarray,
    table,
    label: str,
    *,
    out: str | Path,
    strength: float = 1.0,
    gain: float = 8.0,
    chart_name: str | None = None,
) -> Path:
    """Three rows: original / through LUT / x8 amplified difference.

    *chart* is any float image of sRGB code values (the T3 charts, but a
    synthetic ramp works just as well); *table* is anything
    :func:`tools.render.table_of` accepts.  The difference is computed at the
    chart's native resolution and only then downsampled, so a 1-code twist
    survives the trip to the sheet.
    """
    src = np.asarray(chart, dtype=np.float64)
    if src.ndim != 3 or src.shape[-1] != 3:
        raise ValueError(f"chart must be (H, W, 3) float code values, got {src.shape}")
    out_img = render.apply_table(src, table, strength=strength)
    diff = render.amplified_difference(src, out_img, gain=gain)
    dmax = float(np.abs(out_img - src).max())
    # fraction of the difference row that hit the 0/1 rails at this gain
    clipped = float(np.mean(np.abs(out_img - src) * gain > 0.5))
    name = chart_name or "chart"
    s_pct = int(round(strength * 100))
    tiles = [
        Tile(src, f"original {name}", look="base", crop=name, strength=0.0),
        Tile(out_img, f"through LUT: {_ascii(label)} s{s_pct:03d}", look=label, crop=name, strength=strength),
        Tile(
            diff,
            f"x{gain:g} amplified diff (grey = no change; max {dmax * 255:.1f}/255, "
            f"{clipped * 100:.0f}% of pixels railed)",
            look=label,
            crop=name,
            strength=strength,
            extra={"gain": gain, "max_abs_delta_code8": dmax * 255.0, "railed_fraction": clipped},
        ),
    ]
    canvas, meta = compose(
        tiles,
        kind="chart",
        title=f"{name} through {_ascii(label)} at {s_pct}%",
        subtitle="row 3 = 0.5 + 8 x (look - base), clipped",
        cols=1,
        tile_long=CHART_LONG,
    )
    meta["chart"] = name
    meta["gain"] = gain
    meta["max_abs_delta_code8"] = dmax * 255.0
    meta["railed_fraction"] = clipped
    return save_sheet(canvas, out, meta)


def look_strip(
    base_img: np.ndarray,
    base_id: str,
    looks: Sequence[tuple[str, object]],
    *,
    out_dir: str | Path,
    stem: str | None = None,
    strength: float = 1.0,
    crop_name: str | None = None,
) -> list[Path]:
    """One scene through N looks, split into QUADs anchored on the base tile.

    Every page's first tile is the untouched base, so a judge always has the
    same reference scale.  Returns the page paths.
    """
    out_dir = Path(out_dir)
    stem = stem or f"strip__{base_id}"
    per_page = TILE_COUNT["quad"] - 1  # tile 1 is always the anchor
    pages = math.ceil(len(looks) / per_page) if looks else 0
    if pages == 0:
        raise ValueError("look_strip needs at least one look")
    s_pct = int(round(strength * 100))
    paths = []
    for page in range(pages):
        chunk = list(looks)[page * per_page : (page + 1) * per_page]
        tiles = [
            Tile(base_img, f"ANCHOR base {base_id}", base=base_id, look="base", strength=0.0, crop=crop_name)
        ]
        for name, spec in chunk:
            tiles.append(
                Tile(
                    render.apply_table(base_img, spec, strength=strength),
                    f"{_ascii(name)} s{s_pct:03d}",
                    base=base_id,
                    look=str(name),
                    strength=strength,
                    crop=crop_name,
                    source=str(spec) if isinstance(spec, (str, Path)) else None,
                )
            )
        paths.append(
            sheet(
                tiles,
                kind="quad",
                out=out_dir / f"{stem}__page{page + 1}of{pages}.jpg",
                title=f"{base_id} through {len(looks)} looks - page {page + 1}/{pages}",
                cols=2 if len(tiles) > 2 else len(tiles),
                allow_partial=True,
            )
        )
    return paths


# --------------------------------------------------------------------------- #
# self-test (TOOLS_SPEC T4)
# --------------------------------------------------------------------------- #
REF_CUBE = WORK / "refs" / "reverse" / "Contax_ND_STD_33_sRGB.cube"


def _synthetic_chart(h: int = 380, w: int = 1200) -> np.ndarray:
    """Fallback chart used when tools/charts.py has not produced its npz yet."""
    x = np.linspace(0.0, 1.0, w)
    chart = np.zeros((h, w, 3))
    ramp_h = h // 2
    chart[:ramp_h] = x[None, :, None]
    steps = np.clip(np.floor(x * 21) / 20.0, 0, 1)
    chart[ramp_h:] = steps[None, :, None]
    hue = np.stack(
        [0.5 + 0.45 * np.cos(2 * np.pi * x), 0.5 + 0.45 * np.cos(2 * np.pi * x - 2.094),
         0.5 + 0.45 * np.cos(2 * np.pi * x + 2.094)],
        axis=-1,
    )
    chart[ramp_h - 60 : ramp_h] = hue[None, :, :]
    return chart


def selftest(out_dir: str | Path | None = None, base: str | None = None, verbose: bool = True) -> dict:
    """Build one sheet of every kind and verify the limits programmatically."""
    out_dir = Path(out_dir) if out_dir is not None else WORK / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not REF_CUBE.exists():
        raise FileNotFoundError(f"reference LUT missing: {REF_CUBE}")
    table = render.table_of(REF_CUBE)
    ids = render.base_ids()
    if not ids:
        raise FileNotFoundError("no base .npz found under $W/bases*")
    base_id = base or ("PANA9997" if "PANA9997" in ids else ids[0])
    img = render.load_base(base_id)
    lut = render.apply_table(img, table)
    lut70 = render.apply_table(img, table, strength=0.7)
    look = "Contax_ND_STD"
    results: dict[str, Path] = {}

    results["solo"] = solo(
        Tile(lut, f"{base_id} + {look} s100", base=base_id, look=look, strength=1.0, source=str(REF_CUBE)),
        out=out_dir / "selftest_solo.jpg",
        title=f"SOLO  {base_id}  {look} at 100%",
    )

    boxes = {"upper-left": (0.10, 0.12, 0.26, 0.30), "centre": (0.42, 0.42, 0.58, 0.58),
             "lower-right": (0.70, 0.70, 0.88, 0.90)}
    strip = patch_strip(img, lut, boxes)
    results["pair"] = pair(
        Tile(img, f"base {base_id}", base=base_id, look="base", strength=0.0),
        Tile(lut, f"{look} s100", base=base_id, look=look, strength=1.0, source=str(REF_CUBE)),
        out=out_dir / "selftest_pair.jpg",
        title=f"PAIR  {base_id}  base vs {look}",
        strip=strip,
    )

    results["triad"] = triad(
        Tile(img, "base", base=base_id, look="base", strength=0.0),
        Tile(lut70, f"{look} s070", base=base_id, look=look, strength=0.7),
        Tile(lut, f"{look} s100", base=base_id, look=look, strength=1.0),
        out=out_dir / "selftest_triad.jpg",
        title=f"TRIAD  {base_id}  base / 70% / 100%",
    )

    others = [i for i in ids if i != base_id][:3]
    quad_tiles = [Tile(img, f"ANCHOR base {base_id}", base=base_id, look="base", strength=0.0)]
    for other in others:
        o = render.load_base(other)
        quad_tiles.append(Tile(render.apply_table(o, table), f"{other} + {look}", base=other, look=look, strength=1.0))
    while len(quad_tiles) < 4:
        quad_tiles.append(Tile(lut70, f"{base_id} {look} s070", base=base_id, look=look, strength=0.7))
    results["quad"] = quad(quad_tiles, out=out_dir / "selftest_quad.jpg", title=f"QUAD  {look} across scenes")

    six_tiles = []
    for i in range(6):
        strength = i / 5.0
        six_tiles.append(
            Tile(
                render.apply_table(img, table, strength=strength) if strength else img,
                f"s{int(round(strength * 100)):03d}",
                base=base_id,
                look=look if strength else "base",
                strength=strength,
            )
        )
    results["six"] = six(six_tiles, out=out_dir / "selftest_six.jpg", title=f"SIX  {base_id}  {look} strength ladder")

    try:
        from tools import charts as charts_mod

        chart_img = charts_mod.load_chart("C1")
        chart_name = "C1"
    except Exception:  # charts not built yet -> synthetic fallback
        chart_img = _synthetic_chart()
        chart_name = "synthetic-ramp"
    results["chart"] = chart_sheet(
        chart_img, table, look, out=out_dir / "selftest_chart.jpg", chart_name=chart_name
    )

    try:  # worst case for the three-row layout: a tall chart
        from tools import charts as charts_mod

        tall = charts_mod.load_chart("C2")
        results["chart_tall"] = chart_sheet(
            tall, table, look, out=out_dir / "selftest_chart_tall.jpg", chart_name="C2"
        )
    except Exception:
        pass

    crop_box = (0.30, 0.18, 0.72, 0.62)
    c_before = render.crop(img, crop_box)
    c_after = render.crop(lut, crop_box)
    results["crop_pair"] = pair(
        Tile(c_before, f"base {base_id} [crop]", base=base_id, look="base", crop="selftest-box", strength=0.0),
        Tile(c_after, f"{look} s100 [crop]", base=base_id, look=look, crop="selftest-box", strength=1.0),
        out=out_dir / "selftest_crop_pair.jpg",
        title=f"PAIR  {base_id}  crop, base vs {look}",
        strip=patch_strip(c_before, c_after, {"crop centre": (0.35, 0.35, 0.65, 0.65)}),
    )

    report = {}
    for kind, path in results.items():
        meta = json.loads(path.with_suffix(".json").read_text())
        w, h = meta["canvas_px"]
        tile_max = max(max(t["tile_px"]) for t in meta["tiles"])
        limit = meta["tile_long_edge_limit"]
        assert w * h <= MAX_PIXELS and max(w, h) <= MAX_EDGE and tile_max <= limit
        with Image.open(path) as im:
            assert im.info.get("icc_profile"), f"{path.name}: no ICC profile embedded"
        report[kind] = {
            "path": str(path),
            "canvas": [w, h],
            "mp": round(w * h / 1e6, 3),
            "tiles": len(meta["tiles"]),
            "tile_max_px": tile_max,
            "tile_limit": limit,
            "kb": round(path.stat().st_size / 1024),
        }
        if verbose:
            print(
                f"{kind:10s} {w:5d}x{h:<5d} {w * h / 1e6:5.3f} MP  "
                f"{len(meta['tiles'])} tiles  tile<= {tile_max}/{limit}  {report[kind]['kb']:5d} KB  {path.name}"
            )
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _look_name(spec: str) -> str:
    if str(spec).lower() in {"base", "identity", "none", "id"}:
        return "base"
    return Path(spec).stem


def _tile_for(img, spec, base_id, strength, crop_name):
    name = _look_name(spec)
    if name == "base":
        return Tile(img, f"base {base_id}", base=base_id, look="base", strength=0.0, crop=crop_name)
    out = render.apply_table(img, spec, strength=strength)
    return Tile(
        out,
        f"{name} s{int(round(strength * 100)):03d}",
        base=base_id,
        look=name,
        strength=strength,
        crop=crop_name,
        source=str(spec),
    )


def _load_scene(base_id: str, crop_name: str | None):
    if crop_name:
        # crops.json boxes are in LibRaw half-size pixels, NOT in the 1600 px frame:
        # the frozen 1:1 crop lives in bases_hi/<base>__<crop>.npz — use it.
        hi = render.WORK / "bases_hi" / f"{base_id}__{crop_name}.npz"
        if hi.exists():
            return render._read_npz(hi)
        raise FileNotFoundError(f"no frozen hi-res crop {hi.name}; crops.json boxes cannot be applied to the 1600 px base")
    return render.load_base(base_id)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sheets.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_out=True):
        p.add_argument("--base", required=True)
        p.add_argument("--crop", default=None)
        p.add_argument("--strength", type=float, default=1.0)
        p.add_argument("--title", default=None)
        if need_out:
            p.add_argument("--out", required=True)

    p_solo = sub.add_parser("solo"); common(p_solo); p_solo.add_argument("--look", default="base")
    p_pair = sub.add_parser("pair"); common(p_pair)
    p_pair.add_argument("--a", default="base"); p_pair.add_argument("--b", required=True)
    p_pair.add_argument("--patches", action="store_true", help="append a measured patch strip")
    p_triad = sub.add_parser("triad"); common(p_triad)
    p_triad.add_argument("--a", default="base"); p_triad.add_argument("--b", required=True); p_triad.add_argument("--c", required=True)
    p_quad = sub.add_parser("quad"); common(p_quad); p_quad.add_argument("--look", action="append", required=True)
    p_six = sub.add_parser("six"); common(p_six); p_six.add_argument("--look", action="append", required=True)
    p_chart = sub.add_parser("chart")
    p_chart.add_argument("--chart", required=True, help="chart name (C1..C7) or path to an .npz/.npy")
    p_chart.add_argument("--look", required=True); p_chart.add_argument("--strength", type=float, default=1.0)
    p_chart.add_argument("--gain", type=float, default=8.0); p_chart.add_argument("--out", required=True)
    p_strip = sub.add_parser("strip"); common(p_strip, need_out=False)
    p_strip.add_argument("--look", action="append", required=True); p_strip.add_argument("--out-dir", required=True)
    p_self = sub.add_parser("selftest")
    p_self.add_argument("--out-dir", default=str(WORK / "review")); p_self.add_argument("--base", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        selftest(args.out_dir, base=args.base)
        return 0

    if args.cmd == "chart":
        name = args.chart
        path = Path(name)
        if path.exists():
            arr = np.load(path)
            chart = arr["chart"] if hasattr(arr, "files") else arr
            label_name = path.stem
        else:
            from tools import charts as charts_mod

            chart = charts_mod.load_chart(name)
            label_name = name
        chart_sheet(
            chart, args.look, _look_name(args.look), out=args.out,
            strength=args.strength, gain=args.gain, chart_name=label_name,
        )
        return 0

    img = _load_scene(args.base, args.crop)
    if args.cmd == "solo":
        solo(_tile_for(img, args.look, args.base, args.strength, args.crop), out=args.out,
             title=args.title or f"{args.base} {_look_name(args.look)}")
    elif args.cmd == "pair":
        ta = _tile_for(img, args.a, args.base, args.strength, args.crop)
        tb = _tile_for(img, args.b, args.base, args.strength, args.crop)
        strip = patch_strip(ta.image, tb.image,
                            {"upper": (0.12, 0.12, 0.30, 0.30), "centre": (0.42, 0.42, 0.58, 0.58),
                             "lower": (0.70, 0.68, 0.88, 0.88)}) if args.patches else None
        pair(ta, tb, out=args.out,
             title=args.title or f"{args.base} {_look_name(args.a)} vs {_look_name(args.b)}", strip=strip)
    elif args.cmd == "triad":
        triad(*[_tile_for(img, s, args.base, args.strength, args.crop) for s in (args.a, args.b, args.c)],
              out=args.out, title=args.title or f"{args.base} triad")
    elif args.cmd in {"quad", "six"}:
        tiles = [_tile_for(img, s, args.base, args.strength, args.crop) for s in args.look]
        want = TILE_COUNT[args.cmd]
        if len(tiles) > want:
            raise SystemExit(f"{args.cmd} takes at most {want} looks, got {len(tiles)}")
        while len(tiles) < want:
            tiles.append(Tile(np.full_like(img, 0.11), "(empty)"))
        (quad if args.cmd == "quad" else six)(tiles, out=args.out, title=args.title or f"{args.base} {args.cmd}")
    elif args.cmd == "strip":
        look_strip(img, args.base, [(_look_name(s), s) for s in args.look], out_dir=args.out_dir,
                   strength=args.strength, crop_name=args.crop)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
