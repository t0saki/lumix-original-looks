"""Render real-photo and chart previews for the generated looks.

Per look: a contact sheet (rows = sample images, cols = [original, 70%, 100%]).
Per image: a cross-look sheet (original + every look at 70% and 100%).
Charts: gray ramp + RGB/CMY ramps through each look at 100%.

70% simulates in-camera LUT opacity as 0.7*LUT(x) + 0.3*x in output sRGB code
(working assumption, recorded in the manifest).

Missing sample files are skipped with a warning; if none resolve (NAS not
mounted) the script degrades to charts only and still exits 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from looks.cubeio import read_lut, tetrahedral_interpolation

LABEL_H = 26
PAD = 6


def load_image(path: Path, long_edge: int) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img.thumbnail((long_edge, long_edge), Image.LANCZOS)
    return np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0


def apply_lut(lut, img: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    out = tetrahedral_interpolation(lut, img.reshape(-1, 3)).reshape(img.shape)
    if opacity < 1.0:
        out = opacity * out + (1.0 - opacity) * img
    return out


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8))


def labeled(img: Image.Image, text: str) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + LABEL_H), (24, 24, 24))
    canvas.paste(img, (0, LABEL_H))
    ImageDraw.Draw(canvas).text((6, 6), text, fill=(230, 230, 230))
    return canvas


def grid(cells: list[list[Image.Image]]) -> Image.Image:
    rows = len(cells)
    cols = max(len(r) for r in cells)
    cw = max(c.size[0] for r in cells for c in r)
    ch = max(c.size[1] for r in cells for c in r)
    canvas = Image.new("RGB", (cols * (cw + PAD) + PAD, rows * (ch + PAD) + PAD), (24, 24, 24))
    for y, row in enumerate(cells):
        for x, cell in enumerate(row):
            canvas.paste(cell, (PAD + x * (cw + PAD), PAD + y * (ch + PAD)))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--luts", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--charts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-edge", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    lut_paths = sorted(args.luts.glob("*.cube"))
    luts = {p.stem: read_lut(p) for p in lut_paths}
    print(f"{len(luts)} looks: {', '.join(luts)}")

    images: dict[str, np.ndarray] = {}
    if args.images.exists():
        for line in args.images.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.exists():
                print(f"WARN missing sample, skipped: {p}")
                continue
            images[p.stem] = load_image(p, args.long_edge)
    if not images:
        print("WARN no sample images available (NAS unmounted?) - charts only")

    # per-look contact sheets
    for name, lut in luts.items():
        rows = []
        for stem, img in images.items():
            out100 = apply_lut(lut, img)
            out70 = apply_lut(lut, img, 0.7)
            rows.append([
                labeled(to_pil(img), f"{stem}  original"),
                labeled(to_pil(out70), f"{name} 70%"),
                labeled(to_pil(out100), f"{name} 100%"),
            ])
        if rows:
            path = args.output / f"sheet_{name}.jpg"
            grid(rows).save(path, quality=90)
            print(f"wrote {path}")

    # per-image cross-look sheets: row0 original + looks at 100%, row1 looks at 70%
    for stem, img in images.items():
        row100 = [labeled(to_pil(img), f"{stem}  original")]
        row70 = [labeled(to_pil(img), "original")]
        for name, lut in luts.items():
            row100.append(labeled(to_pil(apply_lut(lut, img)), f"{name} 100%"))
            row70.append(labeled(to_pil(apply_lut(lut, img, 0.7)), f"{name} 70%"))
        path = args.output / f"xlook_{stem}.jpg"
        grid([row100, row70]).save(path, quality=90)
        print(f"wrote {path}")

    # charts at 100%
    for chart in ["00_gray_full.png", "03_rgb_cmy_ramps.png"]:
        cp = args.charts / chart
        if not cp.exists():
            print(f"WARN chart missing: {cp}")
            continue
        base = load_image(cp, 1600)
        for name, lut in luts.items():
            out = apply_lut(lut, base)
            stacked = grid([
                [labeled(to_pil(base), f"{chart} original")],
                [labeled(to_pil(out), f"{name} 100%")],
            ])
            path = args.output / f"chart_{name}_{chart.split('.')[0]}.jpg"
            stacked.save(path, quality=92)
        print(f"charts rendered for {chart}")

    print("previews done")


if __name__ == "__main__":
    main()
