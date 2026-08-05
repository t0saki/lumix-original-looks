"""Generate the original-look .cube files + manifest.json.

Usage:
    .venv/bin/python generate.py --output luts
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from looks.cubeio import LUT3D, sha256_file, write_cube
from looks.params import GENERALISTS, LOOKS, SPECIALISTS, VERSION
from looks.pipeline import generate_look_table


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--looks", nargs="*", default=None)
    parser.add_argument("--size", type=int, default=33)
    args = parser.parse_args()

    names = args.looks or list(LOOKS)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_looks = []
    for name in names:
        look = LOOKS[name]
        table = generate_look_table(look, size=args.size).reshape(
            args.size, args.size, args.size, 3
        )
        lut = LUT3D(table=table, title=f"{name} - Original look {VERSION} (Standard/sRGB base)")
        path = args.output / f"{name}.cube"
        comments = (
            "# Original creative look, generated from scratch (not converted, not sampled)",
            f"# {look['description']}",
            f"# Generator: lumix-original-looks {VERSION} | OKLab pipeline, monotone Hermite tone,",
            "#   constant-L/hue gamut projection | designed for in-camera opacity 70-100%",
            f"# Recommended: opacity {look['opacity_rec']} | grain {look['grain_rec']}",
        )
        write_cube(path, lut, photo_style="STD", comments=comments)
        manifest_looks.append(
            {
                "name": name,
                "file": path.name,
                "version": VERSION,
                "group": "generalist" if name in GENERALISTS else "specialist",
                "description": look["description"],
                "opacity_rec": look["opacity_rec"],
                "grain_rec": look["grain_rec"],
                "sha256": sha256_file(path),
                "recipe": _jsonable({k: v for k, v in look.items() if k != "name"}),
            }
        )
        print(f"wrote {path}")

    manifest = {
        "project": "lumix-original-looks",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "generator_version": VERSION,
        "method": (
            "Parametric original looks computed from scratch: display sRGB in -> linear -> "
            "monotone Hermite tone on OKLab L -> OKLCh hue/sat/split operators with "
            "suite-standard skin window (OKLCh 30-70deg, +/-12 raised-cosine feather) -> "
            "constant-L constant-hue gamut projection -> display sRGB out"
        ),
        "output_grid": args.size,
        "input_photo_style": "STD",
        "input_colourspace": "sRGB (display-referred), Rec.709 primaries",
        "opacity_note": (
            "70% previews simulate in-camera LUT opacity as 0.7*LUT(x) + 0.3*x in output "
            "sRGB code; this matches the assumed camera blend and is a working assumption "
            "pending an on-camera A/B check"
        ),
        "looks": manifest_looks,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
