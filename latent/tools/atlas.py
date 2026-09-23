"""Reference atlas — the sheets the lead looks at to calibrate taste against rivals.

Five maximally-different core scenes x twelve reference LUTs, laid out as QUAD
sheets whose first tile is always the untouched base, plus one C1 chart sheet
per reference.  Nothing here judges anything: the only job is to put the rivals
side by side, at 100 %, legibly, with a sidecar that says exactly what is in
each tile.

Outputs (all under ``$W/review/atlas/``)
----------------------------------------
``atlas_<scene>_<n>.jpg`` + ``.json``   20 QUAD sheets (5 scenes x 4 pages)
``atlas_chart_<ref>.jpg``  + ``.json``  12 chart sheets (C1 through each ref)
``atlas_index.json``                    scene list, reference list, sheet map

CLI::

    py -m tools.atlas build [--scenes ...] [--no-charts] [--out DIR]
    py -m tools.atlas verify [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, JpegImagePlugin

from tools import render, sheets

__all__ = [
    "SCENES",
    "REFERENCES",
    "GROUPS",
    "OUT_DIR",
    "build",
    "build_scene",
    "build_charts",
    "verify",
    "reference_table",
]

WORK = render.WORK
REFS = WORK / "refs"
OUT_DIR = WORK / "review" / "atlas"

# --------------------------------------------------------------------------- #
# the five scenes
# --------------------------------------------------------------------------- #
#: (slug, base id, one-line ASCII description) — all tier=core, maximally apart:
#: skin in shade / tropical foliage + saturated red / warm sky+sea gradient /
#: night with saturated artificial light / high-key neutral white interior.
SCENES: list[tuple[str, str, str]] = [
    (
        "portrait",
        "PANA0116",
        "close-up face in cap shade, off-white cardigan, hazy glass behind (skin)",
    ),
    (
        "foliage",
        "PANA0202",
        "red powder-puff flower on glossy green leaves, yellow-green bokeh (tropical)",
    ),
    (
        "skysea",
        "P1038265",
        "sunset sea, sun behind cloud deck, specular path, black basalt (sky/sea)",
    ),
    (
        "night",
        "PANA0005",
        "Merlion Park at night, magenta CBD towers, warm floodlit white statue (neon)",
    ),
    (
        "white",
        "PANA0049",
        "daylit white gallery atrium, rainbow doorway the only chroma (high-key)",
    ),
]

# --------------------------------------------------------------------------- #
# the twelve references
# --------------------------------------------------------------------------- #
#: (slug, path relative to $W/refs without extension, tile label, note)
#: ``note`` is ASCII and burnt into the label — the alchemy cubes are
#: V-Log-output grades, so their whites land well below 1.0 on an sRGB base and
#: the lead must not read their flatness as a taste decision.
REFERENCES: list[dict] = [
    # --- page 1: reverse-engineered camera reversal/slide renderings ---------
    dict(slug="Pentax_K5_Reversal", rel="reverse/Pentax_K5_Reversal_Film_33_sRGB",
         label="Pentax K5 Reversal Film", family="reverse", note=""),
    dict(slug="Contax_ND_STD", rel="reverse/Contax_ND_STD_33_sRGB",
         label="Contax ND STD", family="reverse", note=""),
    dict(slug="Leica_X2_VIVID", rel="reverse/Leica_X2_VIVID_33_sRGB",
         label="Leica X2 VIVID", family="reverse", note=""),
    # --- page 2: film-print emulations --------------------------------------
    dict(slug="FujiGFX50S_PROVIA", rel="reverse/FujiGFX50S_PROVIA_33_sRGB",
         label="Fuji GFX50S PROVIA", family="reverse", note=""),
    dict(slug="Kodak2383", rel="std_cst/Kodak2383",
         label="Kodak 2383 print", family="std_cst", note=""),
    dict(slug="FujiClassicNeg", rel="std_cst/FujiClassicNeg-CN",
         label="Fuji Classic Neg CN", family="std_cst", note=""),
    # --- page 3: alchemy (log-domain) + another model's social look ----------
    dict(slug="ClassicNegStrong", rel="alchemy/ClassicNegStrong",
         label="alchemy ClassicNegStrong", family="alchemy", note="log-out"),
    dict(slug="LeicaNatural", rel="alchemy/LeicaNatural",
         label="alchemy Leica Natural", family="alchemy", note="log-out"),
    dict(slug="social_01_Album", rel="social/01_Album",
         label="social 01 Album (peer)", family="social", note=""),
    # --- page 4: peer chromatic + the lead's own two ------------------------
    dict(slug="chromatic_01Rubin", rel="chromatic/01Rubin",
         label="chromatic 01 Rubin (peer)", family="chromatic", note=""),
    dict(slug="Skylight", rel="original/Skylight",
         label="OURS Skylight (prev)", family="original", note=""),
    dict(slug="Meridian", rel="original/Meridian",
         label="OURS Meridian (prev)", family="original", note=""),
]

#: page title for each group of three (references are taken three at a time)
GROUP_TITLES = [
    "reverse-engineered camera reversal",
    "film print emulations",
    "alchemy log grades + peer social",
    "peer chromatic + our previous originals",
]

#: references grouped three per QUAD page (tile 1 is the base, tiles 2-4 these)
GROUPS: list[list[dict]] = [REFERENCES[i : i + 3] for i in range(0, len(REFERENCES), 3)]

CHART_NAME = "C1"
STRENGTH = 1.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def reference_path(ref: dict) -> Path:
    """Absolute path of a reference ``.cube``."""
    return REFS / f"{ref['rel']}.cube"


def reference_table(ref: dict) -> np.ndarray:
    """LUT table of a reference, raising a clear error when the cube is absent."""
    path = reference_path(ref)
    if not path.is_file():
        raise FileNotFoundError(f"reference cube missing: {path}")
    return render.table_of(path)


def _white_out(table: np.ndarray) -> float:
    """Mean output code of white (1,1,1) — flags V-Log-output grades."""
    out = render.apply_table(np.ones((1, 1, 3)), table)
    return float(np.mean(out))


def _tile_label(index: int, ref: dict) -> str:
    """``2 Contax ND STD 100%`` (+ a short ``log-out`` warning where it applies)."""
    tag = f" [{ref['note']}]" if ref["note"] else ""
    return f"{index} {ref['label']}{tag} 100%"


def _missing(refs: Sequence[dict]) -> list[str]:
    return [r["slug"] for r in refs if not reference_path(r).is_file()]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_scene(
    slug: str,
    base_id: str,
    description: str,
    *,
    out_dir: Path = OUT_DIR,
    verbose: bool = True,
) -> list[dict]:
    """Four QUAD sheets for one scene: base + three references per sheet."""
    base = render.load_base(base_id)
    records = []
    for page, group in enumerate(GROUPS, start=1):
        tiles = [
            sheets.Tile(
                base,
                f"1 BASE {base_id} (no LUT)",
                base=base_id,
                look="base",
                strength=0.0,
                extra={"role": "anchor"},
            )
        ]
        for j, ref in enumerate(group, start=2):
            table = reference_table(ref)
            tiles.append(
                sheets.Tile(
                    render.apply_table(base, table, strength=STRENGTH),
                    _tile_label(j, ref),
                    base=base_id,
                    look=ref["slug"],
                    strength=STRENGTH,
                    source=str(reference_path(ref)),
                    extra={
                        "ref_rel": ref["rel"],
                        "family": ref["family"],
                        "note": ref["note"],
                        "white_out_mean": round(_white_out(table), 4),
                    },
                )
            )
        out = out_dir / f"atlas_{slug}_{page}.jpg"
        path = sheets.sheet(
            tiles,
            kind="quad",
            out=out,
            title=f"ATLAS {slug} / {base_id} - sheet {page}/4 - {GROUP_TITLES[page - 1]}",
            # keep the subtitle short: it must survive the auto-shrink at the
            # sheet's ~680 px width (the tile labels already carry "100%").
            subtitle=description,
        )
        records.append(
            {
                "sheet": path.name,
                "scene": slug,
                "base": base_id,
                "page": page,
                "group": GROUP_TITLES[page - 1],
                "references": [r["slug"] for r in group],
            }
        )
        if verbose:
            print(f"  {path.name}: base + {', '.join(r['slug'] for r in group)}")
    return records


def build_charts(*, out_dir: Path = OUT_DIR, verbose: bool = True) -> list[dict]:
    """One C1 chart sheet (original / through LUT / x8 diff) per reference."""
    try:
        from tools import charts
    except Exception as exc:  # pragma: no cover - charts is a hard dependency of T3
        if verbose:
            print(f"  charts unavailable ({exc}); skipping chart sheets")
        return []
    chart_file = WORK / "charts" / f"{CHART_NAME}.npz"
    if not chart_file.is_file():
        if verbose:
            print(f"  {chart_file} missing; skipping chart sheets")
        return []
    chart = charts.load_chart(CHART_NAME)
    records = []
    for ref in REFERENCES:
        table = reference_table(ref)
        out = out_dir / f"atlas_chart_{ref['slug']}.jpg"
        path = sheets.chart_sheet(
            chart,
            table,
            ref["label"],
            out=out,
            strength=STRENGTH,
            chart_name=CHART_NAME,
        )
        meta = json.loads(path.with_suffix(".json").read_text())
        records.append(
            {
                "sheet": path.name,
                "chart": CHART_NAME,
                "reference": ref["slug"],
                "max_abs_delta_code8": round(meta["max_abs_delta_code8"], 2),
                "railed_fraction": round(meta["railed_fraction"], 4),
            }
        )
        if verbose:
            print(
                f"  {path.name}: max delta {meta['max_abs_delta_code8']:.1f}/255, "
                f"railed {meta['railed_fraction'] * 100:.0f}%"
            )
    return records


def build(
    *,
    out_dir: Path | str = OUT_DIR,
    scenes: Sequence[str] | None = None,
    with_charts: bool = True,
    verbose: bool = True,
) -> dict:
    """Render the whole atlas and write ``atlas_index.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = _missing(REFERENCES)
    if missing:
        raise FileNotFoundError(f"missing reference cubes: {missing}")
    wanted = [s for s in SCENES if scenes is None or s[0] in scenes or s[1] in scenes]
    if not wanted:
        raise ValueError(f"no scene matched {scenes!r}")

    t0 = time.time()
    sheet_records: list[dict] = []
    for slug, base_id, description in wanted:
        if verbose:
            print(f"{slug} ({base_id}):")
        sheet_records += build_scene(slug, base_id, description, out_dir=out_dir, verbose=verbose)
    chart_records = build_charts(out_dir=out_dir, verbose=verbose) if with_charts else []

    index = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "strength": STRENGTH,
        "scenes": [
            {"slug": s, "base": b, "description": d}
            for s, b, d in wanted
        ],
        "references": [
            dict(
                r,
                path=str(reference_path(r)),
                sha256=render.sha256_file(reference_path(r)),
                white_out_mean=round(_white_out(reference_table(r)), 4),
            )
            for r in REFERENCES
        ],
        "sheets": sheet_records,
        "chart_sheets": chart_records,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "atlas_index.json").write_text(json.dumps(index, indent=2))
    if verbose:
        print(
            f"{len(sheet_records)} scene sheets + {len(chart_records)} chart sheets "
            f"in {index['seconds']} s -> {out_dir}"
        )
    return index


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify(out_dir: Path | str = OUT_DIR, *, verbose: bool = True) -> dict:
    """Re-open every written sheet and re-check the T4 hard limits from disk.

    Checks the JPEG itself (pixels, long edge, sRGB ICC present, 4:4:4), the
    sidecar (exists, tile count, per-tile long edge against the kind's limit)
    and that the two agree on the canvas size.
    """
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("atlas_*.jpg"))
    failures: list[str] = []
    rows = []
    for path in files:
        side = path.with_suffix(".json")
        with Image.open(path) as im:
            w, h = im.size
            icc = bool(im.info.get("icc_profile"))
            try:
                sampling = JpegImagePlugin.get_sampling(im)
            except Exception:
                sampling = None
        mp = w * h / 1e6
        if not side.is_file():
            failures.append(f"{path.name}: no sidecar")
            continue
        meta = json.loads(side.read_text())
        kind = meta.get("kind", "?")
        limit = meta.get("tile_long_edge_limit", sheets.TILE_LONG.get(kind, sheets.CHART_LONG))
        tiles = meta.get("tiles", [])
        tile_max = max((max(t["tile_px"]) for t in tiles), default=0)
        if w * h > sheets.MAX_PIXELS:
            failures.append(f"{path.name}: {w}x{h} = {mp:.3f} MP > 1.15 MP")
        if max(w, h) > sheets.MAX_EDGE:
            failures.append(f"{path.name}: long edge {max(w, h)} > {sheets.MAX_EDGE}")
        if len(tiles) > sheets.MAX_TILES:
            failures.append(f"{path.name}: {len(tiles)} tiles > {sheets.MAX_TILES}")
        if tile_max > limit:
            failures.append(f"{path.name}: tile long edge {tile_max} > {limit} ({kind})")
        if not icc:
            failures.append(f"{path.name}: no embedded ICC profile")
        if sampling != 0:  # 0 == 4:4:4
            failures.append(f"{path.name}: JPEG subsampling {sampling} != 4:4:4")
        if meta.get("canvas_px") != [w, h]:
            failures.append(f"{path.name}: sidecar says {meta.get('canvas_px')}, file is {[w, h]}")
        rows.append(
            {
                "file": path.name,
                "kind": kind,
                "px": [w, h],
                "MP": round(mp, 4),
                "tiles": len(tiles),
                "tile_long_edge": tile_max,
                "limit": limit,
                "icc": icc,
                "subsampling": sampling,
                "kbytes": round(path.stat().st_size / 1024),
            }
        )
    result = {"dir": str(out_dir), "checked": len(rows), "failures": failures, "sheets": rows}
    if verbose:
        for r in rows:
            print(
                f"{r['file']:34s} {r['kind']:6s} {r['px'][0]:4d}x{r['px'][1]:<4d} "
                f"{r['MP']:.3f} MP  tiles {r['tiles']}  tile<= {r['tile_long_edge']:4d}/{r['limit']:4d} "
                f"icc {int(r['icc'])} sub {r['subsampling']}  {r['kbytes']:5d} kB"
            )
        print(f"{len(rows)} sheets checked, {len(failures)} failures")
        for f in failures:
            print(f"  FAIL {f}")
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="reference atlas sheets")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build", help="render every atlas sheet")
    p_build.add_argument("--out", default=str(OUT_DIR))
    p_build.add_argument("--scenes", nargs="*", default=None)
    p_build.add_argument("--no-charts", action="store_true")

    p_verify = sub.add_parser("verify", help="re-check the T4 limits on disk")
    p_verify.add_argument("--out", default=str(OUT_DIR))

    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd = args.cmd or "build"
    if cmd == "build":
        build(
            out_dir=getattr(args, "out", OUT_DIR),
            scenes=getattr(args, "scenes", None),
            with_charts=not getattr(args, "no_charts", False),
        )
        result = verify(getattr(args, "out", OUT_DIR))
        return 1 if result["failures"] else 0
    result = verify(args.out)
    return 1 if result["failures"] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
