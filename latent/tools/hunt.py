"""P5 artefact hunters: build the sheet sets a hunter agent looks through.

Four topics, each a different kind of damage a 33-point display-referred LUT can
do, and each rendered the way that damage is actually visible:

``gradients``
    The C3 sky charts and the four real skies, as chart sheets: original /
    through the LUT / **x8 amplified difference**.  Banding, hue reversals and
    blotchy transitions show up in the difference row long before they show up
    in the picture.
``skin``
    Every frozen face / skin crop in ``$W/bases_hi`` (the holdout scenes
    included), 1:1, through every look at 100 % and 70 %, as QUAD pages anchored
    on the untouched crop, each with a measured patch strip underneath
    (dL / chroma ratio / hue shift on the crop's centre).
``night``
    The three night scenes, 1:1 on their darkest frozen crops, clean and with
    the ISO 12800 perturbation applied BEFORE the LUT — the only way to see
    colour blotches, posterised lamps and purple shadows.
``highlights``
    The near-white and specular crops (cardigans, fuselages, the white
    interior, cloud roll-off) at 100 % and 70 %, with the patch strip reading
    the near-white patch's cast.

Every sheet obeys the T4 budget (``tools.sheets`` asserts it) and every sheet
lands in ``index.json`` with its subject, looks, strength and size, so a hunter
agent can walk the set instead of guessing filenames.

CLI::

    py tools/hunt.py --cubes out/LUTs_r1 --topic gradients --out $W/review/p5/hunt_gradients
    py tools/hunt.py --cubes out/LUTs_r1 --topic all --out $W/review/p5
    py tools/hunt.py --cubes out/LUTs_r1 --topic skin --looks 01Glaze,02Burin --out /tmp/x
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from tools import metrics, perturb as pb, render, sheets
from tools.render import ROOT, WORK

__all__ = ["TOPICS", "GRADIENT_SCENES", "NIGHT_SCENES", "HIGHLIGHT_CROPS", "run", "main"]

TOPICS = ("gradients", "skin", "night", "highlights")

#: real skies, with the part of the frame that IS sky (normalised box).
GRADIENT_SCENES: dict[str, tuple[float, float, float, float]] = {
    "PANA9817": (0.0, 0.0, 1.0, 0.62),   # aircraft against a deep blue gradient
    "P1038291": (0.0, 0.0, 1.0, 0.55),   # blue-hour bridge, huge smooth sky
    "P1038265": (0.0, 0.0, 1.0, 0.55),   # sunset sea, sun behind cloud deck
    "PANA0078": (0.0, 0.0, 1.0, 0.50),   # hazy overcast sky over towers
}
#: the same scenes' frozen 1:1 gradient crops (only used with --extra-crops).
GRADIENT_CROPS = (
    "PANA9817__sky_gradient_top",
    "PANA9817__sky_horizon",
    "P1038291__dusk_sky",
    "P1038291__water_reflection",
    "P1038265__sun_gradient",
    "PANA0078__hazy_sky_towers",
)

NIGHT_SCENES = ("PANA0005", "PANA9951", "P1038291")

#: the near-white / specular subjects the spec names, plus their obvious siblings.
HIGHLIGHT_CROPS = (
    "PANA0116__white_cardigan",
    "PANA0049__white_cardigan",
    "PANA0049__white_wall",
    "PANA9817__white_fuselage",
    "PANA9769__white_fuselage",
    "P1038291__bridge_white",
    "P1038206__facade_white",
    "P1038265__sun_gradient",
)

STRENGTHS = (1.0, 0.7)
DIFF_GAIN = 8.0

WHAT_TO_LOOK_FOR = {
    "gradients": (
        "In the x8 difference row: stair steps across a smooth sky (banding), a change of "
        "difference COLOUR along the gradient (hue reversal), patchy islands (blotchy "
        "transition). Rank the looks by gradient smoothness and name the scene + the part "
        "of the frame for every defect. Say so when you cannot see one."
    ),
    "skin": (
        "Grey or ashen skin, orange or magenta shadow skin, plasticky (chroma-collapsed) "
        "highlight skin, and a hue SPLIT between the lit and the shaded side of one face. "
        "The patch strip under each page gives the measured dL / chroma ratio / hue shift "
        "for that page's three looks - use it to separate 'I think it looks warm' from "
        "'it moved +4 degrees'."
    ),
    "night": (
        "Colour blotches in the shadows, posterised or flat-topped lamps, neon clipping to "
        "a flat slab, and blue-black shadows turning purple. The ISO 12800 pages show the "
        "same crop with sensor noise added BEFORE the LUT: look for the LUT turning luma "
        "noise into colour noise."
    ),
    "highlights": (
        "A cast in near-whites (cardigan, fuselage, white wall), red-channel clipping that "
        "turns a highlight cyan or yellow as it blows, and loss of texture in a bright sky. "
        "Compare 100 % and 70 %: a defect that survives 70 % is in the shape of the look, "
        "not in its strength."
    ),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _looks(cubes_dir: Path, only: Sequence[str] | None) -> list[tuple[str, np.ndarray, Path]]:
    paths = sorted(Path(cubes_dir).glob("*.cube"))
    if only:
        wanted = {s.strip() for s in only}
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise FileNotFoundError(f"no .cube files in {cubes_dir}")
    return [(p.stem, render.table_of(p), p) for p in paths]


def _crops_index() -> dict[str, dict]:
    """``{"BASE__crop": {"base":…, "crop":…, "tag":…, "path":…}}`` for bases_hi."""
    data = json.loads((WORK / "bases_hi" / "crops.json").read_text())
    table = data.get("crops", data)
    out: dict[str, dict] = {}
    for base_id, entries in table.items():
        for name, meta in entries.items():
            key = f"{base_id}__{name}"
            path = WORK / "bases_hi" / f"{key}.npz"
            if path.exists():
                out[key] = {
                    "base": base_id,
                    "crop": name,
                    "tag": str(meta.get("tag", "")) if isinstance(meta, dict) else "",
                    "path": str(path),
                }
    return out


def _skin_keys(index: dict[str, dict]) -> list[str]:
    keys = [
        k for k, v in sorted(index.items())
        if (v["tag"].startswith("face") or v["tag"].startswith("skin"))
        and v["tag"] != "skin_adjacent_neutral"
    ]
    return keys


def _centre_1to1(img: np.ndarray, edge: int) -> np.ndarray:
    """Centre crop of *edge* px (no resampling) — a true 1:1 window."""
    h, w = img.shape[:2]
    e = min(edge, h, w)
    y0, x0 = (h - e) // 2, (w - e) // 2
    return img[y0 : y0 + e, x0 : x0 + e]


def _mean_L(img: np.ndarray) -> float:
    return float(metrics.oklch_from_code(np.asarray(img)[::4, ::4].reshape(-1, 3))[0].mean())


def _patch_boxes(kind: str) -> dict[str, tuple[float, float, float, float]]:
    if kind == "skin":
        return {"centre": (0.30, 0.30, 0.70, 0.70)}
    if kind == "highlight":
        return {"centre": (0.30, 0.30, 0.70, 0.70)}
    return {"centre": (0.35, 0.35, 0.65, 0.65)}


def _look_pages(
    img: np.ndarray,
    subject: str,
    looks: list[tuple[str, np.ndarray, Path]],
    *,
    out_dir: Path,
    stem: str,
    strength: float,
    patch_box: tuple[float, float, float, float] | None,
    scene: str | None,
    crop: str | None,
    title: str,
) -> list[dict]:
    """QUAD pages (anchor + 3 looks) with an optional measured patch strip."""
    per_page = sheets.TILE_COUNT["quad"] - 1
    pages = math.ceil(len(looks) / per_page)
    s_pct = int(round(strength * 100))
    written: list[dict] = []
    for page in range(pages):
        chunk = looks[page * per_page : (page + 1) * per_page]
        tiles = [sheets.Tile(img, f"ANCHOR {subject} (no LUT)", base=scene, look="base",
                             strength=0.0, crop=crop)]
        measurements = []
        for name, table, path in chunk:
            out_img = render.apply_table(img, table, strength=strength)
            tiles.append(
                sheets.Tile(out_img, f"{name} s{s_pct:03d}", base=scene, look=name,
                            strength=strength, crop=crop, source=str(path))
            )
            if patch_box is not None:
                m = render.patch_measure(img, out_img, patch_box)
                m["name"] = f"{name} s{s_pct:03d}"
                m["box"] = list(patch_box)
                measurements.append(m)
        strip = sheets.PatchStrip(measurements=measurements) if measurements else None
        out = out_dir / f"{stem}__s{s_pct:03d}__p{page + 1}of{pages}.jpg"
        sheets.sheet(
            tiles,
            kind="quad",
            out=out,
            title=f"{title} - {s_pct}% - page {page + 1}/{pages}",
            cols=2,
            allow_partial=True,
            strip=strip,
        )
        written.append(
            {
                "file": out.name,
                "sidecar": out.with_suffix(".json").name,
                "kind": "quad",
                "subject": subject,
                "scene": scene,
                "crop": crop,
                "strength": strength,
                "looks": [n for n, _t, _p in chunk],
                "bytes": out.stat().st_size,
            }
        )
    return written


def _chart_pages(
    img: np.ndarray,
    subject: str,
    looks: list[tuple[str, np.ndarray, Path]],
    *,
    out_dir: Path,
    stem: str,
    scene: str | None,
    crop: str | None,
) -> list[dict]:
    """One three-row sheet per look: original / through LUT / x8 difference."""
    written = []
    for name, table, _path in looks:
        out = out_dir / f"{stem}__{name}.jpg"
        sheets.chart_sheet(img, table, name, out=out, gain=DIFF_GAIN, chart_name=subject)
        meta = json.loads(out.with_suffix(".json").read_text())
        written.append(
            {
                "file": out.name,
                "sidecar": out.with_suffix(".json").name,
                "kind": "chart",
                "subject": subject,
                "scene": scene,
                "crop": crop,
                "strength": 1.0,
                "looks": [name],
                "gain": DIFF_GAIN,
                "max_abs_delta_code8": meta.get("max_abs_delta_code8"),
                "railed_fraction": meta.get("railed_fraction"),
                "bytes": out.stat().st_size,
            }
        )
    return written


# --------------------------------------------------------------------------- #
# topics
# --------------------------------------------------------------------------- #
def _keep(subjects: set[str] | None, *names: str) -> bool:
    """``--subjects`` filter: keep when unset or when any *name* was asked for."""
    return subjects is None or any(n in subjects for n in names)


def _topic_gradients(looks, out_dir: Path, extra_crops: bool, progress: bool,
                     subjects: set[str] | None) -> list[dict]:
    from tools import charts as charts_mod

    written: list[dict] = []
    if _keep(subjects, "C3"):
        chart = charts_mod.load_chart("C3")
        written += _chart_pages(chart, "C3 sky charts", looks, out_dir=out_dir, stem="C3",
                                scene=None, crop=None)
        if progress:
            print(f"  C3: {len(written)} sheets", flush=True)
    for scene, box in GRADIENT_SCENES.items():
        if not _keep(subjects, scene):
            continue
        img = render.crop(render.load_base(scene), box)
        n = len(written)
        written += _chart_pages(img, f"{scene} sky", looks, out_dir=out_dir,
                                stem=f"sky_{scene}", scene=scene, crop="sky_region")
        if progress:
            print(f"  {scene}: {len(written) - n} sheets ({img.shape[1]}x{img.shape[0]} px)", flush=True)
    if extra_crops:
        index = _crops_index()
        for key in GRADIENT_CROPS:
            if key not in index or not _keep(subjects, key):
                continue
            img = _centre_1to1(render._read_npz(Path(index[key]["path"])), 900)
            written += _chart_pages(img, f"{key} 1:1", looks, out_dir=out_dir, stem=f"crop_{key}",
                                    scene=index[key]["base"], crop=index[key]["crop"])
            if progress:
                print(f"  {key}: 1:1 crop sheets", flush=True)
    return written


def _topic_skin(looks, out_dir: Path, progress: bool, subjects: set[str] | None) -> list[dict]:
    index = _crops_index()
    written: list[dict] = []
    for key in _skin_keys(index):
        if not _keep(subjects, key, index[key]["base"]):
            continue
        info = index[key]
        img = _centre_1to1(render._read_npz(Path(info["path"])), 920)
        for strength in STRENGTHS:
            written += _look_pages(
                img, key, looks, out_dir=out_dir, stem=f"skin_{key}", strength=strength,
                patch_box=_patch_boxes("skin")["centre"], scene=info["base"], crop=info["crop"],
                title=f"{key} ({info['tag']})",
            )
        if progress:
            print(f"  {key:28s} {info['tag']:22s} {img.shape[1]}x{img.shape[0]} 1:1", flush=True)
    return written


def _topic_night(looks, out_dir: Path, night_crops: int, progress: bool,
                 subjects: set[str] | None) -> list[dict]:
    index = _crops_index()
    written: list[dict] = []
    for scene in NIGHT_SCENES:
        if not _keep(subjects, scene):
            continue
        keys = [k for k, v in index.items() if v["base"] == scene]
        ranked = sorted(keys, key=lambda k: _mean_L(render._read_npz(Path(index[k]["path"]))))
        for key in ranked[: max(1, night_crops)]:
            info = index[key]
            img = _centre_1to1(render._read_npz(Path(info["path"])), 920)
            for variant, src in (("clean", img), ("iso12800", pb.apply_named(img, "iso12800")[0])):
                written += _look_pages(
                    src, f"{key} {variant}", looks, out_dir=out_dir,
                    stem=f"night_{key}__{variant}", strength=1.0,
                    patch_box=_patch_boxes("night")["centre"], scene=scene, crop=info["crop"],
                    title=f"{key} 1:1 shadow crop - {variant}",
                )
            if progress:
                print(f"  {key:28s} mean L {_mean_L(img):.3f}  clean + iso12800", flush=True)
        # the whole frame too, clean and noisy, so the hunter sees the lamps in context
        full = render.load_base(scene)
        for variant, src in (("clean", full), ("iso12800", pb.apply_named(full, "iso12800")[0])):
            written += _look_pages(
                src, f"{scene} full {variant}", looks, out_dir=out_dir,
                stem=f"night_{scene}_full__{variant}", strength=1.0, patch_box=None,
                scene=scene, crop=None, title=f"{scene} full frame - {variant}",
            )
    return written


def _topic_highlights(looks, out_dir: Path, all_crops: bool, progress: bool,
                      subjects: set[str] | None) -> list[dict]:
    index = _crops_index()
    if all_crops:
        keys = [k for k, v in sorted(index.items())
                if v["tag"].startswith("near_white") or v["tag"] in
                {"high_key_interior", "high_key_background", "specular_water",
                 "highlight_rolloff", "clipped_sun_backlit_leaf"}]
    else:
        keys = [k for k in HIGHLIGHT_CROPS if k in index]
    keys = [k for k in keys if _keep(subjects, k, index[k]["base"])]
    written: list[dict] = []
    for key in keys:
        info = index[key]
        img = _centre_1to1(render._read_npz(Path(info["path"])), 920)
        for strength in STRENGTHS:
            written += _look_pages(
                img, key, looks, out_dir=out_dir, stem=f"hi_{key}", strength=strength,
                patch_box=_patch_boxes("highlight")["centre"], scene=info["base"],
                crop=info["crop"], title=f"{key} ({info['tag']})",
            )
        if progress:
            near_white = float(np.mean(np.asarray(img).max(axis=-1) > 0.90)) * 100
            print(f"  {key:28s} {info['tag']:22s} {near_white:5.1f}% of pixels above code 0.90", flush=True)
    return written


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(
    cubes_dir: str | Path,
    topic: str,
    out_dir: str | Path,
    *,
    looks: Sequence[str] | None = None,
    subjects: Sequence[str] | None = None,
    extra_crops: bool = False,
    all_crops: bool = False,
    night_crops: int = 2,
    progress: bool = True,
) -> dict:
    """Build one topic's sheet set and its ``index.json``.

    *subjects* restricts the set to the named scenes / crop keys / charts
    (``"PANA0116"``, ``"PANA0116__face"``, ``"C3"``) — the whole topic when None.
    """
    if topic not in TOPICS:
        raise ValueError(f"unknown topic {topic!r}; have {TOPICS}")
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    look_tables = _looks(Path(cubes_dir), looks)
    subject_set = {s.strip() for s in subjects} if subjects else None
    if progress:
        print(f"hunt {topic}: {len(look_tables)} looks -> {out_dir}", flush=True)

    if topic == "gradients":
        written = _topic_gradients(look_tables, out_dir, extra_crops, progress, subject_set)
    elif topic == "skin":
        written = _topic_skin(look_tables, out_dir, progress, subject_set)
    elif topic == "night":
        written = _topic_night(look_tables, out_dir, night_crops, progress, subject_set)
    else:
        written = _topic_highlights(look_tables, out_dir, all_crops, progress, subject_set)

    groups: dict[str, list[str]] = {}
    for item in written:
        groups.setdefault(item["subject"], []).append(item["file"])
    index = {
        "topic": topic,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cubes": str(Path(cubes_dir).resolve()),
        "looks": [n for n, _t, _p in look_tables],
        "subjects_filter": sorted(subject_set) if subject_set else None,
        "what_to_look_for": WHAT_TO_LOOK_FOR[topic],
        "reporting": (
            "Per look: every defect as (scene/crop, where in the frame, severity 1-3), and an "
            "explicit 'could not see' for the looks and subjects where you found nothing."
        ),
        "n_sheets": len(written),
        "total_bytes": sum(i["bytes"] for i in written),
        "groups": groups,
        "sheets": written,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    if progress:
        print(
            f"hunt {topic}: {len(written)} sheets, "
            f"{index['total_bytes'] / 1e6:.1f} MB, {index['elapsed_s']} s -> {out_dir / 'index.json'}",
            flush=True,
        )
    return index


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hunt.py", description=__doc__.split("\n")[0])
    ap.add_argument("--cubes", default=str(ROOT / "out" / "LUTs"))
    ap.add_argument("--topic", required=True, choices=(*TOPICS, "all"))
    ap.add_argument("--out", default=None, help="output dir (default $W/review/p5/hunt_<topic>)")
    ap.add_argument("--looks", default=None)
    ap.add_argument("--subjects", default=None,
                    help="comma-separated scenes / crop keys / chart names to restrict to")
    ap.add_argument("--extra-crops", action="store_true", help="gradients: add the 1:1 sky crops")
    ap.add_argument("--all-crops", action="store_true", help="highlights: every near-white crop")
    ap.add_argument("--night-crops", type=int, default=2)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    topics = TOPICS if args.topic == "all" else (args.topic,)
    base_out = Path(args.out) if args.out else WORK / "review" / "p5"
    for topic in topics:
        out = base_out / f"hunt_{topic}" if (args.topic == "all" or args.out is None) else base_out
        run(
            args.cubes,
            topic,
            out,
            looks=args.looks.split(",") if args.looks else None,
            subjects=args.subjects.split(",") if args.subjects else None,
            extra_crops=args.extra_crops,
            all_crops=args.all_crops,
            night_crops=args.night_crops,
            progress=not args.quiet,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
