"""P5 strength identity + subset distinctiveness.

Two questions the whole-frame pairwise dE00 build gate cannot answer.

1. **70 % <-> 100 % identity.**  The camera's Real Time LUT strength is assumed
   to be ``s*LUT(x) + (1-s)*x`` in code values, so every look also ships as a
   70 % version.  A look whose 70 % render is closer to ANOTHER look's 100 %
   render than to its own is not a strength setting, it is a second copy of its
   neighbour.  For every look and every core scene this measures
   ``d_ii = mean dE00(look_i @70 %, look_i @100 %)`` against
   ``d_ij = mean dE00(look_i @70 %, look_j @100 %)`` for all ``j != i``, and
   reports the margin ``min_j d_ij - d_ii``.  Negative margin = failure, and the
   confusable pair is named.

2. **Subset dE00.**  Two looks can be 4 dE00 apart over the whole frame and
   identical on the only pixels a portrait or a landscape is judged on.  Skin
   and sky masks are taken in OKLCh on the UNTOUCHED base (so the mask does not
   move with the look), and the pairwise 100 % dE00 is recomputed inside each
   mask, pooled over the core scenes by pixel count.

CLI::

    py tools/identity70.py --cubes out/LUTs_r1 --out $W/review/p5/identity70.json
    py tools/identity70.py --cubes out/LUTs_r1 --scenes PANA0116 --target-px 40000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from tools import metrics, render
from tools.render import ROOT, WORK

__all__ = ["SKIN_MASK", "SKY_MASK", "STRENGTH", "masks_for", "run", "format_table"]

#: in-camera strength the 70 % render models.
STRENGTH = 0.70

#: OKLCh windows, measured on the untouched base.
SKIN_MASK = {"L": (0.35, 0.92), "C": (0.020, 0.160), "h": (15.0, 75.0)}
SKY_MASK = {"L": (0.40, 1.00), "C": (0.015, 0.300), "h": (195.0, 285.0)}

MIN_MASK_PIXELS = 500


def masks_for(base: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean skin / sky masks from the base's own OKLCh values."""
    L, C, h = metrics.oklch_from_code(base)
    out = {}
    for name, spec in (("skin", SKIN_MASK), ("sky", SKY_MASK)):
        lo_h, hi_h = spec["h"]
        in_h = (h >= lo_h) & (h <= hi_h) if lo_h <= hi_h else ((h >= lo_h) | (h <= hi_h))
        out[name] = (
            (L >= spec["L"][0]) & (L <= spec["L"][1])
            & (C >= spec["C"][0]) & (C <= spec["C"][1])
            & in_h
        )
    return out


def _load_cubes(cubes_dir: Path, only: Sequence[str] | None) -> list[tuple[str, np.ndarray, Path]]:
    paths = sorted(Path(cubes_dir).glob("*.cube"))
    if only:
        wanted = {s.strip() for s in only}
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise FileNotFoundError(f"no .cube files in {cubes_dir}")
    return [(p.stem, render.table_of(p), p) for p in paths]


def _core_scenes() -> list[str]:
    return [s for s in (WORK / "bases_core.txt").read_text().split() if s.strip()]


def _stride_to(img: np.ndarray, target_px: int) -> tuple[np.ndarray, int]:
    h, w = img.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / float(target_px)))))
    return img[::step, ::step], step


def run(
    cubes_dir: str | Path,
    out_path: str | Path | None = None,
    *,
    looks: Sequence[str] | None = None,
    scenes: Sequence[str] | None = None,
    target_px: int = 60_000,
    progress: bool = True,
) -> dict:
    """Run both tests; write JSON when *out_path* is given; return the report."""
    t_start = time.perf_counter()
    look_tables = _load_cubes(Path(cubes_dir), looks)
    names = [n for n, _t, _p in look_tables]
    scene_ids = list(scenes) if scenes else _core_scenes()
    n = len(names)
    if progress:
        print(f"identity70: {n} looks x {len(scene_ids)} core scenes", flush=True)

    per_scene: dict[str, dict] = {}
    # pooled subset accumulators: sum of dE00 and pixel count per (pair, subset)
    subset_sum = {k: np.zeros((n, n)) for k in ("skin", "sky", "all")}
    subset_cnt = {k: np.zeros((n, n)) for k in ("skin", "sky", "all")}
    mask_px = {"skin": 0, "sky": 0, "all": 0}

    for si, sid in enumerate(scene_ids, 1):
        t0 = time.perf_counter()
        base, step = _stride_to(render.load_base(sid), target_px)
        masks = masks_for(base)
        flat_masks = {"all": np.ones(base.shape[:2], dtype=bool), **masks}
        r100 = [render.apply_table(base, t) for _n, t, _p in look_tables]
        r070 = [STRENGTH * r + (1.0 - STRENGTH) * base for r in r100]
        lab100 = [metrics.srgb_to_lab(r) for r in r100]
        lab070 = [metrics.srgb_to_lab(r) for r in r070]

        d = np.zeros((n, n))       # d[i][j] = mean dE00(look_i @70, look_j @100)
        for i in range(n):
            for j in range(n):
                d[i, j] = float(metrics.de00(lab070[i], lab100[j]).mean())

        pair100 = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                de = metrics.de00(lab100[i], lab100[j])
                pair100[i, j] = pair100[j, i] = float(de.mean())
                for key, m in flat_masks.items():
                    if m.sum() >= MIN_MASK_PIXELS:
                        subset_sum[key][i, j] += float(de[m].sum())
                        subset_cnt[key][i, j] += int(m.sum())
        for key, m in flat_masks.items():
            if m.sum() >= MIN_MASK_PIXELS:   # the same population the pairs used
                mask_px[key] += int(m.sum())

        scene_entry = {
            "px": int(base.shape[0] * base.shape[1]),
            "stride": int(step),
            "mask_px": {k: int(v.sum()) for k, v in flat_masks.items()},
            "d70_to_100": {names[i]: {names[j]: d[i, j] for j in range(n)} for i in range(n)},
            "looks": {},
        }
        for i, name in enumerate(names):
            others = [(d[i, j], names[j]) for j in range(n) if j != i]
            nearest, nearest_name = min(others)
            scene_entry["looks"][name] = {
                "self": float(d[i, i]),
                "nearest_other": float(nearest),
                "nearest_other_look": nearest_name,
                "margin": float(nearest - d[i, i]),
                "ok": bool(nearest > d[i, i]),
            }
        per_scene[sid] = scene_entry
        if progress:
            worst = min(scene_entry["looks"].items(), key=lambda kv: kv[1]["margin"])
            print(
                f"  [{si:2d}/{len(scene_ids)}] {sid:10s} {base.shape[1]}x{base.shape[0]} "
                f"skin {scene_entry['mask_px']['skin'] / 1000:5.1f} kpx  sky {scene_entry['mask_px']['sky'] / 1000:5.1f} kpx"
                f"  worst margin {worst[1]['margin']:+.3f} ({worst[0]} vs {worst[1]['nearest_other_look']})"
                f"  {time.perf_counter() - t0:.1f} s",
                flush=True,
            )

    looks_report: dict[str, dict] = {}
    for i, name in enumerate(names):
        rows = [(per_scene[s]["looks"][name]["margin"], s,
                 per_scene[s]["looks"][name]["nearest_other_look"],
                 per_scene[s]["looks"][name]["self"],
                 per_scene[s]["looks"][name]["nearest_other"]) for s in scene_ids]
        rows.sort()
        failures = [
            {"scene": s, "confusable_with": o, "self": sf, "other": ot, "margin": m}
            for m, s, o, sf, ot in rows if m <= 0.0
        ]
        looks_report[name] = {
            "ok": not failures,
            "min_margin": float(rows[0][0]),
            "min_margin_scene": rows[0][1],
            "min_margin_confusable_with": rows[0][2],
            "self_de00_mean": float(np.mean([per_scene[s]["looks"][name]["self"] for s in scene_ids])),
            "failures": failures,
        }

    subsets = {}
    for key in ("all", "skin", "sky"):
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                cnt = subset_cnt[key][i, j]
                if cnt > 0:
                    pairs.append({"a": names[i], "b": names[j],
                                  "mean_de00": float(subset_sum[key][i, j] / cnt),
                                  "pixels": int(cnt)})
        pairs.sort(key=lambda p: p["mean_de00"])
        subsets[key] = {
            "pixels": mask_px[key],
            "n_pairs": len(pairs),
            "min": pairs[0] if pairs else None,
            "pairs": pairs,
        }

    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cubes": str(Path(cubes_dir).resolve()),
        "strength": STRENGTH,
        "masks": {"skin": SKIN_MASK, "sky": SKY_MASK},
        "scenes": scene_ids,
        "looks": looks_report,
        "identity_ok": all(v["ok"] for v in looks_report.values()),
        "subset_pairwise_de00": subsets,
        "per_scene": per_scene,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
    }
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
        report["file"] = str(out)
    return report


# --------------------------------------------------------------------------- #
# printed table
# --------------------------------------------------------------------------- #
def format_table(report: dict, subset_limit: int = 10) -> str:
    lines = []
    lines.append(f"70% <-> 100% identity   ({len(report['scenes'])} core scenes, strength {report['strength']:.2f})")
    lines.append(f"{'look':12s} {'self dE00':>9s} {'min margin':>11s} {'scene':>10s}  {'confusable with':<16s} {'ok':>4s}")
    lines.append("-" * 70)
    for name, v in report["looks"].items():
        lines.append(
            f"{name:12s} {v['self_de00_mean']:9.3f} {v['min_margin']:+11.3f} {v['min_margin_scene']:>10s}  "
            f"{v['min_margin_confusable_with']:<16s} {'OK' if v['ok'] else 'FAIL':>4s}"
        )
    fails = [(n, f) for n, v in report["looks"].items() for f in v["failures"]]
    if fails:
        lines.append("")
        lines.append("FAILURES (a 70% render closer to another look's 100% render):")
        for n, f in fails:
            lines.append(
                f"  {n} on {f['scene']}: self {f['self']:.3f} dE00, "
                f"{f['confusable_with']} {f['other']:.3f} dE00 (margin {f['margin']:+.3f})"
            )
    else:
        lines.append("")
        lines.append("identity: every look's 70% render is closest to its own 100% render on every core scene.")

    for key in ("all", "skin", "sky"):
        block = report["subset_pairwise_de00"][key]
        lines.append("")
        lines.append(f"pairwise 100% dE00 on the {key.upper()} subset  ({block['pixels']:,} px pooled) "
                     f"- closest {min(subset_limit, block['n_pairs'])} of {block['n_pairs']} pairs")
        for p in block["pairs"][:subset_limit]:
            lines.append(f"  {p['a']:10s} {p['b']:10s} {p['mean_de00']:7.3f}   ({p['pixels']:,} px)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="identity70.py", description=__doc__.split("\n")[0])
    ap.add_argument("--cubes", default=str(ROOT / "out" / "LUTs"))
    ap.add_argument("--out", default=str(WORK / "review" / "p5" / "identity70.json"))
    ap.add_argument("--looks", default=None)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--target-px", type=int, default=60_000)
    ap.add_argument("--subset-limit", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    report = run(
        args.cubes,
        args.out,
        looks=args.looks.split(",") if args.looks else None,
        scenes=args.scenes.split(",") if args.scenes else None,
        target_px=args.target_px,
        progress=not args.quiet,
    )
    print()
    print(format_table(report, subset_limit=args.subset_limit))
    print()
    print(f"wrote {report.get('file')}  in {report['elapsed_s']} s")
    return 0 if report["identity_ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
