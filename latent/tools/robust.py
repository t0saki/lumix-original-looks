"""P5 perturbation robustness: does the look survive a mis-set camera?

For every look x perturbation x scene this measures what the LUT ADDS to a
capture-side mistake, not what the mistake itself does:

``stability``
    ``mean dE00(LUT(perturbed), LUT(base)) - mean dE00(perturbed, base)``.
    Positive = the LUT amplifies the perturbation; 0 = the LUT is transparent to
    it; negative = the LUT compresses it.  Reported per scene, plus the worst
    8x8 image region so the lead can see WHERE it goes wrong.
``skin drift``
    On the frozen face/skin crops of ``$W/bases_hi``: the hue / chroma-ratio /
    lightness move between LUT(perturbed) and LUT(base), RELATIVE to the same
    move without the LUT.
``noise amplification``
    Chroma-noise sigma (CIELAB a*b* RMS of the noise itself, which is known
    exactly because we added it) in shadow pixels (OKLab L < 0.25), with the LUT
    over without.  Gate: <= 1.35x at ISO 12800.
``clip creep``
    Fraction of pixels the LUT pushes onto the 8-bit 0 / 255 rail that were not
    already there, minus the same number on the unperturbed base.  Gate: the
    perturbation may cost the LUT <= 0.5 % extra railed pixels (exposure
    +1.5 EV excepted, per spec).  The spec's literal net difference of clip
    fractions is reported too, as ``clip_creep``; see ``gates`` in robust.json
    for why it cannot carry the gate.
``hue monotonicity``
    On each skin crop the mean hue through the LUT must move monotonically
    across -1.5 ... +1.5 EV: a sign flip means the look re-routes skin hue at
    some exposure and a half-stop error changes its mind.

Verdict per look: ``pass`` / ``fix`` / ``cut`` with the numbers behind it.
Writes ``robust.json`` and, per look, one TRIAD sheet of the WORST perturbation:
``base(pert) | LUT(pert) | LUT(base)``.

CLI::

    py tools/robust.py --cubes out/LUTs_r1 --out $W/review/p5
    py tools/robust.py --cubes out/LUTs_r1 --out /tmp/x --looks 01Glaze,02Burin --stride 8
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

from tools import metrics, perturb as pb, render, sheets
from tools.render import ROOT, WORK

__all__ = [
    "GATE_NOISE_AMP",
    "GATE_CLIP_CREEP",
    "AMP_WARN",
    "AMP_FAIL",
    "STABILITY_WARN",
    "STABILITY_FAIL",
    "SHADOW_L",
    "clip_fraction",
    "clip_mask",
    "clip_added",
    "block_means",
    "scene_list",
    "skin_crops",
    "run",
]

# --------------------------------------------------------------------------- #
# gates (P5 spec; the two stability lines are this module's own, see below)
# --------------------------------------------------------------------------- #
# P5_RULINGS.md (lead) is authoritative for every threshold below.
GATE_NOISE_AMP = 1.35        # HARD: chroma-noise sigma with LUT / without, at ISO 12800
GATE_CLIP_CREEP = 0.005      # HARD: WB / tint / noise perturbations may add <= 0.5 % railed pixels
#: exposure clip creep is WARN only (a film toe crushing an under-exposed frame
#: is the look's design); +-1.5 EV is informational.
EXPOSURE_CREEP_WARN = {0.5: 0.015, 1.0: 0.04}
#: clip creep level whose EV crossing is the practical exposure bound (使用说明)
CREEP_RECOMMEND = 0.03
#: fine exposure sweep used only to locate that crossing
EV_SWEEP: tuple[float, ...] = tuple(round(s * 0.25, 2) for s in range(-12, 13) if s != 0)
#: stability verdict: p90 of the amplification ratio over every
#: perturbation x scene cell of a look (FAIL above, WARN above AMP_WARN)
AMP_P90_GATE = 1.30
#: The spec's ``stability`` is a DIFFERENCE of dE00 means, so its size is set by
#: how large the perturbation is (a 1.5 EV error is worth ~20 dE00 before any
#: LUT touches it, a WB error ~3).  It is the right thing to REPORT, but a
#: useful verdict line has to be scale free, so the gates below run on
#: ``amplification`` = dE00(LUT) / dE00(no LUT) — "how many times the mistake
#: the LUT makes of it".  Calibrated on out/LUTs_r1: see robust.json's
#: ``amplification_distribution``.
AMP_WARN = 1.20
AMP_FAIL = AMP_P90_GATE
MIN_DE_FOR_RATIO = 0.30      # dE00 below which the ratio is meaningless
#: ``stability`` is reported for every cell, but as a VERDICT line it is nearly
#: useless: on out/LUTs_r1 it is monotone in the size of the perturbation (every
#: look peaks at -1.5 EV, where the perturbation alone is ~20 dE00), so these two
#: are a gross backstop only.  The real colour gate is ``amplification`` above.
STABILITY_WARN = 8.0
STABILITY_FAIL = 15.0
SHADOW_L = 0.25              # OKLab L below which a pixel counts as shadow
MIN_SHADOW_PIXELS = 1000     # fewer than this -> the scene has no shadow to measure
REGION_GRID = 8              # worst-region search grid
HUE_MONO_TOL = 0.25          # degrees; below this a hue step is measurement noise
HUE_REVERSAL_WARN = HUE_MONO_TOL  # rulings: any reversal below 3 deg is a WARN
HUE_REVERSAL_FAIL = 3.0      # degrees of reversal that is a real re-route (FAIL)
#: skin window (OKLCh on the untouched crop) the skin-drift numbers are taken in,
#: so a crowd crop is measured on its skin and not on the pavement behind it.
SKIN_WINDOW = {"L": (0.35, 0.92), "C": (0.020, 0.160), "h": (15.0, 75.0)}
MIN_SKIN_PIXELS = 200

_CLIP_EXEMPT = set(pb.CLIP_EXEMPT)


# --------------------------------------------------------------------------- #
# small numerics
# --------------------------------------------------------------------------- #
#: codes from the rail at which a PRE-LUT pixel already counts as railed: a
#: near-white wall at 254 that the LUT's white point rounds to 255 is not creep
#: (without it +0.25 EV showed 3 % "creep" that the perturbation itself rails at +0.5 EV).
RAIL_TOL = 1


def clip_mask(img: np.ndarray, tol: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of pixels with ANY channel within *tol* codes of 8-bit 0 / 255."""
    q = np.round(np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 255.0)
    return (q <= float(tol)).any(axis=-1), (q >= 255.0 - tol).any(axis=-1)


def clip_fraction(img: np.ndarray) -> tuple[float, float, float]:
    """Fraction of pixels with ANY channel at 8-bit 0 / at 255 / at either."""
    lo, hi = clip_mask(img)
    return float(lo.mean()), float(hi.mean()), float((lo | hi).mean())


def clip_added(before: np.ndarray | tuple, after: np.ndarray | tuple) -> tuple[float, float, float]:
    """Fraction of pixels the LUT pushes ONTO a rail that were not on it before.

    A plain difference of clip fractions is not usable here: a look that tints
    the shadows lifts a whole night scene's zero-channel pixels off code 0, so
    the difference goes to -20 % and says nothing about clipping.  Counting the
    pixels that become railed is monotone, non-negative and is exactly what
    "the LUT adds N % clipped pixels" means.  Arguments are images or the
    ``(lo, hi)`` masks from :func:`clip_mask`.
    """
    b_lo, b_hi = before if isinstance(before, tuple) else clip_mask(before)
    a_lo, a_hi = after if isinstance(after, tuple) else clip_mask(after)
    lo = float((a_lo & ~b_lo).mean())
    hi = float((a_hi & ~b_hi).mean())
    both = float(((a_lo | a_hi) & ~(b_lo | b_hi)).mean())
    return lo, hi, both


def block_means(values: np.ndarray, grid: int = REGION_GRID) -> np.ndarray:
    """Mean of *values* (H, W) over a ``grid x grid`` tiling (edges dropped)."""
    arr = np.asarray(values, dtype=np.float64)
    h, w = arr.shape[:2]
    bh, bw = h // grid, w // grid
    if bh < 1 or bw < 1:
        return arr.mean(keepdims=True).reshape(1, 1)
    trimmed = arr[: bh * grid, : bw * grid]
    return trimmed.reshape(grid, bh, grid, bw).mean(axis=(1, 3))


def _oklch_mean(img: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float, float]:
    """Mean OKLab of an image (optionally masked) -> (L, C, h_deg)."""
    arr = np.asarray(img, dtype=np.float64)
    if mask is not None:
        arr = arr[mask]
    L, C, h = metrics.oklch_from_code(arr.reshape(-1, 3))
    a = C * np.cos(np.deg2rad(h))
    b = C * np.sin(np.deg2rad(h))
    Lm, am, bm = float(L.mean()), float(a.mean()), float(b.mean())
    return Lm, float(np.hypot(am, bm)), float(np.rad2deg(np.arctan2(bm, am)) % 360.0)


def skin_mask(img: np.ndarray) -> np.ndarray:
    """Pixels of *img* inside :data:`SKIN_WINDOW` (OKLCh)."""
    L, C, h = metrics.oklch_from_code(np.asarray(img, dtype=np.float64))
    w = SKIN_WINDOW
    return (
        (L >= w["L"][0]) & (L <= w["L"][1])
        & (C >= w["C"][0]) & (C <= w["C"][1])
        & (h >= w["h"][0]) & (h <= w["h"][1])
    )


def _wrap180(d: float) -> float:
    return float((d + 180.0) % 360.0 - 180.0)


def _chroma_rms(lab_a: np.ndarray, lab_b: np.ndarray, mask: np.ndarray) -> float:
    """RMS CIELAB chroma difference ``sqrt(mean(da**2 + db**2))`` over *mask*."""
    d = lab_a[..., 1:] - lab_b[..., 1:]
    sel = d[mask]
    if sel.size == 0:
        return float("nan")
    return float(np.sqrt((sel**2).sum(axis=-1).mean()))


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def scene_list(work: Path | None = None) -> list[tuple[str, str]]:
    """``[(scene_id, "core"|"holdout"), ...]`` from the frozen split files."""
    work = Path(work) if work is not None else WORK
    out: list[tuple[str, str]] = []
    for tier, name in (("core", "bases_core.txt"), ("holdout", "bases_holdout.txt")):
        path = work / name
        if not path.exists():
            raise FileNotFoundError(f"missing scene split file {path}")
        for line in path.read_text().split():
            if line.strip():
                out.append((line.strip(), tier))
    return out


def skin_crops(work: Path | None = None) -> list[dict]:
    """The frozen face / skin crops (``bases_hi``) P5 measures skin drift on."""
    work = Path(work) if work is not None else WORK
    crops = json.loads((work / "bases_hi" / "crops.json").read_text())
    table = crops.get("crops", crops)
    out = []
    for base_id, entries in sorted(table.items()):
        for name, meta in sorted(entries.items()):
            tag = str(meta.get("tag", "")) if isinstance(meta, dict) else ""
            if not (tag.startswith("face") or tag.startswith("skin")):
                continue
            if tag == "skin_adjacent_neutral":  # a beige dress, not skin
                continue
            path = work / "bases_hi" / f"{base_id}__{name}.npz"
            if path.exists():
                out.append({"base": base_id, "crop": name, "tag": tag, "path": str(path)})
    return out


def _load_cubes(cubes_dir: Path, only: Sequence[str] | None = None) -> list[tuple[str, np.ndarray, Path]]:
    paths = sorted(Path(cubes_dir).glob("*.cube"))
    if only:
        wanted = {s.strip() for s in only}
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise FileNotFoundError(f"no .cube files in {cubes_dir}")
    return [(p.stem, render.table_of(p), p) for p in paths]


def _stride_to(img: np.ndarray, target_px: int) -> tuple[np.ndarray, int]:
    """Decimate by an integer stride to ~*target_px* pixels.

    Decimation, not resampling: the pixel VALUES stay exactly what the camera
    would have produced, which is what the noise and clipping metrics need.
    """
    h, w = img.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / float(target_px)))))
    return img[::step, ::step], step


# --------------------------------------------------------------------------- #
# per-scene measurement
# --------------------------------------------------------------------------- #
def _measure_scene(
    scene_id: str,
    tier: str,
    looks: list[tuple[str, np.ndarray, Path]],
    pert_names: list[str],
    *,
    target_px: int,
    seed: int,
    ev_sweep: Sequence[float] = (),
) -> dict:
    base_full = render.load_base(scene_id)
    base, step = _stride_to(base_full, target_px)
    h, w = base.shape[:2]
    base_lab = metrics.srgb_to_lab(base)
    okL = metrics.oklch_from_code(base)[0]
    shadow = okL < SHADOW_L
    n_shadow = int(shadow.sum())
    base_clip = clip_mask(base)
    clip_base = float((base_clip[0] | base_clip[1]).mean())

    # every perturbation of this scene, once
    pert_img: dict[str, np.ndarray] = {}
    pert_lab: dict[str, np.ndarray] = {}
    pert_clip: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    nolut: dict[str, dict] = {}
    for name in pert_names:
        img, _meta = pb.apply_named(base, name, seed=seed)
        lab = metrics.srgb_to_lab(img)
        pert_img[name] = img
        pert_lab[name] = lab
        pert_clip[name] = clip_mask(img, RAIL_TOL)
        de = metrics.de00(lab, base_lab)
        lo, hi, any_ = clip_fraction(img)
        nolut[name] = {
            "de00_mean": float(de.mean()),
            "de00_p95": float(np.percentile(de, 95)),
            "de_map": de,
            "clip": any_,
            "clip_lo": lo,
            "clip_hi": hi,
            "chroma_sigma": _chroma_rms(lab, base_lab, shadow) if n_shadow >= MIN_SHADOW_PIXELS else float("nan"),
        }

    sweep_img = {ev: pb.perturb(base, ev=ev, seed=seed)[0] for ev in ev_sweep}
    sweep_clip = {ev: clip_mask(img, RAIL_TOL) for ev, img in sweep_img.items()}

    out = {
        "scene": scene_id,
        "tier": tier,
        "px": int(h * w),
        "stride": int(step),
        "shape": [int(h), int(w)],
        "shadow_pixels": n_shadow,
        "clip_base": clip_base,
        "nolut": {k: {kk: vv for kk, vv in v.items() if kk != "de_map"} for k, v in nolut.items()},
        "looks": {},
    }

    for look_name, table, _path in looks:
        lut_base = render.apply_table(base, table)
        lut_base_lab = metrics.srgb_to_lab(lut_base)
        lut_base_clip = clip_mask(lut_base)
        clip_lut_base = float((lut_base_clip[0] | lut_base_clip[1]).mean())
        # what the LUT rails on the UNPERTURBED base: its own black crush /
        # white clip, which no perturbation is responsible for.
        ref_lo, ref_hi, ref_any = clip_added(clip_mask(base, RAIL_TOL), lut_base_clip)
        clip_creep_ref = clip_lut_base - clip_base
        per_pert = {}
        for name in pert_names:
            lut_pert = render.apply_table(pert_img[name], table)
            lut_pert_lab = metrics.srgb_to_lab(lut_pert)
            de_lut = metrics.de00(lut_pert_lab, lut_base_lab)
            delta = de_lut - nolut[name]["de_map"]
            regions = block_means(delta)
            iy, ix = np.unravel_index(int(np.argmax(regions)), regions.shape)
            lut_pert_clip = clip_mask(lut_pert)
            lo = float(lut_pert_clip[0].mean())
            hi = float(lut_pert_clip[1].mean())
            any_ = float((lut_pert_clip[0] | lut_pert_clip[1]).mean())
            add_lo, add_hi, add_any = clip_added(pert_clip[name], lut_pert_clip)
            de_nolut_mean = nolut[name]["de00_mean"]
            entry = {
                "stability": float(delta.mean()),
                # the same thing normalised: how many times the perturbation the
                # LUT makes of it.  Undefined when the perturbation itself is
                # almost invisible, so it is only reported above 0.30 dE00.
                "amplification": (
                    float(de_lut.mean() / de_nolut_mean) if de_nolut_mean >= MIN_DE_FOR_RATIO else None
                ),
                "de00_lut_mean": float(de_lut.mean()),
                "de00_lut_p95": float(np.percentile(de_lut, 95)),
                "de00_nolut_mean": de_nolut_mean,
                "worst_region": {
                    "grid": REGION_GRID,
                    "row": int(iy),
                    "col": int(ix),
                    "stability": float(regions[iy, ix]),
                    "box_norm": [
                        round(ix / REGION_GRID, 4),
                        round(iy / REGION_GRID, 4),
                        round((ix + 1) / REGION_GRID, 4),
                        round((iy + 1) / REGION_GRID, 4),
                    ],
                },
                "clip_lut": any_,
                "clip_lut_lo": lo,
                "clip_lut_hi": hi,
                "clip_nolut": nolut[name]["clip"],
                "clip_nolut_lo": nolut[name]["clip_lo"],
                "clip_nolut_hi": nolut[name]["clip_hi"],
            }
            entry["clip_creep"] = entry["clip_lut"] - entry["clip_nolut"]
            entry["clip_creep_lo"] = lo - nolut[name]["clip_lo"]
            entry["clip_creep_hi"] = hi - nolut[name]["clip_hi"]
            entry["clip_creep_ref"] = clip_creep_ref
            entry["clip_added"] = add_any
            entry["clip_added_lo"] = add_lo
            entry["clip_added_hi"] = add_hi
            entry["clip_added_ref"] = ref_any
            entry["clip_creep_excess"] = add_any - ref_any
            entry["clip_creep_excess_lo"] = add_lo - ref_lo
            entry["clip_creep_excess_hi"] = add_hi - ref_hi
            if pb.family(name) == "noise" and n_shadow >= MIN_SHADOW_PIXELS:
                sig_lut = _chroma_rms(lut_pert_lab, lut_base_lab, shadow)
                sig_nolut = nolut[name]["chroma_sigma"]
                entry["chroma_sigma_lut"] = sig_lut
                entry["chroma_sigma_nolut"] = sig_nolut
                entry["noise_amp"] = float(sig_lut / sig_nolut) if sig_nolut > 0 else float("nan")
            per_pert[name] = entry
        sweep = {}
        for ev in ev_sweep:
            a_lo, a_hi, a_any = clip_added(sweep_clip[ev], clip_mask(render.apply_table(sweep_img[ev], table)))
            sweep[f"{ev:+.2f}"] = [a_any - ref_any, a_lo - ref_lo, a_hi - ref_hi]
        out["looks"][look_name] = {
            "clip_lut_base": clip_lut_base,
            "perturbations": per_pert,
            "ev_sweep": sweep,
        }
    return out


# --------------------------------------------------------------------------- #
# skin crops
# --------------------------------------------------------------------------- #
def _measure_skin(
    crops: list[dict],
    looks: list[tuple[str, np.ndarray, Path]],
    pert_names: list[str],
    *,
    target_px: int,
    seed: int,
    progress: bool,
) -> dict:
    ev_order = [f"ev{ev:+.1f}" for ev in (-1.5, -1.0, -0.5)] + ["base"] + [f"ev{ev:+.1f}" for ev in (0.5, 1.0, 1.5)]
    results: dict[str, dict] = {name: {} for name, _t, _p in looks}
    for crop in crops:
        img_full = render._read_npz(Path(crop["path"]))
        img, step = _stride_to(img_full, target_px)
        key = f"{crop['base']}__{crop['crop']}"
        # measure on the crop's SKIN pixels, chosen once on the untouched crop so
        # the population cannot drift with the look or with the perturbation.
        mask = skin_mask(img)
        n_skin = int(mask.sum())
        mask_kind = "skin_window"
        if n_skin < MIN_SKIN_PIXELS:
            mask, mask_kind = None, "whole_crop"
            n_skin = int(img.shape[0] * img.shape[1])
        perts = {n: pb.apply_named(img, n, seed=seed)[0] for n in pert_names}
        base_mean = _oklch_mean(img, mask)
        nolut = {n: (_oklch_mean(perts[n], mask), base_mean) for n in pert_names}
        for look_name, table, _path in looks:
            lut_base = render.apply_table(img, table)
            Lb, Cb, hb = _oklch_mean(lut_base, mask)
            per_pert = {}
            hues = {}
            for name in pert_names:
                lut_p = render.apply_table(perts[name], table)
                Lp, Cp, hp = _oklch_mean(lut_p, mask)
                (Ln, Cn, hn), (L0, C0, h0) = nolut[name]
                dh_lut, dh_nolut = _wrap180(hp - hb), _wrap180(hn - h0)
                cr_lut = Cp / Cb if Cb > 1e-9 else float("nan")
                cr_nolut = Cn / C0 if C0 > 1e-9 else float("nan")
                per_pert[name] = {
                    "dh_lut": dh_lut,
                    "dh_nolut": dh_nolut,
                    "dh_rel": _wrap180(dh_lut - dh_nolut),
                    "cr_lut": cr_lut,
                    "cr_nolut": cr_nolut,
                    "cr_rel": float(cr_lut / cr_nolut) if cr_nolut and np.isfinite(cr_nolut) else float("nan"),
                    "dl_lut": Lp - Lb,
                    "dl_nolut": Ln - L0,
                    "dl_rel": (Lp - Lb) - (Ln - L0),
                }
                hues[name] = hp
            hues["base"] = hb
            seq = [hues[k] for k in ev_order if k in hues]
            unwrapped = [seq[0]]
            for v in seq[1:]:
                unwrapped.append(unwrapped[-1] + _wrap180(v - unwrapped[-1]))
            steps = np.diff(np.asarray(unwrapped))
            signed = steps[np.abs(steps) > HUE_MONO_TOL]
            monotone = bool(signed.size == 0 or np.all(signed > 0) or np.all(signed < 0))
            reversal = 0.0
            if signed.size:
                pos, neg = signed[signed > 0], signed[signed < 0]
                if pos.size and neg.size:
                    reversal = float(min(pos.max(), -neg.min()))
            results[look_name][key] = {
                "tag": crop["tag"],
                "px": int(img.shape[0] * img.shape[1]),
                "measured_px": n_skin,
                "mask": mask_kind,
                "stride": int(step),
                "base_hue_lut": hb,
                "hue_sequence": {k: float(hues[k]) for k in ev_order if k in hues},
                "hue_monotone": monotone,
                "hue_reversal_deg": reversal,
                "perturbations": per_pert,
            }
        if progress:
            print(
                f"    skin crop {key:28s} {img.shape[1]}x{img.shape[0]} px (stride {step}), "
                f"{n_skin} px in the {mask_kind}",
                flush=True,
            )
    return results


# --------------------------------------------------------------------------- #
# aggregation + verdict
# --------------------------------------------------------------------------- #
def _creep_limit(name: str) -> tuple[float, str] | None:
    """(limit, "HARD"|"WARN") for a perturbation's clip creep, None = informational."""
    if name in _CLIP_EXEMPT:
        return None
    if pb.family(name) != "exposure":
        return GATE_CLIP_CREEP, "HARD"
    ev = abs(float(pb.PERTURBATIONS[name]["ev"]))
    lim = EXPOSURE_CREEP_WARN.get(round(ev, 2))
    return None if lim is None else (lim, "WARN")


def _ev_curve(look_name: str, scenes: list[dict]) -> dict[str, list]:
    """``{"+0.25": [worst-scene creep excess, scene], ...}`` from the EV sweep."""
    curve: dict[str, list] = {}
    for sc in scenes:
        for k, v in sc["looks"][look_name].get("ev_sweep", {}).items():
            if k not in curve or v[0] > curve[k][0]:
                curve[k] = [float(v[0]), sc["scene"]]
    return dict(sorted(curve.items(), key=lambda kv: float(kv[0])))


def _crossing(curve: dict[str, list], sign: int, level: float = CREEP_RECOMMEND) -> float | None:
    """First EV (walking outward from 0 in direction *sign*) where the curve reaches
    *level*, linearly interpolated from the previous point (creep 0 at 0 EV).
    None = no crossing inside the sweep."""
    pts = sorted(((float(k), v[0]) for k, v in curve.items() if float(k) * sign > 0), key=lambda t: abs(t[0]))
    prev_ev, prev_v = 0.0, 0.0
    for ev, v in pts:
        if v >= level:
            t = (level - prev_v) / (v - prev_v) if v > prev_v else 1.0
            return float(prev_ev + t * (ev - prev_ev))
        prev_ev, prev_v = ev, v
    return None


def _recommend(neg: float | None, pos: float | None) -> str:
    """使用说明 line: the crossings rounded INWARD to the camera's 1/3-stop grid."""
    lim = max(abs(e) for e in EV_SWEEP) if EV_SWEEP else 3.0

    def fmt(ev: float | None, sign: int) -> str:
        if ev is None:
            return f"{'−' if sign < 0 else '+'}{lim:.1f}+"
        thirds = np.floor(abs(ev) * 3.0 + 1e-6) / 3.0
        return f"{'−' if sign < 0 else '+'}{thirds:.1f}"

    return f"safe {fmt(neg, -1)} … {fmt(pos, +1)} EV"


def _aggregate(look_name: str, scenes: list[dict], skin: dict, pert_names: list[str]) -> dict:
    per_pert: dict[str, dict] = {}
    for name in pert_names:
        stabs, creeps, amps, ratios, excess, added = [], [], [], [], [], []
        for sc in scenes:
            e = sc["looks"][look_name]["perturbations"][name]
            stabs.append((e["stability"], sc["scene"], e["worst_region"]))
            creeps.append((e["clip_creep"], sc["scene"]))
            added.append((e["clip_added"], sc["scene"]))
            excess.append((e["clip_creep_excess"], sc["scene"],
                           e["clip_creep_excess_lo"], e["clip_creep_excess_hi"]))
            if e.get("amplification") is not None:
                ratios.append((e["amplification"], sc["scene"]))
            if "noise_amp" in e and np.isfinite(e["noise_amp"]):
                amps.append((e["noise_amp"], sc["scene"]))
        stabs.sort(key=lambda t: -t[0])
        creeps.sort(key=lambda t: -t[0])
        added.sort(key=lambda t: -t[0])
        excess.sort(key=lambda t: -t[0])
        amps.sort(key=lambda t: -t[0])
        ratios.sort(key=lambda t: -t[0])
        entry = {
            "stability_mean": float(np.mean([s[0] for s in stabs])),
            "stability_worst": float(stabs[0][0]),
            "stability_worst_scene": stabs[0][1],
            "stability_worst_region": stabs[0][2],
            "amplification_worst": float(ratios[0][0]) if ratios else None,
            "amplification_worst_scene": ratios[0][1] if ratios else None,
            "amplification_mean": float(np.mean([r[0] for r in ratios])) if ratios else None,
            "amplification_scenes": len(ratios),
            "clip_creep_worst": float(creeps[0][0]),
            "clip_creep_worst_scene": creeps[0][1],
            "clip_added_worst": float(added[0][0]),
            "clip_added_worst_scene": added[0][1],
            "clip_creep_excess_worst": float(excess[0][0]),
            "clip_creep_excess_worst_scene": excess[0][1],
            "clip_creep_excess_worst_lo": float(excess[0][2]),
            "clip_creep_excess_worst_hi": float(excess[0][3]),
            "clip_creep_excess_worst_end": (
                "shadow (code 0)" if excess[0][2] >= excess[0][3] else "highlight (code 255)"
            ),
            "clip_exempt": name in _CLIP_EXEMPT,
        }
        if amps:
            entry["noise_amp_worst"] = float(amps[0][0])
            entry["noise_amp_worst_scene"] = amps[0][1]
            entry["noise_amp_mean"] = float(np.mean([a[0] for a in amps]))
            entry["noise_amp_scenes"] = len(amps)
        # one number per perturbation: how many "units of badness", where one
        # unit is a 1.35x colour amplification, 0.35 of chroma-noise
        # amplification, or 0.5 % of clip creep.
        sev = 0.0
        if entry["amplification_worst"] is not None:
            sev = max(sev, (entry["amplification_worst"] - 1.0) / (AMP_WARN - 1.0))
        if "noise_amp_worst" in entry:
            sev = max(sev, (entry["noise_amp_worst"] - 1.0) / (GATE_NOISE_AMP - 1.0))
        lim = _creep_limit(name)
        if lim is not None:
            sev = max(sev, entry["clip_creep_excess_worst"] / lim[0])
        entry["severity"] = float(sev)
        per_pert[name] = entry

    worst_pert = max(per_pert, key=lambda k: per_pert[k]["severity"])
    crops = skin.get(look_name, {})

    hard, fails, warns = [], [], []   # hard -> cut, fails -> fix
    stab_worst = max(per_pert[n]["stability_worst"] for n in per_pert)
    stab_worst_pert = max(per_pert, key=lambda k: per_pert[k]["stability_worst"])
    _amps = [(per_pert[n]["amplification_worst"], n) for n in per_pert
             if per_pert[n]["amplification_worst"] is not None]
    amp_worst, amp_worst_pert = max(_amps) if _amps else (None, None)
    # every amplification cell of this look (perturbation x scene)
    cells = [
        (e["amplification"], name, sc["scene"], e["worst_region"])
        for sc in scenes
        for name, e in sc["looks"][look_name]["perturbations"].items()
        if e.get("amplification") is not None
    ]
    amp_p90 = float(np.percentile([c[0] for c in cells], 90)) if cells else None
    worst_cell = max(cells, key=lambda c: c[0]) if cells else None
    if amp_p90 is not None and amp_p90 > AMP_P90_GATE:
        fails.append(f"stability: p90 amplification {amp_p90:.3f}x (gate {AMP_P90_GATE})")
    elif amp_p90 is not None and amp_p90 > AMP_WARN:
        warns.append(f"stability: p90 amplification {amp_p90:.3f}x (warn {AMP_WARN})")

    amp = per_pert.get("iso12800", {}).get("noise_amp_worst")
    if amp is not None and amp > GATE_NOISE_AMP:
        hard.append(
            f"noise amplification {amp:.3f}x at ISO 12800 on "
            f"{per_pert['iso12800']['noise_amp_worst_scene']} (HARD gate {GATE_NOISE_AMP})"
        )

    # clip creep: WB / tint / noise HARD 0.5 %; exposure +-0.5 / +-1.0 WARN; +-1.5 info
    creep_rows = {}
    for n in per_pert:
        lim = _creep_limit(n)
        v = per_pert[n]["clip_creep_excess_worst"]
        creep_rows[n] = v
        if lim is None or v <= lim[0]:
            continue
        msg = (f"clip creep {v * 100:.2f} % at {n} on {per_pert[n]['clip_creep_excess_worst_scene']}, "
               f"{per_pert[n]['clip_creep_excess_worst_end']} ({lim[1]} {lim[0] * 100:.1f} %)")
        (hard if lim[1] == "HARD" else warns).append(msg)
    hard_creep = [n for n in per_pert if pb.family(n) != "exposure"]
    creep_pert = max(hard_creep, key=lambda k: creep_rows[k]) if hard_creep else None
    creep = creep_rows[creep_pert] if creep_pert else 0.0
    wb_perts = [n for n in per_pert if pb.family(n) == "wb"]
    wb_creep = max((creep_rows[n] for n in wb_perts), default=None)
    creep_raw = max(per_pert[n]["clip_added_worst"] for n in per_pert)

    # exposure sweep -> EV where the worst-scene clip creep crosses 3 %
    ev_curve = _ev_curve(look_name, scenes)
    cross_neg, cross_pos = _crossing(ev_curve, -1), _crossing(ev_curve, +1)

    reversals = [(k, v["hue_reversal_deg"]) for k, v in crops.items() if not v["hue_monotone"]]
    for key, deg in sorted(reversals, key=lambda t: -t[1]):
        if deg >= HUE_REVERSAL_FAIL:
            fails.append(f"skin hue reverses under exposure on {key} ({deg:.2f} deg >= {HUE_REVERSAL_FAIL})")
        elif deg > HUE_REVERSAL_WARN:
            warns.append(f"skin hue reverses under exposure on {key} ({deg:.2f} deg)")
    hue_rev_max = max((v["hue_reversal_deg"] for v in crops.values()), default=0.0)

    skin_worst = None
    for key, v in crops.items():
        for name, e in v["perturbations"].items():
            cand = (abs(e["dh_rel"]), key, name, e)
            if skin_worst is None or cand[0] > skin_worst[0]:
                skin_worst = cand

    # rulings: "cut" is reserved for a HARD failure
    verdict = "cut" if hard else ("fix" if fails else "pass")

    return {
        "verdict": verdict,
        "hard_fails": hard,
        "fails": hard + fails,
        "warns": warns,
        "worst_perturbation": worst_pert,
        "worst_perturbation_severity": per_pert[worst_pert]["severity"],
        "worst_perturbation_scene": (
            per_pert[worst_pert]["amplification_worst_scene"]
            or per_pert[worst_pert]["stability_worst_scene"]
        ),
        "exposure": {
            "creep_level": CREEP_RECOMMEND,
            "curve_worst_scene": ev_curve,
            "cross_neg_ev": cross_neg,
            "cross_pos_ev": cross_pos,
            "recommendation": _recommend(cross_neg, cross_pos),
        },
        "headline": {
            "amplification_p90": amp_p90,
            "amplification_worst_cell": None if worst_cell is None else {
                "amplification": worst_cell[0], "perturbation": worst_cell[1],
                "scene": worst_cell[2], "region": worst_cell[3],
            },
            "clip_creep_wb_worst": wb_creep,
            "hue_reversal_max_deg": float(hue_rev_max),
            "exposure_cross_neg_ev": cross_neg,
            "exposure_cross_pos_ev": cross_pos,
            "amplification_worst": amp_worst,
            "amplification_worst_perturbation": amp_worst_pert,
            "stability_worst": float(stab_worst),
            "stability_worst_perturbation": stab_worst_pert,
            "stability_mean_all": float(np.mean([per_pert[n]["stability_mean"] for n in per_pert])),
            "noise_amp_iso12800": amp,
            "noise_amp_iso3200": per_pert.get("iso3200", {}).get("noise_amp_worst"),
            "clip_creep_excess_worst": float(creep),
            "clip_creep_excess_worst_perturbation": creep_pert,
            "clip_added_worst": float(creep_raw),
            "clip_creep_net_worst": float(max(per_pert[n]["clip_creep_worst"] for n in per_pert)),
            "clip_added_ev+1.5": per_pert.get("ev+1.5", {}).get("clip_added_worst"),
            "skin_dh_rel_worst": None if skin_worst is None else float(skin_worst[3]["dh_rel"]),
            "skin_dh_rel_worst_where": None if skin_worst is None else f"{skin_worst[1]} / {skin_worst[2]}",
            "hue_reversals": len(reversals),
        },
        "perturbations": per_pert,
        "skin": crops,
    }


# --------------------------------------------------------------------------- #
# TRIAD sheet for the worst perturbation
# --------------------------------------------------------------------------- #
def _triad_sheet(look_name: str, table, cube_path: Path, pert_name: str, scene_id: str,
                 out_dir: Path, seed: int, note: str) -> Path:
    full = render.load_base(scene_id)
    crop_1to1 = pb.family(pert_name) == "noise"
    if crop_1to1:
        # noise lives at the pixel level: show a 1:1 window, never a downsample.
        edge = sheets.TILE_LONG["triad"]
        h, w = full.shape[:2]
        okL = metrics.oklch_from_code(full[::4, ::4])[0]
        blocks = block_means(okL)
        iy, ix = np.unravel_index(int(np.argmin(blocks)), blocks.shape)
        cy = int((iy + 0.5) / REGION_GRID * h)
        cx = int((ix + 0.5) / REGION_GRID * w)
        y0 = int(np.clip(cy - edge // 2, 0, max(0, h - edge)))
        x0 = int(np.clip(cx - edge // 2, 0, max(0, w - edge)))
        src = full[y0 : y0 + edge, x0 : x0 + edge]
        where = f"1:1 crop {edge}px at ({x0},{y0}) - darkest region"
    else:
        src = full
        where = "full frame"
    pert_img, _meta = pb.apply_named(src, pert_name, seed=seed)
    lut_pert = render.apply_table(pert_img, table)
    lut_base = render.apply_table(src, table)
    out = out_dir / f"robust__{look_name}__{pert_name.replace('+', 'p').replace('.', '')}.jpg"
    return sheets.triad(
        sheets.Tile(pert_img, f"base + {pert_name} (no LUT)", base=scene_id, look="base",
                    strength=0.0, extra={"perturbation": pert_name}),
        sheets.Tile(lut_pert, f"{look_name} on {pert_name}", base=scene_id, look=look_name,
                    strength=1.0, source=str(cube_path), extra={"perturbation": pert_name}),
        sheets.Tile(lut_base, f"{look_name} on unperturbed base", base=scene_id, look=look_name,
                    strength=1.0, source=str(cube_path)),
        out=out,
        title=f"{look_name} worst perturbation: {pert_name} on {scene_id}",
        subtitle=f"{where} - {note}",
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(
    cubes_dir: str | Path,
    out_dir: str | Path,
    *,
    looks: Sequence[str] | None = None,
    scenes: Sequence[str] | None = None,
    perturbations: Sequence[str] | None = None,
    target_px: int = 110_000,
    skin_px: int = 30_000,
    seed: int = pb.SEED,
    clip_exempt: Sequence[str] | None = None,
    ev_sweep: Sequence[float] | None = None,
    sheets_on: bool = True,
    progress: bool = True,
) -> dict:
    """Measure every look x perturbation x scene; write ``robust.json`` + sheets.

    *clip_exempt* overrides which perturbations the clip-creep gate ignores
    (default: the spec's ``ev+1.5`` alone).
    """
    t_start = time.perf_counter()
    global _CLIP_EXEMPT
    if clip_exempt is not None:
        _CLIP_EXEMPT = {s.strip() for s in clip_exempt if s.strip()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    look_tables = _load_cubes(Path(cubes_dir), looks)
    pert_names = list(perturbations) if perturbations else list(pb.PERTURBATIONS)
    all_scenes = scene_list()
    if scenes:
        wanted = {s.strip() for s in scenes}
        all_scenes = [s for s in all_scenes if s[0] in wanted]
    if not all_scenes:
        raise ValueError("no scenes selected")

    if progress:
        print(
            f"robust: {len(look_tables)} looks x {len(pert_names)} perturbations x {len(all_scenes)} scenes "
            f"= {len(look_tables) * len(pert_names) * len(all_scenes)} cells",
            flush=True,
        )

    scene_results = []
    for i, (sid, tier) in enumerate(all_scenes, 1):
        t0 = time.perf_counter()
        res = _measure_scene(sid, tier, look_tables, pert_names, target_px=target_px, seed=seed,
                             ev_sweep=EV_SWEEP if ev_sweep is None else tuple(ev_sweep))
        scene_results.append(res)
        if progress:
            print(
                f"  [{i:2d}/{len(all_scenes)}] {sid:10s} {tier:7s} {res['shape'][1]}x{res['shape'][0]} "
                f"(stride {res['stride']}, {res['px'] / 1000:.0f} kpx, shadow {res['shadow_pixels'] / 1000:.0f} kpx)"
                f"  {time.perf_counter() - t0:5.1f} s   [{time.perf_counter() - t_start:6.1f} s total]",
                flush=True,
            )

    crops = skin_crops()
    if scenes:
        crops = [c for c in crops if c["base"] in {s.strip() for s in scenes}]
    if progress:
        print(f"  skin drift on {len(crops)} frozen face/skin crops", flush=True)
    t0 = time.perf_counter()
    skin = _measure_skin(crops, look_tables, pert_names, target_px=skin_px, seed=seed, progress=progress)
    if progress:
        print(f"  skin crops done in {time.perf_counter() - t0:.1f} s", flush=True)

    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cubes": str(Path(cubes_dir).resolve()),
        "seed": int(seed),
        "gates": {
            "noise_amp_iso12800": GATE_NOISE_AMP,
            "rulings": "docs/P5_RULINGS.md",
            "verdict_rule": "cut = any HARD failure (noise amp, WB/tint/noise clip creep); "
                            "fix = stability p90 or skin hue reversal >= 3 deg; else pass",
            "clip_creep": GATE_CLIP_CREEP,
            "clip_creep_hard_families": ["wb", "noise"],
            "exposure_creep_warn": {f"+-{k}": v for k, v in EXPOSURE_CREEP_WARN.items()},
            "exposure_creep_recommend_level": CREEP_RECOMMEND,
            "ev_sweep": list(EV_SWEEP if ev_sweep is None else ev_sweep),
            "amplification_p90_gate": AMP_P90_GATE,
            "clip_creep_measured_as": (
                "clip_added(x) = fraction of pixels RAILED by the LUT that were not railed before it; "
                "clip_creep_excess = clip_added(perturbed) - clip_added(unperturbed base). "
                "The spec's literal net difference clip(LUT(pert)) - clip(pert) is still reported as "
                "clip_creep, but it is not usable as a gate on out/LUTs_r1: it is dominated by each look's "
                "own black crush (10Splice adds ~5.5 % on P1038265 under EVERY perturbation, wb6000 "
                "included) and it goes strongly NEGATIVE on night scenes, where any look with a shadow "
                "tint lifts ~20 % of pixels off an exactly-zero channel. Counting newly railed pixels is "
                "non-negative and monotone, and subtracting the unperturbed reference isolates what the "
                "perturbation costs."
            ),
            "clip_creep_exempt": sorted(_CLIP_EXEMPT),
            "rail_tol_codes_pre_lut": RAIL_TOL,
            "amplification_warn": AMP_WARN,
            "amplification_fail": AMP_FAIL,
            "stability_warn": STABILITY_WARN,
            "stability_fail": STABILITY_FAIL,
            "hue_reversal_warn_deg": HUE_REVERSAL_WARN,
            "hue_reversal_fail_deg": HUE_REVERSAL_FAIL,
            "shadow_L": SHADOW_L,
            "skin_window": SKIN_WINDOW,
        },
        "perturbations": {n: dict(pb.PERTURBATIONS[n], family=pb.family(n)) for n in pert_names},
        "scenes": [{"scene": s["scene"], "tier": s["tier"], "px": s["px"], "stride": s["stride"],
                    "shadow_pixels": s["shadow_pixels"]} for s in scene_results],
        "skin_crops": crops,
        "looks": {},
    }

    for look_name, table, path in look_tables:
        agg = _aggregate(look_name, scene_results, skin, pert_names)
        agg["cube"] = str(path)
        agg["per_scene"] = {
            s["scene"]: s["looks"][look_name]["perturbations"] for s in scene_results
        }
        if sheets_on:
            note = (
                f"verdict {agg['verdict']}  amp p90 {(agg['headline']['amplification_p90'] or 0):.3f}x  "
                f"stability {agg['headline']['stability_worst']:+.3f} dE00  {agg['exposure']['recommendation'].replace('−', '-').replace('…', '..')}"
            )
            agg["sheet"] = str(
                _triad_sheet(look_name, table, path, agg["worst_perturbation"],
                             agg["worst_perturbation_scene"], out_dir, seed, note)
            )
        report["looks"][look_name] = agg
        if progress:
            h = agg["headline"]
            print(
                f"  {look_name:10s} {agg['verdict']:5s}  p90 {(h['amplification_p90'] or float('nan')):.3f}x "
                f"max {(h['amplification_worst'] or float('nan')):.3f}x "
                f"({h['amplification_worst_perturbation']})  stab {h['stability_worst']:+.3f} dE00 "
                f"({h['stability_worst_perturbation']})  noise x{(h['noise_amp_iso12800'] or float('nan')):.3f}  "
                f"creep {h['clip_creep_excess_worst'] * 100:+.3f}% (added {h['clip_added_worst'] * 100:.3f}%)  "
                f"hue rev {h['hue_reversals']}",
                flush=True,
            )

    ratios = [
        e["amplification"]
        for s in scene_results
        for lk in s["looks"].values()
        for e in lk["perturbations"].values()
        if e.get("amplification") is not None
    ]
    if ratios:
        arr = np.asarray(ratios)
        report["amplification_distribution"] = {
            "n": int(arr.size),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        }
    report["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    report["verdicts"] = {k: v["verdict"] for k, v in report["looks"].items()}
    (out_dir / "robust_table.md").write_text(summary_table(report))
    path = out_dir / "robust.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=True))
    if progress:
        counts = {v: sum(1 for x in report["verdicts"].values() if x == v) for v in ("pass", "fix", "cut")}
        print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KB)  "
              f"pass {counts['pass']} / fix {counts['fix']} / cut {counts['cut']}  "
              f"in {report['elapsed_s']} s", flush=True)
    return report


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.2f}"


def _ev(v: float | None) -> str:
    return "none" if v is None else f"{v:+.2f}"


def summary_table(report: dict) -> str:
    """Compact per-look table (markdown) + the 使用说明 exposure lines."""
    rows = [
        "| look | verdict | amp p90 | worst amp cell | noise×12800 | WB creep % | 3 % creep EV (−/+) | skin hue rev ° | 使用说明 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, lk in report["looks"].items():
        h = lk["headline"]
        c = h["amplification_worst_cell"]
        cell = "—" if c is None else f"{c['amplification']:.2f} {c['perturbation']} {c['scene']} r{c['region']['row']}c{c['region']['col']}"
        rows.append(
            f"| {name} | {lk['verdict']} | {(h['amplification_p90'] or float('nan')):.3f} | {cell} | "
            f"{(h['noise_amp_iso12800'] or float('nan')):.3f} | {_pct(h['clip_creep_wb_worst'])} | "
            f"{_ev(h['exposure_cross_neg_ev'])} / {_ev(h['exposure_cross_pos_ev'])} | "
            f"{h['hue_reversal_max_deg']:.2f} | {lk['exposure']['recommendation']} |"
        )
    notes = [f"- **{n}**: " + "; ".join(lk["fails"] + [f"(warn) {w}" for w in lk["warns"]])
             for n, lk in report["looks"].items() if lk["fails"] or lk["warns"]]
    return "\n".join(["# P5 robustness (rulings applied)", ""] + rows + ["", "## fails / warns", ""] + notes) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="robust.py", description=__doc__.split("\n")[0])
    ap.add_argument("--cubes", default=str(ROOT / "out" / "LUTs"))
    ap.add_argument("--out", default=str(WORK / "review" / "p5"))
    ap.add_argument("--looks", default=None, help="comma-separated look names (default: all)")
    ap.add_argument("--scenes", default=None, help="comma-separated scene ids (default: core + holdout)")
    ap.add_argument("--perturbations", default=None, help="comma-separated names (default: all 12)")
    ap.add_argument("--target-px", type=int, default=110_000, help="pixels per scene after decimation")
    ap.add_argument("--skin-px", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=pb.SEED)
    ap.add_argument("--clip-exempt", default=None,
                    help=f"perturbations the clip gate ignores (default: {','.join(sorted(pb.CLIP_EXEMPT))})")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    run(
        args.cubes,
        args.out,
        looks=args.looks.split(",") if args.looks else None,
        scenes=args.scenes.split(",") if args.scenes else None,
        perturbations=args.perturbations.split(",") if args.perturbations else None,
        target_px=args.target_px,
        skin_px=args.skin_px,
        seed=args.seed,
        clip_exempt=args.clip_exempt.split(",") if args.clip_exempt is not None else None,
        sheets_on=not args.no_sheets,
        progress=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
