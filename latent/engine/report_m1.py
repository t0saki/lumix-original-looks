"""Milestone-1 report (ENGINE_SPEC §8): N + P + G, both stage orders.

**SUPERSEDED — this module targets engine v1.0 and no longer runs.**  It is kept
because ``work.nosync/review/m1_engine_report.json`` is a lead artifact and this
is the code that produced it.  ENGINE_SPEC_v1_1 R2 deleted ``gamut.soft`` and
``gamut.eps_dark`` and replaced the tanh tail with a measured limit, so the
``eps_dark`` sweep and the ``(knee, soft)`` scan below no longer describe
anything the engine does.  The v1.1 report is ``engine/report_m1b.py``:

    $R/py -m engine.report_m1b

Run in ONE interpreter (ENV.md):  ``$R/py -m engine.report_m1``

Writes ``$R/work.nosync/review/m1_engine_report.json`` and prints the table.
Everything it reports is measured here, now; nothing is quoted from a doc.

Blocks
------
``identity``      identity look, both orders, sizes 17 and 33
``looks``         both demo looks x both orders: grey error vs T, d2 (full +
                  interior), folds at 100 % and 70 %, final-clip excursion,
                  pre-gamut nm, lattice time
``eps_dark``      the {0, .02, .04, .06} sweep of ENGINE_SPEC §4
``ablation``      each operator switched off one at a time (§8: required when a
                  demo look exceeds d2 p99.9 = 6)
``gamut_alone``   [G] on an otherwise identity look, over (knee, soft) — isolates
                  how much of the d2 and of the folding is the compressor itself
``chroma_headroom`` how much of each look's chroma table the knee can carry
``fingerprint``   measured chroma ratio / hue rotation at the pinned probe
                  (L = 0.65, C = 0.10) next to PLAN appendix A's declared values
``validate``      what validate() says about each look as typed, and the
                  smallest change that clears it
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

from engine import curves as _curves, field as _field, pipeline, spec
from engine.spec import Arch, Field, Gamut, LookSpec, Neutral, Skin, SpecError, Vibrance
from tools import metrics

ROOT = Path(os.environ.get("LATENT_ROOT", Path(__file__).resolve().parent.parent))
OUT = Path(os.environ.get("LATENT_WORK", ROOT / "work.nosync")) / "review" / "m1_engine_report.json"

DEMOS = ("_demo_glaze", "_demo_arcade")
ORDERS = ("NPG", "PGN")
EPS_SWEEP = (0.0, 0.02, 0.04, 0.06)

#: PLAN appendix A, the numbers the demo looks were typed from
DECLARED = {
    "_demo_glaze": {
        "cr": [1.22, 1.20, 1.16, 1.18, 1.24, 1.26, 1.28, 1.30, 1.32, 1.26, 1.20, 1.20],
        "dh10": [-1, -2, -3, -3, -2, 0, 1, 2, 2, 1, 0, -1],
        "tint_rg": [0, 0, 0, 0, 0, 0, 0], "tint_bg": [0, 0, 0, 0, 0, 0, 0],
        "tone": [0, 5, 24, 58, 117, 174, 218, 247, 255],
    },
    "_demo_arcade": {
        "cr": [0.92, 0.90, 0.84, 0.78, 0.80, 0.94, 1.04, 1.00, 0.86, 0.82, 0.88, 0.94],
        "dh10": [8, 5, -2, -4, 6, 11, 12, 7, 2, 4, 9, 11],
        "tint_rg": [-5, -5, -3, 0, 3, 5, 2], "tint_bg": [2, 1, 0, 1, 3, 5, 2],
        "tone": [0, 3, 18, 50, 114, 177, 218, 246, 255],
    },
}
TONE_IN = [0, 11, 31, 64, 115, 166, 209, 240, 255]


def _p(x, n=3):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------------------


_DARK_MASK: dict[int, np.ndarray] = {}


def _dark_mask(size: int) -> np.ndarray:
    """Lattice points whose input is dark (max channel <= 0.25) — the only
    population ``eps_dark`` can move, and the one R06 says a naive eps crushes."""
    m = _DARK_MASK.get(size)
    if m is None:
        m = metrics.identity_table(size).max(axis=-1) <= 0.25
        _DARK_MASK[size] = m
    return m


def _dark_d2(tab: np.ndarray) -> float:
    mask = _dark_mask(tab.shape[0])
    parts = []
    for axis in range(3):
        d2 = np.abs(np.diff(tab, n=2, axis=axis)) * 255.0
        sl = [slice(None)] * 3
        sl[axis] = slice(1, -1)          # the centre point of each triple
        parts.append(d2[mask[tuple(sl)]].ravel())
    v = np.concatenate(parts)
    return float(np.percentile(v, 99.9))


def quick(c, size: int = 33) -> dict:
    """lattice + the four gate numbers, cheaply."""
    t0 = time.perf_counter()
    raw, nm = pipeline._run(c, metrics.identity_table(size))
    dt = time.perf_counter() - t0
    tab = np.clip(raw, 0.0, 1.0)
    d2 = metrics.second_diff_stats(tab)
    fold = metrics.fold_stats(tab)
    fold70 = metrics.fold_stats(metrics.blend_table(tab, 0.7))
    _, grey = pipeline.grey_response(c)
    return {
        "d2_full_p99_9": _p(d2["full"]["p99_9"], 2),
        "d2_int_p99_9": _p(d2["interior"]["p99_9"], 2),
        "d2_dark_p99_9": _p(_dark_d2(tab), 2),
        "d2_full_max": _p(d2["full"]["max"], 2),
        "neg_tetra": fold["neg_count"],
        "neg_tetra_70": fold70["neg_count"],
        "min_ratio": _p(fold["min_ratio"], 4),
        "nm_p99_9": _p(np.percentile(nm, 99.9)) if nm is not None else None,
        "nm_max": _p(nm.max()) if nm is not None else None,
        "clip_below_codes": _p(max(0.0, -raw.min()) * 255.0),
        "clip_above_codes": _p(max(0.0, raw.max() - 1.0) * 255.0),
        "grey_err_codes": _p(np.max(np.abs(grey - c.neutral.T)) * 255.0, 5),
        "lattice_time_s": _p(dt, 4),
    }


def isolations(s: LookSpec) -> dict:
    """The opposite of the ablation: each operator ALONE on top of [N] + [G].

    ``no X`` tables cannot separate operators that only misbehave together;
    these can.
    """
    F = s.field
    bare = dataclasses.replace(s, field=Field(gates=F.gates), skin=Skin(), ops=(), caps=())
    cases = {
        "[N]+[G] only": bare,
        "+ rotation tables only": dataclasses.replace(
            bare, field=dataclasses.replace(bare.field, dh10=F.dh10, dh18=F.dh18,
                                            rot_l_scale=F.rot_l_scale)),
        "+ chroma table CR only": dataclasses.replace(
            bare, field=dataclasses.replace(bare.field, cr=F.cr)),
        "+ arch only": dataclasses.replace(
            bare, field=dataclasses.replace(bare.field, arch=F.arch)),
        "+ vibrance/sat only": dataclasses.replace(
            bare, field=dataclasses.replace(bare.field, sat=F.sat, vibrance=F.vibrance)),
        "+ DL table only": dataclasses.replace(
            bare, field=dataclasses.replace(bare.field, dl=F.dl)),
        "+ skin protocol only": dataclasses.replace(bare, skin=s.skin),
        "+ local ops only": dataclasses.replace(bare, ops=s.ops),
        "+ all chroma (CR*arch*vib)": dataclasses.replace(
            bare, field=dataclasses.replace(bare.field, cr=F.cr, arch=F.arch,
                                            sat=F.sat, vibrance=F.vibrance)),
    }
    return {k: quick(pipeline.compile(v, strict=False)) for k, v in cases.items()}


def probe_fingerprint(c) -> dict:
    """Measured chroma ratio and hue rotation at the pinned probe L = 0.65,
    C = 0.10 (TOOLS_SPEC T2's `c10` column), on the 12 declared hue knots."""
    h = np.arange(0.0, 360.0, 30.0)
    L = np.full(12, 0.65)
    C = np.full(12, 0.10)
    code, in_gamut = metrics.code_from_oklch(L, C, h)
    out = pipeline.apply(c, code)
    Lo, Co, ho = metrics.oklch_from_code(out)
    dh = ((ho - h + 180.0) % 360.0) - 180.0
    return {
        "in_gamut": [bool(x) for x in in_gamut],
        "cr": [_p(x) for x in Co / C],
        "dh": [_p(x, 2) for x in dh],
        "dl": [_p(x, 4) for x in (Lo - L)],
    }


def measured_tint(c) -> dict:
    t = c.neutral.t
    T = c.neutral.T
    ts = np.asarray(metrics.NEUTRAL_T)
    return {
        "tint_rg": [_p(x, 2) for x in np.interp(ts, t, (T[:, 0] - T[:, 1]) * 255.0)],
        "tint_bg": [_p(x, 2) for x in np.interp(ts, t, (T[:, 2] - T[:, 1]) * 255.0)],
        "tone_out": [_p(float(np.interp(v / 255.0, t, T[:, 1]) * 255.0), 2) for v in TONE_IN],
    }


# ---------------------------------------------------------------------------


def ablations(s: LookSpec) -> dict:
    """Every operator switched off one at a time (ENGINE_SPEC §8)."""
    F = s.field
    cases = {
        "as typed": s,
        "no rotation tables": dataclasses.replace(
            s, field=dataclasses.replace(F, dh10=(0.0,) * 12, dh18=(0.0,) * 12)),
        "no chroma table CR": dataclasses.replace(s, field=dataclasses.replace(F, cr=(1.0,) * 12)),
        "no arch": dataclasses.replace(s, field=dataclasses.replace(F, arch=Arch())),
        "no vibrance": dataclasses.replace(s, field=dataclasses.replace(F, vibrance=Vibrance())),
        "no sat": dataclasses.replace(s, field=dataclasses.replace(F, sat=1.0)),
        "no DL table": dataclasses.replace(s, field=dataclasses.replace(F, dl=(0.0,) * 12)),
        "no local ops": dataclasses.replace(s, ops=()),
        "no caps": dataclasses.replace(s, caps=()),
        "no skin protocol": dataclasses.replace(s, skin=Skin()),
        "no [P] at all": dataclasses.replace(s, field=Field(), skin=Skin(), ops=(), caps=()),
        "no [N] tone+tint": dataclasses.replace(s, neutral=Neutral()),
        "no [N] tint only": dataclasses.replace(
            s, neutral=dataclasses.replace(s.neutral, tint_rg=(), tint_bg=())),
        "no [G] (gamut null)": dataclasses.replace(s, gamut=None),
    }
    out = {}
    for name, sp in cases.items():
        out[name] = quick(pipeline.compile(sp, strict=False))
    return out


def eps_dark_sweep(s: LookSpec) -> dict:
    out = {}
    for eps in EPS_SWEEP:
        g = Gamut() if s.gamut is None else s.gamut
        sp = dataclasses.replace(s, gamut=dataclasses.replace(g, eps_dark=eps))
        out[f"{eps:.2f}"] = quick(pipeline.compile(sp, strict=False))
    return out


def gamut_alone() -> dict:
    """[G] on an otherwise identity look: what the compressor costs by itself.

    Stepped by 0.01 through 0.72..0.65 as well, because the first draft of this
    sweep skipped 0.61-0.69 and so mis-stated the fold boundary.
    """
    out = {}
    grid = metrics.identity_table(33)
    knees = [(None, None)] + [(k, round(1.0 - k, 2)) for k in
                              (0.90, 0.80, 0.72, 0.71, 0.70, 0.69, 0.68, 0.67, 0.66,
                               0.65, 0.60, 0.55, 0.50, 0.40, 0.30)]
    for knee, soft in knees:
        g = None if knee is None else Gamut(knee=knee, soft=soft, eps_dark=0.0)
        key = "null" if knee is None else f"knee {knee:.2f} / soft {soft:.2f}"
        c = pipeline.compile(LookSpec(name="Ident", gamut=g), strict=False)
        q = quick(c)
        if knee is not None:
            # the radial gain of the knee at the cube's own corner nm = 3**0.25
            x = (3.0 ** 0.25 - knee) / soft
            q["radial_gain_at_cube_corner"] = _p(1.0 / np.cosh(x) ** 2, 5)
            # what declaring [G] costs an identity look, in codes
            tab = pipeline.lattice(c, 33)
            d = np.abs(tab - grid) * 255.0
            q["cost_vs_identity_max_codes"] = _p(d.max(), 2)
            q["cost_vs_identity_mean_codes"] = _p(d.mean(), 3)
        out[key] = q
    return out


def chroma_headroom(s: LookSpec) -> dict:
    """Scale every chroma-raising operator toward 1 and watch the gates."""
    F = s.field
    out = {}
    for a in (1.0, 0.75, 0.50, 0.35, 0.25, 0.10, 0.0):
        cr = tuple(1.0 + a * (x - 1.0) for x in F.cr)
        vib = Vibrance(gain=1.0 + a * (F.vibrance.gain - 1.0), c=F.vibrance.c)
        arch = Arch(*(tuple(1.0 + a * (x - 1.0) for x in v)
                      for v in (F.arch.warm, F.arch.green, F.arch.blue)))
        sp = dataclasses.replace(
            s, field=dataclasses.replace(F, cr=cr, vibrance=vib, arch=arch))
        out[f"{a:.2f}"] = quick(pipeline.compile(sp, strict=False))
    return out


def scale_field(s: LookSpec, b: float) -> LookSpec:
    """Scale every [P] operator toward the identity by ``b`` (1 = as typed).

    [N] is left alone: the grey axis already meets its gate exactly, and the
    tone/tint is the part of a look a reviewer reads off a picture directly.
    """
    F = s.field
    sk = s.skin
    field = dataclasses.replace(
        F,
        dh10=tuple(b * x for x in F.dh10),
        dh18=tuple(b * x for x in F.dh18),
        cr=tuple(1.0 + b * (x - 1.0) for x in F.cr),
        dl=tuple(b * x for x in F.dl),
        arch=Arch(*(tuple(1.0 + b * (x - 1.0) for x in v)
                    for v in (F.arch.warm, F.arch.green, F.arch.blue))),
        sat=1.0 + b * (F.sat - 1.0),
        vibrance=Vibrance(gain=1.0 + b * (F.vibrance.gain - 1.0), c=F.vibrance.c),
    )
    skin = dataclasses.replace(
        sk, pull=b * sk.pull, hue_offset=b * sk.hue_offset, l_lift=b * sk.l_lift,
        chroma=tuple(1.0 + b * (x - 1.0) for x in sk.chroma))
    ops = tuple(dataclasses.replace(
        o, dh=(b * o.dh[0], b * o.dh[1]), gain_c=1.0 + b * (o.gain_c - 1.0), dl=b * o.dl)
        for o in s.ops)
    return dataclasses.replace(s, field=field, skin=skin, ops=ops)


def strength_ladder(s: LookSpec) -> dict:
    """How much of the look the gates can actually carry.

    The whole [P] side is scaled toward the identity; the report names the
    largest scale that clears zero folds, d2 p99.9 <= 6 (full) and <= 4
    (interior) and nm p99.9 <= 1.15, under the best [G] found by `gamut_alone`.
    """
    out = {}
    for knee, soft, eps in ((0.70, 0.30, 0.0), (0.65, 0.35, 0.0), (0.50, 0.50, 0.0)):
        rows = {}
        for b in (1.0, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.0):
            sp = dataclasses.replace(scale_field(s, b),
                                     gamut=Gamut(knee=knee, soft=soft, eps_dark=eps))
            q = quick(pipeline.compile(sp, strict=False))
            q["pass"] = bool(q["neg_tetra"] == 0 and q["neg_tetra_70"] == 0
                             and q["d2_full_p99_9"] <= 6.0 and q["d2_int_p99_9"] <= 4.0
                             and q["clip_below_codes"] < 1.0)
            q["pass_nm"] = bool(q["nm_p99_9"] is not None and q["nm_p99_9"] <= 1.15)
            rows[f"{b:.2f}"] = q
        ok = [float(k) for k, v in rows.items() if v["pass"]]
        rows["largest_passing_scale"] = max(ok) if ok else None
        out[f"knee {knee:.2f} / soft {soft:.2f} / eps {eps:.2f}"] = rows
    return out


#: third-party / previously shipped .cube files used to calibrate the gates
REF_CUBES = (
    ("Pentax K5 Reversal", "refs/reverse/Pentax_K5_Reversal_Film_33_sRGB.cube"),
    ("Contax N Digital", "refs/reverse/Contax_ND_STD_33_sRGB.cube"),
    ("Leica X2 Vivid", "refs/reverse/Leica_X2_VIVID_33_sRGB.cube"),
    ("Lumix S1R Vivid", "refs/reverse/Lumix_S1R_Vivid_sRGB_33.cube"),
    ("Fuji GFX50S Provia", "refs/reverse/FujiGFX50S_PROVIA_33_sRGB.cube"),
    ("ours 2025 Skylight", "refs/original/Skylight.cube"),
    ("ours 2025 Canopy", "refs/original/Canopy.cube"),
    ("Chromatic 01Rubin", "refs/chromatic/01Rubin.cube"),
)


def reference_calibration() -> dict:
    """The same metrics on shipped third-party LUTs — is the gate reachable?

    The reference LUTs buy a low full-lattice d2 partly by CLIPPING the outer
    shell flat (Pentax: 9.7 % of the table sits at exactly 0), which is exactly
    what ENGINE_SPEC forbids us.  ``zero`` / ``one`` and ``neg_strict`` are
    reported so the comparison is honest.
    """
    work = Path(os.environ.get("LATENT_WORK", ROOT / "work.nosync"))
    out = {}
    for label, rel in REF_CUBES:
        p = work / rel
        if not p.exists():
            out[label] = {"error": f"missing: {p}"}
            continue
        tab = metrics.sampler_from_cube(p).table
        if tab.shape[0] != 33:
            out[label] = {"error": f"size {tab.shape[0]}"}
            continue
        d2 = metrics.second_diff_stats(tab)
        f = metrics.fold_stats(tab)
        cl = metrics.clip_stats(tab)
        out[label] = {
            "d2_full_p99_9": _p(d2["full"]["p99_9"], 2),
            "d2_int_p99_9": _p(d2["interior"]["p99_9"], 2),
            "d2_dark_p99_9": _p(_dark_d2(tab), 2),
            "d2_full_max": _p(d2["full"]["max"], 2),
            "neg_tetra": f["neg_count"],
            "neg_strict": f["neg_strict_count"],
            "zero_frac": _p(cl["zero"], 5),
            "one_frac": _p(cl["one"], 5),
        }
    return out


#: the dense (C0, L0) grid the pinned §3.7 probe is checked against
DENSE_C = np.arange(0.04, 0.3301, 0.005)
DENSE_L = np.arange(0.10, 0.9601, 0.0125)


def validate_block(s: LookSpec) -> dict:
    d = {"name": s.name, "errors": None, "warnings": []}
    try:
        d["warnings"] = spec.validate(s)
    except SpecError as exc:
        d["errors"] = str(exc).splitlines()[1:]
    cf = _field.compile_field(s)
    j, where = _field.fold_bound(cf, spec.FOLD_PROBE_C, spec.FOLD_PROBE_L)
    jd, whered = _field.fold_bound(cf, DENSE_C, DENSE_L)
    d["hue_fold_bound"] = {"min_1_plus_ddh": _p(j, 4),
                           "at": {"h0": where[0], "C0": where[1], "L0": where[2]},
                           "dense_min": _p(jd, 4),
                           "dense_at": {"h0": whered[0], "C0": _p(whered[1], 4),
                                        "L0": _p(whered[2], 4)},
                           "probe_overestimate": _p(j - jd, 4)}
    lo, hi, w = _field.arch_range(cf)
    d["cr_times_arch"] = {"min": _p(lo), "max": _p(hi), "at_h0": _p(w[0], 1), "at_L0": _p(w[1])}
    glo, ghi, gw = _field.total_gain_range(cf)
    d["total_chroma_gain"] = {"min": _p(glo), "max": _p(ghi),
                              "at": {"h0": _p(gw[0], 1), "L0": _p(gw[1]), "C0": _p(gw[2], 3)},
                              "design_budget": [spec.GAIN_LO, spec.GAIN_HI],
                              "hard_bound": spec.GAIN_HARD_HI}
    gerr, gt, gch_ = pipeline.grey_axis_error(s)
    d["grey_axis"] = {"max_err_codes": _p(gerr, 5), "at_t": _p(gt, 4), "channel": gch_,
                      "gate": spec.GREY_ERR_MAX}
    # smallest skin-protocol change that clears the fold bound, PINNED vs DENSE:
    # the two fixes the first review proposed (hue_residual 0.43, feather 18 deg
    # as 21/39/61/79) clear the old sparse probe and still fold on the dense one.
    fixes = {}
    for hr in (0.25, 0.35, 0.40, 0.43, 0.45, 0.50, 0.60):
        cf2 = _field.compile_field(dataclasses.replace(
            s, skin=dataclasses.replace(s.skin, hue_residual=hr)))
        fixes[f"hue_residual {hr:.2f}"] = {
            "pinned": _p(_field.fold_bound(cf2, spec.FOLD_PROBE_C, spec.FOLD_PROBE_L)[0], 4),
            "dense": _p(_field.fold_bound(cf2, DENSE_C, DENSE_L)[0], 4)}
    for fe in (12.0, 15.0, 17.0, 18.0, 20.0, 24.0):
        w4 = (s.skin.window[1] - fe, s.skin.window[1], s.skin.window[2],
              s.skin.window[2] + fe)
        cf2 = _field.compile_field(dataclasses.replace(
            s, skin=dataclasses.replace(s.skin, window=w4)))
        fixes[f"skin feather {fe:.0f} deg {tuple(int(x) for x in w4)}"] = {
            "pinned": _p(_field.fold_bound(cf2, spec.FOLD_PROBE_C, spec.FOLD_PROBE_L)[0], 4),
            "dense": _p(_field.fold_bound(cf2, DENSE_C, DENSE_L)[0], 4)}
    d["fold_bound_vs_fix"] = fixes
    return d


# ---------------------------------------------------------------------------
# what the two adversarial reviews found, reproduced and then re-measured
# ---------------------------------------------------------------------------


def _cap_ratio_at_chroma(c, L0: float, C0: float, h0: float) -> float:
    """C_out / C_in for one in-gamut probe — how much a cap actually moved it."""
    L = np.array([float(L0)])
    h = np.array([float(h0)])
    Cin = np.array([float(C0)])
    code, _ = metrics.code_from_oklch(L, Cin, h)
    _, Co, _ = metrics.oklch_from_code(pipeline.apply(c, code))
    return float(Co[0] / Cin[0])


def _cap_ratio(c, L0: float, frac: float, h0: float) -> float:
    L = np.array([float(L0)])
    h = np.array([float(h0)])
    return _cap_ratio_at_chroma(c, L0, float(frac * metrics.cmax(L, h)[0]), h0)


def _accepts(s: LookSpec) -> str:
    try:
        w = spec.validate(s)
    except SpecError as exc:
        return "REJECTED: " + str(exc).splitlines()[1].strip(" -")[:150]
    return "accepted" + (f" (+{len(w)} warning(s))" if w else "")


def review_regressions() -> dict:
    """Every confirmed finding: the offending spec, what it used to do, and what
    the engine says about it now.  ``before`` numbers were measured on the
    pre-fix engine and are quoted; ``now`` is measured here."""
    tinted = Neutral(
        tone=tuple(tuple(p) for p in zip(TONE_IN, DECLARED["_demo_glaze"]["tone"])),
        tint_rg=(spec.TintBump(a=-9.0, mu=0.30, sigma=0.25),),
        tint_bg=(spec.TintBump(a=12.0, mu=0.35, sigma=0.25),))
    big_tint = Neutral(
        tone=tuple(tuple(p) for p in zip(TONE_IN, DECLARED["_demo_glaze"]["tone"])),
        tint_rg=(spec.TintBump(a=-40.0, mu=0.28, sigma=0.15),),
        tint_bg=(spec.TintBump(a=40.0, mu=0.28, sigma=0.15),))

    cases = {
        "cap fires on the tinted grey (NPG) — the same cap, now legal": {
            "spec": "tint -9/+12 + cap(center 30, sigma 60, start .005, cap .30), knee .70",
            "before": "with cap .015 instead of .30: validate() -> [], grey error 2.3879 "
                      "codes NPG / 0.0000 PGN (gate 0.02).  The cap window came from the "
                      "CURRENT colour, which in NPG is the grey [N] has tinted",
            "look": LookSpec(name="CapG", neutral=tinted,
                             caps=(spec.Cap(center=30, sigma=60, start=0.005, cap=0.30),),
                             gamut=Gamut(0.70, 0.30, 0.0)),
        },
        "the original start .005 / cap .015 pair": {
            "spec": "cap(center 30, sigma 60, start .005, cap .015)",
            "before": "validate() -> []",
            "look": LookSpec(name="CapG", neutral=tinted,
                             caps=(spec.Cap(center=30, sigma=60, start=0.005, cap=0.015),),
                             gamut=Gamut(0.70, 0.30, 0.0)),
        },
        "cap ignores grd and the chroma gate — same cap, now legal": {
            "spec": "cap(center 30, sigma 45, start .02, cap .30) alone",
            "before": "with cap .015: C_out/C_in = 0.958 at L0 = .030 (where grd == 0), "
                      "0.783 at L0 = .050, hue-dependent at C0 = 0.0102, 3,472 folded",
            "look": LookSpec(name="Cap2",
                             caps=(spec.Cap(center=30, sigma=45, start=0.02, cap=0.30),)),
        },
        "ENGINE_SPEC §7's own example cap": {
            "spec": "identity look + cap(center 140, sigma 35, start .11, cap .15)",
            "before": "validate() -> [], 20 negative tetrahedra, min ratio -1.307e-05",
            "look": LookSpec(name="CapX",
                             caps=(spec.Cap(center=140, sigma=35, start=0.11, cap=0.15),)),
        },
        "ops[].dl has no ceiling": {
            "spec": "ops[0] = (center 140, sigma 30, dl 0.5)",
            "before": "validate() -> [], 55,847 folded, d2 p99.9 32.68; dl 0.30 gave "
                      "OKLab dL +0.3297 (5.5x the |DL| <= 0.06 ceiling)",
            "look": LookSpec(name="X1", ops=(spec.Op(center=140, sigma=30, dl=0.5),)),
        },
        "skin.l_lift has no ceiling": {
            "spec": "skin.l_lift = 0.5",
            "before": "validate() -> [], 11,964 folded, min ratio -71.23, d2 p99.9 142.25",
            "look": LookSpec(name="X2", skin=Skin(l_lift=0.5)),
        },
        "field.sat is unchecked": {
            "spec": "field.sat = 4.0",
            "before": "validate() -> [], 185,513 of 196,608 folded, d2 p99.9 86.31",
            "look": LookSpec(name="X3", field=Field(sat=4.0)),
        },
        "field.sat may be negative": {
            "spec": "field.sat = -1.0",
            "before": "validate() -> [], 106,316 folded (the chroma vector flips sign)",
            "look": LookSpec(name="X4", field=Field(sat=-1.0)),
        },
        "skin.chroma is unchecked": {
            "spec": "skin.chroma = (3, 3, 3)",
            "before": "validate() -> [], 13,711 folded, min ratio -151.25",
            "look": LookSpec(name="X5", skin=Skin(chroma=(3.0, 3.0, 3.0))),
        },
        "mono.dl has no ceiling": {
            "spec": "mono.dl = 0.40 on all 12 knots",
            "before": "validate() -> [], pre-clip max 1.1089 (+27.7 codes above white)",
            "look": LookSpec(name="X7", mono=spec.Mono(dl=(0.40,) * 12)),
        },
        "coincident S5 edges = a step in the path": {
            "spec": "ops[0].l_band = (0.2, 0.2, 0.8, 0.8)",
            "before": "validate() -> [], two probes 7.2e-07 codes apart came out 28.64 "
                      "codes apart; 11,469 folded tetrahedra",
            "look": LookSpec(name="Step", ops=(spec.Op(center=140, sigma=30, dh=(8.0, 8.0),
                                                       l_band=(0.2, 0.2, 0.8, 0.8)),)),
        },
        "reversed skin.c_fade": {
            "spec": "skin.c_fade = (0.24, 0.16)",
            "before": "validate() -> [] (the skin window was installed inverted)",
            "look": LookSpec(name="Y3", skin=Skin(c_fade=(0.24, 0.16))),
        },
        "scrambled skin.l_gate": {
            "spec": "skin.l_gate = (0.9, 0.2, 0.35, 0.97)",
            "before": "validate() -> []",
            "look": LookSpec(name="Y4", skin=Skin(l_gate=(0.9, 0.2, 0.35, 0.97))),
        },
        "[G] eats a big tint in NPG": {
            "spec": "tint +-40 at mu 0.28 + gamut(knee .30, soft .70)",
            "before": "validate() -> [], grey error 0.8506 codes NPG / 0.0000 PGN",
            "look": LookSpec(name="Tw", neutral=big_tint, gamut=Gamut(0.30, 0.70, 0.0)),
        },
        "the same tint at the §4 default knee": {
            "spec": "tint +-40 at mu 0.28 + gamut(knee .70, soft .30)",
            "before": "grey error 0.0000 codes — the exposure is to looks that lower the knee",
            "look": LookSpec(name="Tw2", neutral=big_tint, gamut=Gamut(0.70, 0.30, 0.0)),
        },
    }
    out = {}
    for name, c in cases.items():
        entry = {"spec": c["spec"], "before": c["before"], "now": _accepts(c["look"])}
        if name.startswith(("cap fires", "[G] eats", "the same tint")):
            entry["grey_err_codes_now"] = _p(pipeline.grey_axis_error(c["look"])[0], 5)
        if name.startswith("cap ignores"):
            cc = pipeline.compile(c["look"])
            entry["chroma_ratio_now"] = {
                f"L0={L0:.3f} (grd={float(_curves.S5(0.04, 0.14, L0)):.4f})":
                    _p(_cap_ratio(cc, L0, 0.85, 30.0), 6)
                for L0 in (0.030, 0.050, 0.35)}
            entry["hue_dependence_at_C0_0.0102"] = {
                f"h0={h:.0f}": _p(_cap_ratio_at_chroma(cc, 0.50, 0.0102, h), 6)
                for h in (0.0, 180.0)}
            entry["neg_tetra_now"] = quick(cc)["neg_tetra"]
        out[name] = entry

    # the ones that are measurements rather than rejections
    m = {}
    s = LookSpec(name="Argent", mono=spec.Mono(filter=(1.15, 0.0, -0.15)))
    cm = pipeline.compile(s)
    t = np.linspace(0.0, 0.30, 20001)
    ramp = np.stack([np.zeros_like(t), np.zeros_like(t), t], -1)
    raw, _ = pipeline._run(cm, ramp)
    d = np.diff(raw[:, 0])
    m["mono clamp plateau (filter 1.15/0/-0.15)"] = {
        "before": "cbrt(max(Y,0)): all 20,000 steps of a blue ramp exactly 0 "
                  "(76.5-code plateau INSIDE the path), clip excursion reported as 0",
        "now": {"zero_steps": int(np.count_nonzero(d == 0.0)),
                "min_abs_step_codes": _p(float(np.abs(d).min()) * 255.0, 6),
                "preclip_min_codes": _p(float(raw.min()) * 255.0, 3),
                "validate": _accepts(s)},
    }
    md = pipeline.diagnostics(
        pipeline.compile(LookSpec(name="Argent", mono=spec.Mono(filter=(0.36, 0.53, 0.11)))),
        grid_n=50_000)
    m["mono fold gate"] = {
        "before": "diagnostics reported neg_count = 196,608 / 196,608 with no note; "
                  "§8's 'zero negative tetrahedra' is unreachable for a rank-1 map",
        "now": {"neg_count": md["fold"]["neg_count"],
                "neg_strict_count": md["fold"]["neg_strict_count"],
                "gate_field": md["fold"]["gate_field"],
                "grey_min_step_codes": _p(md["mono"]["grey_min_step_codes"], 5),
                "grey_strictly_increasing": md["mono"]["grey_strictly_increasing"]},
    }
    a = _field._arch_spline([0.92, 1.02, 1.10, 1.14, 1.10])
    lo = np.linspace(0.08, 0.25, 201)
    m["arch flat extension"] = {
        "before": "declared A_warm(0.25) = 0.920, actual range on [0.08, 0.25] "
                  "0.9113..0.9200 (-0.9 %); §3.4 says 'extended flat'",
        "now": {"low_tail_min": _p(float(a(np.clip(lo, 0.25, 0.85)).min()), 6),
                "low_tail_max": _p(float(a(np.clip(lo, 0.25, 0.85)).max()), 6)},
    }
    bad = pipeline.compile(LookSpec(name="X3", field=Field(sat=4.0)), strict=False)
    try:
        pipeline.write_cube_file(bad, ROOT / "work.nosync" / "tmp" / "X3.cube")
        m["strict=False can still ship"] = {"now": "STILL WRITES — not fixed"}
    except SpecError as exc:
        m["strict=False can still ship"] = {
            "before": "write_cube_file() checked only the 8-char stem, so a look "
                      "validate() rejects could be written to a .cube deliverable",
            "now": "refused: " + str(exc).splitlines()[0]}
    out["measurements"] = m
    return out


def build() -> dict:
    rep: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "spec": "ENGINE_SPEC §8 milestone 1 (N + P + G)",
                 "gates": {"identity": "< 1e-6", "grey_err_codes": "< 0.02",
                           "neg_tetra": "== 0", "d2_full_p99_9": "<= 6"}}

    # --- identity ------------------------------------------------------
    ident = {}
    s_id = spec.load_look("_identity")
    for order in ORDERS:
        c = pipeline.compile(dataclasses.replace(s_id, order=order))
        ident[order] = {"err_33": _p(pipeline.identity_error(c, 33), 18),
                        "err_17": _p(pipeline.identity_error(c, 17), 18),
                        "warnings": list(c.warnings)}
    rep["identity"] = ident

    # --- the two demo looks --------------------------------------------
    looks = {}
    for name in DEMOS:
        s = spec.load_look(name)
        entry = {"validate": validate_block(s), "declared": DECLARED[name]}
        for order in ORDERS:
            c = pipeline.compile(dataclasses.replace(s, order=order), strict=False)
            d = pipeline.diagnostics(c)
            entry[order] = {
                "d2_dark_p99_9": _p(_dark_d2(pipeline.lattice(c, 33)), 2),
                "grey_err_codes": _p(d["grey"]["max_err_codes"], 5),
                "grey_err_at_code": _p(d["grey"]["max_err_at_code"], 1),
                "white_err": _p(d["grey"]["white_err"], 15),
                "grey_monotone_per_channel": d["grey"]["monotone_per_channel"],
                "grey_min_step_codes": _p(d["grey"]["min_step_codes"], 5),
                "d2_full": {k: _p(v, 3) for k, v in d["d2"]["full"].items()},
                "d2_interior": {k: _p(v, 3) for k, v in d["d2"]["interior"].items()},
                "fold_100": d["fold"],
                "fold_70": d["fold_blend70"],
                "clip": {k: _p(v, 5) for k, v in d["clip"].items()},
                "nm": {k: (_p(v, 4) if isinstance(v, float) else v)
                       for k, v in d.get("nm", {}).items() if k != "gamut"},
                "grid_error": {k: (_p(v, 4) if isinstance(v, float) else v)
                               for k, v in d["grid_error"].items()},
                "lattice_time_s": _p(d["lattice_time_s"], 4),
            }
        c = pipeline.compile(s, strict=False)
        entry["fingerprint_measured"] = probe_fingerprint(c)
        entry["fingerprint_gamut_off"] = probe_fingerprint(
            pipeline.compile(dataclasses.replace(s, gamut=None), strict=False))
        entry["neutral_measured"] = measured_tint(c)
        entry["eps_dark_sweep"] = eps_dark_sweep(s)
        entry["ablation_NPG"] = ablations(s)
        entry["isolation_NPG"] = isolations(s)
        entry["chroma_headroom"] = chroma_headroom(s)
        entry["strength_ladder"] = strength_ladder(s)
        looks[name] = entry
    rep["looks"] = looks

    rep["gamut_alone"] = gamut_alone()
    rep["reference_calibration"] = reference_calibration()
    rep["review_regressions"] = review_regressions()

    # --- gate summary ---------------------------------------------------
    gates = {}
    for order in ORDERS:
        gates[f"identity {order}"] = {
            "value": rep["identity"][order]["err_33"], "gate": "< 1e-6",
            "pass": rep["identity"][order]["err_33"] < 1e-6}
    for name, e in rep["looks"].items():
        for order in ORDERS:
            o = e[order]
            gates[f"{name} {order} grey err"] = {
                "value": o["grey_err_codes"], "gate": "< 0.02 codes",
                "pass": o["grey_err_codes"] < 0.02}
            gates[f"{name} {order} neg tetra 100%"] = {
                "value": o["fold_100"]["neg_count"], "gate": "== 0",
                "pass": o["fold_100"]["neg_count"] == 0}
            gates[f"{name} {order} neg tetra 70%"] = {
                "value": o["fold_70"]["neg_count"], "gate": "== 0",
                "pass": o["fold_70"]["neg_count"] == 0}
            gates[f"{name} {order} d2 p99.9 full"] = {
                "value": o["d2_full"]["p99_9"], "gate": "<= 6",
                "pass": o["d2_full"]["p99_9"] <= 6.0}
            gates[f"{name} {order} d2 p99.9 interior"] = {
                "value": o["d2_interior"]["p99_9"], "gate": "<= 4",
                "pass": o["d2_interior"]["p99_9"] <= 4.0}
            gates[f"{name} {order} nm p99.9"] = {
                "value": o["nm"]["p99_9"], "gate": "<= 1.15",
                "pass": o["nm"]["p99_9"] <= 1.15}
            gates[f"{name} {order} clip excursion"] = {
                "value": o["clip"]["excursion_below_codes"], "gate": "< 1 code",
                "pass": o["clip"]["excursion_below_codes"] < 1.0}
            ge = o["grid_error"]
            gates[f"{name} {order} grid err mean"] = {
                "value": ge["mean"], "gate": "<= 0.35 codes (R06 §4)",
                "pass": ge["mean"] <= 0.35}
            gates[f"{name} {order} grid err p99"] = {
                "value": ge["p99"], "gate": "<= 1.6 codes (R06 §4)",
                "pass": ge["p99"] <= 1.6}
            gates[f"{name} {order} grid err max"] = {
                "value": ge["max"], "gate": "<= 5.0 codes (R06 §4)",
                "pass": ge["max"] <= 5.0}
        gates[f"{name} validate()"] = {
            "value": "rejected" if e["validate"]["errors"] else "accepted",
            "gate": "accepted (ENGINE_SPEC §7 makes validate the gate)",
            "pass": not e["validate"]["errors"]}
    rep["gate_summary"] = gates
    rep["conclusions"] = conclusions(rep)
    return rep


def conclusions(rep: dict) -> dict:
    """Everything here is derived from the numbers above, not asserted."""
    out: dict = {}

    # eps_dark: best d2 subject to clip excursion < 1 code (ENGINE_SPEC §4)
    eps_pick = {}
    for name, e in rep["looks"].items():
        ok = {k: v for k, v in e["eps_dark_sweep"].items() if v["clip_below_codes"] < 1.0}
        if ok:
            best = min(ok, key=lambda k: (ok[k]["d2_full_p99_9"], ok[k]["neg_tetra"]))
            eps_pick[name] = {"choice": float(best), "feasible": sorted(float(k) for k in ok),
                              "d2_full": ok[best]["d2_full_p99_9"],
                              "d2_dark": ok[best]["d2_dark_p99_9"],
                              "neg_tetra": ok[best]["neg_tetra"]}
        else:
            eps_pick[name] = {"choice": None, "feasible": []}
    out["eps_dark"] = eps_pick

    # stage order: which one measures better, per look
    order_pick = {}
    for name, e in rep["looks"].items():
        a, b = e["NPG"], e["PGN"]
        order_pick[name] = {
            "d2_full": {"NPG": a["d2_full"]["p99_9"], "PGN": b["d2_full"]["p99_9"]},
            "d2_interior": {"NPG": a["d2_interior"]["p99_9"], "PGN": b["d2_interior"]["p99_9"]},
            "neg_tetra": {"NPG": a["fold_100"]["neg_count"], "PGN": b["fold_100"]["neg_count"]},
            "clip_below": {"NPG": a["clip"]["excursion_below_codes"],
                           "PGN": b["clip"]["excursion_below_codes"]},
            "grey_err": {"NPG": a["grey_err_codes"], "PGN": b["grey_err_codes"]},
            "better_on_d2_and_folds":
                "PGN" if (b["d2_full"]["p99_9"] < a["d2_full"]["p99_9"]
                          and b["fold_100"]["neg_count"] <= a["fold_100"]["neg_count"])
                else "NPG",
        }
    out["order"] = order_pick

    # gamut knee: the softest knee that keeps [G] alone fold-free
    ga = rep["gamut_alone"]
    fold_free = [k for k, v in ga.items() if k != "null" and v["neg_tetra"] == 0]
    knees = sorted((float(k.split()[1]), v) for k, v in ga.items() if k != "null")
    boundary = max((kn for kn, v in knees if v["neg_tetra"] == 0), default=None)
    out["gamut_knee"] = {
        "identity_plus_G_fold_free": fold_free,
        "largest_fold_free_knee": boundary,
        "default_0.70_0.30_neg_tetra": ga["knee 0.70 / soft 0.30"]["neg_tetra"],
        "default_0.70_0.30_d2_full": ga["knee 0.70 / soft 0.30"]["d2_full_p99_9"],
        "declaring_G_costs_on_an_identity_look": {
            k: {"max_codes": v.get("cost_vs_identity_max_codes"),
                "mean_codes": v.get("cost_vs_identity_mean_codes"),
                "d2_interior": v["d2_int_p99_9"]}
            for k, v in ga.items() if k != "null"},
        "radial_gain_at_cube_corner":
            {k: v.get("radial_gain_at_cube_corner") for k, v in ga.items() if k != "null"},
        "note": "ENGINE_SPEC §4 still states knee 0.70 / soft 0.30, so the engine "
                "default is unchanged and validate() warns instead; the measured "
                "boundary is a LEAD decision (0.65/0.35 costs ~0.4 codes more mean "
                "chroma and has ~4x the fold margin of 0.69)",
    }

    # grid error (R06 §4) — the shipped table vs the continuous engine
    out["grid_error"] = {
        name: {order: e[order]["grid_error"] for order in rep["looks"][name]
               if order in ORDERS}
        for name, e in rep["looks"].items()}

    # look strength the gates can carry
    out["strength_ceiling"] = {
        name: {k: v["largest_passing_scale"] for k, v in e["strength_ladder"].items()}
        for name, e in rep["looks"].items()
    }

    # chroma over-drive at the pinned probe
    out["chroma_overdrive_at_probe"] = {}
    for name, e in rep["looks"].items():
        r = [e["fingerprint_measured"]["cr"][i] / e["declared"]["cr"][i] for i in range(12)]
        out["chroma_overdrive_at_probe"][name] = {
            "min": _p(min(r)), "median": _p(float(np.median(r))), "max": _p(max(r))}

    # d2 caliber: is "interior instead of full" a fix?  Not for Glaze it isn't.
    out["d2_caliber"] = {
        name: {order: {"full_p99_9": e[order]["d2_full"]["p99_9"],
                       "interior_p99_9": e[order]["d2_interior"]["p99_9"],
                       "interior_over_full":
                           _p(e[order]["d2_interior"]["p99_9"] / e[order]["d2_full"]["p99_9"])}
               for order in ORDERS}
        for name, e in rep["looks"].items()}
    out["d2_caliber"]["note"] = (
        "switching ENGINE_SPEC §8's d2 gate from the full lattice to the interior "
        "mask helps Arcade (ratio ~0.8) and makes Glaze WORSE (ratio > 1): Glaze's "
        "curvature is in the interior, not on the shell.  The 'interior is only "
        "1.45-2.77 codes' number is [N]+[G] with the field switched off, not a "
        "demo look.  If the gate moves it should be the full-lattice threshold "
        "that moves, not the caliber")

    out["reference_lut_gate_status"] = {
        k: {"d2_full_le_6": (v.get("d2_full_p99_9") or 1e9) <= 6.0,
            "zero_neg_tetra": v.get("neg_tetra") == 0,
            "zero_strict_folds": v.get("neg_strict") == 0}
        for k, v in rep["reference_calibration"].items() if "error" not in v}
    return out


# ---------------------------------------------------------------------------


def _row(label, q, w=26):
    return (f"{label:<{w}} {q['d2_full_p99_9']:>8} {q['d2_int_p99_9']:>8} "
            f"{q['d2_dark_p99_9']:>7} {q['neg_tetra']:>8} {str(q['nm_p99_9']):>7} "
            f"{str(q['clip_below_codes']):>8}")


def print_table(rep: dict) -> None:
    W = 26
    print("=" * 96)
    print("LATENT-2026 ENGINE — MILESTONE 1 (N + P + G)   ", rep["generated"])
    print("=" * 96)
    print("\nIDENTITY LOOK (gate < 1e-6)")
    for order, v in rep["identity"].items():
        print(f"  {order}: 33^3 max|out-in| = {v['err_33']:.3e}   17^3 = {v['err_17']:.3e}")

    hdr = (f"{'':<{W}} {'d2 p99.9':>8} {'d2 int':>8} {'d2 drk':>7} {'negtet':>8} "
           f"{'nm p99.9':>7} {'clip<0':>8}")
    for name, e in rep["looks"].items():
        print(f"\n{'-' * 96}\n{name}   (declared numbers: PLAN appendix A, typed verbatim)")
        v = e["validate"]
        if v["errors"]:
            print("  validate() REJECTS this look:")
            for x in v["errors"]:
                print(f"     ! {x.strip()}")
        else:
            print("  validate(): passes")
        for x in v["warnings"]:
            print(f"     ~ {x}")
        print(f"  hue fold bound min(1+dDh/dh0) = {v['hue_fold_bound']['min_1_plus_ddh']} "
              f"at {v['hue_fold_bound']['at']}")
        print(f"     dense-grid check: {v['hue_fold_bound']['dense_min']} at "
              f"{v['hue_fold_bound']['dense_at']}")
        print(f"  total chroma gain {v['total_chroma_gain']['min']}..."
              f"{v['total_chroma_gain']['max']} at {v['total_chroma_gain']['at']} "
              f"(budget {v['total_chroma_gain']['design_budget']}, hard "
              f"{v['total_chroma_gain']['hard_bound']})")
        print(f"\n  {'order':<{W}} {'grey err':>9} {'white err':>10} {'lattice s':>10} "
              f"{'grid mean':>10} {'grid p99':>9} {'grid max':>9}")
        for order in ORDERS:
            o = e[order]
            print(f"  {order:<{W}} {o['grey_err_codes']:>9} {o['white_err']:>10.2e} "
                  f"{o['lattice_time_s']:>10} {o['grid_error']['mean']:>10} "
                  f"{o['grid_error']['p99']:>9} {o['grid_error']['max']:>9}")
        print(f"\n  {hdr}")
        for order in ORDERS:
            o = e[order]
            print(f"  {order:<{W}} {o['d2_full']['p99_9']:>8} {o['d2_interior']['p99_9']:>8} "
                  f"{o['d2_dark_p99_9']:>7} {o['fold_100']['neg_count']:>8} "
                  f"{o['nm']['p99_9']:>7} {o['clip']['excursion_below_codes']:>8}")
            print(f"  {'  (70 % blend)':<{W}} {'':>8} {'':>8} {'':>7} "
                  f"{o['fold_70']['neg_count']:>8}")

        print(f"\n  eps_dark sweep (NPG, ENGINE_SPEC §4)\n  {hdr}")
        for k, q in e["eps_dark_sweep"].items():
            print(_row("  eps_dark = " + k, q, W + 2))

        print(f"\n  operator ablation — each one switched OFF (NPG)\n  {hdr}")
        for k, q in e["ablation_NPG"].items():
            print(_row("  " + k, q, W + 2))

        print(f"\n  operator isolation — each one ALONE on [N]+[G] (NPG)\n  {hdr}")
        for k, q in e["isolation_NPG"].items():
            print(_row("  " + k, q, W + 2))

        print(f"\n  chroma headroom: CR/arch/vibrance scaled toward 1 by a\n  {hdr}")
        for k, q in e["chroma_headroom"].items():
            print(_row("  a = " + k, q, W + 2))

        print("\n  hue fold bound vs the proposed skin fixes   pinned probe / dense grid")
        for k, q in e["validate"]["fold_bound_vs_fix"].items():
            flag = "" if q["dense"] >= 0.3 else "   <- still folds"
            print(f"     {k:<34} {q['pinned']:>8} {q['dense']:>8}{flag}")

        for gk, rows in e["strength_ladder"].items():
            best = rows["largest_passing_scale"]
            print(f"\n  strength ladder: whole [P] scaled toward identity — {gk}")
            print(f"  {hdr} {'gates':>6}")
            for k, q in rows.items():
                if k == "largest_passing_scale":
                    continue
                print(_row("  scale = " + k, q, W + 2)
                      + f" {'ok' if q['pass'] else 'FAIL':>6}")
            print(f"     largest [P] scale that clears folds + d2 + clip: {best}")

        fp = e["fingerprint_measured"]
        fg = e["fingerprint_gamut_off"]
        dec = e["declared"]
        print("\n  measured vs declared at the pinned probe L = 0.65, C = 0.10")
        print(f"  {'h0':>5} {'cr decl':>8} {'cr meas':>8} {'cr noG':>8} "
              f"{'dh10 decl':>10} {'dh meas':>8} {'dh noG':>8}")
        for i in range(12):
            print(f"  {i * 30:>5} {dec['cr'][i]:>8} {fp['cr'][i]:>8} {fg['cr'][i]:>8} "
                  f"{dec['dh10'][i]:>10} {fp['dh'][i]:>8} {fg['dh'][i]:>8}")
        rat = [fp["cr"][i] / dec["cr"][i] for i in range(12)]
        print(f"     measured/declared chroma ratio: min {min(rat):.3f} "
              f"median {float(np.median(rat)):.3f} max {max(rat):.3f}")
        nm = e["neutral_measured"]
        print(f"\n  grey target T: tone out for input {TONE_IN}")
        print(f"     declared {dec['tone']}")
        print(f"     measured {nm['tone_out']}")
        print(f"     R-G declared {dec['tint_rg']}  measured {nm['tint_rg']}")
        print(f"     B-G declared {dec['tint_bg']}  measured {nm['tint_bg']}")

    print(f"\n{'-' * 96}\n[G] ALONE on an identity look — what the compressor itself costs")
    print(f"  {hdr}")
    for k, q in rep["gamut_alone"].items():
        extra = q.get("radial_gain_at_cube_corner")
        print(_row("  " + k, q, W + 2) + (f"   radial gain @nm=1.316: {extra}" if extra else ""))

    print(f"\n{'-' * 96}\nREVIEW REGRESSIONS — what each confirmed finding does NOW")
    for k, v in rep["review_regressions"].items():
        if k == "measurements":
            continue
        print(f"  {k}")
        print(f"     spec:   {v['spec']}")
        print(f"     before: {v['before']}")
        extra = (f"   grey err now {v['grey_err_codes_now']} codes"
                 if "grey_err_codes_now" in v else "")
        print(f"     now:    {v['now']}{extra}")
    for k, v in rep["review_regressions"]["measurements"].items():
        print(f"  {k}\n     before: {v.get('before')}\n     now:    {v['now']}")

    print(f"\n{'-' * 96}\nGATE CALIBRATION — the same metrics on SHIPPED reference .cube files")
    print(f"  {'':<26} {'d2 p99.9':>8} {'d2 int':>8} {'d2 drk':>7} {'negtet':>8} "
          f"{'strict':>7} {'==0':>8} {'==1':>8}")
    for k, q in rep["reference_calibration"].items():
        if "error" in q:
            print(f"  {k:<26} {q['error']}")
            continue
        print(f"  {k:<26} {q['d2_full_p99_9']:>8} {q['d2_int_p99_9']:>8} "
              f"{q['d2_dark_p99_9']:>7} {q['neg_tetra']:>8} {q['neg_strict']:>7} "
              f"{q['zero_frac']:>8} {q['one_frac']:>8}")

    print(f"\n{'-' * 96}\nGATE SUMMARY")
    for k, v in rep["gate_summary"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k:<44} {str(v['value']):>12}  "
              f"gate {v['gate']}")
    n_fail = sum(1 for v in rep["gate_summary"].values() if not v["pass"])
    print(f"  {n_fail} of {len(rep['gate_summary'])} gates fail.")

    cc = rep["conclusions"]
    print(f"\n{'-' * 96}\nDERIVED CONCLUSIONS (all computed from the tables above)")
    for name, v in cc["eps_dark"].items():
        print(f"  eps_dark for {name}: {v['choice']}  (values with clip < 1 code: "
              f"{v['feasible']})")
    for name, v in cc["order"].items():
        print(f"  order for {name}: {v['better_on_d2_and_folds']} measures better — "
              f"d2 full {v['d2_full']}, neg tetra {v['neg_tetra']}")
    print(f"  [G] alone is fold-free at: {cc['gamut_knee']['identity_plus_G_fold_free']}")
    print(f"  largest fold-free knee (soft = 1-knee): {cc['gamut_knee']['largest_fold_free_knee']}")
    for name, v in cc["d2_caliber"].items():
        if name == "note":
            continue
        print(f"  d2 interior/full for {name}: "
              + ", ".join(f"{o} {v[o]['interior_over_full']}" for o in ORDERS))
    for name, v in cc["grid_error"].items():
        print(f"  grid error {name}: "
              + ", ".join(f"{o} mean {v[o]['mean']} / p99 {v[o]['p99']} / max {v[o]['max']}"
                          for o in ORDERS))
    print(f"  [G] at the §4 default (0.70/0.30) leaves {cc['gamut_knee']['default_0.70_0.30_neg_tetra']}"
          f" folded tetrahedra and d2 {cc['gamut_knee']['default_0.70_0.30_d2_full']} on an "
          f"otherwise IDENTITY look")
    for name, v in cc["strength_ceiling"].items():
        print(f"  largest [P] scale of {name} that clears folds+d2+clip: {v}")
    for name, v in cc["chroma_overdrive_at_probe"].items():
        print(f"  {name} delivers {v['median']}x the declared chroma at L=.65,C=.10 "
              f"(min {v['min']}, max {v['max']})")
    print("=" * 96)


def main() -> None:
    raise SystemExit(
        "engine/report_m1.py is the ENGINE v1.0 milestone-1 generator and does not "
        "run against v1.1: ENGINE_SPEC_v1_1 R2 removed gamut.soft / gamut.eps_dark, "
        "which this module sweeps.  It is kept only as the provenance of "
        "work.nosync/review/m1_engine_report.json.\n"
        "Use:  $R/py -m engine.report_m1b"
    )


def _main_v1_0() -> None:
    t0 = time.perf_counter()
    rep = build()
    rep["build_time_s"] = round(time.perf_counter() - t0, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print_table(rep)
    print(f"\nwritten: {OUT}   ({rep['build_time_s']} s)")


if __name__ == "__main__":
    main()
