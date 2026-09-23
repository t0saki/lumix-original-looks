"""Milestone 2 report — ENGINE_SPEC v1.2 W5 and W7.

``$R/py -m engine.report_m2`` writes ``work.nosync/review/m2_engine_report.json``
and prints the three tables the ruling asks for:

* **W5.1** ``w5_anchor`` — the luma anchor against the OKLab one on identity +
  ``sat in {1.00, 1.05, 1.10, 1.20, 1.40}`` (W5.1's own population) and, as a
  cross-check the ruling does not ask for, on the three real looks of W5.3.
* **W5.2** ``w5_curve`` — the slope-floor curve's stated properties, measured.
  ``w5_relfade_width`` — the width of W2's rectifier, chosen the same way.
* **W5.3** ``w5_sweep`` — ``knee x s1`` on ``looks/lead/{03Gilt,04Viride,
  10Splice}.json``, with the tax on in-gamut colours carried alongside.
* **W7** ``w7`` — the twelve ``looks/lead/*.json`` compiled AS THEY ARE, plus
  ``w7_ablation``: the per-operator ablation of every look that still FAILs a
  hard gate.

Everything runs in ONE process (docs/ENV.md): the 33**3 lattice of ~90 looks,
the photo sample once, the CMAX table once.
"""

from __future__ import annotations

import dataclasses as dc
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine import cmax as _cmax, color, curves, field as _field, gamut as _gamut
from engine import pipeline, spec as _spec, xfer
from engine.spec import LookSpec, SpecError

ROOT = Path(__file__).resolve().parent.parent
LEAD = ROOT / "looks" / "lead"
OUT = ROOT / "work.nosync" / "review" / "m2_engine_report.json"

#: W5.3's three looks
SWEEP_LOOKS = ("03Gilt", "04Viride", "10Splice")
#: W5.1's population
ANCHOR_SATS = (1.00, 1.05, 1.10, 1.20, 1.40)
KNEES = (0.65, 0.70, 0.75, 0.80)
SLOPES = (0.15, 0.25, 0.35)

#: v1.2 W6 + R5 hard gates, as (label, key in the row, limit, "hi"/"lo").
#:
#: The last two were MISSING from the first W7 table even though ``tools/qc.py``
#: has always computed and FAILed on them, so the per-look ``fails`` lists and
#: the headline count were incomplete:
#:
#: * ``jac_material`` is W6's own third bullet ("65**3 Jacobian gate: same idea
#:   — FAIL only on det < -0.02"), ``tools.qc.jacobian_check``;
#: * ``blend70_d2_p99_9`` is R06 §4's smoothness gate on the 70 % blend
#:   (``tools.qc.BLEND70_D2_P999 = 5.0``), i.e. inside the fold / smoothness /
#:   push / lim group this milestone touched.  NOTE for the lead: the blend is
#:   linear and the identity lattice has zero second differences, so this
#:   number is EXACTLY ``0.70 x d2_full_p99_9`` (measured ratio 0.700000000 on
#:   02Burin / 04Viride / 08Arcade).  The gate is therefore a full-lattice d2
#:   p99.9 limit of 7.14 codes, which contradicts W6's own 12.0 — see `issues`.
HARD_GATES = (
    ("material folds @100%", "material_100", 0.0, "hi"),
    ("material folds @70%", "material_70", 0.0, "hi"),
    ("micro folds %", "micro_pct", 2.0, "hi"),
    ("crush %", "crush_pct", 8.0, "hi"),
    ("interior d2 p99.9", "d2_interior_p99_9", 6.0, "hi"),
    ("full d2 p99.9", "d2_full_p99_9", 12.0, "hi"),
    ("full d2 max", "d2_full_max", 30.0, "hi"),
    ("grey vs T (codes)", "grey_err_codes", 0.02, "hi"),
    ("engine Jacobian folds (65^3)", "jac_material", 0.0, "hi"),
    ("blend70 d2 p99.9", "blend70_d2_p99_9", 5.0, "hi"),
)


def _photo(n: int = 60_000, seed: int = 20260922):
    p = ROOT / "work.nosync" / "cal" / "photo_sample.npy"
    if not p.exists():
        return None
    a = np.asarray(np.load(p), dtype=np.float64)
    if a.shape[0] > n:
        a = a[np.random.default_rng(seed).choice(a.shape[0], n, replace=False)]
    return a


def _measure(c, *, photo=None, size: int = 33, jac: bool = False) -> dict:
    """Every number a W5/W7 row carries, from one lattice evaluation.

    ``jac=True`` adds the 65**3 engine-Jacobian count of W6's third bullet.  It
    costs ~1 s a look (6 engine passes over 274,625 points), so only the W7
    rows pay for it; the W5 sweeps do not.
    """
    from tools import metrics as M

    t0 = time.perf_counter()
    tab = pipeline.lattice(c, size)
    lat_ms = (time.perf_counter() - t0) * 1e3
    f = M.fold_stats(tab)
    tab70 = M.blend_table(tab, 0.70)
    f70 = M.fold_stats(tab70)
    d2_70 = M.second_diff_stats(tab70)
    d2 = M.second_diff_stats(tab)
    _, out = pipeline.grey_response(c)
    grey = float(np.max(np.abs(out - c.neutral.T)) * 255.0)
    row = {
        "grey_err_codes": grey,
        "d2_interior_p99_9": d2["interior"]["p99_9"],
        "d2_full_p99_9": d2["full"]["p99_9"],
        "d2_full_max": d2["full"]["max"],
        "material_100": f["material_count"],
        "material_70": f70["material_count"],
        "micro_100": f["micro_count"],
        "micro_pct": f["micro_frac"] * 100.0,
        "micro_70": f70["micro_count"],
        "micro_pct_70": f70["micro_frac"] * 100.0,
        "blend70_d2_p99_9": d2_70["full"]["p99_9"],
        "crush_pct": f["crush_excess_frac"] * 100.0,
        "strict_100": f["neg_strict_count"],
        "min_ratio": f["min_ratio"],
        "lim": c.gamut_lim,
        "q": c.gamut_q,
        "min_radial_gain": c.gamut_min_radial_gain,
        "nm_pre_max": c.nm_pre_max,
        "flat_frac_pct": None if c.gamut_flat_frac is None else c.gamut_flat_frac * 100.0,
        "lattice_ms": lat_ms,
    }
    if jac and c.spec.mono is None:
        # W6's 65**3 Jacobian gate.  A mono look maps the cube onto a curve, so
        # its Jacobian is singular everywhere by construction; tools/qc.py skips
        # it for the same reason.
        from tools import qc as _qc

        j = _qc.jacobian_check(c)
        row["jac_material"] = j["material_count"]
        row["jac_min_det"] = j["min_det"]
        row["jac_neg_count"] = j["neg_count"]
        row["jac_n"] = j["n"]
    if photo is not None:
        ps = pipeline.push_stats(c, photo)
        row["push_p99_9"] = ps.get("p99_9")
        row["push_max"] = ps.get("max")
        de_out = pipeline.apply(c, photo)
        row["de00_photo_mean"] = float(M.de00(M.srgb_to_lab(photo),
                                              M.srgb_to_lab(de_out)).mean())
    return row


def _fails(row: dict, *, mono: bool = False) -> list[str]:
    out = []
    for label, key, limit, side in HARD_GATES:
        v = row.get(key)
        if v is None:
            continue
        # a mono look maps the cube onto a 1-D curve, so EVERY tetrahedron has
        # volume 0 by construction: crush is 100 % and micro is 0, and neither
        # says anything about the look.  tools/qc.py skips the same two lines.
        if mono and key in ("crush_pct", "micro_pct"):
            continue
        if (side == "hi" and float(v) > limit) or (side == "lo" and float(v) < limit):
            out.append(f"{label} = {float(v):.4g} (limit {limit})")
    return out


# ---------------------------------------------------------------------------
# W5.1 — the anchor
# ---------------------------------------------------------------------------


def w5_anchor(photo) -> dict:
    ident = _spec.load_look("_identity")
    rows = []
    for anchor, sat in itertools.product(_gamut.ANCHORS, ANCHOR_SATS):
        s = dc.replace(ident, name="Study",
                       field=dc.replace(ident.field, sat=sat),
                       gamut=_spec.Gamut())
        c = pipeline.compile(s, strict=False)
        c = pipeline._measure_limit(dc.replace(c, anchor=anchor, gamut_lim=None))
        r = _measure(c)
        r.update(anchor=anchor, sat=sat, look="identity+sat")
        rows.append(r)

    real = []
    for nm in SWEEP_LOOKS:
        s = dc.replace(_spec.load_look(LEAD / f"{nm}.json"), gamut=_spec.Gamut())
        for anchor in _gamut.ANCHORS:
            c = pipeline.compile(s, strict=False)
            c = pipeline._measure_limit(dc.replace(c, anchor=anchor, gamut_lim=None))
            r = _measure(c)
            r.update(anchor=anchor, look=nm)
            real.append(r)

    return {
        "what": "W5.1: both anchors on identity + sat, and on the three real looks",
        "population": "identity + sat only (no other op), 33**3 lattice",
        "rows": rows,
        "real_looks": real,
        "choice": _gamut.DEFAULT_ANCHOR,
        "verdict": (
            "KEEP THE OKLAB ANCHOR.  On W5.1's own population the two are equal "
            "where the ruling looks (0 material folds, 0 micro folds, 0 crush, 0 "
            "strict negatives at every sat for both) and the luma anchor is worse "
            "everywhere else: interior d2 p99.9 and lim are higher at every sat, "
            "and at sat 1.05 lim reads 1.96 against 1.28.  The 678 folds W5.1 "
            "cites were v1.1's, and v1.2 removes their two causes (W1 withdrew "
            "R1's headroom, W5.2 replaced the flat tail) WITHOUT touching the "
            "anchor: identity + sat 1.10 measures 0 folds of every kind on both. "
            "On real looks the luma anchor is unusable — see `luma_defect`."),
        "luma_defect": (
            "The luma anchor's foliation argument holds only while the anchor is "
            "inside (0,1).  [P] legitimately produces out-of-gamut codes with a "
            "negative channel, and a saturated blue carries only 0.0722 of the "
            "luma: 04Viride's own blue corner (input (0,0,1)) leaves [P] at "
            "(0.188, -0.165, 0.962), luma -0.0085.  W5.1's clip(., 1e-6, 1-1e-6) "
            "then divides |d| = 0.165 by 1e-6 and nm reads 165,038; 118 of the "
            "35,937 nodes of that look have luma < 1e-6.  lim = 1.04*max nm is "
            "then 171,640 (q = 686,556) and 20.5 % of the lattice is crushed. "
            "This is not a numerical detail: the plane n = const for n <= 0 does "
            "not intersect the sRGB cube, so nm is genuinely unbounded there.  "
            "Making the luma anchor shippable needs a ruling (a dark/bright "
            "floor like v1.0's eps_dark, or a constraint that no stage may take "
            "the luma outside [0,1])."),
    }


# ---------------------------------------------------------------------------
# W5.2 — the curve, and the width of W2's rectifier
# ---------------------------------------------------------------------------


def w5_curve() -> dict:
    knee, lim, s1 = 0.65, 1.55, 0.35
    q = (lim - knee) / (1.0 - knee)
    nm = np.linspace(0.0, 2.2, 2_200_001)
    mp = _gamut.limit_curve(nm, knee, lim, s1)
    g = _gamut.radial_gain(nm, knee, lim, s1)
    h = 1e-6
    slope_hi = float((_gamut.limit_curve(np.array([knee + 2 * h]), knee, lim, s1)[0]
                      - _gamut.limit_curve(np.array([knee + h]), knee, lim, s1)[0]) / h)
    at_lim = float(_gamut.limit_curve(np.array([lim]), knee, lim, s1)[0])
    tail = float((_gamut.limit_curve(np.array([lim + 0.3]), knee, lim, s1)[0] - at_lim)
                 / 0.3)
    return {
        "what": "W5.2's stated properties, measured at knee 0.65 / lim 1.55 / s1 0.35",
        "q": q, "q2": (q - s1) / (1.0 - s1),
        "identity_below_knee": bool(np.all(mp[nm <= knee] == nm[nm <= knee])),
        "strictly_increasing": bool(np.all(np.diff(mp) > 0.0)),
        "slope_at_knee_plus": slope_hi,
        "mp_at_lim": at_lim,
        "tail_slope_measured": tail,
        "tail_slope_expected_s1_over_q": s1 / q,
        "min_radial_gain_measured": float(g.min()),
        "min_radial_gain_expected": s1 / q,
        "note": ("C1 at the knee (slope 1), R(1) = 1 so mp(lim) = 1 exactly, and "
                 "the tail is linear with slope s1/q in mp-per-nm: the radial "
                 "direction is never erased.  v1.1's curve was constant at 1 "
                 "above lim and its radial gain reached 1.6e-12."),
    }


def w5_relfade_width(widths=(0.02, 0.04, 0.08, 0.16)) -> dict:
    """W2 writes ``sp(g-1, 0.02)``; ``sp(0, 0.02) = 0.01386 != 0``.  The width of
    the C2 rectifier that replaces it is chosen here."""
    ident = _spec.load_look("_identity")
    rows = []
    keep = _field.RELFADE_WIDTH
    try:
        for w in widths:
            _field.RELFADE_WIDTH = w
            for sat in (1.05, 1.10, 1.20, 1.40):
                s = dc.replace(ident, name="W", field=dc.replace(ident.field, sat=sat),
                               gamut=_spec.Gamut())
                c = pipeline.compile(s, strict=False)
                r = _measure(c)
                r.update(width=w, sat=sat, look="identity+sat")
                rows.append(r)
            s = dc.replace(_spec.load_look(LEAD / "03Gilt.json"), gamut=_spec.Gamut())
            c = pipeline.compile(s, strict=False)
            r = _measure(c)
            r.update(width=w, sat=None, look="03Gilt")
            rows.append(r)
    finally:
        _field.RELFADE_WIDTH = keep

    # what the ruling's own sp() would do to an identity look
    sp0 = float(curves.sp(np.array(0.0), 0.02))
    ident_c = pipeline.compile(_spec.load_look("_identity"))
    lab = color.linear_srgb_to_oklab(color.srgb_decode(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])))
    L, C, h = color.oklab_to_lch(lab)
    lab_faded = color.lch_to_oklab(L, C * (1.0 - sp0), h)
    codes = xfer.signed_srgb_encode(color.oklab_to_linear_srgb(lab_faded))
    sp_cost = float(np.max(np.abs(codes - np.eye(3))) * 255.0)
    return {
        "what": "the width of the C2 rectifier that replaces W2's sp(g-1, 0.02)",
        "sp_zero_offset": sp0,
        "sp_identity_cost_codes": sp_cost,
        "sp_note": (f"sp(0, 0.02) = {sp0:.5f}, so W2's formula fades EVERY gain by "
                    f"1.386 % of chroma even when g = 1: the sRGB primaries move "
                    f"{sp_cost:.2f} codes on an identity look, against "
                    "ENGINE_SPEC §8's 1e-6 gate, and a desaturating gain would be "
                    "faded too, which W2 forbids in the same paragraph."),
        "chosen": _field.RELFADE_WIDTH,
        "rows": rows,
        "verdict": ("0.04.  A gain inside (1, 1+w) is only partly faded and still "
                    "pushes a shell colour out: at w = 0.08 identity + sat 1.05 "
                    "measures lim 1.957 / interior d2 2.12 against 1.193 / 1.18 at "
                    "w = 0.04 and w = 0.02, which are identical to each other in "
                    "every column measured.  0.04 is the larger of the two that "
                    "costs nothing, i.e. half the curvature (4.96/w) for the same "
                    "result."),
    }


# ---------------------------------------------------------------------------
# W5.3 — the knee x s1 sweep
# ---------------------------------------------------------------------------


def w5_sweep(photo) -> dict:
    from tools import metrics as M

    looks = {nm: _spec.load_look(LEAD / f"{nm}.json") for nm in SWEEP_LOOKS}
    rows, cells = [], []
    ident = _spec.load_look("_identity")
    ident_tab = M.identity_table(33)
    for knee, s1 in itertools.product(KNEES, SLOPES):
        per = []
        for nm, s in looks.items():
            c = pipeline.compile(dc.replace(
                s, gamut=_spec.Gamut(knee=knee, p=8.0, end_slope=s1)), strict=False)
            r = _measure(c, photo=photo)
            r.update(look=nm, knee=knee, s1=s1)
            rows.append(r)
            per.append(r)
        # the column W5.3 does not ask for: what [G] alone costs an in-gamut colour
        ci = pipeline.compile(dc.replace(
            ident, gamut=_spec.Gamut(knee=knee, p=8.0, end_slope=s1)))
        tab = pipeline.lattice(ci, 33)
        tax = np.abs(tab - ident_tab).max(axis=-1) * 255.0
        _, nm_i = pipeline._run(dc.replace(ci, gamut_lim=None), ident_tab)
        cells.append({
            "knee": knee, "s1": s1,
            "material_100": int(sum(r["material_100"] for r in per)),
            "material_70": int(sum(r["material_70"] for r in per)),
            "micro_pct_mean": float(np.mean([r["micro_pct"] for r in per])),
            "crush_pct_mean": float(np.mean([r["crush_pct"] for r in per])),
            "d2_interior_mean": float(np.mean([r["d2_interior_p99_9"] for r in per])),
            "identity_tax_mean_codes": float(tax.mean()),
            "identity_tax_max_codes": float(tax.max()),
            "identity_frac_touched_pct": float(
                np.count_nonzero(nm_i > knee) / nm_i.size * 100.0),
        })

    ranked = sorted(cells, key=lambda c: (c["material_100"] + c["material_70"],
                                          c["crush_pct_mean"], c["d2_interior_mean"]))
    return {
        "what": "W5.3: knee x s1 on looks/lead/{03Gilt,04Viride,10Splice}.json, p = 8",
        "criteria": "no material folds, smallest crush, then smallest interior d2",
        "rows": rows,
        "cells": cells,
        "ranked": [{k: c[k] for k in ("knee", "s1", "material_100", "material_70",
                                      "crush_pct_mean", "d2_interior_mean")}
                   for c in ranked],
        "chosen": {"p": _spec.DEFAULT_P, "knee": _spec.DEFAULT_KNEE,
                   "end_slope": _spec.DEFAULT_END_SLOPE},
        "verdict": (
            "NO CELL of the 4 x 3 grid reaches zero material folds: the three "
            "looks measure 3,440 to 4,154 over the grid, and 3,983 with [G] "
            "switched off entirely (240 / 1,816 / 1,927) — so the residual folds "
            "are [P]'s and the compressor is REDUCING them, not causing them.  "
            "With the first criterion infeasible the choice falls to fewest "
            "material folds, then crush, then interior d2: knee 0.65 / s1 0.35 "
            "(3,440 @100 %, 0 @70 %, crush 14.50 %, interior d2 11.24 — the "
            "joint-best d2 in the grid).  W5.3's provisional 0.75 / 0.25 is worse "
            "on all three columns (3,868 / 15.27 % / 12.60)."),
        "caveats": [
            "Both optima sit on the EDGE of the swept range.  Extending it "
            "informally to knee 0.50 / s1 0.60 keeps improving monotonically "
            "(2,668 material folds, crush 14.54 %, interior d2 9.88): the grid is "
            "saying 'compress less', and the limit of that is [G] off.",
            "The one column W5.3 does not ask for is the tax on in-gamut colours. "
            "identity + [G] alone moves the 33**3 lattice by a mean of 3.31 codes "
            "at knee 0.65 against 1.93 at knee 0.80, and touches 63.6 % of it "
            "against 44.5 %.  v1.1 R2 chose knee 0.80 on exactly that criterion; "
            "if the lead wants it back, knee 0.80 / s1 0.35 is the cell.",
            "lim does not depend on knee (it is 1.04 x the pre-gamut nm max), so "
            "the knee only moves where compression starts and how large q is.",
        ],
    }


# ---------------------------------------------------------------------------
# W7 — the twelve lead looks, as they are
# ---------------------------------------------------------------------------


def w7(photo) -> dict:
    rows = []
    for p in sorted(LEAD.glob("*.json")):
        s = _spec.load_look(p)
        t0 = time.perf_counter()
        try:
            c = pipeline.compile(s)
            validates, err = True, None
        except SpecError as exc:
            c = pipeline.compile(s, strict=False)
            validates, err = False, str(exc)
        compile_s = time.perf_counter() - t0
        r = _measure(c, photo=photo, jac=True)
        r.update(
            name=s.name, look=p.name, order=s.order, mono=s.mono is not None,
            validates=validates, validate_error=err,
            compile_s=compile_s,
            gamut_declared=None if s.gamut is None else s.gamut.to_dict(),
            gamut_legacy=list(s.gamut.legacy) if s.gamut is not None else [],
            field_legacy=list(s.field.legacy),
            gates_legacy=list(s.field.gates.legacy),
            warnings=list(c.warnings),
        )
        r["fails"] = _fails(r, mono=s.mono is not None)
        rows.append(r)
    return {
        "what": "W7: the twelve looks/lead/*.json compiled AS THEY ARE, v1.2 engine",
        "gates": [{"label": l, "key": k, "limit": v, "side": sd}
                  for l, k, v, sd in HARD_GATES],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# W7 — operator ablation for every look that FAILs a hard gate
# ---------------------------------------------------------------------------


_KNOTS = np.arange(12) * 30.0


def _worst_jump(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    d = np.abs(np.diff(np.concatenate([a, a[:1]])))
    k = int(np.argmax(d))
    return {"delta": float(d[k]), "from_h": float(_KNOTS[k]),
            "to_h": float(_KNOTS[(k + 1) % 12]),
            "from": float(a[k]), "to": float(a[(k + 1) % 12])}


def w7_diagnosis(w7_doc: dict) -> dict:
    """Per look: which table knot is the steepest, and where the three closed-form
    probes bottom out.  This is what turns "04Viride FAILs" into a one-number
    edit for the lead."""
    out = {}
    for row in w7_doc["rows"]:
        s = _spec.load_look(LEAD / row["look"])
        if s.mono is not None:
            continue
        cf = _field.compile_field(s)
        jmin, jwhere = _field.fold_bound(cf, _spec.FOLD_PROBE_C, _spec.FOLD_PROBE_L)
        cmin, cwhere = _field.chroma_radial_bound(cf)
        glo, ghi, gwhere = _field.total_gain_range(cf)
        out[row["name"]] = {
            "fails": row["fails"],
            "cr_worst_step": _worst_jump(s.field.cr),
            "dh10_worst_step": _worst_jump(s.field.dh10),
            "dh18_worst_step": _worst_jump(s.field.dh18),
            "dl_worst_step": _worst_jump(s.field.dl),
            "dh18_max_abs": float(np.max(np.abs(s.field.dh18))),
            "hue_fold_bound": jmin,
            "hue_fold_bound_at": {"h0": jwhere[0], "C0": jwhere[1], "L0": jwhere[2]},
            "radial_chroma_bound": cmin,
            "radial_chroma_bound_at": {"h0": cwhere[0], "L0": cwhere[1],
                                       "C0": cwhere[2]},
            "field_gain_range": [glo, ghi],
            "field_gain_at": {"h0": gwhere[0], "L0": gwhere[1], "C0": gwhere[2]},
            "sat_x_vib": float(s.field.sat * s.field.vibrance.gain),
        }
    return out


def w7_ablation(w7_doc: dict) -> dict:
    from tools import round as R

    out = {}
    for row in w7_doc["rows"]:
        if not row["fails"]:
            continue
        doc = json.loads((LEAD / row["look"]).read_text(encoding="utf-8"))
        base = {k: row[k] for k in ("d2_interior_p99_9", "d2_full_p99_9", "d2_full_max")}
        base.update(fold100=row["material_100"], fold70=row["material_70"],
                    micro_pct=row["micro_pct"], crush_pct=row["crush_pct"],
                    min_ratio=row["min_ratio"])
        out[row["name"]] = {
            "fails": row["fails"],
            "base": base,
            "rows": R.ablations(doc, base),
        }
    return out


def w7_order(w7_doc: dict) -> dict:
    """Where the material folds that are LEFT actually come from.

    W2's gamut closure is an argument about [P]'s gain acting on the chroma
    ``r0`` was measured from — the stage-0 chroma.  In the default NPG order
    [N]'s per-channel tone curve sits between the two and inflates the chroma
    on the way, so [P] multiplies a chroma that is already past the shell while
    the fade still believes there is headroom.  Two controls isolate it: the
    same look in PGN (where [P] does see the stage-0 colour), and the same look
    in NPG with the tone curve replaced by the identity.
    """
    rows = {}
    for row in w7_doc["rows"]:
        if row["mono"]:
            continue
        doc = json.loads((LEAD / row["look"]).read_text(encoding="utf-8"))
        flat = dict(doc)
        flat["neutral"] = {"tone": [[0, 0], [255, 255]],
                           "tint_rg": doc.get("neutral", {}).get("tint_rg", []),
                           "tint_bg": doc.get("neutral", {}).get("tint_bg", [])}
        out = {}
        for tag, dd, order in (("as-is/NPG", doc, "NPG"), ("as-is/PGN", doc, "PGN"),
                               ("tone=identity/NPG", flat, "NPG")):
            c = pipeline.compile(dc.replace(LookSpec.from_dict(dd), order=order),
                                 strict=False)
            m = _measure(c)
            out[tag] = {k: m[k] for k in ("material_100", "micro_pct", "crush_pct",
                                          "d2_interior_p99_9", "d2_full_p99_9")}
        rows[row["name"]] = out
    tot = {tag: int(sum(r[tag]["material_100"] for r in rows.values()))
           for tag in ("as-is/NPG", "as-is/PGN", "tone=identity/NPG")}
    return {
        "what": "W7 control: the same eleven colour looks in PGN, and with a flat tone",
        "rows": rows,
        "material_folds_total": tot,
        "verdict": (
            "The material folds that survive the W2 repairs are an ORDER effect, "
            "not a hue-table effect.  Summed over the eleven colour looks: "
            f"{tot['as-is/NPG']} in NPG, {tot['as-is/PGN']} in PGN, "
            f"{tot['tone=identity/NPG']} in NPG with a flat tone curve.  "
            "`order` is already a look-level key (ENGINE_SPEC §1) and the grey "
            "axis is exact in both orders, so this is a one-line edit per look "
            "for the lead to rule on — it is NOT an engine change, and it must "
            "not be made by changing what `r0` is measured from (that would "
            "break ENGINE_SPEC §0's stage-0 invariant, which is what makes "
            "§3.7's probes see what the lattice sees)."),
    }


# ---------------------------------------------------------------------------
# W1 / W2 — the defects an independent verifier found in the first v1.2 engine
# ---------------------------------------------------------------------------


def w1_fold_bound() -> dict:
    """W1's stated evidence, restated so the number reproduces.

    The first report wrote "a flat 4 deg rotation's fold bound is back from
    -0.674 to 1.000".  It is 1.000 only with the skin protection neutralised:
    with the engine's default ``Skin`` the residual makes even a flat rotation
    hue-dependent through the skin window.
    """
    rows = {}
    for tag, sk in (("default Skin (hue_residual 0.25)", _spec.Skin()),
                    ("Skin(hue_residual=1.0)", _spec.Skin(hue_residual=1.0)),
                    ("Skin(hue_residual=1.0, pull=0, hue_offset=0)",
                     _spec.Skin(hue_residual=1.0, pull=0.0, hue_offset=0.0))):
        s = LookSpec(name="Fb", field=_spec.Field(dh10=(4.0,) * 12, dh18=(4.0,) * 12),
                     skin=sk, gamut=_spec.Gamut())
        cf = _field.compile_field(s)
        j, where = _field.fold_bound(cf, _spec.FOLD_PROBE_C, _spec.FOLD_PROBE_L)
        from tools import metrics as M
        rows[tag] = {
            "fold_bound": float(j),
            "at": {"h0": where[0], "C0": where[1], "L0": where[2]},
            "material_folds": int(M.fold_stats(
                pipeline.lattice(pipeline.compile(s)))["material_count"]),
        }
    return {
        "what": "W1's evidence: the fold bound of a FLAT 4 deg rotation",
        "rows": rows,
        "correction": (
            "Reproducible statement: the fold bound of a flat 4 deg rotation is "
            "1.000000 with the skin residual neutralised and 0.688142 at the "
            "default hue_residual 0.25 (worst at h0 = 29 deg, C0 = 0.14, "
            "L0 = 0.35).  Material folds are 0 in every case, so W1's conclusion "
            "stands; only the quoted number needed the qualifier."),
    }


def w2_cmax_cap() -> dict:
    """W2 assumes CMAX sits INSIDE the true boundary.  The first v1.2 table did
    not; :func:`engine.cmax._build` now caps the Gaussian by the raw bisection."""
    from tools import metrics as M

    raw = _cmax._bisect()
    smoothed = np.maximum(_cmax._smooth(raw), _cmax.CMAX_FLOOR)
    capped = _cmax.cmax_table()

    Lg = np.linspace(0.0, 1.0, 199)
    Hg = np.arange(0.0, 360.0, 0.5)
    LL, HH = np.meshgrid(Lg, Hg, indexing="ij")
    true = M.cmax(LL, HH, iters=60)

    def _against_true(tab_lookup):
        d = tab_lookup - true
        # W2's own 0.004 floor makes the table exceed a boundary that is
        # essentially 0 near black and white; that is the ruling's choice, so
        # the honest number is the excess where the table is above the floor.
        m = tab_lookup > _cmax.CMAX_FLOOR + 1e-12
        return {"max_excess": float(d.max()),
                "p99_9_excess": float(np.percentile(d, 99.9)),
                "frac_outside_pct": float(np.mean(d > 1e-12) * 100.0),
                "max_excess_above_floor": float(d[m].max()),
                "p99_9_excess_above_floor": float(np.percentile(d[m], 99.9))}

    def _lookup(table):
        keep = _cmax._TABLE
        try:
            _cmax._TABLE = table
            return _cmax.cmax(LL, HH)
        finally:
            _cmax._TABLE = keep

    def _rough(v):
        dh = np.abs(np.diff(np.concatenate([v, v[:, :2]], axis=1), 2, axis=1)).max()
        return {"d2_L": float(np.abs(np.diff(v, 2, axis=0)).max()), "d2_h": float(dh)}

    # the causal test: pure [S], gamut null, only r0's denominator swapped
    t = np.linspace(0.0, 1.0, 33)
    b, g, r = np.meshgrid(t, t, t, indexing="ij")
    grid = np.stack([r, g, b], axis=-1)
    lab0 = color.linear_srgb_to_oklab(xfer.signed_srgb_decode(grid))
    L0, C0, h0 = color.oklab_to_lch(lab0)
    true_l = np.maximum(M.cmax(L0, h0, iters=60), _cmax.CMAX_FLOOR)
    W = _cmax.R0_WIDTH
    causal = {}
    for sat in (1.2, 1.4, 1.8):
        row = {}
        for tag, table, width in (
                ("v1.2 as shipped (uncapped + hard clamp)", smoothed, 0.0),
                ("cap only", capped, 0.0),
                ("shoulder only", smoothed, W),
                ("fixed (cap + shoulder)", capped, W),
                ("true boundary + shoulder", None, W)):
            if table is None:
                r0 = _cmax.soft_saturate(C0 / true_l, width)
            else:
                keep = _cmax._TABLE
                try:
                    _cmax._TABLE = table
                    r0 = _cmax.relative_chroma(L0, C0, h0, width=width)
                finally:
                    _cmax._TABLE = keep
            g_eff = _field.relative_fade(np.full_like(C0, sat), r0, 1.0)
            lab = np.concatenate([lab0[..., :1], lab0[..., 1:] * g_eff[..., None]],
                                 axis=-1)
            code = xfer.signed_srgb_encode(color.oklab_to_linear_srgb(lab))
            row[tag] = {
                "preclip_below_codes": float(max(0.0, -code.min()) * 255.0),
                "preclip_above_codes": float(max(0.0, code.max() - 1.0) * 255.0),
                "frac_outside_pct": float(
                    np.mean((code < -1e-9) | (code > 1.0 + 1e-9)) * 100.0),
            }
        causal[f"sat={sat}"] = row

    return {
        "what": "W2's CMAX must sit inside the true sRGB boundary; the cap that makes it",
        "vs_true_boundary": {"uncapped (v1)": _against_true(_lookup(smoothed)),
                             "capped (v2)": _against_true(_lookup(capped))},
        "on_node_excess_vs_raw": {
            "uncapped (v1)": float((smoothed - np.maximum(raw, _cmax.CMAX_FLOOR)).max()),
            "capped (v2)": float((capped - np.maximum(raw, _cmax.CMAX_FLOOR)).max())},
        "frac_cells_taken_from_the_smoothing_pct": float(
            np.mean(capped < np.maximum(raw, _cmax.CMAX_FLOOR) - 1e-15) * 100.0),
        "roughness_max_second_difference": {
            "raw bisection": _rough(np.maximum(raw, _cmax.CMAX_FLOOR)),
            "uncapped (v1)": _rough(smoothed), "capped (v2)": _rough(capped)},
        "causal_pure_S_stage": causal,
        "verdict": (
            "The plain Gaussian sat OUTSIDE the true boundary on 83.8 % of the "
            "(L, h) grid (max +0.0302 at L = 1.00, h = 111 deg, where the true "
            "boundary is 0), so a colour on the true shell read r0 < 1, kept part "
            "of its boost and left the gamut: pure [S] at sat 1.4 drove the "
            "pre-clip lattice 12.333 codes below 0 on 2.85 % of the nodes, "
            "against 1.4e-05 codes on 0.041 % with both W2 repairs in.  Capping "
            "by the raw bisection costs no chroma (it removes only what the "
            "Gaussian ADDED — the mean deficit against the raw boundary is "
            "unchanged at 0.00106) and still smooths the cusp ridge, which is "
            "the only real roughness: the cap takes the smoothed value on the "
            "16 % of cells where the Gaussian lowered the surface.  What is "
            "left is bilinear-interpolation noise BETWEEN grid nodes (max "
            "+0.0027, p99.9 +0.0014); on the nodes themselves the inequality is "
            "exact."),
    }


def w2_r0_width(widths=(0.0, 0.04, 0.08, 0.12, 0.16, 0.24)) -> dict:
    """W2's ``r0 = min(C0/CMAX, 1)`` is a hard clamp in the middle of the input
    set.  The width of the C2 shoulder that replaces it is chosen here."""
    from tools import metrics as M

    t = np.linspace(0.0, 1.0, 33)
    b, g, r = np.meshgrid(t, t, t, indexing="ij")
    lab0 = color.linear_srgb_to_oklab(xfer.signed_srgb_decode(
        np.stack([r, g, b], axis=-1)))
    L0, C0, h0 = color.oklab_to_lch(lab0)
    s_lat = C0 / _cmax.cmax(L0, h0)

    def _crease(sat=1.4, L0v=0.50, h0v=30.0):
        c = pipeline.compile(LookSpec(name="Ka", field=_spec.Field(sat=sat), gamut=None))
        cm = float(_cmax.cmax(L0v, h0v))
        row = {}
        for h in (1e-2, 1e-3, 1e-4, 1e-5):
            off = np.linspace(-3.0 * h, 3.0 * h, 201)
            def code(Cv):
                lin = color.oklab_to_linear_srgb(np.stack(
                    [np.full_like(Cv, L0v), Cv * np.cos(np.radians(h0v)),
                     Cv * np.sin(np.radians(h0v))], axis=-1))
                return xfer.signed_srgb_encode(lin)
            d2 = np.abs(pipeline.apply(c, code(cm + off - h))
                        - 2.0 * pipeline.apply(c, code(cm + off))
                        + pipeline.apply(c, code(cm + off + h))).max(axis=-1) / h ** 2
            row[f"h={h:g}"] = float(d2.max())
        return row

    rows = []
    keep = _cmax.R0_WIDTH
    try:
        for w in widths:
            _cmax.R0_WIDTH = w
            row = {"width": w, "crease_D": _crease()}
            for nm in SWEEP_LOOKS:
                c = pipeline.compile(_spec.load_look(LEAD / f"{nm}.json"), strict=False)
                m = _measure(c)
                row[nm] = {k: m[k] for k in ("material_100", "micro_pct", "crush_pct",
                                             "d2_interior_p99_9", "d2_full_p99_9")}
            for sat in (1.05, 1.40):
                c = pipeline.compile(dc.replace(
                    _spec.load_look("_identity"), name="W",
                    field=dc.replace(_spec.load_look("_identity").field, sat=sat),
                    gamut=_spec.Gamut()), strict=False)
                m = _measure(c)
                row[f"identity+sat{sat}"] = {
                    k: m[k] for k in ("material_100", "d2_interior_p99_9",
                                      "d2_full_p99_9", "lim")}
            rows.append(row)
    finally:
        _cmax.R0_WIDTH = keep

    # the properties the shoulder must have, measured
    ss = np.linspace(0.0, 1.5, 3_000_001)
    rho = _cmax.soft_saturate(ss)
    dT = np.diff(ss * rho) / np.diff(ss)
    return {
        "what": "the width of the C2 shoulder that replaces W2's min(C0/CMAX, 1)",
        "chosen": _cmax.R0_WIDTH,
        "reach_on_lattice": {
            "frac_r0_saturated_pct": float(np.mean(s_lat >= 1.0) * 100.0),
            "frac_s_gt_0p95_pct": float(np.mean(s_lat > 0.95) * 100.0)},
        "properties": {
            "rho(1)": float(_cmax.soft_saturate(np.array(1.0))),
            "rho(1.5)": float(_cmax.soft_saturate(np.array(1.5))),
            "linear_region_scale": float(_cmax.soft_saturate(np.array(0.3)) / 0.3),
            "monotone": bool(np.all(np.diff(rho) >= -1e-15)),
            "max_d_s_rho_ds": float(dT.max()),
            "bound_needed": 2.0},
        "rows": rows,
        "verdict": (
            "The hard min() put a C1 crease on the whole gamut shell — "
            "D(h) = 798 / 1987 / 18382 / 182387 for h = 1e-2..1e-5, i.e. D ~ 1/h "
            "— and 15.03 % of the 33**3 lattice sits exactly on it.  The shoulder "
            "removes it (D converges from w = 0.04 up) and, because the "
            "rectifier is soft_ramp rather than soft_relu, keeps "
            "d(s*r0)/ds <= 2, which is what W2's monotonicity algebra needs. "
            "0.12 is where the fold gains have arrived while the shoulder's own "
            "price (a 6.4 % STRONGER fade in the linear region) is still small."),
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------


def _n(v, fmt="{:.3f}", dash="--"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def print_tables(doc: dict) -> None:
    a = doc["w5_anchor"]
    print("\nW5.1 ANCHOR STUDY  (identity + sat, 33**3, knee 0.65 / s1 0.35 / p 8)")
    print(f"{'anchor':7s} {'sat':>5s} {'lim':>10s} {'q':>10s} {'mat':>5s} {'mat70':>6s} "
          f"{'micro%':>7s} {'strict':>7s} {'crush%':>7s} {'min ratio':>10s} "
          f"{'d2 int':>7s} {'d2 full':>8s}")
    for r in a["rows"]:
        print(f"{r['anchor']:7s} {r['sat']:5.2f} {_n(r['lim'], '{:10.4f}')} "
              f"{_n(r['q'], '{:10.3f}')} {r['material_100']:5d} {r['material_70']:6d} "
              f"{r['micro_pct']:7.3f} {r['strict_100']:7d} {r['crush_pct']:7.3f} "
              f"{r['min_ratio']:+10.4f} {r['d2_interior_p99_9']:7.2f} "
              f"{r['d2_full_p99_9']:8.2f}")
    print("  cross-check on the three real looks:")
    for r in a["real_looks"]:
        print(f"  {r['look']:9s} {r['anchor']:6s} lim {_n(r['lim'], '{:13.4f}')} "
              f"q {_n(r['q'], '{:12.2f}')} mat {r['material_100']:6d} "
              f"mat70 {r['material_70']:6d} crush% {r['crush_pct']:6.2f} "
              f"d2int {r['d2_interior_p99_9']:7.2f}")
    print(f"  -> {a['choice']}")

    s = doc["w5_sweep"]
    print("\nW5.3 KNEE x S1 SWEEP  (sum / mean over 03Gilt, 04Viride, 10Splice)")
    print(f"{'knee':>5s} {'s1':>5s} {'mat@100':>8s} {'mat@70':>7s} {'micro%':>7s} "
          f"{'crush%':>7s} {'d2 int':>7s} {'tax mean':>9s} {'tax max':>8s} "
          f"{'touched%':>9s}")
    for c in s["cells"]:
        star = "  <=" if (c["knee"] == _spec.DEFAULT_KNEE
                          and c["s1"] == _spec.DEFAULT_END_SLOPE) else ""
        print(f"{c['knee']:5.2f} {c['s1']:5.2f} {c['material_100']:8d} "
              f"{c['material_70']:7d} {c['micro_pct_mean']:7.3f} "
              f"{c['crush_pct_mean']:7.3f} {c['d2_interior_mean']:7.2f} "
              f"{c['identity_tax_mean_codes']:9.3f} "
              f"{c['identity_tax_max_codes']:8.2f} "
              f"{c['identity_frac_touched_pct']:9.2f}{star}")
    print(f"  chosen: {s['chosen']}")

    print("\nW7  looks/lead/*.json, v1.2 engine, as they are")
    print(f"{'look':9s} {'ok':3s} {'grey':>7s} {'d2int':>7s} {'d2full':>7s} "
          f"{'b70d2':>6s} {'mat100':>7s} {'mat70':>6s} {'jac65':>6s} {'micro%':>7s} "
          f"{'crush%':>7s} {'lim':>8s} {'q':>7s} {'push':>7s} {'mingain':>8s} "
          f"{'dE00':>6s}")
    for r in doc["w7"]["rows"]:
        print(f"{r['name']:9s} {'Y' if r['validates'] else 'n':3s} "
              f"{r['grey_err_codes']:7.4f} {r['d2_interior_p99_9']:7.2f} "
              f"{r['d2_full_p99_9']:7.2f} {r['blend70_d2_p99_9']:6.2f} "
              f"{r['material_100']:7d} {r['material_70']:6d} "
              f"{_n(r.get('jac_material'), '{:6.0f}')} "
              f"{r['micro_pct']:7.3f} {r['crush_pct']:7.2f} {_n(r['lim'], '{:8.3f}')} "
              f"{_n(r['q'], '{:7.2f}')} {_n(r.get('push_p99_9'), '{:7.3f}')} "
              f"{_n(r['min_radial_gain'], '{:8.4f}')} "
              f"{_n(r.get('de00_photo_mean'), '{:6.2f}')}")
    n_fail = sum(1 for r in doc["w7"]["rows"] if r["fails"])
    print(f"  {n_fail} of {len(doc['w7']['rows'])} looks FAIL at least one hard gate")
    for r in doc["w7"]["rows"]:
        if r["fails"]:
            print(f"  {r['name']}: FAIL " + "; ".join(r["fails"]))

    print("\nW7 DIAGNOSIS  steepest table step and where each probe bottoms out")
    print(f"{'look':9s} {'cr step':>22s} {'dh18 step':>22s} {'foldbnd':>8s} "
          f"{'radial':>7s} {'gain hi':>8s}")
    for name, g in doc["w7_diagnosis"].items():
        cr, dh = g["cr_worst_step"], g["dh18_worst_step"]
        print(f"{name:9s} {cr['delta']:6.3f} @{cr['from_h']:3.0f}->{cr['to_h']:3.0f} deg  "
              f"{dh['delta']:6.1f} @{dh['from_h']:3.0f}->{dh['to_h']:3.0f} deg  "
              f"{g['hue_fold_bound']:8.3f} {g['radial_chroma_bound']:7.3f} "
              f"{g['field_gain_range'][1]:8.3f}")

    o = doc["w7_order"]
    print("\nW7 ORDER CONTROL  where the surviving material folds come from")
    print(f"{'look':9s} {'NPG mat':>8s} {'PGN mat':>8s} {'flat-tone mat':>14s} "
          f"{'NPG d2int':>10s} {'PGN d2int':>10s}")
    for name, r in o["rows"].items():
        print(f"{name:9s} {r['as-is/NPG']['material_100']:8d} "
              f"{r['as-is/PGN']['material_100']:8d} "
              f"{r['tone=identity/NPG']['material_100']:14d} "
              f"{r['as-is/NPG']['d2_interior_p99_9']:10.2f} "
              f"{r['as-is/PGN']['d2_interior_p99_9']:10.2f}")
    print(f"  totals: {o['material_folds_total']}")

    print("\nW7 ABLATION  (only looks that FAIL a hard gate)")
    for name, ab in doc["w7_ablation"].items():
        print(f"  {name}  fails: {', '.join(ab['fails'])}")
        b = ab["base"]
        print(f"    {'switch':<11} {'d2int':>7s} {'d2full':>7s} {'mat@100':>8s} "
              f"{'mat@70':>7s} {'micro%':>7s} {'crush%':>7s}  clears")
        print(f"    {'(none)':<11} {_n(b['d2_interior_p99_9'], '{:7.2f}')} "
              f"{_n(b['d2_full_p99_9'], '{:7.2f}')} {_n(b['fold100'], '{:8.0f}')} "
              f"{_n(b['fold70'], '{:7.0f}')} {_n(b['micro_pct'], '{:7.3f}')} "
              f"{_n(b['crush_pct'], '{:7.2f}')}  -")
        for r in ab["rows"]:
            if r.get("error"):
                print(f"    {r['switch']:<11} {r['error'][:70]}")
                continue
            print(f"    {r['switch']:<11} {_n(r.get('d2_interior_p99_9'), '{:7.2f}')} "
                  f"{_n(r.get('d2_full_p99_9'), '{:7.2f}')} "
                  f"{_n(r.get('fold100'), '{:8.0f}')} {_n(r.get('fold70'), '{:7.0f}')} "
                  f"{_n(r.get('micro_pct'), '{:7.3f}')} "
                  f"{_n(r.get('crush_pct'), '{:7.2f}')}  "
                  + (", ".join(r.get("clears") or []) or "-"))


# ---------------------------------------------------------------------------


def build() -> dict:
    t0 = time.perf_counter()
    photo = _photo()
    _cmax.cmax_table()
    doc = {
        "spec": "ENGINE_SPEC_v1_2.md — W1..W7",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_defaults": {
            "gamut": _spec.Gamut().to_dict(),
            "anchor": _gamut.DEFAULT_ANCHOR,
            "lim_margin": _gamut.LIM_MARGIN,
            "relfade": _spec.DEFAULT_RELFADE,
            "relfade_width": _field.RELFADE_WIDTH,
            "gates": _spec.Gates().to_dict(),
            "gate_min_width": _spec.GATE_MIN_WIDTH,
            "radial_min_slope": _spec.RADIAL_MIN_SLOPE,
            "gain_hard_hi": _spec.GAIN_HARD_HI,
            "cmax": {"grid": [_cmax.N_L, _cmax.N_H], "sigma_L": _cmax.SIGMA_L,
                     "sigma_h": _cmax.SIGMA_H, "floor": _cmax.CMAX_FLOOR,
                     "version": _cmax.VERSION, "r0_width": _cmax.R0_WIDTH,
                     "cache": str(_cmax.cache_path())},
            "radial_probe": {"h_step": float(_field.RADIAL_H[1] - _field.RADIAL_H[0]),
                             "n_h": int(_field.RADIAL_H.size),
                             "n_L": len(_field.RADIAL_L),
                             "n_C": int(_field.RADIAL_C.size)},
        },
        "photo_sample": None if photo is None else {"n": int(photo.shape[0])},
    }
    doc["w1_fold_bound"] = w1_fold_bound()
    doc["w2_cmax_cap"] = w2_cmax_cap()
    doc["w2_r0_width"] = w2_r0_width()
    doc["w5_anchor"] = w5_anchor(photo)
    doc["w5_curve"] = w5_curve()
    doc["w5_relfade_width"] = w5_relfade_width()
    doc["w5_sweep"] = w5_sweep(photo)
    doc["w7"] = w7(photo)
    doc["w7_diagnosis"] = w7_diagnosis(doc["w7"])
    doc["w7_order"] = w7_order(doc["w7"])
    doc["w7_ablation"] = w7_ablation(doc["w7"])
    doc["build_time_s"] = time.perf_counter() - t0
    return doc


def main() -> int:
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    print_tables(doc)
    print(f"\nwritten: {OUT}  ({OUT.stat().st_size / 1024:.0f} KiB, "
          f"{doc['build_time_s']:.1f} s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
