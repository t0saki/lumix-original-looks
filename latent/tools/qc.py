"""tools/qc.py — the numeric gates, run on the **shipped `.cube` file**.

Authority for the thresholds, in order:

-2. ``docs/GATES_v3.md`` — the lead's gate rulings, which amend PLAN §验证, R5
   and W6 and win over everything below.  What v3 changed:

   * the HARD list is now closed and explicit (file/header/stem, grey vs T,
     white, monotone per channel *and* in OKLab L, hair slope, black lift,
     material folds @100 and @70, micro folds, crush, the 65³ Jacobian,
     interior / full d2, grid error on the **photo** population, clip excess,
     the skin hue band, the two skin chroma ratios and highlight cleanliness);
   * ``ap.wb_preset`` is **replaced** by ``ap.highlight_clean``: the same cast
     statistic, measured over near-neutral **highlight** pixels only
     (input ``C0 <= 0.03`` and input ``L >= 0.80``), because the all-tones
     population fails every look with a designed shadow/mid tint.  The
     population size ``n`` is reported, and below 500 px the gate falls back to
     the near-neutral population at input ``L >= 0.70``;
   * the skin chroma gates are **absolute** (bright ratio >= 0.78 over input
     L > 0.75, dark ratio >= 0.85 over input L 0.45-0.62), not a comparison
     against ``looks/targets/<name>.json``, which is obsolete;
   * ``blend70.d2_p99_9`` (exactly 0.7 × the full-lattice d2) and
     ``fingerprint.residual`` are REMOVED;
   * grid error on RANDOM RGB, ``push``, ``lim``, ``min radial gain``, the dE00
     band, ``skin.displacement`` (FAIL only above 0.030), pairwise dE00 and
     ``neutral.vs_target > 0.2`` are WARN-only;
   * a MONO look skips every colour-only gate, and its highlight-cleanliness
     line becomes ``|tint| <= 1.5`` codes on the grey ramp at ``t >= 0.80``.

-1. ``docs/ENGINE_SPEC_v1_2.md`` §W6 — the lead's ruling before that, which amends
   R5 and wins.  What W6 changed: **folds are judged by magnitude**, not by
   sign — ``material`` (ratio < -0.02) must be 0 at 100 % and 70 % (FAIL),
   ``micro`` (-0.02 .. -1e-6) WARNs above 0.5 % and FAILs above 2 %, ``crush``
   (|ratio| < 0.02, excess over the identity) WARNs above 3 % and FAILs above
   8 %; the 65³ Jacobian FAILs only on ``det < -0.02``; interior d2 p99.9 WARNs
   above 4.0 and FAILs above 6.0; and the ``push`` / ``lim`` gates keep R5's
   numbers but are **WARN-only**.

0. ``docs/ENGINE_SPEC_v1_1.md`` §R5 — the lead's rulings after milestone 1,
   which amend PLAN §验证 and win over everything below.  What R5 changed:
   the smoothness lines (interior p99.9 4 / warn 2.5, full p99.9 12 / warn 9,
   full max 30); the fold gate now reads **strict** negatives past a 1e-6 noise
   floor with *collapsed* tetrahedra split out as a WARN; PLAN's unreachable
   ``nm p99.9 <= 1.15`` replaced by ``push`` (photo p99.9 of ``nm_pre - nm_in``)
   and the engine's own measured ``lim``; ``C'`` gated only when [F] is on and
   the tone slope gated at 0.04 for t >= 1/255 instead; the dE00 bands demoted
   to WARN; the declared-chroma "dark skin" population moved to input L
   0.45-0.62; and the ``norm_residual`` key bug in the fingerprint gate fixed.

1. ``docs/PLAN.md`` §验证 "数值门" — the contract.  Where PLAN names a number,
   that number is used, even when R06 is looser (``nm`` p99.9 and the final-clip
   no-op are the two places they disagree; PLAN wins, R06's value is carried in
   the gate's ``note``).
2. ``docs/research/06_engine_math_and_robustness.md`` §4 — the threshold table,
   used wherever PLAN is silent (warn levels, black point, skin displacement,
   dE00-vs-identity band, file-level checks).
3. PLAN's twelve anti-pattern gates: PLAN pins a number for four of them
   (hair slope ≥ 0.18, black lift ≤ 8 codes, "really a WB preset" ≤ ±1.5 codes,
   lamp/skin hue collision ≥ 8°, warm-cool ΔL ≥ 0.015).  The other eight name
   the defect but not the number; the thresholds chosen here are constants at
   the top of this file, each with the reason, and each gate reports its
   measured value so the lead can move the line.

Everything is measured on the table read back from disk.  The three checks that
need the continuous engine — grid error, the 65³ Jacobian and the pre-clip
excursion / pre-gamut ``nm`` — run only when a compiled look is handed in, and
report ``skip`` otherwise.

Public API::

    qc_table(source, compiled=None, ...) -> dict      # source: .cube path | table | LUT3D
    qc_cube(path, compiled=None, ...) -> dict         # alias, path only
    pairwise_de00(tables) -> dict                     # the build-level gate
    format_table(report) -> str                       # the printed table
    main(argv) -> int                                 # CLI, non-zero on any FAIL

CLI::

    ./py -m tools.qc out/LUTs/*.cube
    ./py -m tools.qc --look _demo_glaze out/LUTs/Glaze.cube
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from engine import color
from engine.cubeio import LUT3D, read_lut, tetrahedral_interpolation
from tools import metrics as M

__all__ = [
    "qc_table",
    "qc_cube",
    "pairwise_de00",
    "format_table",
    "gate_summary",
    "hard_fails",
    "GATES_V3_HARD",
    "GATES_V3_WARN_ONLY",
    "GATES_V3_REMOVED",
    "main",
    "PASS",
    "WARN",
    "FAIL",
    "SKIP",
    "INFO",
]

FloatArray = np.ndarray

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"
#: ``docs/REVIEW_r1.md`` tooling ruling: a gate row that is an ALIAS of another
#: gate's statistic reports the number and nothing else — it may never turn into
#: a second FAIL or WARN for a defect that is already gated once.  ``INFO`` is
#: that status: it is printed, it is stored, it never enters ``fails``/``warns``
#: and it never moves ``summary["worst"]``.
INFO = "info"
_RANK = {PASS: 0, SKIP: 0, INFO: 0, WARN: 1, FAIL: 2}

# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------

#: PLAN: 33-point cube, 8-char ASCII stem, LUMIX STD header.
CUBE_SIZE = 33
_STEM_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")
STD_HEADER = "#LUMIXPHOTOSTYLE STD"

#: PLAN 数值门.  ``(fail, warn)``; ``None`` = no level at that severity.
GREY_VS_T = (0.5, 0.2)          # codes, max |out - T| on the 4097 ramp
WHITE_ERR = 1e-6                # codes/1.0 units: max |white - 1|
BLACK_VS_DESIGNED = 1.0 / 255.0  # R06 §4, in [0,1] units
TINT_DECLARED = 1.0             # codes, |measured - designed| per t
TINT_NEUTRAL_AXIS = (2.0, 1.0)  # R06 §4, for looks whose spec declares no tint
SLOPE_C_FAIL = (0.33, 3.0)      # PLAN: N's correction slope C' — v1.1 R5: only
SLOPE_C_WARN = (0.5, 2.0)       # R06 §4    when [F] is on (C == T otherwise)
TONE_SLOPE_MIN = 0.04           # v1.1 R5: tone slope >= 0.04 for t >= 1/255
#: v1.1 R5 smoothness.  Interior is the shipping budget; the full lattice
#: population measures the compressor at the cube's corners (identity+G alone
#: reads 7.6–10 there), so its line sits at 12 / warn 9, max 30.
D2_FULL_P999 = (12.0, 9.0)      # v1.1 R5 / v1.2 W6 (was PLAN 6 / R06 warn 4)
#: v1.2 W6: interior d2 p99.9 WARN > 4.0, FAIL > 6.0.  (Contax measures 5.5,
#: Pentax 4.2, Leica X2 10.4, the rival AI sets 14-18.)
D2_INTERIOR_P999 = (6.0, 4.0)
D2_FULL_MAX = (30.0, None)      # v1.1 R5 / W6: "full max <= 30"; no warn level
FOLD_MIN_RATIO_WARN = 1e-4      # R06 §4
FOLD_EPS = M.FOLD_EPS           # noise floor, inside metrics.fold_stats
FOLD_COLLAPSED_WARN = 0.0005    # v1.1 R5: collapsed > 0.05 % -> WARN (informational)

#: --- v1.2 W6: folds judged by MAGNITUDE -----------------------------------
#: "A tetrahedron whose volume ratio is -3e-5 is a crushed sliver on the gamut
#: shell, not a visible reversal; the v1.1 gate cannot tell a sliver from a real
#: fold, and every commercial reference LUT fails it by thousands."
FOLD_MATERIAL = (0.0, None)         # material (< -0.02): must be 0 -> FAIL
FOLD_MICRO_PCT = (2.0, 0.5)         # micro folds, % of tetrahedra: FAIL / WARN
FOLD_CRUSH_PCT = (8.0, 3.0)         # |ratio| < 0.02, excess over identity: FAIL / WARN
#: 65**3 Jacobian: FAIL only on det < -0.02 * (identity det).  The identity map
#: is the 3x3 identity, so its determinant is exactly 1 and the line is -0.02.
JAC_MATERIAL_DET = -0.02
#: v1.1 R5 replaces PLAN's "nm p99.9 <= 1.15" (the identity already reads 1.27:
#: nm is bounded by 3**0.25 = 1.316 and every saturated in-gamut colour
#: approaches it).  What is gated instead: how much further out of gamut the
#: look pushes a real pixel, and the compressor's own measured limit.
#: v1.2 W6: "push / lim gates stay as in R5 but are WARN-only."  R5's two
#: levels are kept in the note; neither can FAIL.
PUSH_P999 = (None, 0.35)        # fail / warn, p99.9 of (nm_pre - nm_in), photo
GAMUT_LIM = (None, 1.6)         # fail / warn, W5.2's lattice-measured `lim`
CLIP_NOOP = 1e-9                # PLAN: "终裁切为空操作（1e-9）", in [0,1] units
CLIP_PLATEAU = (0.005, 0.001)   # PLAN fail 0.5 %; R06 warn 0.1 %
#: GATES_v3: "grid error on RANDOM RGB (all three numbers; the population is
#: dominated by the gamut shell)" is WARN-only.  R5's FAIL numbers become the
#: warn lines; R5's inner warn on the mean (0.20) is carried in the note.
GRID_RANDOM = {"mean": (None, 0.35), "p99": (None, 1.6), "max": (None, 5.0)}
GRID_RANDOM_INNER_WARN = 0.20   # R5's old mean warn, reported not gated
#: GATES_v3 HARD: "grid error on PHOTO pixels mean <= 0.25 / p99 <= 1.0 / max <= 3.5".
GRID_PHOTO = {"mean": (0.25, None), "p99": (1.0, None), "max": (3.5, None)}
#: R06 §4's 70 %-blend smoothness line.  GATES_v3 REMOVED the gate ("exactly
#: 0.7 x the full-lattice d2 — redundant and contradictory"); the constant is
#: kept because ``engine/report_m2.py`` still quotes it.
BLEND70_D2_P999 = 5.0           # R06 §4 — NOT a gate any more (GATES_v3)
SKIN_HUE_BAND = (25.0, 80.0)    # R06 §4, output hue for h_in in [35,70], C>=0.06
#: GATES_v3 WARN-only: "`skin.displacement` (FAIL only above 0.030)".  R06's
#: 0.014 becomes the warn line.
SKIN_DISPLACEMENT = (0.030, 0.014)   # 2*C_out*sin(dh/2)
#: GATES_v3 HARD, ABSOLUTE (the declared-target gates are gone with
#: ``looks/targets``): "bright-skin chroma ratio >= 0.78 (population L > 0.75),
#: dark-skin chroma ratio >= 0.85 (population L 0.45-0.62)".
SKIN_CHROMA_BRIGHT = 0.78
SKIN_CHROMA_DARK = 0.85
#: the two populations, as input OKLab L.  Bright is "L > 0.75" — sampled from
#: just above the line to the top of a photographic highlight on skin.
SKIN_BRIGHT_L = (0.76, 0.90)
SKIN_DARK_L = (0.45, 0.62)
#: v1.1 R5: "dE00 bands: WARN only — the lead judges strength by eye."  Both
#: bands are kept (the numbers are unchanged) but neither can FAIL any more.
DE00_IDENTITY_BAND = (3.0, 16.0)    # outer band, R06 §4
DE00_IDENTITY_WARN = (5.0, 13.0)    # inner band, R06 §4 (reported in the note)
#: ``(fail_below, warn_below)`` for the build-level DISTINCTIVENESS gate.
#: History: PLAN had fail < 2.5 / warn < 3.5; GATES_v3 made it WARN-only at
#: < 2.5; ``docs/REVIEW_r1.md`` ("Tooling rulings") reinstates a FAIL, lower:
#: "pairwise photo dE00 < 1.8 = FAIL, < 2.5 = WARN (was WARN only)".  Two looks
#: closer than 1.8 dE00 on the photo sample are one look, and that blocks the
#: build.
PAIRWISE_DE00 = (1.8, 2.5)
#: the two superseded lines, kept so a stored report can still be read and so
#: nobody quietly restores them.
PAIRWISE_DE00_PLAN = (2.5, 3.5)

# --- the twelve anti-pattern gates -----------------------------------------
# PLAN pins these four numbers:
AP_HAIR_SLOPE = 0.18        # mean grey slope over t in [0, 0.25]
AP_BLACK_LIFT = 8.0         # codes
#: GATES_v3 replaces anti-pattern 9 (`ap.wb_preset`, "really a WB preset") with
#: HIGHLIGHT CLEANLINESS.  Same statistic — the mean cast the look ADDS, in
#: codes — but over near-neutral HIGHLIGHT pixels only: the all-tones
#: population "fails every look with a designed shadow/mid tint".
AP_HILITE_CLEAN = 1.5       # codes, max(|mean d(R-G)|, |mean d(B-G)|)
AP_HILITE_C0 = 0.03         # input OKLab chroma: "near-neutral"
AP_HILITE_L = 0.80          # input OKLab L: "highlight"
AP_HILITE_L_FALLBACK = 0.70  # if n < AP_HILITE_MIN_N, fall back to this L
AP_HILITE_MIN_N = 500       # "if n < 500 fall back to input L >= 0.70"
#: MONO variant of the same line: "highlight cast becomes |tint| <= 1.5 codes
#: at t >= 0.80" — measured on the grey ramp, where a mono look's only possible
#: cast lives.
AP_HILITE_MONO_T = 0.80
MONO_FOLD_EPS = FOLD_EPS    # kept as a name; v1.1 R5 moved the noise floor into
                            # tools.metrics.fold_stats, where it now applies to
                            # every look, not only to the mono branch.
AP_LAMP_SKIN_SEP = 8.0      # degrees between lamp-light and skin output hue
AP_WARM_COOL_DL = 0.015     # OKLab L separation (only gated for declared 德味 looks)
# PLAN names the defect but not the number; chosen here, with the reason:
AP_WHITE_TINT = (1.0, 0.5)  # codes.  Measured over AP_WHITE_T: the default
AP_WHITE_T = 0.97           # white guard (ENGINE_SPEC §2) fades 0.88 -> 1.0, so
                            # a *designed* tint is still ~full at t = 0.90 and
                            # only the top ~3 % of the ramp is promised clean.
AP_GREY_SKIN = (0.70, 0.85)     # min skin-patch chroma ratio.  A look may declare
                                # a muted skin (§3.4 skin.chroma) and R06 §4
                                # tolerates the declared value +-0.08, so the
                                # anti-pattern line has to sit below any
                                # plausible declaration: below 0.70 the patch
                                # reads grey rather than muted.
AP_NEON_GREEN = (1.80, 1.50)    # max chroma ratio over greens (h 110-165).  1.8
                                # is §3.7's own chroma-gain ceiling; a green
                                # past it is outside the engine's budget, which
                                # is the strongest non-invented line available.
AP_MUDDY_CYAN = (0.70, 0.85)    # min chroma ratio over cyan/blue (h 190-250).
                                # §3.7 allows down to 0.4 globally; below 0.70 a
                                # blue sky reads grey, which is the defect named.
AP_SKIN_SHADOW_H = (25.0, 30.0)  # min OUTPUT hue on dark skin.  Same 25 deg
                                 # floor R06 §4 puts on the whole skin band,
                                 # applied to the dark sub-population, which is
                                 # where the magenta shows.
AP_PRIMARY_CRUSH = (1.0, 2.0)   # min dE00 between neighbouring max-chroma hues:
                                # below ~1 the primaries have merged into a
                                # single clipped ridge.

#: probe grids for the anti-pattern gates (OKLCh, stage-0 space).
_GREEN_H = np.arange(110.0, 166.0, 5.0)
_CYAN_H = np.arange(190.0, 251.0, 5.0)
_PROBE_C = (0.08, 0.11, 0.14)
_PROBE_L = (0.35, 0.50, 0.65)


# ---------------------------------------------------------------------------
# docs/GATES_v3.md, as a list — the one place the ruling is written down
# ---------------------------------------------------------------------------

#: The HARD list of ``docs/GATES_v3.md``, **in the order the ruling writes it**.
#: ``tools/round.py`` prints these first and ``tools/build.py`` puts them at the
#: left of the build table, so the two stay in step by construction.
#: ``(gate key, short label)``.
GATES_V3_HARD: tuple[tuple[str, str], ...] = (
    # "file/header/stem"
    ("file.size", "LUT_3D_SIZE 33"),
    ("file.rows", "rows == size^3"),
    ("file.finite", "finite, in [0,1]"),
    ("file.header", "LUMIX STD header"),
    ("file.stem", "stem <= 8 alnum"),
    # "grey vs T <= 0.5 codes; white exact; per-channel + OKLab-L monotone"
    ("neutral.vs_target", "grey vs T"),
    ("neutral.white_err", "white exact"),
    ("neutral.monotone_rgb", "per-channel monotone"),
    ("neutral.monotone_L", "OKLab-L monotone"),
    # "hair slope >= 0.18; black lift <= 8 codes"
    ("ap.hair_black", "hair slope"),
    ("ap.black_lift", "black lift"),
    # "material folds @100 = 0 and @70 = 0; micro folds <= 2 %; crush <= 8 %"
    ("fold.neg_count", "material folds @100%"),
    ("blend70.neg_count", "material folds @70%"),
    ("fold.micro", "micro folds %"),
    ("blend70.micro", "micro folds % @70%"),
    ("fold.crush", "crushed %"),
    # "65^3 Jacobian material = 0"
    ("jacobian.neg_count", "65^3 Jacobian folds"),
    # "interior d2 p99.9 <= 6.0; full d2 p99.9 <= 12; full d2 max <= 30"
    ("d2.interior_p99_9", "interior d2 p99.9"),
    ("d2.full_p99_9", "full d2 p99.9"),
    ("d2.full_max", "full d2 max"),
    # "grid error on PHOTO pixels mean <= 0.25 / p99 <= 1.0 / max <= 3.5"
    ("grid.photo_mean", "grid err mean, photo"),
    ("grid.photo_p99", "grid err p99, photo"),
    ("grid.photo_max", "grid err max, photo"),
    # "clip excess <= 0.5 %"
    ("clip.at_zero", "clip excess at 0"),
    ("clip.at_one", "clip excess at 1"),
    # "skin output hue band inside [25, 80]"
    ("skin.hue_band_lo", "skin hue band lo"),
    ("skin.hue_band_hi", "skin hue band hi"),
    # "bright-skin chroma ratio >= 0.78 ...; dark-skin chroma ratio >= 0.85 ..."
    ("skin.skin_chroma_bright", "bright-skin chroma"),
    ("skin.skin_chroma_dark", "dark-skin chroma"),
    # "highlight cleanliness ... <= 1.5 codes (this replaces `ap.wb_preset`)"
    ("ap.highlight_clean", "highlight cleanliness"),
)

#: The WARN-only list of ``docs/GATES_v3.md``.  None of these can FAIL; they are
#: printed under their own heading so a colourist can see at a glance that they
#: do not block shipping.  (``pairwise mean dE00`` is build-level — it is not a
#: per-look gate row — and ``neutral.vs_target > 0.2`` is the *warn level* of a
#: gate that is HARD at 0.5, so both are footnotes rather than rows here.)
GATES_V3_WARN_ONLY: tuple[tuple[str, str], ...] = (
    ("grid.random_mean", "grid err mean, random"),
    ("grid.random_p99", "grid err p99, random"),
    ("grid.random_max", "grid err max, random"),
    ("gamut.push_p99_9", "push"),
    ("gamut.lim", "lim"),
    ("gamut.min_radial_gain", "min radial gain"),
    ("de00.vs_identity", "dE00 band"),
    ("skin.displacement", "skin displacement"),
)

#: GATES_v3's REMOVED list, kept so a stored report can be read and so nothing
#: silently reintroduces one of them.
GATES_V3_REMOVED: tuple[str, ...] = (
    "blend70.d2_p99_9",         # exactly 0.7 x the full-lattice d2
    "fingerprint.residual",     # looks/targets is obsolete
    "skin.skin_chroma_bright_declared",
    "skin.skin_chroma_dark_declared",
    "ap.wb_preset",             # replaced by ap.highlight_clean
)


# ---------------------------------------------------------------------------
# gate records
# ---------------------------------------------------------------------------


def _fmt(v: float | None, fmt: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return fmt.format(v)


def _gate(
    key: str,
    group: str,
    label: str,
    value: float | None,
    *,
    fail_hi: float | None = None,
    warn_hi: float | None = None,
    fail_lo: float | None = None,
    warn_lo: float | None = None,
    fmt: str = "{:.4g}",
    unit: str = "",
    note: str = "",
    status: str | None = None,
    threshold: str | None = None,
) -> dict:
    """One gate row.  ``value is None`` (or an explicit ``status``) -> skip."""
    if status is None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            status = SKIP
        else:
            v = float(value)
            if (fail_hi is not None and v > fail_hi) or (fail_lo is not None and v < fail_lo):
                status = FAIL
            elif (warn_hi is not None and v > warn_hi) or (warn_lo is not None and v < warn_lo):
                status = WARN
            else:
                status = PASS
    if threshold is None:
        parts = []
        if fail_lo is not None:
            parts.append(f">= {_fmt(fail_lo, fmt)}")
        if fail_hi is not None:
            parts.append(f"<= {_fmt(fail_hi, fmt)}")
        threshold = " and ".join(parts) if parts else ""
        if warn_hi is not None or warn_lo is not None:
            w = []
            if warn_lo is not None:
                w.append(f">= {_fmt(warn_lo, fmt)}")
            if warn_hi is not None:
                w.append(f"<= {_fmt(warn_hi, fmt)}")
            threshold = (threshold + "  (warn " + " and ".join(w) + ")").strip()
    return {
        "key": key,
        "group": group,
        "label": label,
        "value": None if value is None else float(value),
        "text": _fmt(value, fmt),
        "unit": unit,
        "status": status,
        "threshold": threshold,
        "note": note,
    }


def _bool_gate(key: str, group: str, label: str, ok: bool | None, *,
               detail: str = "", note: str = "", warn_only: bool = False) -> dict:
    if ok is None:
        status = SKIP
    elif ok:
        status = PASS
    else:
        status = WARN if warn_only else FAIL
    return {
        "key": key, "group": group, "label": label,
        "value": None if ok is None else float(bool(ok)),
        "text": detail or ("ok" if ok else "violated") if ok is not None else "--",
        "unit": "", "status": status, "threshold": "must hold", "note": note,
    }


def gate_summary(gates: Sequence[dict]) -> dict:
    out = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0, INFO: 0}
    for g in gates:
        out[g["status"]] = out.get(g["status"], 0) + 1
    out["worst"] = FAIL if out[FAIL] else (WARN if out[WARN] else PASS)
    out["ok"] = out[FAIL] == 0
    return out


def hard_fails(report_or_gates) -> list[str]:
    """The failing gate keys that sit on ``docs/GATES_v3.md``'s HARD list.

    In GATES_v3 order, so the list reads the way the ruling does.  Anything
    else that FAILs is a gate the ruling does not mention — it still blocks a
    build (``tools/build.py`` counts every FAIL), but it is not one of the
    lead's hard lines, and the report prints it under its own heading.
    """
    gates = (report_or_gates.get("gates", ())
             if isinstance(report_or_gates, dict) else report_or_gates)
    bad = {str(g.get("key")) for g in gates if g.get("status") == FAIL}
    return [k for k, _ in GATES_V3_HARD if k in bad]


# ---------------------------------------------------------------------------
# probe helpers
# ---------------------------------------------------------------------------


def _wrap180(d: FloatArray) -> FloatArray:
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def _probe(sampler: Callable[[FloatArray], FloatArray], L, C, h) -> dict:
    """Push an OKLCh probe grid through *sampler*.

    Out-of-gamut probe points are dropped (mask), matching the fingerprint
    probe's in-gamut rule (eps 5e-4 on linear values).  Returns flat arrays of
    the *kept* points only.
    """
    L, C, h = np.broadcast_arrays(
        np.asarray(L, dtype=np.float64),
        np.asarray(C, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
    )
    code, ok = M.code_from_oklch(L, C, h)
    code = code.reshape(-1, 3)
    ok = ok.reshape(-1)
    if not np.any(ok):
        return {"n": 0}
    inp = code[ok]
    out = np.asarray(sampler(inp), dtype=np.float64)
    L1, C1, h1 = np.asarray(L).reshape(-1)[ok], np.asarray(C).reshape(-1)[ok], np.asarray(h).reshape(-1)[ok]
    L2, C2, h2 = M.oklch_from_code(out)
    return {
        "n": int(ok.sum()),
        "L_in": L1, "C_in": C1, "h_in": h1,
        "L_out": L2, "C_out": C2, "h_out": h2,
        "dh": _wrap180(h2 - h1),
        "cr": C2 / np.maximum(C1, 1e-12),
        "dl": L2 - L1,
        "rgb_in": inp, "rgb_out": out,
    }


def _grid(L, C, h) -> tuple[FloatArray, FloatArray, FloatArray]:
    return np.meshgrid(
        np.atleast_1d(np.asarray(L, dtype=np.float64)),
        np.atleast_1d(np.asarray(C, dtype=np.float64)),
        np.atleast_1d(np.asarray(h, dtype=np.float64)),
        indexing="ij",
    )


def _probe_grid(sampler, L, C, h) -> dict:
    """``_probe`` over the full outer product of the three axis lists."""
    return _probe(sampler, *_grid(L, C, h))


def _skin_scan(sampler) -> dict:
    """Port of ``lumix-original-looks/qc_looks.py::skin_scan``.

    Core region h 42-58°, C 0.06-0.13, L* 30-85; the perceptual displacement
    ``2*C_out*sin(Δh/2)`` is what R06 §4 gates (degrees alone over-penalise
    near-neutral colours).  ``cross_exposure`` is the same scan measured at two
    lightnesses and differenced — a look that rotates dark skin one way and
    bright skin the other reads as "the face changes hue with exposure".
    """
    def drift_over(hs, cs):
        ls = np.linspace(30.0, 85.0, 12)
        H, Ls, Cs = np.meshgrid(hs, ls, cs, indexing="ij")
        L_ok = color.oklabL_from_lstar(Ls)
        p = _probe(sampler, L_ok, Cs, H)
        if p["n"] == 0:
            return np.zeros(0), np.zeros(0)
        deg = np.abs(p["dh"])
        return deg, 2.0 * p["C_out"] * np.sin(np.radians(deg) / 2.0)

    core_deg, core_disp = drift_over(np.linspace(42.0, 58.0, 7), np.linspace(0.06, 0.13, 4))
    full_deg, _ = drift_over(np.linspace(30.0, 70.0, 9), np.linspace(0.05, 0.13, 5))

    def hue_at(lstar_val: float, hs: FloatArray) -> FloatArray:
        L1 = color.oklabL_from_lstar(np.full(hs.shape, lstar_val))
        lab1 = color.lch_to_oklab(L1, np.full(hs.shape, 0.09), hs)
        code1 = color.srgb_encode(np.clip(color.oklab_to_linear_srgb(lab1), 0.0, 1.0))
        o = np.asarray(sampler(code1), dtype=np.float64)
        return M.oklch_from_code(o)[2]

    hs_core = np.linspace(42.0, 58.0, 7)
    cross_deg = np.abs(_wrap180(hue_at(75.0, hs_core) - hue_at(40.0, hs_core)))
    cross_disp = 2.0 * 0.09 * np.sin(np.radians(cross_deg) / 2.0)
    return {
        "core_hue_disp_max": float(core_disp.max()) if core_disp.size else float("nan"),
        "core_hue_drift_deg_max": float(core_deg.max()) if core_deg.size else float("nan"),
        "full_hue_drift_deg_max": float(full_deg.max()) if full_deg.size else float("nan"),
        "cross_exposure_disp_max": float(cross_disp.max()),
        "cross_exposure_deg_max": float(cross_deg.max()),
    }


#: the three sRGB8 skin patches TOOLS_SPEC §T2 pins.
SKIN_PATCHES = ((200, 150, 125), (150, 100, 80), (235, 195, 170))


def _skin_patches(sampler) -> dict:
    inp = np.asarray(SKIN_PATCHES, dtype=np.float64) / 255.0
    out = np.asarray(sampler(inp), dtype=np.float64)
    L1, C1, h1 = M.oklch_from_code(inp)
    L2, C2, h2 = M.oklch_from_code(out)
    lab1, lab2 = M.srgb_to_lab(inp), M.srgb_to_lab(out)
    return {
        "dh": [float(x) for x in _wrap180(h2 - h1)],
        "cr": [float(x) for x in (C2 / np.maximum(C1, 1e-12))],
        "dl": [float(x) for x in (L2 - L1)],
        "de00": [float(x) for x in M.de00(lab1, lab2)],
        "L_out": [float(x) for x in L2],
    }


# ---------------------------------------------------------------------------
# continuous-engine checks (need a compiled look)
# ---------------------------------------------------------------------------


def _fold_block(table: FloatArray, is_mono: bool) -> dict:
    """``metrics.fold_stats`` plus the gate count this look is judged on.

    ENGINE_SPEC v1.2 W6: the gate is **material folds** — ``ratio < -0.02`` —
    and must be 0 at 100 % and at 70 %.  Two further populations are reported
    with their own lines: ``micro`` (``-0.02 <= ratio < -1e-6``, a crushed
    sliver on the gamut shell) WARN above 0.5 % / FAIL above 2 %, and ``crush``
    (``|ratio| < 0.02``, reversed or not) WARN above 3 % / FAIL above 8 %.

    A mono lattice is rank 1, so every volume is 0 by construction: its
    ``crush`` is 100 % and its ``material`` is 0, which is exactly right — the
    map is degenerate, not reversed — and the crush line is skipped for it.
    """
    table = np.asarray(table, dtype=np.float64)
    fold = dict(M.fold_stats(table))
    fold["gate_count"] = int(fold["material_count"])
    fold["gate_field"] = "material_count"
    if is_mono:
        fold["gate_note"] = (
            f"mono: rank-1 lattice, every volume is 0 by construction "
            f"({fold['crush_count']} crushed, {fold['neg_strict_raw_count']} "
            f"raw ratio<0 of float noise); the gate reads "
            f"ratio < -{M.MATERIAL_EPS} plus the 1-D grey monotonicity gate")
    else:
        fold["gate_note"] = ""
    return fold


def _clip_block(table: FloatArray) -> dict:
    """``metrics.clip_stats`` plus the plateau *excess* over the identity.

    PLAN gates "lattice points stuck at 0 / at 1" at 0.5 % each, but the
    identity lattice itself has 3.03 % of its channel values exactly 0 (the
    whole ``r = 0`` face, and likewise for G and B) and 3.03 % exactly 1 — so
    the raw fraction can never reach 0.5 % for any LUT that keeps black black.
    What the gate means is a **plateau**: a lattice value driven to the rail
    that the identity did not already have there.  Both numbers are reported;
    the gate reads the excess.
    """
    table = np.asarray(table, dtype=np.float64)
    out = dict(M.clip_stats(table))
    ident = M.identity_table(table.shape[0])
    out["excess_zero"] = float(((table == 0.0) & (ident > 0.0)).mean())
    out["excess_one"] = float(((table == 1.0) & (ident < 1.0)).mean())
    out["identity_zero"] = float((ident == 0.0).mean())
    out["identity_one"] = float((ident == 1.0).mean())
    return out


def _engine_raw(compiled, rgb: FloatArray) -> FloatArray:
    """Pre-clip engine output — the continuous map, before the one final clip."""
    from engine import pipeline

    run = getattr(pipeline, "_run", None)
    if run is None:  # pragma: no cover - defensive
        return np.asarray(pipeline.apply(compiled, rgb), dtype=np.float64)
    return np.asarray(run(compiled, rgb)[0], dtype=np.float64)


def jacobian_check(compiled, *, size: int = 65, h: float = 1.0 / 256.0) -> dict:
    """R06 §4: central-difference Jacobian determinant on a 65³ engine sample.

    The grid runs over ``[h, 1-h]`` so every offset point stays inside the
    LUT's own domain; the faces and corners are covered by the lattice fold
    test, which is exact there.  The determinant is taken of the **pre-clip**
    map: the final clip is a projection whose Jacobian is 0 wherever it bites,
    which would report a "fold" that is really a clip (and the clip has its own
    gate).
    """
    t = np.linspace(h, 1.0 - h, int(size))
    b, g, r = np.meshgrid(t, t, t, indexing="ij")
    base = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    cols = []
    for axis in range(3):
        off = np.zeros(3)
        off[axis] = h
        plus = _engine_raw(compiled, base + off)
        minus = _engine_raw(compiled, base - off)
        cols.append((plus - minus) / (2.0 * h))
    j0, j1, j2 = cols  # d out / d in_r, d in_g, d in_b   (each (N, 3))
    det = (
        j0[:, 0] * (j1[:, 1] * j2[:, 2] - j1[:, 2] * j2[:, 1])
        - j0[:, 1] * (j1[:, 0] * j2[:, 2] - j1[:, 2] * j2[:, 0])
        + j0[:, 2] * (j1[:, 0] * j2[:, 1] - j1[:, 1] * j2[:, 0])
    )
    # v1.2 W6: "65**3 Jacobian gate: same idea — FAIL only on
    # det < -0.02*(identity det); report the rest."  The identity map's
    # Jacobian is the 3x3 identity, det = 1 exactly.
    material = int(np.count_nonzero(det < JAC_MATERIAL_DET))
    return {
        "size": int(size),
        "h": float(h),
        "n": int(det.size),
        "min_det": float(det.min()),
        "neg_count": int(np.count_nonzero(det <= 0.0)),
        "neg_frac": float(np.count_nonzero(det <= 0.0) / det.size),
        "material_count": material,
        "material_frac": float(material / det.size),
        "material_det": JAC_MATERIAL_DET,
        "p001": float(np.percentile(det, 0.1)),
        "median": float(np.median(det)),
    }


def grid_error(compiled, table: FloatArray, points: FloatArray, *, name: str = "look") -> dict:
    """max-channel |continuous engine − tetrahedral(table)| in 8-bit codes."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    from engine import pipeline

    cont = np.asarray(pipeline.apply(compiled, points), dtype=np.float64)
    samp = tetrahedral_interpolation(LUT3D(table=np.asarray(table), title=name), points)
    e = np.abs(cont - samp).max(axis=-1) * 255.0
    return {
        "n": int(points.shape[0]),
        "mean": float(e.mean()),
        "p99": float(np.percentile(e, 99.0)),
        "p99_9": float(np.percentile(e, 99.9)),
        "max": float(e.max()),
    }


def _nm_stats(nm: FloatArray, knee: float) -> dict:
    nm = np.asarray(nm, dtype=np.float64)
    return {
        "p99": float(np.percentile(nm, 99.0)),
        "p99_9": float(np.percentile(nm, 99.9)),
        "max": float(nm.max()),
        "frac_touched": float(np.count_nonzero(nm > knee) / nm.size),
        "knee": float(knee),
        "n": int(nm.size),
    }


def _nm_of_input(compiled, rgb: FloatArray) -> FloatArray | None:
    """``nm`` of the INPUT colour itself — R1's ``nm0``, the headroom measure.

    Computed with the engine's own ``[G]`` front half so that the exponent
    ``p``, the denominators and any dark-side floor are whatever the engine
    currently uses (R2 is changing the tail, not this part).  Falls back to a
    local copy of R1's formula if the engine's signature has moved, and returns
    ``None`` if neither works — the push gate then reports ``skip``.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    try:
        from engine import color, gamut as G, xfer

        L0 = color.linear_srgb_to_oklab(xfer.signed_srgb_decode(rgb))[..., 0]
        fn = getattr(G, "norm", None)
        if fn is not None:  # v1.1+ engine: the norm is its own function
            g = compiled.spec.gamut
            p = getattr(g if g is not None else G.DEFAULT_DIAG, "p", 4.0)
            kind = getattr(compiled, "anchor", None) or G.DEFAULT_ANCHOR
            return np.asarray(fn(rgb, L0, p, kind)[0], dtype=np.float64)
        return np.asarray(G.apply_gamut(compiled.spec.gamut, rgb, L0)[1], dtype=np.float64)
    except Exception:  # pragma: no cover - engine API drift
        pass
    try:  # R1's own words, p = 4, dark floor 1e-6
        from engine import color, xfer

        L0 = color.linear_srgb_to_oklab(xfer.signed_srgb_decode(rgb))[..., 0]
        n0 = color.srgb_encode(np.clip(L0, 0.0, 1.0) ** 3)[..., None]
        d0 = rgb - n0
        u0 = np.where(d0 >= 0.0, d0 / (1.0 - n0 + 1e-12), -d0 / np.maximum(n0, 1e-6))
        return np.sum(np.maximum(u0, 0.0) ** 4, axis=-1) ** 0.25
    except Exception:  # pragma: no cover
        return None


def push_stats(compiled, points: FloatArray) -> dict | None:
    """v1.1 R5's out-of-gamut **pressure**: ``nm_pre - nm_in`` over a sample.

    ``nm_in`` is how close the input colour already sits to the gamut shell
    (R1's ``nm0``); ``nm_pre`` is where the look has put it by the time [G]
    sees it.  The difference is what the look actually did, which is the
    quantity PLAN's absolute ``nm <= 1.15`` was trying and failing to express.
    """
    from engine import pipeline

    run = getattr(pipeline, "_run", None)
    if run is None:  # pragma: no cover - defensive
        return None
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nm_pre = run(compiled, pts)[1]
    if nm_pre is None:
        return None
    nm_in = _nm_of_input(compiled, pts)
    if nm_in is None:
        return None
    nm_pre = np.asarray(nm_pre, dtype=np.float64).reshape(-1)
    push = nm_pre - np.asarray(nm_in, dtype=np.float64).reshape(-1)
    return {
        "n": int(push.size),
        "p99": float(np.percentile(push, 99.0)),
        "p99_9": float(np.percentile(push, 99.9)),
        "max": float(push.max()),
        "mean": float(push.mean()),
        "frac_positive": float(np.count_nonzero(push > 0.0) / push.size),
        "nm_in_p99_9": float(np.percentile(nm_in, 99.9)),
        "nm_pre_p99_9": float(np.percentile(nm_pre, 99.9)),
    }


#: Set to False once a ``diagnostics()`` call has been seen without a ``lim``,
#: so a whole build does not pay for the call twelve times over.
_DIAG_HAS_LIM: bool | None = None


def _find_lim(obj, depth: int = 0):
    """First numeric ``lim`` found in a nested dict, with its path."""
    if depth > 4 or not isinstance(obj, dict):
        return None, ""
    v = obj.get("lim")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v), "lim"
    for k, sub in obj.items():
        got, path = _find_lim(sub, depth + 1)
        if got is not None:
            return got, f"{k}.{path}"
    return None, ""


def engine_lim(compiled, *, use_diagnostics: bool = True) -> tuple[float | None, str]:
    """R2's per-look compressor limit ``lim``, as the ENGINE publishes it.

    ``lim = max(1.02 * max lattice pre-gamut nm, knee + 0.05)`` is measured at
    compile time by [G]; QC must read it, not re-derive it.  Looked for first
    on the compiled look (cheap), then in ``pipeline.diagnostics()``.  Returns
    ``(None, reason)`` when the engine does not publish it yet — the gate then
    reports ``skip`` rather than crashing.
    """
    global _DIAG_HAS_LIM
    for holder, attr, path in (
        (compiled, "lim", "compiled.lim"),
        (compiled, "gamut_lim", "compiled.gamut_lim"),
        (getattr(compiled, "gamut", None), "lim", "compiled.gamut.lim"),
        (getattr(compiled, "field", None), "lim", "compiled.field.lim"),
        (getattr(compiled.spec, "gamut", None), "lim", "spec.gamut.lim"),
    ):
        v = getattr(holder, attr, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v), path
    if not use_diagnostics or _DIAG_HAS_LIM is False:
        return None, "engine does not publish `lim` (R2 not landed yet)"
    try:
        from engine import pipeline

        try:
            d = pipeline.diagnostics(compiled, grid_n=1024)
        except TypeError:  # pragma: no cover - signature drift
            d = pipeline.diagnostics(compiled)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"diagnostics() raised {type(exc).__name__}: {exc}"
    v, where = _find_lim(d)
    _DIAG_HAS_LIM = v is not None
    if v is None:
        return None, "diagnostics() has no 'lim' key (R2 not landed yet)"
    return v, f"diagnostics()[{where}]"


def _preclip_and_nm(compiled, size: int = CUBE_SIZE,
                    photo: FloatArray | None = None) -> tuple[dict, dict | None, dict | None]:
    """Final-clip excursion and pre-gamut 4-norm ``nm`` over the lattice.

    The lattice is what ENGINE_SPEC §4 and PLAN name, but it *contains the sRGB
    primary corners*, where ``nm = 3**0.25 = 1.316`` for any look including the
    identity — so ``nm`` is also measured over the photo sample when one exists,
    which is the population a shipping decision can actually act on.
    """
    from engine import pipeline

    t = np.linspace(0.0, 1.0, int(size))
    b, g, r = np.meshgrid(t, t, t, indexing="ij")
    grid = np.stack([r, g, b], axis=-1)
    run = getattr(pipeline, "_run", None)
    if run is None:  # pragma: no cover
        return {"excursion": float("nan")}, None, None
    raw, nm = run(compiled, grid)
    raw = np.asarray(raw, dtype=np.float64)
    below = float(max(0.0, -raw.min()))
    above = float(max(0.0, raw.max() - 1.0))
    clip = {
        "excursion": max(below, above),
        "below": below,
        "above": above,
        "below_codes": below * 255.0,
        "above_codes": above * 255.0,
        "frac_below": float(np.count_nonzero(raw < 0.0) / raw.size),
        "frac_above": float(np.count_nonzero(raw > 1.0) / raw.size),
    }
    nm_d = nm_photo = None
    if nm is not None:
        from engine.spec import DEFAULT_KNEE

        knee = (DEFAULT_KNEE if compiled.spec.gamut is None
                else float(compiled.spec.gamut.knee))
        nm_d = _nm_stats(nm, knee)
        if photo is not None:
            nm_photo = _nm_stats(run(compiled, photo)[1], knee)
    return clip, nm_d, nm_photo


# ---------------------------------------------------------------------------
# declared targets
# ---------------------------------------------------------------------------


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_declared(name: str, targets_dir: Path | None = None) -> dict | None:
    """``looks/targets/<name>.json`` — the look's declared numbers, if any.

    Recognised keys (all optional): ``tint_rg`` / ``tint_bg`` (7 codes at
    t = 5/18/35/50/65/80/95 %), ``skin_chroma_bright`` / ``skin_chroma_dark``,
    ``neutral_axis`` (bool), ``tags`` (e.g. ``["deutsch"]``), ``fingerprint``
    (a ``latent-probe-1`` fingerprint or a compare-style target profile).
    """
    d = targets_dir or (_root() / "looks" / "targets")
    p = Path(d) / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _photo_sample(photo_sample) -> FloatArray | None:
    if photo_sample is False:
        return None
    if photo_sample is None:
        p = _root() / "work.nosync" / "cal" / "photo_sample.npy"
        if not p.exists():
            return None
        return np.asarray(np.load(p), dtype=np.float64)
    return np.asarray(photo_sample, dtype=np.float64).reshape(-1, 3)


# ---------------------------------------------------------------------------
# the main entry point
# ---------------------------------------------------------------------------


def qc_table(
    source,
    compiled=None,
    *,
    name: str | None = None,
    declared: dict | None = None,
    photo_sample=None,
    grid_n: int = 200_000,
    seed: int = 20260922,
    jacobian: bool = True,
    jac_size: int = 65,
    blend_s: float = 0.70,
    lim_from_diagnostics: bool = True,
) -> dict:
    """Every numeric gate of PLAN §验证, measured on *source*.

    *source* is the shipped ``.cube`` **path** (the intended use: QC reads the
    artifact back from disk), or a raw ``(N,N,N,3)`` table / ``LUT3D`` for
    tests.  *compiled* is the ``engine.pipeline.Compiled`` (or a ``LookSpec``,
    which is compiled here) that produced it; without it the checks that need
    the continuous engine report ``skip``.

    Returns ``{"name", "source", "gates": [...], "summary": {...},
    "metrics": {...}}``.  Every gate carries its measured number, so a FAIL is
    readable without re-running anything.
    """
    from engine import pipeline
    from engine import spec as spec_mod

    if compiled is not None and isinstance(compiled, spec_mod.LookSpec):
        compiled = pipeline.compile(compiled)

    path: Path | None = None
    lut: LUT3D | None = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        lut = read_lut(path)
        table = lut.table
    elif isinstance(source, LUT3D):
        lut = source
        table = source.table
    else:
        table = np.asarray(source, dtype=np.float64)
    table = np.asarray(table, dtype=np.float64)

    if name is None:
        if compiled is not None:
            name = compiled.spec.name
        elif path is not None:
            name = path.stem
        else:
            name = getattr(lut, "title", "table")

    is_mono = compiled is not None and compiled.spec.mono is not None
    sampler = M.sampler_from_table(table, title=name)
    gates: list[dict] = []
    mt: dict = {}

    # -- file / format ------------------------------------------------------
    size = table.shape[0]
    gates.append(_gate("file.size", "file", "LUT_3D_SIZE", float(size),
                       fail_lo=CUBE_SIZE, fail_hi=CUBE_SIZE, fmt="{:.0f}",
                       note="the S9 rejects 65-point cubes"))
    gates.append(_bool_gate("file.rows", "file", "rows == size^3",
                            bool(table.size == size ** 3 * 3),
                            detail=f"{size ** 3} rows"))
    finite = bool(np.all(np.isfinite(table)))
    in01 = bool(np.all((table >= 0.0) & (table <= 1.0)))
    gates.append(_bool_gate("file.finite", "file", "finite and in [0,1]",
                            finite and in01,
                            detail="ok" if (finite and in01) else
                            ("non-finite" if not finite else "out of [0,1]")))
    if path is not None:
        head_ok = any(STD_HEADER in c for c in (lut.comments or ()))
        gates.append(_bool_gate("file.header", "file", "#LUMIXPHOTOSTYLE STD",
                                head_ok, detail="present" if head_ok else "missing"))
        stem_ok = bool(_STEM_RE.match(path.stem))
        gates.append(_bool_gate("file.stem", "file", "stem <= 8 ASCII alnum",
                                stem_ok, detail=path.stem))
    else:
        gates.append(_bool_gate("file.header", "file", "#LUMIXPHOTOSTYLE STD", None))
        gates.append(_bool_gate("file.stem", "file", "stem <= 8 ASCII alnum", None))

    # -- neutral axis -------------------------------------------------------
    neu = M.neutral_stats(sampler)
    mt["neutral"] = neu
    gates.append(_bool_gate("neutral.monotone_rgb", "neutral",
                            "grey monotone per channel", bool(neu["monotone"]),
                            detail=f"min step {neu['min_step'] * 255:+.3e} codes"))
    gates.append(_bool_gate("neutral.monotone_L", "neutral",
                            "grey monotone in OKLab L", bool(neu["monotone_L"]),
                            detail=f"min step {neu['min_step_L']:+.3e}"))
    gates.append(_gate("neutral.white_err", "neutral", "white (1,1,1) error",
                       neu["white_err"], fail_hi=WHITE_ERR, fmt="{:.2e}"))

    ramp_t = np.linspace(0.0, 1.0, 4097)
    ramp_out = np.asarray(sampler(np.stack([ramp_t] * 3, axis=-1)), dtype=np.float64)

    if compiled is not None:
        T = np.asarray(compiled.neutral.T, dtype=np.float64)
        t_T = np.asarray(compiled.neutral.t, dtype=np.float64)
        T_on = np.stack([np.interp(ramp_t, t_T, T[:, c]) for c in range(3)], axis=-1)
        err = np.abs(ramp_out - T_on) * 255.0
        k = int(np.argmax(err.max(axis=1)))
        mt["grey_vs_T"] = {
            "max_codes": float(err.max()),
            "at_t": float(ramp_t[k]),
            "channel": "RGB"[int(np.argmax(err[k]))],
        }
        gates.append(_gate("neutral.vs_target", "neutral", "grey vs designed T",
                           err.max(), fail_hi=GREY_VS_T[0], warn_hi=GREY_VS_T[1],
                           unit="codes", fmt="{:.3f}",
                           note=f"worst at t={ramp_t[k]:.3f} ch {'RGB'[int(np.argmax(err[k]))]}"))
        black_err = float(np.max(np.abs(ramp_out[0] - T_on[0])))
        gates.append(_gate("neutral.black_vs_designed", "neutral",
                           "black vs designed", black_err * 255.0,
                           fail_hi=BLACK_VS_DESIGNED * 255.0, unit="codes", fmt="{:.3f}"))
        # declared tint: the look file's own designed tint, read off T
        des_rg = np.interp(np.asarray(M.NEUTRAL_T), t_T, (T[:, 0] - T[:, 1]) * 255.0)
        des_bg = np.interp(np.asarray(M.NEUTRAL_T), t_T, (T[:, 2] - T[:, 1]) * 255.0)
        dev = max(float(np.max(np.abs(np.asarray(neu["tint_rg"]) - des_rg))),
                  float(np.max(np.abs(np.asarray(neu["tint_bg"]) - des_bg))))
        mt["tint"] = {"measured_rg": neu["tint_rg"], "measured_bg": neu["tint_bg"],
                      "designed_rg": [float(x) for x in des_rg],
                      "designed_bg": [float(x) for x in des_bg]}
        gates.append(_gate("neutral.tint_declared", "neutral", "tint vs declared",
                           dev, fail_hi=TINT_DECLARED, unit="codes", fmt="{:.3f}",
                           note="declared = the look file's own designed tint"))
        ns = compiled.spec.neutral
        if not ns.tint_rg and not ns.tint_bg:
            worst = max(float(np.max(np.abs(neu["tint_rg"]))),
                        float(np.max(np.abs(neu["tint_bg"]))))
            gates.append(_gate("neutral.tint_neutral_axis", "neutral",
                               "tint, neutral-axis look", worst,
                               fail_hi=TINT_NEUTRAL_AXIS[0], warn_hi=TINT_NEUTRAL_AXIS[1],
                               unit="codes", fmt="{:.3f}"))
        else:
            gates.append(_gate("neutral.tint_neutral_axis", "neutral",
                               "tint, neutral-axis look", None, status=SKIP,
                               note="look declares a tint"))
        C = np.asarray(compiled.neutral.C, dtype=np.float64)
        slope = np.gradient(C, t_T, axis=0)
        # the smallest step the camera can actually produce is one 8-bit code;
        # the endpoint derivative of the tone PCHIP at t = 0 is reported
        # separately so a toe that only dips below the gate in the very first
        # sample is visible as such rather than hidden.
        keep = t_T >= 1.0 / 255.0
        inner = slope[keep]
        k_min = int(np.argmin(slope.min(axis=1)))
        k_in = int(np.argmin(inner.min(axis=1)))
        t_in = t_T[keep]
        mt["slope_C"] = {
            "min": float(slope.min()), "max": float(slope.max()),
            "min_at_t": float(t_T[k_min]),
            "min_above_1code": float(inner.min()),
            "min_above_1code_at_t": float(t_in[k_in]),
            "at_t0": [float(x) for x in slope[0]],
        }
        # v1.1 R5: the tone-slope line is >= 0.04 for t >= 1/255 (ENGINE_SPEC
        # §2's own validate number), measured on every channel of C.
        gates.append(_gate("neutral.tone_slope_min", "neutral",
                           "tone slope min, t >= 1/255", inner.min(),
                           fail_lo=TONE_SLOPE_MIN, fmt="{:.4f}",
                           note=f"worst at t={t_in[k_in]:.4f} ch "
                                f"{'RGB'[int(np.argmin(inner[k_in]))]}; the t = 0 "
                                f"PCHIP endpoint derivative is "
                                f"{slope[0].min():.4f} and is outside the gate"))
        # v1.1 R5: "C' gate applies only when film is on."  With [F] off,
        # C_i == T_i exactly, so PLAN's [0.33, 3.0] lands on the tone curve's
        # own toe slope, where ENGINE_SPEC §2 explicitly permits 0.04.
        film_on = getattr(compiled.spec, "film", None) is not None
        gates.append(_gate("neutral.slope_C_min", "neutral", "correction slope C' min",
                           slope.min() if film_on else None,
                           fail_lo=SLOPE_C_FAIL[0], warn_lo=SLOPE_C_WARN[0],
                           fmt="{:.3f}", status=None if film_on else SKIP,
                           note=f"at t={t_T[k_min]:.4f}; min for t >= 1/255 is "
                                f"{inner.min():.3f}"
                                + ("" if film_on else
                                   "; [F] is off so C == T and R5 scopes this gate "
                                   "to film looks — neutral.tone_slope_min is the "
                                   "line that applies")))
        gates.append(_gate("neutral.slope_C_max", "neutral", "correction slope C' max",
                           slope.max() if film_on else None,
                           fail_hi=SLOPE_C_FAIL[1], warn_hi=SLOPE_C_WARN[1],
                           fmt="{:.3f}", status=None if film_on else SKIP,
                           note=f"measured {slope.max():.3f}"
                                + ("" if film_on else "; [F] off -> R5 skips the gate")))
    else:
        for k_, lab_ in (("neutral.vs_target", "grey vs designed T"),
                         ("neutral.black_vs_designed", "black vs designed"),
                         ("neutral.tint_declared", "tint vs declared"),
                         ("neutral.tint_neutral_axis", "tint, neutral-axis look"),
                         ("neutral.tone_slope_min", "tone slope min, t >= 1/255"),
                         ("neutral.slope_C_min", "correction slope C' min"),
                         ("neutral.slope_C_max", "correction slope C' max")):
            gates.append(_gate(k_, "neutral", lab_, None, status=SKIP,
                               note="needs the compiled look"))

    # -- smoothness ---------------------------------------------------------
    d2 = M.second_diff_stats(table)
    mt["d2"] = {"full": d2["full"], "interior": d2["interior"]}
    gates.append(_gate("d2.full_p99_9", "smooth", "d2 p99.9, full lattice",
                       d2["full"]["p99_9"], fail_hi=D2_FULL_P999[0],
                       warn_hi=D2_FULL_P999[1], unit="codes", fmt="{:.2f}"))
    gates.append(_gate("d2.interior_p99_9", "smooth", "d2 p99.9, interior",
                       d2["interior"]["p99_9"], fail_hi=D2_INTERIOR_P999[0],
                       warn_hi=D2_INTERIOR_P999[1], unit="codes", fmt="{:.2f}"))
    gates.append(_gate("d2.full_max", "smooth", "d2 max, full lattice",
                       d2["full"]["max"], fail_hi=D2_FULL_MAX[0],
                       warn_hi=D2_FULL_MAX[1], unit="codes", fmt="{:.2f}"))

    # -- injectivity --------------------------------------------------------
    fold = _fold_block(table, is_mono)
    mt["fold"] = fold
    fold_note = fold["gate_note"]
    # v1.2 W6.  The key stays `fold.neg_count` (tools/round.py's HARD_GATES and
    # every stored report read it); what it *counts* is now material folds.
    gates.append(_gate("fold.neg_count", "fold", "folded tetrahedra (material)",
                       float(fold["gate_count"]), fail_hi=FOLD_MATERIAL[0], fmt="{:.0f}",
                       note=fold_note or
                            f"ratio < -{M.MATERIAL_EPS} of {fold['n_tetra']}; "
                            f"{fold['micro_count']} micro, {fold['crush_count']} crushed, "
                            f"{fold['neg_strict_raw_count']} raw ratio<0"))
    # GATES_v3: "MONO looks skip ... crush, micro, Jacobian" — a rank-1 lattice
    # has zero volume everywhere, so micro and crush are statements about float
    # noise, not about the map.
    gates.append(_gate("fold.micro", "fold", "micro folds (slivers)",
                       None if is_mono else fold["micro_frac"] * 100.0,
                       fail_hi=FOLD_MICRO_PCT[0], warn_hi=FOLD_MICRO_PCT[1],
                       unit="%", fmt="{:.4f}", status=SKIP if is_mono else None,
                       note=(fold_note if is_mono else
                             f"-{M.MATERIAL_EPS} <= ratio < -{FOLD_EPS:.0e}: a crushed "
                             f"sliver on the gamut shell, not a reversal "
                             f"({fold['micro_count']} of {fold['n_tetra']})")))
    gates.append(_gate("fold.crush", "fold", "crushed tetrahedra",
                       None if is_mono else fold["crush_excess_frac"] * 100.0,
                       fail_hi=FOLD_CRUSH_PCT[0], warn_hi=FOLD_CRUSH_PCT[1],
                       unit="%", fmt="{:.3f}", status=SKIP if is_mono else None,
                       note=(fold_note if is_mono else
                             f"|ratio| < {M.CRUSH_EPS}, excess over the identity "
                             f"lattice ({fold['crush_count']} of {fold['n_tetra']})")))
    gates.append(_gate("fold.collapsed", "fold", "collapsed tetrahedra",
                       None if is_mono else fold["collapsed_frac"] * 100.0,
                       warn_hi=FOLD_COLLAPSED_WARN * 100.0, unit="%", fmt="{:.4f}",
                       status=SKIP if is_mono else None,
                       threshold=f"(warn <= {FOLD_COLLAPSED_WARN * 100:.2f} %)",
                       note=(fold_note if is_mono else
                             f"|ratio| <= {FOLD_EPS:.0e}: degenerate, not reversed "
                             f"({fold['collapsed_count']} of {fold['n_tetra']})")))
    gates.append(_gate("fold.min_ratio", "fold", "min tetra volume ratio",
                       fold["min_ratio"], fmt="{:+.3e}", note=fold_note,
                       threshold=f">= -{M.MATERIAL_EPS}  (warn >= +1.000e-04)",
                       status=(FAIL if fold["gate_count"] > 0 else
                               # a mono lattice is collapsed everywhere by
                               # construction: min_ratio ~ 0 is the definition,
                               # not a warning sign
                               PASS if is_mono else
                               WARN if fold["min_ratio"] < FOLD_MIN_RATIO_WARN
                               else PASS)))

    if compiled is not None and jacobian and not is_mono:
        jac = jacobian_check(compiled, size=jac_size)
        mt["jacobian"] = jac
        gates.append(_gate("jacobian.neg_count", "fold",
                           f"engine Jacobian det < {JAC_MATERIAL_DET} ({jac_size}^3)",
                           float(jac["material_count"]), fail_hi=0.0, fmt="{:.0f}",
                           note=f"min det {jac['min_det']:+.3e} of {jac['n']} points; "
                                f"{jac['neg_count']} have det <= 0 (v1.2 W6 gates the "
                                f"material ones only — identity det is 1)"))
    else:
        gates.append(_gate("jacobian.neg_count", "fold",
                           f"engine Jacobian det < {JAC_MATERIAL_DET} ({jac_size}^3)",
                           None, fail_hi=0.0, fmt="{:.0f}", status=SKIP,
                           note="mono: rank-1 map, det is 0 by construction "
                                "(GATES_v3 skips the Jacobian for mono)" if is_mono
                           else "needs the compiled look"))

    # 70 % blend
    tb = M.blend_table(table, blend_s)
    f70 = _fold_block(tb, is_mono)
    d70 = M.second_diff_stats(tb)
    n70 = M.neutral_stats(M.sampler_from_table(tb, title=name))
    mt["blend70"] = {"fold": f70, "d2": d70["full"], "monotone": n70["monotone"]}
    gates.append(_gate("blend70.neg_count", "blend70",
                       f"folded tetrahedra (material) @ {blend_s:.0%}",
                       float(f70["gate_count"]), fail_hi=FOLD_MATERIAL[0], fmt="{:.0f}",
                       note=fold_note or
                            f"{f70['micro_count']} micro, {f70['crush_count']} crushed, "
                            f"{f70['neg_strict_raw_count']} raw ratio<0"))
    gates.append(_gate("blend70.micro", "blend70", f"micro folds @ {blend_s:.0%}",
                       None if is_mono else f70["micro_frac"] * 100.0,
                       fail_hi=FOLD_MICRO_PCT[0], warn_hi=FOLD_MICRO_PCT[1],
                       unit="%", fmt="{:.4f}", status=SKIP if is_mono else None,
                       note=(fold_note if is_mono else
                             f"{f70['micro_count']} of {f70['n_tetra']}")))
    # GATES_v3 REMOVED `blend70.d2_p99_9`: "exactly 0.7 x the full-lattice d2 —
    # redundant and contradictory".  The number is still measured and carried in
    # metrics["blend70"]["d2"]; it is no longer a gate row.
    gates.append(_bool_gate("blend70.monotone", "blend70",
                            f"grey monotone @ {blend_s:.0%}", bool(n70["monotone"])))

    # -- gamut / clipping ---------------------------------------------------
    photo = _photo_sample(photo_sample)
    clip = _clip_block(table)
    mt["clip"] = clip
    gates.append(_gate("clip.at_zero", "clip", "clip plateau at 0",
                       clip["excess_zero"] * 100.0, fail_hi=CLIP_PLATEAU[0] * 100.0,
                       warn_hi=CLIP_PLATEAU[1] * 100.0, unit="%", fmt="{:.3f}",
                       note=f"excess over the identity lattice; raw "
                            f"{clip['zero'] * 100:.3f} % against an identity floor of "
                            f"{clip['identity_zero'] * 100:.3f} %"))
    gates.append(_gate("clip.at_one", "clip", "clip plateau at 1",
                       clip["excess_one"] * 100.0, fail_hi=CLIP_PLATEAU[0] * 100.0,
                       warn_hi=CLIP_PLATEAU[1] * 100.0, unit="%", fmt="{:.3f}",
                       note=f"excess over the identity lattice; raw "
                            f"{clip['one'] * 100:.3f} % against an identity floor of "
                            f"{clip['identity_one'] * 100:.3f} %"))

    if compiled is not None:
        pre, nm_d, nm_p = _preclip_and_nm(compiled, size=size, photo=photo)
        mt["preclip"] = pre
        gates.append(_gate("clip.final_noop", "clip", "final clip is a no-op",
                           pre["excursion"], fail_hi=CLIP_NOOP, fmt="{:.3e}",
                           note=f"{pre['below_codes']:.2f} below / "
                                f"{pre['above_codes']:.2f} above, in codes"))
        if nm_d is not None:
            mt["nm"] = nm_d
            mt["nm_photo"] = nm_p
            # v1.1 R5: `push` over the PHOTO sample, not an absolute nm.
            pu = push_stats(compiled, photo) if photo is not None else None
            mt["push"] = pu
            gates.append(_gate("gamut.push_p99_9", "clip",
                               "out-of-gamut push p99.9 (photo)",
                               None if pu is None else pu["p99_9"],
                               fail_hi=PUSH_P999[0], warn_hi=PUSH_P999[1], fmt="{:.3f}",
                               status=None if pu is not None else SKIP,
                               threshold="(warn <= 0.35; v1.2 W6: WARN-only, "
                                         "R5's 0.60 FAIL line withdrawn)",
                               note=("no photo sample — R5 pins this gate to the photo "
                                     "population" if pu is None else
                                     f"p99.9 of (nm_pre - nm_in) over {pu['n']} px; "
                                     f"max {pu['max']:+.3f}, mean {pu['mean']:+.4f}, "
                                     f"{pu['frac_positive'] * 100:.1f} % pushed outward; "
                                     f"lattice nm p99.9 {nm_d['p99_9']:.3f} / max "
                                     f"{nm_d['max']:.3f}")))
            lim, lim_src = engine_lim(compiled, use_diagnostics=lim_from_diagnostics)
            est = 1.04 * nm_d["max"]
            mt["nm_lim"] = {"lim": lim, "source": lim_src, "qc_estimate": est,
                            "knee": nm_d["knee"]}
            gates.append(_gate("gamut.lim", "clip", "compressor limit lim (lattice)",
                               lim, fail_hi=GAMUT_LIM[0], warn_hi=GAMUT_LIM[1],
                               fmt="{:.3f}", status=None if lim is not None else SKIP,
                               threshold="(warn <= 1.6; v1.2 W6: WARN-only, "
                                         "R5's 1.9 FAIL line withdrawn)",
                               note=(f"from {lim_src}" if lim is not None else
                                     f"{lim_src}; qc's own W5.2 arithmetic on this "
                                     f"lattice would give 1.04*max(nm) = {est:.3f} "
                                     f"(informational, NOT the gate)")))
            # W5.2's slope floor: the compressor's own worst radial gain, s1/q.
            mrg = getattr(compiled, "gamut_min_radial_gain", None)
            mt["nm_lim"]["min_radial_gain"] = mrg
            gates.append(_gate("gamut.min_radial_gain", "clip",
                               "compressor min radial gain", mrg,
                               warn_lo=0.05, fmt="{:.4f}",
                               status=None if mrg is not None else SKIP,
                               threshold="(warn >= 0.05)",
                               note="s1/q — W5.2's slope floor; below 0.05 the "
                                    "compressor is collapsing rather than compressing"))
        else:
            for k_, lab_ in (("gamut.push_p99_9", "out-of-gamut push p99.9 (photo)"),
                             ("gamut.lim", "compressor limit lim (lattice)"),
                             ("gamut.min_radial_gain", "compressor min radial gain")):
                gates.append(_gate(k_, "clip", lab_, None, status=SKIP,
                                   note="mono branch skips [G]" if is_mono else "no gamut stage"))
    else:
        gates.append(_gate("clip.final_noop", "clip", "final clip is a no-op",
                           None, status=SKIP, note="needs the compiled look"))
        for k_, lab_ in (("gamut.push_p99_9", "out-of-gamut push p99.9 (photo)"),
                         ("gamut.lim", "compressor limit lim (lattice)"),
                         ("gamut.min_radial_gain", "compressor min radial gain")):
            gates.append(_gate(k_, "clip", lab_, None, status=SKIP,
                               note="needs the compiled look"))

    # -- grid error (33-point table vs the continuous engine) ---------------
    if compiled is not None:
        rng = np.random.default_rng(seed)
        px = rng.uniform(0.002, 0.998, size=(int(grid_n), 3))
        ge = grid_error(compiled, table, px, name=name)
        mt["grid_random"] = ge
        for key, lab_ in (("mean", "grid err mean, random"),
                          ("p99", "grid err p99, random"),
                          ("max", "grid err max, random")):
            f_, w_ = GRID_RANDOM[key]
            gates.append(_gate(f"grid.random_{key}", "grid", lab_, ge[key],
                               fail_hi=f_, warn_hi=w_, unit="codes", fmt="{:.3f}",
                               threshold=f"(warn <= {w_}; GATES_v3: WARN-only)",
                               note=f"n={ge['n']}; uniform RGB is dominated by the "
                                    f"gamut shell, so GATES_v3 gates the PHOTO "
                                    f"population instead"
                                    + (f" (R5's inner warn was "
                                       f"{GRID_RANDOM_INNER_WARN})" if key == "mean"
                                       else "")))
        if photo is not None:
            gp = grid_error(compiled, table, photo, name=name)
            mt["grid_photo"] = gp
            for key, lab_ in (("mean", "grid err mean, photo"),
                              ("p99", "grid err p99, photo"),
                              ("max", "grid err max, photo")):
                f_, w_ = GRID_PHOTO[key]
                gates.append(_gate(f"grid.photo_{key}", "grid", lab_, gp[key],
                                   fail_hi=f_, warn_hi=w_, unit="codes", fmt="{:.3f}",
                                   note=f"n={gp['n']}"))
        else:
            for key, lab_ in (("mean", "grid err mean, photo"),
                              ("p99", "grid err p99, photo"),
                              ("max", "grid err max, photo")):
                gates.append(_gate(f"grid.photo_{key}", "grid", lab_, None,
                                   status=SKIP, note="no $W/cal/photo_sample.npy"))
    else:
        for key in ("mean", "p99", "max"):
            gates.append(_gate(f"grid.random_{key}", "grid", f"grid err {key}, random",
                               None, status=SKIP, note="needs the compiled look"))
            gates.append(_gate(f"grid.photo_{key}", "grid", f"grid err {key}, photo",
                               None, status=SKIP, note="needs the compiled look"))

    # -- skin ---------------------------------------------------------------
    # GATES_v3: "MONO looks skip every colour-only gate (skin hue band, ...)".
    # A mono output has C == 0, so its hue is undefined and its chroma ratio is
    # 0 by construction: gating either would be gating the mono-ness itself.
    mono_note = "mono: colour-only gate, skipped (GATES_v3)"
    sp = _probe_grid(sampler, np.linspace(0.30, 0.85, 8), (0.06, 0.09, 0.12, 0.16),
                     np.arange(35.0, 70.1, 2.5))
    h_lo = h_hi = None
    if sp["n"]:
        h_lo, h_hi = float(sp["h_out"].min()), float(sp["h_out"].max())
        mt["skin_band"] = {"h_out_min": h_lo, "h_out_max": h_hi, "n": sp["n"]}
    band_skip = SKIP if (is_mono or not sp["n"]) else None
    band_note = mono_note if is_mono else ("" if sp["n"] else "no in-gamut skin probe")
    gates.append(_gate("skin.hue_band_lo", "skin", "skin output hue min",
                       None if band_skip else h_lo, fail_lo=SKIN_HUE_BAND[0],
                       unit="deg", fmt="{:.2f}", status=band_skip, note=band_note))
    gates.append(_gate("skin.hue_band_hi", "skin", "skin output hue max",
                       None if band_skip else h_hi, fail_hi=SKIN_HUE_BAND[1],
                       unit="deg", fmt="{:.2f}", status=band_skip, note=band_note))
    ss = _skin_scan(sampler)
    mt["skin_scan"] = ss
    # GATES_v3 WARN-only: "`skin.displacement` (FAIL only above 0.030)".
    gates.append(_gate("skin.displacement", "skin", "skin hue displacement",
                       None if is_mono else ss["core_hue_disp_max"],
                       fail_hi=SKIN_DISPLACEMENT[0], warn_hi=SKIN_DISPLACEMENT[1],
                       fmt="{:.4f}", status=SKIP if is_mono else None,
                       note=(mono_note if is_mono else
                             f"core drift {ss['core_hue_drift_deg_max']:.2f} deg; "
                             f"GATES_v3 FAILs only above {SKIN_DISPLACEMENT[0]}")))
    patches = _skin_patches(sampler)
    mt["skin_patches"] = patches

    # GATES_v3 HARD: the skin chroma ratios are ABSOLUTE lines now, not a
    # comparison against looks/targets/<name>.json (obsolete).  Bright is the
    # "population L > 0.75" the ruling names; dark stays R5's L 0.45-0.62.
    bright = _probe_grid(sampler, np.linspace(SKIN_BRIGHT_L[0], SKIN_BRIGHT_L[1], 4),
                         (0.07, 0.10), np.arange(38.0, 66.1, 4.0))
    dark = _probe_grid(sampler, np.linspace(SKIN_DARK_L[0], SKIN_DARK_L[1], 4),
                       (0.06, 0.09), np.arange(38.0, 66.1, 4.0))
    cr_bright = float(np.mean(bright["cr"])) if bright["n"] else float("nan")
    cr_dark = float(np.mean(dark["cr"])) if dark["n"] else float("nan")
    mt["skin_chroma"] = {
        "bright": cr_bright, "dark": cr_dark,
        "dark_population": f"L {SKIN_DARK_L[0]:.2f}-{SKIN_DARK_L[1]:.2f} (v1.1 R5)",
        "bright_population": f"L {SKIN_BRIGHT_L[0]:.2f}-{SKIN_BRIGHT_L[1]:.2f} "
                             f"(GATES_v3: L > 0.75)",
        "n_bright": int(bright["n"]), "n_dark": int(dark["n"]),
        "fail_below": {"bright": SKIN_CHROMA_BRIGHT, "dark": SKIN_CHROMA_DARK},
    }
    dec = declared if declared is not None else load_declared(name)
    mt["declared"] = dec
    for key, lab_, meas, line, pop, npx in (
            ("skin_chroma_bright", "bright-skin chroma ratio", cr_bright,
             SKIN_CHROMA_BRIGHT, "L > 0.75", bright["n"]),
            ("skin_chroma_dark", "dark-skin chroma ratio", cr_dark,
             SKIN_CHROMA_DARK, f"L {SKIN_DARK_L[0]}-{SKIN_DARK_L[1]}", dark["n"])):
        gates.append(_gate(f"skin.{key}", "skin", lab_,
                           None if (is_mono or not npx) else meas,
                           fail_lo=line, fmt="{:.3f}",
                           status=SKIP if (is_mono or not npx) else None,
                           note=(mono_note if is_mono else
                                 f"population {pop}, n={int(npx)} probe points; "
                                 f"GATES_v3 absolute line (looks/targets is obsolete)")))

    # -- dE00 vs identity ---------------------------------------------------
    de_points = photo if photo is not None else M.identity_table(17).reshape(-1, 3)
    de_out = np.asarray(sampler(de_points), dtype=np.float64)
    de = M.de00(M.srgb_to_lab(de_points), M.srgb_to_lab(de_out))
    mt["de00_vs_identity"] = {
        "n": int(de.size), "mean": float(de.mean()),
        "p95": float(np.percentile(de, 95.0)), "max": float(de.max()),
        "population": "photo" if photo is not None else "lattice17",
    }
    # v1.1 R5: the dE00 bands are WARN only — the lead judges strength by eye.
    gates.append(_gate("de00.vs_identity", "colour", "dE00 vs identity (mean)",
                       float(de.mean()),
                       warn_lo=DE00_IDENTITY_BAND[0], warn_hi=DE00_IDENTITY_BAND[1],
                       fmt="{:.2f}",
                       note=f"strength of the look, WARN only (R5); inner band "
                            f"{DE00_IDENTITY_WARN[0]}-{DE00_IDENTITY_WARN[1]}; "
                            + ("photo sample" if photo is not None else "17^3 lattice")))

    # -- fingerprint vs declared target -------------------------------------
    # GATES_v3 REMOVED `fingerprint.residual` together with the declared-target
    # skin gates: "looks/targets is obsolete".  tools/fingerprint.py still
    # measures and compares fingerprints; it is simply not a shipping gate.

    # -- the twelve anti-pattern gates --------------------------------------
    gates.extend(_antipattern_gates(sampler, ramp_t, ramp_out, neu, d2, patches,
                                    photo, dec, mt, is_mono))

    report = {
        "name": name,
        "source": str(path) if path is not None else None,
        "size": int(size),
        "mono": bool(is_mono),
        "compiled": compiled is not None,
        "warnings": list(getattr(compiled, "warnings", ()) or ()),
        "gates": gates,
        "metrics": mt,
    }
    report["summary"] = gate_summary(gates)
    report["fails"] = [g["key"] for g in gates if g["status"] == FAIL]
    report["warns"] = [g["key"] for g in gates if g["status"] == WARN]
    # GATES_v3's own HARD list, separated out: these are the lines the lead
    # says block shipping, in the order the ruling writes them.
    report["hard_fails"] = hard_fails(gates)
    report["other_fails"] = [k for k in report["fails"]
                             if k not in set(report["hard_fails"])]
    return report


def qc_cube(path, compiled=None, **kw) -> dict:
    """QC the ``.cube`` at *path* (read back from disk)."""
    return qc_table(Path(path), compiled, **kw)


# ---------------------------------------------------------------------------
# the twelve anti-pattern gates (PLAN §验证)
# ---------------------------------------------------------------------------


def _antipattern_gates(sampler, ramp_t, ramp_out, neu, d2, patches, photo, dec, mt,
                       is_mono: bool = False) -> list[dict]:
    g: list[dict] = []
    tags = set((dec or {}).get("tags", ()))
    #: GATES_v3: "MONO looks skip every colour-only gate (skin hue band, grey
    #: skin, muddy cyan, primary crush, lamp/skin collision, highlight cast
    #: becomes |tint| <= 1.5 codes at t >= 0.80, crush, micro, Jacobian)."  The
    #: tone gates (hair slope, black lift, banding) and the tint gates still
    #: apply to a mono look — that is where a mono look can actually go wrong.
    mono_note = "mono: colour-only gate, skipped (GATES_v3)"

    # 1. 染色白 — a tinted white.
    near_white = ramp_t >= AP_WHITE_T
    wt = float(np.max(np.abs(ramp_out[near_white, 0] - ramp_out[near_white, 1])
                      .max() * 255.0))
    wt = max(wt, float(np.abs(ramp_out[near_white, 2] - ramp_out[near_white, 1]).max() * 255.0))
    mt["ap_white_tint"] = wt
    g.append(_gate("ap.white_tint", "antipattern",
                   f"1 tinted white (t>={AP_WHITE_T})", wt,
                   fail_hi=AP_WHITE_TINT[0], warn_hi=AP_WHITE_TINT[1],
                   unit="codes", fmt="{:.3f}"))

    # 2. 灰肤 — grey skin.
    cr_min = float(np.min(patches["cr"]))
    mt["ap_grey_skin"] = cr_min
    g.append(_gate("ap.grey_skin", "antipattern", "2 grey skin (min patch C ratio)",
                   None if is_mono else cr_min,
                   fail_lo=AP_GREY_SKIN[0], warn_lo=AP_GREY_SKIN[1], fmt="{:.3f}",
                   status=SKIP if is_mono else None, note=mono_note if is_mono else ""))

    # 3. 荧光绿 — fluorescent green.
    gp = _probe_grid(sampler, _PROBE_L, _PROBE_C, _GREEN_H)
    cr_hi = float(np.max(gp["cr"])) if gp["n"] else float("nan")
    mt["ap_neon_green"] = cr_hi
    g.append(_gate("ap.neon_green", "antipattern", "3 fluorescent green (max C ratio)",
                   None if is_mono else cr_hi,
                   fail_hi=AP_NEON_GREEN[0], warn_hi=AP_NEON_GREEN[1], fmt="{:.3f}",
                   status=SKIP if is_mono else None,
                   note=mono_note if is_mono else f"h 110-165, n={gp['n']}"))

    # 4. 浑浊青蓝 — muddy cyan / blue.
    cp = _probe_grid(sampler, _PROBE_L, _PROBE_C, _CYAN_H)
    cr_lo = float(np.min(cp["cr"])) if cp["n"] else float("nan")
    mt["ap_muddy_cyan"] = cr_lo
    g.append(_gate("ap.muddy_cyan", "antipattern", "4 muddy cyan/blue (min C ratio)",
                   None if is_mono else cr_lo,
                   fail_lo=AP_MUDDY_CYAN[0], warn_lo=AP_MUDDY_CYAN[1], fmt="{:.3f}",
                   status=SKIP if is_mono else None,
                   note=mono_note if is_mono else f"h 190-250, n={cp['n']}"))

    # 5. 肤色暗部品红 — magenta in the skin shadows.
    dp = _probe_grid(sampler, (0.28, 0.34, 0.40), (0.05, 0.07, 0.09),
                     np.arange(38.0, 62.1, 3.0))
    dh_lo = float(np.min(dp["dh"])) if dp["n"] else float("nan")
    h_lo = float(np.min(dp["h_out"])) if dp["n"] else float("nan")
    mt["ap_skin_shadow"] = {"h_out_min": h_lo, "dh_min": dh_lo}
    g.append(_gate("ap.skin_shadow_magenta", "antipattern",
                   "5 magenta skin shadows (min h_out)", None if is_mono else h_lo,
                   fail_lo=AP_SKIN_SHADOW_H[0], warn_lo=AP_SKIN_SHADOW_H[1],
                   unit="deg", fmt="{:.2f}", status=SKIP if is_mono else None,
                   note=(mono_note if is_mono else
                         f"L 0.28-0.40, n={dp['n']}, min dh {dh_lo:+.2f} deg")))

    # 6. 发丝死黑 — crushed hair: PLAN pins the 0-25 % grey slope at >= 0.18.
    lo = ramp_t <= 0.25
    slope_lo = float((ramp_out[lo][-1, 1] - ramp_out[lo][0, 1]) / (ramp_t[lo][-1] - ramp_t[lo][0]))
    mt["ap_hair_slope"] = slope_lo
    g.append(_gate("ap.hair_black", "antipattern", "6 crushed hair (0-25 % slope)",
                   slope_lo, fail_lo=AP_HAIR_SLOPE, fmt="{:.3f}"))

    # 7. 渐变断层 — banding.  The measurable form is the interior second
    #    difference, which is *exactly* the statistic `d2.interior_p99_9` (a
    #    GATES_v3 HARD gate) already reads.  docs/REVIEW_r1.md, tooling rulings:
    #    "`ap.banding` is an alias of `d2.interior_p99_9`: report it as info,
    #    never as a second FAIL/WARN."  So the row keeps the number and the
    #    thresholds in its text, and carries the INFO status: one defect, one
    #    gate, counted once.
    g.append(_gate("ap.banding", "antipattern", "7 banding (d2 p99.9 interior)",
                   d2["interior"]["p99_9"], unit="codes", fmt="{:.2f}",
                   status=INFO,
                   threshold=f"alias of d2.interior_p99_9 (<= {D2_INTERIOR_P999[0]}"
                             f", warn > {D2_INTERIOR_P999[1]})",
                   note="info alias of d2.interior_p99_9 — the same statistic, "
                        "gated there (REVIEW_r1); never a second FAIL/WARN"))

    # 8. 黑位过抬 — lifted black, PLAN: <= 8 codes.
    blk = float(np.mean(ramp_out[0]) * 255.0)
    mt["ap_black_lift"] = blk
    g.append(_gate("ap.black_lift", "antipattern", "8 lifted black", blk,
                   fail_hi=AP_BLACK_LIFT, unit="codes", fmt="{:.2f}"))

    # 9. HIGHLIGHT CLEANLINESS (GATES_v3, replacing "实为白平衡预设"/`ap.wb_preset`).
    #    The statistic is unchanged — the mean cast the look ADDS to a pixel,
    #    in codes, for R-G and B-G — but the population is now the near-neutral
    #    HIGHLIGHTS (input C0 <= 0.03 and input L >= 0.80).  The lead's reason:
    #    the all-tones near-neutral population "fails every look with a designed
    #    shadow/mid tint", because a tint crossover is *meant* to move the
    #    shadows.  What a look may not do is leave a cast in the highlights,
    #    where the white guard is supposed to hold.
    #    n (the number of pixels in the population) is reported, and below 500
    #    the gate falls back to input L >= 0.70.
    #    A MONO look has no colour at all, so the same 1.5-code line is read off
    #    the grey ramp for t >= 0.80 instead.
    g.append(_highlight_clean_gate(sampler, ramp_t, ramp_out, photo, mt, is_mono))

    # 10. 原色裁死 — the saturated hues crushed into one ridge.
    hs = np.arange(0.0, 360.0, 15.0)
    cm = M.cmax(np.full(hs.shape, 0.50), hs)
    codes, _ok = M.code_from_oklch(np.full(hs.shape, 0.50), cm * 0.98, hs)
    out = np.asarray(sampler(np.clip(codes, 0.0, 1.0)), dtype=np.float64)
    lab = M.srgb_to_lab(out)
    step = M.de00(lab, np.roll(lab, -1, axis=0))
    prim = float(step.min())
    mt["ap_primary_crush"] = prim
    g.append(_gate("ap.primary_crush", "antipattern",
                   "10 primaries crushed (min adj dE00)", None if is_mono else prim,
                   fail_lo=AP_PRIMARY_CRUSH[0], warn_lo=AP_PRIMARY_CRUSH[1], fmt="{:.2f}",
                   status=SKIP if is_mono else None,
                   note=mono_note if is_mono else "24 hues at L=0.50, C=0.98*Cmax"))

    # 11. 灯光/肤色色相相撞 — lamp light and skin must stay >= 8 deg apart.
    skin_p = _probe_grid(sampler, (0.55, 0.65), (0.09,), (45.0, 50.0, 55.0))
    lamp_p = _probe_grid(sampler, (0.78, 0.86), (0.08, 0.11), (75.0, 82.0, 88.0))
    if skin_p["n"] and lamp_p["n"]:
        sep = float(np.min(np.abs(_wrap180(lamp_p["h_out"][:, None] - skin_p["h_out"][None, :]))))
    else:
        sep = float("nan")
    mt["ap_lamp_skin_sep"] = sep
    g.append(_gate("ap.lamp_skin_collision", "antipattern",
                   "11 lamp/skin hue collision", None if is_mono else sep,
                   fail_lo=AP_LAMP_SKIN_SEP, unit="deg", fmt="{:.2f}",
                   status=SKIP if is_mono else None,
                   note=(mono_note if is_mono else
                         "skin h 45-55 vs lamp h 75-88, output hues")))

    # 12. 德味无冷暖亮度分离 — only a gate for a look that declares the trait.
    warm = _probe_grid(sampler, (0.45, 0.60, 0.75), (0.10,), (45.0, 55.0, 65.0))
    cool = _probe_grid(sampler, (0.45, 0.60, 0.75), (0.10,), (235.0, 250.0, 265.0))
    split = (float(np.mean(warm["dl"]) - np.mean(cool["dl"]))
             if warm["n"] and cool["n"] else float("nan"))
    mt["ap_warm_cool_dl"] = split
    declared_deutsch = bool(tags & {"deutsch", "德味", "germanic"}) and not is_mono
    g.append(_gate("ap.warm_cool_split", "antipattern",
                   "12 warm/cool L separation", abs(split) if declared_deutsch else None,
                   fail_lo=AP_WARM_COOL_DL, fmt="{:.4f}",
                   status=None if declared_deutsch else SKIP,
                   note=(mono_note if is_mono else f"measured {split:+.4f}"
                         + ("" if declared_deutsch
                            else "; only gated for looks tagged 'deutsch'"))))
    return g


def _highlight_clean_gate(sampler, ramp_t, ramp_out, photo, mt,
                          is_mono: bool = False) -> dict:
    """GATES_v3's highlight-cleanliness gate (it replaces ``ap.wb_preset``).

    Colour looks: the mean cast the LUT **adds** over near-neutral highlight
    photo pixels — ``max(|mean d(R-G)|, |mean d(B-G)|)`` in 8-bit codes, over
    the pixels whose *input* has OKLab ``C0 <= 0.03`` and ``L >= 0.80``.  The
    population size ``n`` is always reported; if ``n < 500`` the population
    falls back to the same near-neutral rule at input ``L >= 0.70``, and the
    note says which population was used.

    (The *absolute* output ``|R-G|`` is reported too but is not the line: the
    photo sample's own near-neutral highlights already read 6.7 codes of
    ``|R-G|`` before any LUT touches them, so an absolute line at 1.5 codes
    would fail the identity.  What a look is responsible for is what it adds.)

    Mono looks: "highlight cast becomes ``|tint| <= 1.5`` codes at ``t >=
    0.80``" — read off the grey ramp, since a mono LUT has no colour to measure
    on a photograph.
    """
    if is_mono:
        hi = np.asarray(ramp_t) >= AP_HILITE_MONO_T
        out = np.asarray(ramp_out)[hi]
        rg = float(np.max(np.abs(out[:, 0] - out[:, 1])) * 255.0)
        bg = float(np.max(np.abs(out[:, 2] - out[:, 1])) * 255.0)
        tint = max(rg, bg)
        mt["ap_highlight_clean"] = {"mono": True, "tint_rg": rg, "tint_bg": bg,
                                    "n": int(hi.sum()), "t_min": AP_HILITE_MONO_T,
                                    "population": f"grey ramp t >= {AP_HILITE_MONO_T}"}
        return _gate("ap.highlight_clean", "antipattern",
                     "9 highlight cleanliness (mono |tint|)", tint,
                     fail_hi=AP_HILITE_CLEAN, unit="codes", fmt="{:.3f}",
                     note=f"MONO: max |R-G| {rg:.2f}, |B-G| {bg:.2f} over "
                          f"{int(hi.sum())} ramp samples at t >= {AP_HILITE_MONO_T}")
    if photo is None:
        mt["ap_highlight_clean"] = None
        return _gate("ap.highlight_clean", "antipattern",
                     "9 highlight cleanliness (mean cast)", None, status=SKIP,
                     note="no $W/cal/photo_sample.npy — GATES_v3 pins this gate to "
                          "the photo population")
    photo = np.asarray(photo, dtype=np.float64)
    out = np.asarray(sampler(photo), dtype=np.float64)
    L0, C0, _h0 = M.oklch_from_code(photo)
    near = C0 <= AP_HILITE_C0

    def _cast(mask):
        p, o = photo[mask], out[mask]
        d_rg = float(np.mean((o[:, 0] - o[:, 1]) - (p[:, 0] - p[:, 1])) * 255.0)
        d_bg = float(np.mean((o[:, 2] - o[:, 1]) - (p[:, 2] - p[:, 1])) * 255.0)
        return d_rg, d_bg, {
            "abs_d_rg": float(np.mean(np.abs((o[:, 0] - o[:, 1])
                                             - (p[:, 0] - p[:, 1]))) * 255.0),
            "abs_d_bg": float(np.mean(np.abs((o[:, 2] - o[:, 1])
                                             - (p[:, 2] - p[:, 1]))) * 255.0),
            "out_rg": float(np.mean(np.abs(o[:, 0] - o[:, 1])) * 255.0),
            "out_bg": float(np.mean(np.abs(o[:, 2] - o[:, 1])) * 255.0),
            "in_rg": float(np.mean(np.abs(p[:, 0] - p[:, 1])) * 255.0),
            "in_bg": float(np.mean(np.abs(p[:, 2] - p[:, 1])) * 255.0),
        }

    mask = near & (L0 >= AP_HILITE_L)
    n = int(mask.sum())
    l_used, fell_back = AP_HILITE_L, False
    if n < AP_HILITE_MIN_N:
        mask = near & (L0 >= AP_HILITE_L_FALLBACK)
        l_used, fell_back = AP_HILITE_L_FALLBACK, True
    n_used = int(mask.sum())
    if n_used == 0:
        mt["ap_highlight_clean"] = {"n": 0, "n_highlight": n, "fell_back": fell_back}
        return _gate("ap.highlight_clean", "antipattern",
                     "9 highlight cleanliness (mean cast)", None, status=SKIP,
                     note=f"no near-neutral pixel (C0 <= {AP_HILITE_C0}) at any "
                          f"lightness in this {photo.shape[0]}-px sample")
    d_rg, d_bg, extra = _cast(mask)
    cast = max(abs(d_rg), abs(d_bg))
    mt["ap_highlight_clean"] = {
        "mono": False, "d_rg": d_rg, "d_bg": d_bg, "cast": cast,
        "n": n_used, "n_highlight": n, "n_photo": int(photo.shape[0]),
        "L_min": l_used, "C0_max": AP_HILITE_C0, "fell_back": fell_back,
        "population": f"input C0 <= {AP_HILITE_C0} and input L >= {l_used}",
        **extra,
    }
    note = (f"n={n_used} px (input C0 <= {AP_HILITE_C0}, input L >= {l_used})"
            + (f" — FELL BACK from L >= {AP_HILITE_L}, which held only {n} px "
               f"(< {AP_HILITE_MIN_N})" if fell_back else "")
            + f"; d(R-G) {d_rg:+.2f}, d(B-G) {d_bg:+.2f}; mean |d| "
              f"{extra['abs_d_rg']:.2f}/{extra['abs_d_bg']:.2f}; output |R-G| "
              f"{extra['out_rg']:.2f} against an input |R-G| of {extra['in_rg']:.2f}")
    return _gate("ap.highlight_clean", "antipattern",
                 "9 highlight cleanliness (mean cast)", cast,
                 fail_hi=AP_HILITE_CLEAN, unit="codes", fmt="{:.3f}", note=note)


# ---------------------------------------------------------------------------
# pairwise dE00 (the build-level gate)
# ---------------------------------------------------------------------------


def pairwise_de00(tables: dict, *, photo_sample=None, n: int = 60_000,
                  seed: int = 424242) -> dict:
    """Mean dE00 between every pair of built looks.

    PLAN: mean pairwise dE00 >= 2.5 (warn < 3.5) — two looks that land on top of
    each other are one look.  Measured on the photo sample when it exists
    (subsampled to *n* pixels for speed), otherwise a 17³ lattice.
    """
    names = sorted(tables)
    photo = _photo_sample(photo_sample)
    if photo is not None:
        if photo.shape[0] > n:
            idx = np.random.default_rng(seed).choice(photo.shape[0], int(n), replace=False)
            photo = photo[idx]
        pts = photo
        pop = "photo"
    else:
        pts = M.identity_table(17).reshape(-1, 3)
        pop = "lattice17"
    labs = {}
    for nm in names:
        s = M.sampler_from_table(tables[nm], title=nm)
        labs[nm] = M.srgb_to_lab(np.asarray(s(pts), dtype=np.float64))
    pairs = []
    worst = None
    for i, a in enumerate(names):
        for bnm in names[i + 1:]:
            mean = float(M.de00(labs[a], labs[bnm]).mean())
            pairs.append({"a": a, "b": bnm, "mean_de00": mean})
            if worst is None or mean < worst["mean_de00"]:
                worst = pairs[-1]
    return {
        "population": pop, "n": int(pts.shape[0]), "pairs": pairs,
        "min_mean_de00": None if worst is None else worst["mean_de00"],
        "worst_pair": worst,
        # docs/REVIEW_r1.md "Tooling rulings": "Distinctiveness gate for the
        # build: pairwise photo dE00 < 1.8 = FAIL, < 2.5 = WARN (was WARN
        # only)."  This is the ONE build-level gate that can FAIL besides the
        # per-look GATES_v3 HARD list.
        "gate": {"fail_below": PAIRWISE_DE00[0], "warn_below": PAIRWISE_DE00[1],
                 "plan_fail_below": PAIRWISE_DE00_PLAN[0],
                 "note": "REVIEW_r1: FAIL < 1.8, WARN < 2.5"},
        "status": (SKIP if worst is None else
                   FAIL if worst["mean_de00"] < PAIRWISE_DE00[0] else
                   WARN if worst["mean_de00"] < PAIRWISE_DE00[1] else PASS),
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

_MARK = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "skip", INFO: "info"}


def format_table(report: dict, *, verbose: bool = False, width: int = 34) -> str:
    """The compact printed table: one line per gate."""
    lines: list[str] = []
    s = report["summary"]
    head = (f"{report['name']}  [{report.get('source') or 'in-memory table'}]  "
            f"{s[PASS]} pass / {s[WARN]} warn / {s[FAIL]} FAIL / {s[SKIP]} skip"
            + (f" / {s[INFO]} info" if s.get(INFO) else ""))
    lines.append(head)
    lines.append("-" * max(len(head), 92))
    group = None
    for g in report["gates"]:
        if not verbose and g["status"] == SKIP:
            continue
        if g["group"] != group:
            group = g["group"]
            lines.append(f"[{group}]")
        val = g["text"] + (f" {g['unit']}" if g["unit"] else "")
        line = f"  {_MARK[g['status']]}  {g['label']:<{width}} {val:>13}   {g['threshold']}"
        if g["note"] and (verbose or g["status"] in (WARN, FAIL, INFO)):
            line += f"\n        ^ {g['note']}"
        lines.append(line)
    if report.get("warnings"):
        lines.append("[compile warnings]")
        for w in report["warnings"]:
            lines.append(f"  ..  {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tools/qc.py",
        description="Numeric gates (PLAN §验证 数值门) on shipped .cube files.",
    )
    ap.add_argument("cubes", nargs="+", help=".cube files to check")
    ap.add_argument("--look", action="append", default=[],
                    help="look name or json to compile and pair with the cube of the "
                         "same position (enables the engine-side gates)")
    ap.add_argument("--no-jacobian", action="store_true", help="skip the 65^3 Jacobian")
    ap.add_argument("--grid-n", type=int, default=200_000)
    ap.add_argument("--no-photo", action="store_true", help="ignore photo_sample.npy")
    ap.add_argument("--no-diag-lim", action="store_true",
                    help="do not call engine.pipeline.diagnostics() looking for `lim`")
    ap.add_argument("--json", type=Path, default=None, help="write the reports here")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write qc_<name>.json per cube into this directory")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    from engine import pipeline, spec as spec_mod

    reports = []
    worst = 0
    for i, cube in enumerate(a.cubes):
        compiled = None
        if i < len(a.look):
            compiled = pipeline.compile(spec_mod.load_look(a.look[i]))
        rep = qc_table(
            Path(cube), compiled,
            photo_sample=False if a.no_photo else None,
            grid_n=a.grid_n, jacobian=not a.no_jacobian,
            lim_from_diagnostics=not a.no_diag_lim,
        )
        reports.append(rep)
        print(format_table(rep, verbose=a.verbose))
        print()
        worst = max(worst, _RANK[rep["summary"]["worst"]])
        if a.out_dir:
            a.out_dir.mkdir(parents=True, exist_ok=True)
            (a.out_dir / f"qc_{rep['name']}.json").write_text(
                json.dumps(rep, indent=1, default=float), encoding="utf-8")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(reports, indent=1, default=float), encoding="utf-8")
    return 1 if worst >= _RANK[FAIL] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
