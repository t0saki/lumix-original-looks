"""``tools/fit.py`` — fixed-point table compensation in **measurement space**.

PLAN 附录 A's twelve target profiles are *fingerprints*: "what an end-to-end
measurement of the finished LUT should read", not engine parameters.  Typing
them straight into ``looks/<name>.json`` does not work — the engine's own
per-channel tone curve already contributes a chunk of every chroma ratio
(``C_out/C_in ~ local_slope**0.57``) and of every ΔL, the arch is normalised to
1 at ``L0 = 0.65``, vibrance multiplies the chroma table, and the skin window
attenuates all three.  The demo looks built that way measure d2 p99.9 = 18–28
codes and fold 6,000–26,000 tetrahedra.

The engine's tables are nevertheless *interpretable* measurement-space
quantities (ENGINE_SPEC §3 designs them that way), so the map from parameters to
measurements is close to the identity and a damped fixed point converges in a
handful of iterations::

    loop:
        compile(look)  ->  measure with probe latent-probe-1
        cr_k   *= (target / measured) ** relax
        dh10_k += relax * (target - measured)                    # C = 0.10 column
        dh18_k += relax * (target - measured) / g18_k            # high-chroma column
        dl_k   += relax * (target - measured) / leverage
        arch[f][j] *= (target / measured) ** relax               # normalised at L0 = 0.65
        skin chroma / hue offset / l_lift                        # against the 3 patches
        tint bumps: scipy.least_squares against the 7 samples
    until normalised residual RMS < 0.5 or 12 iterations

Every update is **clamped by** :func:`engine.spec.validate`: a candidate that
does not validate is retried with a halved step, and if it still does not
validate the previous look is kept and a *clamp event* is recorded.  A gate is
never traded for a target.

Normalisation of the residual is the probe's own (``tools.fingerprint.NORM``):
hue °/3, chroma ratio /0.05, 8-bit code /2, OKLab ΔL /0.01.

CLI::

    py tools/fit.py all                      # fit every looks/targets/*.json
    py tools/fit.py fit 01Glaze --iters 12   # one look, verbose
    py tools/fit.py show 01Glaze             # residual table of the look on disk
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from engine import curves, field as _field, pipeline
from engine.neutral import build_target, validate_neutral
from engine.spec import (DEFAULT_END_SLOPE, DEFAULT_KNEE, DEFAULT_P, Gates,
                         LookSpec, Neutral, SpecError, validate)
from tools import fingerprint as FP
from tools import metrics as M

__all__ = [
    "KNOT_H", "KNOT_IDX", "CHI", "NORM",
    "load_target", "measure", "residual_rows", "residual_rms",
    "initial_look", "fit_tint_bumps", "fit_look", "fit_all", "format_residuals",
]

FloatArray = np.ndarray

# ---------------------------------------------------------------------------
# the probe points fit works on (a strict subset of latent-probe-1)
# ---------------------------------------------------------------------------

#: the 12 hue knots of a look table, h0 = 0, 30, ..., 330
KNOT_H: FloatArray = np.arange(12, dtype=np.float64) * 30.0
#: where those knots sit inside the probe's 24-hue arrays (h = 0, 15, ..., 345)
KNOT_IDX: tuple[int, ...] = tuple(range(0, 24, 2))

NORM = FP.NORM

_ARCH_L = np.asarray(FP.ARCH_L, dtype=np.float64)        # .25 .40 .55 .70 .85
_ARCH_H = np.asarray(FP.ARCH_H, dtype=np.float64)        # 55 (warm) 145 (green) 250 (blue)
_PATCHES = np.asarray(FP.SKIN_PATCHES, dtype=np.float64) / 255.0
_RAMP_CODES = np.asarray(FP.RAMP_CODES, dtype=np.float64) / 255.0
_NEUTRAL_T = np.asarray(FP.NEUTRAL_T, dtype=np.float64)

#: the probe's high-chroma column at the 12 knots: ``min(0.18, 0.80*cmax(0.65,h))``
CHI: FloatArray = np.minimum(FP.CHI_CAP,
                             FP.CHI_FRAC * M.cmax(np.full(12, FP.HUE_L), KNOT_H))

#: OKLab L of CIELAB L* = 45 and L* = 60 — PLAN's foliage layering boundary
L_STAR_45 = float(((45.0 + 16.0) / 116.0) ** 3) ** (1.0 / 3.0)
L_STAR_60 = float(((60.0 + 16.0) / 116.0) ** 3) ** (1.0 / 3.0)

# ---------------------------------------------------------------------------
# fit constants (every one of them is a number the lead can move)
# ---------------------------------------------------------------------------

RELAX = 0.8                 # the task's damping factor
MAX_ITER = 12
TOL_RMS = 0.5               # normalised residual RMS: stop below this
LINE_SEARCH = 6             # halvings before a step is abandoned
#: below this gate opening the high-chroma column has no authority at all and
#: its knot is frozen; DIV floors the 1/g18 step amplification, because a knot
#: that has to be driven to 25 deg to move a measurement by 1 deg is a
#: smoothness disaster in a 12-knot / 30 deg table long before it is a fit.
G18_MIN = 0.35
G18_DIV = 0.50
DH_ABS_MAX = 30.0           # hard clamp on a rotation knot, degrees
#: a rotation knot may not exceed its own target by more than this (deg + factor):
#: R06 §1.3i's curvature budget sigma >= 1.8*sqrt(A/C) is enforced by
#: engine.spec on `ops` but NOT on the hue tables, whose effective width is the
#: 30 deg knot spacing.
DH_HEADROOM = (2.0, 4.0)
CR_CLAMP = (0.45, 1.75)     # engine.spec GAIN_LO/GAIN_HI with a margin
SKINC_CLAMP = (0.45, 1.75)
#: normalised ARCH knot bound.  ENGINE_SPEC §9.3 makes the arch a *residual*
#: operator ("supplies what tone slope does not already give"); a normalised knot
#: outside this is no longer a residual and it is always a symptom of a knot
#: chasing a gamut-limited probe point (green at L0 = 0.25 has cmax ~ 0.07, so
#: the C = 0.10 arch probe there is clipped and [G] eats any gain added).
ARCH_CLAMP = (0.60, 1.50)
#: a target that does not move by this fraction of the requested change is
#: "not reachable"; after STALL_MAX such iterations its knot is frozen.
STALL_GAIN = 0.15
STALL_MAX = 2
#: a residual smaller than this (normalised) is not worth freezing over
STALL_MIN_NORM = 0.6
#: stop when the residual RMS has not improved for this many iterations
PATIENCE = 3
DL_BUDGET = 0.06            # ENGINE_SPEC §3.5/§3.7, on the SUM of the three terms
TINT_A_MAX = 60.0           # bump amplitude bound, 8-bit codes
TINT_SIGMA = (0.15, 0.90)   # ENGINE_SPEC §2: sigma >= 0.15


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def load_target(name: str, targets_dir: Path | None = None) -> dict:
    """``looks/targets/<name>.json`` — the transcribed 附录 A profile."""
    d = Path(targets_dir or (_root() / "looks" / "targets"))
    p = d / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"no target profile at {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("probe") != FP.PROBE_ID:
        raise ValueError(f"{p.name}: probe {doc.get('probe')!r} != {FP.PROBE_ID!r}")
    return doc


def target_arrays(target: dict) -> dict:
    """The 附录 A profile as plain arrays in probe space (NaN = not declared)."""
    fp = target["fingerprint"]

    def knots(block: dict | None, key: str) -> FloatArray:
        if block is None or key not in block:
            return np.full(12, np.nan)
        v = np.asarray([np.nan if x is None else x for x in block[key]], dtype=np.float64)
        return v[list(KNOT_IDX)]

    hue = fp.get("hue", {})
    out = {
        "ramp8": np.asarray(fp["neutral"]["ramp8"], dtype=np.float64),
        "tint_rg": np.asarray(fp["neutral"]["tint_rg"], dtype=np.float64),
        "tint_bg": np.asarray(fp["neutral"]["tint_bg"], dtype=np.float64),
        "dh10": knots(hue.get("c10"), "dh"),
        "cr": knots(hue.get("c10"), "cr"),
        "dl": knots(hue.get("c10"), "dl"),
        "dh18": knots(hue.get("chi"), "dh"),
        "arch_cr": (np.asarray(fp["arch"]["cr"], dtype=np.float64)
                    if "arch" in fp else np.full((3, 5), np.nan)),
    }
    sk = fp.get("skin", {})
    for key in ("dh", "cr", "dl"):
        out[f"skin_{key}"] = (np.asarray(sk[key], dtype=np.float64)
                              if key in sk else np.full(3, np.nan))
    return out


# ---------------------------------------------------------------------------
# measurement — the same code path as tools/fingerprint.py, restricted to the
# items the fit uses (a full fingerprint() also builds two dE00 lattices).
# ---------------------------------------------------------------------------


def measure(sampler: Callable[[FloatArray], FloatArray]) -> dict:
    """Measure *sampler* at exactly the probe points 附录 A declares.

    Uses ``tools.fingerprint``'s own ``_probe_points`` and pinned constants, so
    the numbers are bit-identical to the corresponding entries of
    :func:`tools.fingerprint.fingerprint`; :mod:`tests.test_fit` asserts that.
    """
    L65 = np.full(12, FP.HUE_L)
    p10 = FP._probe_points(sampler, L65, np.full(12, FP.C10), KNOT_H)
    pchi = FP._probe_points(sampler, L65, CHI, KNOT_H)

    Lg = np.broadcast_to(_ARCH_L[None, :], (3, 5))          # rows = warm / green / blue
    hg = np.broadcast_to(_ARCH_H[:, None], (3, 5))
    pa = FP._probe_points(sampler, Lg, np.full((3, 5), FP.ARCH_C), hg)

    out_skin = np.asarray(sampler(_PATCHES), dtype=np.float64)
    l_in, c_in, h_in = M.oklch_from_code(_PATCHES)
    l_out, c_out, h_out = M.oklch_from_code(out_skin)

    ns = M.neutral_stats(sampler)
    ramp = np.asarray(sampler(np.stack([_RAMP_CODES] * 3, axis=-1)), dtype=np.float64)

    return {
        "dh10": p10["dh"], "cr": p10["cr"], "dl": p10["dl"],
        "dh18": pchi["dh"], "chi_c_in": pchi["c_in"], "c10_c_in": p10["c_in"],
        "c10_h_in": p10["h_in"],
        "arch_cr": pa["cr"], "arch_dl": pa["dl"],
        "skin_dh": FP._wrap180(h_out - h_in), "skin_cr": c_out / c_in,
        "skin_dl": l_out - l_in, "skin_l_in": l_in, "skin_c_in": c_in, "skin_h_in": h_in,
        "ramp8": ramp[:, 1] * 255.0,
        "tint_rg": np.asarray(ns["tint_rg"], dtype=np.float64),
        "tint_bg": np.asarray(ns["tint_bg"], dtype=np.float64),
        "black": float(np.mean(ns["black"]) * 255.0),
        #: the 18 % grey output code — the lead asked for it explicitly for the
        #: three looks with deliberately deep lower midtones (input code 31 is
        #: 附录 A's 18 % sample, RAMP_CODES[2]).
        "grey18": float(ramp[2, 1] * 255.0),
    }


# ---------------------------------------------------------------------------
# residuals
# ---------------------------------------------------------------------------

#: (item, kind, shape) of everything the fit scores
RESIDUAL_ITEMS: tuple[tuple[str, str], ...] = (
    ("neutral.ramp8", "code"),
    ("neutral.tint_rg", "code"),
    ("neutral.tint_bg", "code"),
    ("hue.c10.dh", "deg"),
    ("hue.c10.cr", "ratio"),
    ("hue.c10.dl", "dl"),
    ("hue.chi.dh", "deg"),
    ("arch.cr", "ratio"),
    ("skin.dh", "deg"),
    ("skin.cr", "ratio"),
    ("skin.dl", "dl"),
)

_ITEM_KEY = {
    "neutral.ramp8": "ramp8", "neutral.tint_rg": "tint_rg", "neutral.tint_bg": "tint_bg",
    "hue.c10.dh": "dh10", "hue.c10.cr": "cr", "hue.c10.dl": "dl", "hue.chi.dh": "dh18",
    "arch.cr": "arch_cr", "skin.dh": "skin_dh", "skin.cr": "skin_cr", "skin.dl": "skin_dl",
}

_INDEX_LABEL = {
    "hue.c10.dh": lambda i: f"h{int(KNOT_H[i[0]]):d}",
    "hue.c10.cr": lambda i: f"h{int(KNOT_H[i[0]]):d}",
    "hue.c10.dl": lambda i: f"h{int(KNOT_H[i[0]]):d}",
    "hue.chi.dh": lambda i: f"h{int(KNOT_H[i[0]]):d}",
    "arch.cr": lambda i: f"{('warm', 'green', 'blue')[i[0]]}@L{_ARCH_L[i[1]]:.2f}",
    "skin.dh": lambda i: ("mid", "dark", "bright")[i[0]],
    "skin.cr": lambda i: ("mid", "dark", "bright")[i[0]],
    "skin.dl": lambda i: ("mid", "dark", "bright")[i[0]],
    "neutral.ramp8": lambda i: f"code{int(FP.RAMP_CODES[i[0]]):d}",
    "neutral.tint_rg": lambda i: f"t{FP.NEUTRAL_T[i[0]]:.2f}",
    "neutral.tint_bg": lambda i: f"t{FP.NEUTRAL_T[i[0]]:.2f}",
}


def residual_rows(meas: dict, tgt: dict) -> list[dict]:
    """One row per declared target item, worst (largest ``norm``) first."""
    rows: list[dict] = []
    for item, kind in RESIDUAL_ITEMS:
        key = _ITEM_KEY[item]
        t = np.asarray(tgt[key], dtype=np.float64)
        m = np.asarray(meas[key], dtype=np.float64)
        if t.shape != m.shape:
            raise ValueError(f"{item}: target {t.shape} vs measured {m.shape}")
        for ix in np.ndindex(t.shape):
            tv, mv = float(t[ix]), float(m[ix])
            if not np.isfinite(tv):
                continue
            r = mv - tv
            rows.append({
                "item": item, "index": _INDEX_LABEL[item](ix),
                "target": tv, "measured": mv, "residual": r, "kind": kind,
                "norm": abs(r) / NORM[kind] if np.isfinite(r) else float("nan"),
            })
    rows.sort(key=lambda r: -(r["norm"] if r["norm"] == r["norm"] else 1e9))
    return rows


def residual_rms(rows: Sequence[dict]) -> float:
    v = [r["norm"] for r in rows if r["norm"] == r["norm"]]
    return float(np.sqrt(np.mean(np.square(v)))) if v else float("nan")


def format_residuals(rows: Sequence[dict], limit: int = 12) -> str:
    out = [f"{'item':<16s}{'idx':>16s}{'target':>10s}{'meas':>10s}{'resid':>10s}{'norm':>7s}"]
    for r in rows[:limit]:
        out.append(f"{r['item']:<16s}{r['index']:>16s}{r['target']:>10.4f}"
                   f"{r['measured']:>10.4f}{r['residual']:>+10.4f}{r['norm']:>7.2f}")
    if len(rows) > limit:
        out.append(f"... {len(rows) - limit} more")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the initial look: 附录 A typed straight in + the qualitative ops
# ---------------------------------------------------------------------------

#: per-look engine setup that comes from the *qualitative* half of 附录 A.
#: Conservative by construction; every entry is listed in the fit report so the
#: lead can review it.  Keys: vibrance, skin overrides, ops, caps, gamut.
QUALITATIVE: dict[str, dict] = {
    "01Glaze": {
        # 附录 A: "vibrance ×1.40@C.05→×1.05@C.20" — S5(0.05, c_hi, 0.20) = 0.875 => c_hi = 0.2555
        "vibrance": {"gain": 1.40, "c": [0.05, 0.2555]},
        "skin": {"pull": 0.25},
        "why": "vibrance 1.40@C.05 -> 1.05@C.20 (附录 A); skin pull 0.25 (附录 A)",
    },
    "02Burin": {
        "skin": {"pull": 0.20},
        "why": "no vibrance declared; skin pull 0.20 = the floor of PLAN §全局协议's 0.20-0.35 band "
               "(附录 A declares none for this look)",
    },
    "03Gilt": {
        "skin": {"pull": 0.35, "hue_residual": 0.10},
        "why": "附录 A: 肤色窗豁免金色旋转 -> skin.hue_residual 0.10 (the dh tables barely reach "
               "the skin window); 吸附 0.35 (the fold bound's ceiling)",
    },
    "04Viride": {
        "skin": {"pull": 0.35, "hue_residual": 0.10, "chroma_residual": 0.10},
        "ops": [
            {"id": "leaf-shadow", "center": 140.0, "sigma": 28.0,
             "dh": [6.0, 0.0], "dh_l": [L_STAR_45, L_STAR_60],
             "c_gate": [0.03, 0.10], "skin_residual": 0.10},
            {"id": "green-ceiling", "center": 140.0, "sigma": 32.0,
             "gain_c": 0.88, "c_gate": [0.15, 0.23], "skin_residual": 0.0},
        ],
        "why": "附录 A: 叶片分层旋转 L*<45 +12 deg / >60 +6 deg -> a +6 deg op that is ZERO at the "
               "probe's L = 0.65 and full below L* 45, so the dh10 target stays clean and the "
               "shadow leaf gets table+6; C upper bound .150 -> a gain_c 0.88 op gated above "
               "C0 = 0.15 (a caps[] entry with cap .15 cannot validate, see the report); "
               "肤色强保护 -> skin hue/chroma residual 0.10",
    },
    "05Clear": {
        "vibrance": {"gain": 1.18, "c": [0.02, 0.16]},
        "skin": {"pull": 0.20},
        "why": "附录 A: vibrance ×1.18 (engine default chroma ramp); skin pull 0.20 (none declared)",
    },
    "06Almond": {
        "vibrance": {"gain": 1.16, "c": [0.02, 0.16]},
        "skin": {"pull": 0.30},
        "why": "附录 A: vibrance ×1.16; 肤 吸附 0.30",
    },
    "07Voile": {
        "skin": {"pull": 0.20},
        "why": "no vibrance declared; skin pull 0.20 (none declared)",
    },
    "08Arcade": {
        "skin": {"pull": 0.20, "hue_residual": 0.15, "chroma_residual": 0.15},
        "ops": [
            {"id": "leaf-shadow", "center": 140.0, "sigma": 28.0,
             "dh": [13.0, 0.0], "dh_l": [L_STAR_45, L_STAR_60],
             "c_gate": [0.03, 0.10], "skin_residual": 0.10},
        ],
        "why": "附录 A: 绿叶旋转按亮度分层 阴影叶 +19 / 日照叶 +6 -> a +13 deg op that is ZERO at "
               "L = 0.65 (the table already carries +6 at h 120); 肤色不吃品红高光与黄色度削减"
               "（残留 0.15）-> skin hue/chroma residual 0.15",
    },
    "09Tinsel": {
        "skin": {"pull": 0.20},
        "why": "no vibrance declared; skin pull 0.20. 附录 A's white-shirt guard "
               "(C_in < 0.015 takes only 40 % of the tint) is NOT implementable — see the report",
    },
    "10Splice": {
        "skin": {"pull": 0.20},
        "why": "no vibrance declared; skin pull 0.20. 附录 A's 绿按光照分 (.72 shade / .88 sun) "
               "contradicts this look's own green arch and is NOT implemented — see the report",
    },
    "11Sodium": {
        "skin": {"pull": 0.20},
        "ops": [
            {"id": "lamp-gold", "center": 72.0, "sigma": 22.0, "dh": [4.0, 4.0],
             "c_gate": [0.03, 0.10], "skin_residual": 0.0},
        ],
        "caps": [
            {"center": 300.0, "sigma": 150.0, "start": 0.20, "cap": 0.26,
             "c_gate": [0.03, 0.10]},
        ],
        "why": "附录 A: 灯光金 (h 60-80) 与肤色 (42-56) 输出相隔 >= 10 deg -> a +4 deg op centred "
               "on 72 deg with skin_residual 0 (dead inside the skin window); 色度膝 "
               "(start .20, cap .26) -> one wide cap (engine caps are hue-windowed; see report)",
    },
    "12Argent": {
        "mono": True,
        "why": "黄滤镜 Tri-X: mono.filter fitted (linear-light weights, sum 1, B weight <= 0.05) "
               "+ mono.dl (12 knots) fitted against the per-hue ΔL target",
    },
}

#: [G] for every shipping look.  ``soft`` / ``eps_dark`` were deleted by
#: ENGINE_SPEC v1.1 R2 (the tanh tail they parametrised is gone; the limit is
#: measured per look at compile time) and ``end_slope`` is v1.2 W5.2's slope
#: floor.  The values are ``engine.spec``'s own defaults, chosen by W5.3's
#: knee x s1 sweep — see ``engine/spec.py::DEFAULT_KNEE``.
GAMUT = {"knee": DEFAULT_KNEE, "p": DEFAULT_P, "end_slope": DEFAULT_END_SLOPE}


def initial_look(target: dict) -> dict:
    """附录 A typed straight into the look tables + the qualitative ops.

    This is the *starting point* of the fit, i.e. exactly the thing the
    milestone-1 demo looks proved does not work on its own.
    """
    name = target["name"]
    q = QUALITATIVE.get(name, {})
    tone = [[float(c), float(o)] for c, o in
            zip(target["tone"]["codes"], target["tone"]["out"])]
    doc: dict = {
        "name": name,
        "cn": target.get("cn", ""),
        "title": target.get("title", name),
        "order": "NPG",
        "neutral": {
            "tone": tone,
            "tint_rg": [], "tint_bg": [],
            "white_guard": [0.88, 1.0], "tint_black_k": 12.0,
            "fade": [0.05, 0.16], "tint_skin_residual": 0.5,
        },
        "film": None,
    }
    ht = target["hue_table"]
    if q.get("mono"):
        doc["skin"] = {"pull": 0.0}
        doc["field"] = {"dl": [0.0] * 12}
        doc["gamut"] = None
        doc["mono"] = {"filter": [0.30, 0.62, 0.08], "dl": list(ht["dl"])}
        return doc

    vib = q.get("vibrance", {"gain": 1.0, "c": [0.02, 0.16]})
    # The vibrance gain at the probe's own chroma is known in closed form, and it
    # multiplies every cr the probe reads.  Dividing it out of the appendix
    # numbers before the loop starts is not a fudge: it is the one factor of the
    # measurement that needs no measuring, and without it the first candidates
    # sit above engine.spec's hard chroma-gain bound (Glaze: vib(0.10) = 1.369 x
    # cr 1.32 = 1.81) and the line search collapses.
    v10 = 1.0 + (vib["gain"] - 1.0) * (1.0 - float(curves.S5(vib["c"][0], vib["c"][1], FP.C10)))
    cr0 = [float(x) / v10 for x in ht["cr"]]
    # The appendix arch is an absolute chroma ratio; ENGINE_SPEC §3.4's ARCH is
    # normalised to 1 at L0 = 0.65, where CR already carries the level.
    arch0 = {}
    for fam in ("warm", "green", "blue"):
        v = np.asarray(target["arch"][fam], dtype=np.float64)
        arch0[fam] = [float(x) for x in v / float(_field._arch_spline(v)(_field.ARCH_NORM_L))]

    doc["field"] = {
        "dh10": list(ht["dh10"]), "dh18": list(ht["dh18"]),
        "cr": cr0, "dl": list(ht["dl"]),
        "rot_l_scale": [1.0, 1.0],
        "arch": arch0,
        "sat": 1.0,
        "vibrance": vib,
        # ENGINE_SPEC v1.2 W4 re-cut every gate (and deleted `iso`, whose ops
        # moved to stage [S]); this used to be a hard-coded copy of the v1.1
        # defaults, which v1.2's `validate` rejects on the width rule.  Take
        # them from the engine so there is one place they live.
        "gates": Gates().to_dict(),
    }
    sk = {
        # PLAN §全局协议 declares the feather 26/74.  With this set's rotation
        # amplitudes that feather folds the hue map (Glaze: 1 + dΔh/dh0 = 0.098
        # at h0 = 67 deg, gate 0.3), so the default here is the wider 20/80 and
        # _repair() widens it further when a look still folds.  The window
        # itself, h in [38,62], is PLAN's.
        "window": [20.0, 38.0, 62.0, 80.0], "c_gate": [0.03, 0.06],
        "c_fade": [0.16, 0.24], "l_gate": [0.20, 0.35, 0.90, 0.97],
        "center": 50.0, "pull": 0.20, "hue_offset": 0.0,
        "hue_residual": 0.25, "chroma_residual": 0.25, "l_residual": 0.25,
        "chroma": list(target["skin"]["cr"]) if target["skin"].get("cr")
                  else [1.0, 1.0, 1.0],
        "l_lift": 0.0,
    }
    if target["skin"].get("pull") is not None:
        sk["pull"] = float(target["skin"]["pull"])
    sk.update(q.get("skin", {}))
    # 附录 A lists the skin chroma as mid / dark / bright; the engine's SKINC
    # knots are dark(L .50) / mid(L .68) / bright(L .86).
    if target["skin"].get("cr"):
        mid, dark, bright = target["skin"]["cr"]
        sk["chroma"] = [float(dark), float(mid), float(bright)]
    doc["skin"] = sk
    doc["ops"] = copy.deepcopy(q.get("ops", []))
    doc["caps"] = copy.deepcopy(q.get("caps", []))
    doc["gamut"] = dict(GAMUT)
    doc["mono"] = None
    return doc


# ---------------------------------------------------------------------------
# the tint sub-fit (scipy.least_squares on bump amplitude / centre / width)
# ---------------------------------------------------------------------------


def _tint_basis(t: FloatArray, neutral: dict) -> FloatArray:
    """``(1 - S5(white_guard)) * T_G/(T_G + k_b)`` at *t*, in [0,1]."""
    ns = Neutral.from_dict(neutral)
    tt, T = build_target(ns)
    TG = np.interp(t, tt, T[:, 1])
    k_b = ns.tint_black_k / 255.0
    guard_w = 1.0 - curves.S5(ns.white_guard[0], ns.white_guard[1], t)
    guard_b = TG / (TG + k_b) if k_b > 0 else np.ones_like(TG)
    return guard_w * guard_b


def _bump_sum(t: FloatArray, p: FloatArray) -> FloatArray:
    out = np.zeros_like(t)
    for j in range(0, len(p), 3):
        a, mu, sg = p[j], p[j + 1], p[j + 2]
        out = out + a * np.exp(-0.5 * ((t - mu) / sg) ** 2)
    return out


def _n_bumps(want: FloatArray) -> int:
    nz = np.abs(want) > 1e-9
    if not nz.any():
        return 0
    sign = np.sign(want[nz])
    changes = int(np.count_nonzero(np.diff(sign) != 0))
    return int(min(3, max(1, changes + 1)))


def fit_tint_bumps(neutral: dict, want: FloatArray, *, seed: int = 7) -> list[list[float]]:
    """Gaussian bumps whose *effective* tint matches *want* (7 codes) best.

    ``want`` is what the probe should read at t = 5/18/35/50/65/80/95 %; the
    effective tint is the bump sum times the white guard times the black guard,
    so the amplitudes are larger than the target wherever a guard is biting.
    """
    from scipy.optimize import least_squares

    want = np.asarray(want, dtype=np.float64)
    n = _n_bumps(want)
    if n == 0:
        return []
    t = _NEUTRAL_T
    basis = _tint_basis(t, neutral)
    # a dense grid keeps the bumps from exploding between the 7 samples
    t_d = np.linspace(0.0, 1.0, 129)
    basis_d = _tint_basis(t_d, neutral)

    def resid(p):
        pred = _bump_sum(t, p) * basis
        r = list(pred - want)
        # regularise: the effective tint must not blow up between samples
        pred_d = _bump_sum(t_d, p) * basis_d
        over = np.maximum(np.abs(pred_d) - (np.abs(want).max() + 3.0), 0.0)
        return np.concatenate([r, 0.5 * over])

    rng = np.random.default_rng(seed)
    order = np.argsort(-np.abs(want))
    best = None
    for trial in range(4):
        p0: list[float] = []
        lo: list[float] = []
        hi: list[float] = []
        for j in range(n):
            k = int(order[min(j, len(order) - 1)])
            mu0 = float(t[k]) + (0.0 if trial == 0 else float(rng.uniform(-0.15, 0.15)))
            a0 = float(want[k]) / max(basis[k], 0.05)
            p0 += [float(np.clip(a0, -TINT_A_MAX, TINT_A_MAX)), mu0, 0.22 + 0.06 * j]
            lo += [-TINT_A_MAX, -0.4, TINT_SIGMA[0]]
            hi += [TINT_A_MAX, 1.4, TINT_SIGMA[1]]
        try:
            sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=4000)
        except Exception:  # pragma: no cover - defensive
            continue
        cost = float(np.sum(resid(sol.x)[: t.size] ** 2))
        if best is None or cost < best[0]:
            best = (cost, sol.x)
    if best is None:  # pragma: no cover
        return []
    p = best[1]
    return [[float(p[j]), float(p[j + 1]), float(p[j + 2])] for j in range(0, len(p), 3)]


def _shrink_tint(neutral: dict) -> tuple[dict, int]:
    """Shrink the bump amplitudes until [N] validates.  Returns (neutral, n)."""
    out = copy.deepcopy(neutral)
    for k in range(12):
        if not validate_neutral(Neutral.from_dict(out)):
            return out, k
        for key in ("tint_rg", "tint_bg"):
            out[key] = [[b[0] * 0.8, b[1], b[2]] for b in out[key]]
    return out, 12


# ---------------------------------------------------------------------------
# the fixed point
# ---------------------------------------------------------------------------


def _sampler(doc: dict) -> Callable[[FloatArray], FloatArray]:
    c = pipeline.compile(LookSpec.from_dict(doc))
    fn = lambda rgb: pipeline.apply(c, rgb)  # noqa: E731
    fn.compiled = c  # type: ignore[attr-defined]
    return fn


def _validates(doc: dict) -> tuple[bool, str]:
    try:
        validate(LookSpec.from_dict(doc))
    except SpecError as exc:
        return False, str(exc).replace("\n", " ")
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


#: remedy ladder for :func:`_repair`.  Each entry is (error fragment, remedy).
#: The order inside a remedy matters: the cheapest, most local trade first.
_REPAIR_MAX = 60


def _repair(doc: dict, clamps: list[dict], it: int) -> tuple[dict, bool]:
    """Walk a candidate back inside ``validate`` without abandoning the target.

    Returns ``(doc, ok)``.  Every remedy applied is appended to *clamps* — a
    gate is never traded for a target, so what the fit gives up is on the record.
    """
    doc = copy.deepcopy(doc)
    for _ in range(_REPAIR_MAX):
        ok, msg = _validates(doc)
        if ok:
            return doc, True
        sk = doc.get("skin", {})
        fl = doc.get("field", {})
        if "hue map folds" in msg or "min(1 + dDh/dh0)" in msg:
            # cheapest first: widen the feather a little (the window h in [38,62]
            # itself is untouched), then give up rotation amplitude.  The skin
            # exemption residual is a *declared design number* for Gilt / Viride
            # / Arcade, so it is the last thing traded and never past 0.40.
            w = list(sk.get("window", [20.0, 38.0, 62.0, 80.0]))
            if w[0] > 16.0 or w[3] < 84.0:
                w[0] = max(16.0, w[0] - 4.0)
                w[3] = min(84.0, w[3] + 4.0)
                sk["window"] = w
                _note(clamps, it, "skin feather", f"widened to {w[0]:.0f}/{w[3]:.0f} "
                                                  "(hue map fold bound)")
                continue
            if fl and max(max(abs(x) for x in fl["dh10"]),
                          max(abs(x) for x in fl["dh18"])) > 1.0:
                fl["dh10"] = [x * 0.96 for x in fl["dh10"]]
                fl["dh18"] = [x * 0.96 for x in fl["dh18"]]
                _note(clamps, it, "dh tables", "scaled 0.96 per step (hue map fold bound) — "
                                               "rotation amplitude given up to keep the gate")
                continue
            if sk.get("hue_residual", 0.25) < 0.40:
                sk["hue_residual"] = round(min(0.40, sk.get("hue_residual", 0.25) + 0.05), 3)
                _note(clamps, it, "skin.hue_residual",
                      f"raised to {sk['hue_residual']} (hue map fold bound) — weakens "
                      "PLAN §全局协议's 0.25 skin exemption")
                continue
        if "total chroma gain" in msg or "CR*ARCH ranges" in msg:
            fl["cr"] = [1.0 + (x - 1.0) * 0.94 for x in fl["cr"]]
            for fam in ("warm", "green", "blue"):
                fl["arch"][fam] = [1.0 + (x - 1.0) * 0.94 for x in fl["arch"][fam]]
            sk["chroma"] = [1.0 + (x - 1.0) * 0.94 for x in sk["chroma"]]
            _note(clamps, it, "chroma tables", "scaled 0.94 (engine chroma-gain budget)")
            continue
        if "lightness budget" in msg:
            doc, _ = _dl_budget(doc)
            _note(clamps, it, "ΔL budget", "terms scaled back (ENGINE_SPEC §3.5)")
            continue
        if "field.vibrance: width" in msg:
            c = list(fl["vibrance"]["c"])
            c[1] = min(0.40, c[1] + 0.02)
            fl["vibrance"]["c"] = c
            _note(clamps, it, "vibrance ramp", f"c_hi -> {c[1]:.3f} (fold rule)")
            continue
        if "neutral" in msg and "slope" in msg:
            doc["neutral"], n = _shrink_tint(doc["neutral"])
            _note(clamps, it, "tint bumps", f"shrunk {n}x0.8 (neutral slope)")
            if n < 12:
                continue
        if "grey axis" in msg:
            g = doc.get("gamut")
            if g is not None and g["knee"] < 0.85:
                g["knee"] = round(min(0.85, g["knee"] + 0.05), 3)
                if "soft" in g:   # v1.1 R2 deleted `soft`; a v0 doc may still carry it
                    g["soft"] = round(min(1.0 - g["knee"], g["soft"]), 3)
                _note(clamps, it, "gamut.knee", f"raised to {g['knee']} (grey axis vs T)")
                continue
        return doc, False
    return doc, False


def _note(clamps: list[dict], it: int, what: str, detail: str) -> None:
    for c in clamps:
        if c["what"] == what and c["detail"] == detail:
            c["count"] = c.get("count", 1) + 1
            return
    clamps.append({"iter": it, "what": what, "detail": detail, "count": 1})


def _g18(doc: dict) -> FloatArray:
    """Authority of the ``dh18`` knot at the probe's high-chroma column."""
    lo, hi = doc["field"]["gates"]["rot18"]
    return curves.S5(lo, hi, CHI)


def _skin_weights(doc: dict, meas: dict) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(w_skin, hue_delta_to_centre, lightness shape)`` at the three patches."""
    spec = LookSpec.from_dict(doc)
    w = _field._skin_weight(spec.skin, meas["skin_l_in"], meas["skin_c_in"], meas["skin_h_in"])
    dh = curves.hue_delta(meas["skin_h_in"], spec.skin.center)
    L = meas["skin_l_in"]
    shape = 4.0 * L * (1.0 - L) / _field.L_SHAPE_NORM
    return np.asarray(w), np.asarray(dh), np.asarray(shape)


def _dl_budget(doc: dict) -> tuple[dict, bool]:
    """Scale the ΔL terms so ENGINE_SPEC §3.5's total budget holds."""
    doc = copy.deepcopy(doc)
    if doc.get("mono"):
        mx = max(abs(x) for x in doc["mono"]["dl"])
        if mx > DL_BUDGET:
            s = DL_BUDGET / mx
            doc["mono"]["dl"] = [x * s for x in doc["mono"]["dl"]]
            return doc, True
        return doc, False
    table = max(abs(x) for x in doc["field"]["dl"])
    ops = sum(abs(o.get("dl", 0.0)) for o in doc.get("ops", ()))
    lift = abs(doc["skin"]["l_lift"])
    total = table + ops + lift
    if total <= DL_BUDGET:
        return doc, False
    # keep the ops (design), then the table (12 targets), shrink the lift first
    room = DL_BUDGET - ops
    if lift > 0 and table <= room:
        doc["skin"]["l_lift"] = math.copysign(min(lift, room - table), doc["skin"]["l_lift"])
    else:
        keep = max(room - min(lift, 0.5 * room), 1e-6)
        s = keep / max(table, 1e-12)
        doc["field"]["dl"] = [x * s for x in doc["field"]["dl"]]
        doc["skin"]["l_lift"] = math.copysign(min(lift, room - keep),
                                              doc["skin"]["l_lift"] or 1.0)
    return doc, True


def _arch_level(five) -> float:
    """The value a 5-knot arch row takes at ``L0 = 0.65`` (ENGINE_SPEC's norm)."""
    v = np.asarray(five, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        v = np.where(np.isfinite(v), v, np.nanmean(v) if np.isfinite(v).any() else 1.0)
    return float(_field._arch_spline(v)(_field.ARCH_NORM_L))


def appendix_consistency(target: dict) -> list[dict]:
    """Where 附录 A's own two chroma blocks disagree.

    The hue table's ``cr`` is the chroma ratio at ``L0 = 0.65``; the arch block
    is the chroma ratio at the same C on a lightness ladder through the same
    three hues.  Read at ``L0 = 0.65`` the arch must therefore agree with the
    hue table.  Where it does not, no look can satisfy both and the fit lands in
    between — this table says by how much, straight from the transcription, with
    no engine involved.
    """
    if not target.get("arch"):
        return []
    from scipy.interpolate import CubicSpline

    cr = target["hue_table"]["cr"]
    if cr is None:
        return []
    y = np.asarray(cr, dtype=np.float64)
    sp = CubicSpline(np.concatenate([KNOT_H, [360.0]]), np.concatenate([y, y[:1]]),
                     bc_type="periodic")
    out = []
    for f, fam in enumerate(("warm", "green", "blue")):
        at65 = _arch_level(target["arch"][fam])
        from_cr = float(sp(_ARCH_H[f]))
        out.append({"family": fam, "h": float(_ARCH_H[f]),
                    "cr_table_at_L065": round(from_cr, 4),
                    "arch_at_L065": round(at65, 4),
                    "disagreement": round(at65 - from_cr, 4),
                    "norm": round(abs(at65 - from_cr) / NORM["ratio"], 2)})
    return out


def reachability(tgt: dict) -> tuple[dict, list[dict]]:
    """Which 附录 A items the sRGB gamut puts out of the engine's reach.

    The probe asks for a fixed chroma (C = 0.10 for the hue table and the whole
    arch block) at a fixed lightness.  At several of those points the *input*
    itself is already outside sRGB — ``cmax(0.85, 55 deg) = 0.095 < 0.10`` — so
    the probe measures a clipped colour and [G] eats anything the look adds;
    at others the input fits but ``C * cr_target`` does not.  Both are targets
    no look can hit, and a fit that chases them walks its chroma tables out of
    ENGINE_SPEC §3.7's budget and drags every other knot back with it through
    :func:`_repair`.  They are frozen at their transcribed value and reported.
    """
    freeze = new_freeze()
    notes: list[dict] = []
    cm_hue = M.cmax(np.full(12, FP.HUE_L), KNOT_H)
    for k in range(12):
        t = tgt["cr"][k]
        if np.isfinite(t) and FP.C10 * t > cm_hue[k]:
            freeze["cr"][k] = True
            notes.append({"item": f"hue.c10.cr h{int(KNOT_H[k])}", "target": float(t),
                          "reason": f"C*cr = {FP.C10 * t:.4f} > cmax(L 0.65, h "
                                    f"{int(KNOT_H[k])}) = {cm_hue[k]:.4f} — outside sRGB"})
    Lg = np.broadcast_to(_ARCH_L[None, :], (3, 5))
    hg = np.broadcast_to(_ARCH_H[:, None], (3, 5))
    cm_arch = M.cmax(Lg, hg)
    for f in range(3):
        for j in range(5):
            t = tgt["arch_cr"][f][j]
            if not np.isfinite(t):
                continue
            fam = ("warm", "green", "blue")[f]
            if FP.ARCH_C > cm_arch[f, j]:
                freeze["arch_cr"][f, j] = True
                notes.append({"item": f"arch.cr {fam}@L{_ARCH_L[j]:.2f}", "target": float(t),
                              "reason": f"the probe's own C = {FP.ARCH_C} is already outside "
                                        f"sRGB there (cmax = {cm_arch[f, j]:.4f})"})
            elif FP.ARCH_C * t > cm_arch[f, j]:
                freeze["arch_cr"][f, j] = True
                notes.append({"item": f"arch.cr {fam}@L{_ARCH_L[j]:.2f}", "target": float(t),
                              "reason": f"C*cr = {FP.ARCH_C * t:.4f} > cmax = "
                                        f"{cm_arch[f, j]:.4f} — outside sRGB"})
    return freeze, notes


def new_freeze() -> dict:
    """Per-knot 'this target is out of the engine's reach' flags (all clear)."""
    return {"dh10": np.zeros(12, bool), "dh18": np.zeros(12, bool),
            "cr": np.zeros(12, bool), "dl": np.zeros(12, bool),
            "arch_cr": np.zeros((3, 5), bool), "skin_dh": np.zeros(3, bool),
            "skin_cr": np.zeros(3, bool), "skin_dl": np.zeros(3, bool)}


def _step(doc: dict, meas: dict, tgt: dict, *, a: float, lev_dl: float,
          freeze: dict) -> dict:
    """One damped update of every table, at step factor *a*.

    A knot flagged in *freeze* is left alone: its target has been shown not to
    move under the update (the engine cannot reach it), and chasing it drives
    the parameter out of the chroma budget and drags every other knot with it
    through :func:`_repair`.
    """
    new = copy.deepcopy(doc)
    fl = new.get("field", {})

    if new.get("mono"):
        want, got = tgt["dl"], meas["dl"]
        dl = np.asarray(new["mono"]["dl"], dtype=np.float64)
        ok = np.isfinite(want) & np.isfinite(got) & ~freeze["dl"]
        dl[ok] = dl[ok] + a * (want[ok] - got[ok]) / lev_dl
        new["mono"]["dl"] = [float(np.clip(x, -DL_BUDGET, DL_BUDGET)) for x in dl]
        return new

    # --- hue rotations -----------------------------------------------------
    for key in ("dh10", "dh18"):
        cur = np.asarray(fl[key], dtype=np.float64)
        want, got = tgt[key], meas[key]
        ok = np.isfinite(want) & np.isfinite(got) & ~freeze[key]
        if key == "dh18":
            g = _g18(new)
            cur[ok] = cur[ok] + a * (want[ok] - got[ok]) / np.maximum(g[ok], G18_DIV)
        else:
            cur[ok] = cur[ok] + a * (want[ok] - got[ok])
        lim = np.minimum(DH_ABS_MAX,
                         np.where(np.isfinite(want), np.abs(want), 0.0) * DH_HEADROOM[0]
                         + DH_HEADROOM[1])
        fl[key] = [float(x) for x in np.clip(cur, -lim, lim)]

    # --- chroma table ------------------------------------------------------
    cur = np.asarray(fl["cr"], dtype=np.float64)
    want, got = tgt["cr"], meas["cr"]
    ok = np.isfinite(want) & np.isfinite(got) & (np.abs(got) > 1e-6) & ~freeze["cr"]
    cur[ok] = cur[ok] * np.power(np.clip(want[ok] / got[ok], 0.25, 4.0), a)
    fl["cr"] = [float(x) for x in np.clip(cur, *CR_CLAMP)]

    # --- lightness table ---------------------------------------------------
    cur = np.asarray(fl["dl"], dtype=np.float64)
    want, got = tgt["dl"], meas["dl"]
    ok = np.isfinite(want) & np.isfinite(got) & ~freeze["dl"]
    cur[ok] = cur[ok] + a * (want[ok] - got[ok]) / lev_dl
    fl["dl"] = [float(x) for x in np.clip(cur, -DL_BUDGET, DL_BUDGET)]

    # --- arch --------------------------------------------------------------
    # ENGINE_SPEC §3.4 normalises ARCH to 1 at L0 = 0.65, where CR carries the
    # level, so only the *shape* of the arch is identifiable here.  Updating on
    # the raw ratio makes the fit chase a level error it structurally cannot
    # fix (附录 A's arch block and its cr table disagree about the level for
    # several looks — see `appendix_consistency` in the report); updating on the
    # shape, each side divided by its own value at L0 = 0.65, does not.
    want, got = tgt["arch_cr"], meas["arch_cr"]
    for f, fam in enumerate(("warm", "green", "blue")):
        cur = np.asarray(fl["arch"][fam], dtype=np.float64)
        ok = (np.isfinite(want[f]) & np.isfinite(got[f]) & (np.abs(got[f]) > 1e-6)
              & ~freeze["arch_cr"][f])
        if ok.any():
            wn = _arch_level(want[f])
            gn = _arch_level(got[f])
            if wn > 1e-6 and gn > 1e-6:
                ratio = (want[f][ok] / wn) / (got[f][ok] / gn)
                cur[ok] = cur[ok] * np.power(np.clip(ratio, 0.25, 4.0), a)
        sp = _field._arch_spline(cur)
        norm = float(sp(_field.ARCH_NORM_L))
        if norm > 1e-6:
            cur = cur / norm        # cosmetic: the engine normalises anyway
        fl["arch"][fam] = [float(x) for x in np.clip(cur, *ARCH_CLAMP)]

    # --- skin --------------------------------------------------------------
    sk = new["skin"]
    w, dhc, shape = _skin_weights(new, meas)
    # hue: only the constant offset is fitted (pull is a declared design number)
    ok = np.isfinite(tgt["skin_dh"]) & np.isfinite(meas["skin_dh"]) & ~freeze["skin_dh"]
    if ok.any() and float(np.sum((w[ok]) ** 2)) > 1e-9:
        r = tgt["skin_dh"][ok] - meas["skin_dh"][ok]
        sk["hue_offset"] = float(sk["hue_offset"]
                                 + a * float(np.sum(r * w[ok]) / np.sum(w[ok] ** 2)))
        sk["hue_offset"] = float(np.clip(sk["hue_offset"], -20.0, 20.0))
    # chroma: SKINC knots are dark(L .50) / mid(L .68) / bright(L .86);
    # the patches are mid / dark / bright.
    ok = (np.isfinite(tgt["skin_cr"]) & np.isfinite(meas["skin_cr"])
          & (meas["skin_cr"] > 1e-6) & ~freeze["skin_cr"])
    ch = list(sk["chroma"])
    for patch, knot in ((1, 0), (0, 1), (2, 2)):
        if ok[patch]:
            ch[knot] = float(np.clip(
                ch[knot] * (tgt["skin_cr"][patch] / meas["skin_cr"][patch]) ** a, *SKINC_CLAMP))
    sk["chroma"] = ch
    # lightness lift: one scalar against three patches
    ok = np.isfinite(tgt["skin_dl"]) & np.isfinite(meas["skin_dl"]) & ~freeze["skin_dl"]
    lev = w * shape
    if ok.any() and float(np.sum(lev[ok] ** 2)) > 1e-9:
        r = tgt["skin_dl"][ok] - meas["skin_dl"][ok]
        sk["l_lift"] = float(sk["l_lift"]
                             + a * float(np.sum(r * lev[ok]) / np.sum(lev[ok] ** 2)))
        sk["l_lift"] = float(np.clip(sk["l_lift"], -DL_BUDGET, DL_BUDGET))
    return new


_STALL_GROUPS = ("dh10", "dh18", "cr", "dl", "arch_cr", "skin_dh", "skin_cr", "skin_dl")
_STALL_KIND = {"dh10": "deg", "dh18": "deg", "cr": "ratio", "dl": "dl", "arch_cr": "ratio",
               "skin_dh": "deg", "skin_cr": "ratio", "skin_dl": "dl"}
_STALL_LABEL = {
    "dh10": lambda i: f"hue.c10.dh h{int(KNOT_H[i[0]])}",
    "dh18": lambda i: f"hue.chi.dh h{int(KNOT_H[i[0]])}",
    "cr": lambda i: f"hue.c10.cr h{int(KNOT_H[i[0]])}",
    "dl": lambda i: f"hue.c10.dl h{int(KNOT_H[i[0]])}",
    "arch_cr": lambda i: f"arch.cr {('warm', 'green', 'blue')[i[0]]}@L{_ARCH_L[i[1]]:.2f}",
    "skin_dh": lambda i: f"skin.dh {('mid', 'dark', 'bright')[i[0]]}",
    "skin_cr": lambda i: f"skin.cr {('mid', 'dark', 'bright')[i[0]]}",
    "skin_dl": lambda i: f"skin.dl {('mid', 'dark', 'bright')[i[0]]}",
}


def _update_stalls(stall: dict, freeze: dict, tgt: dict, before: dict, after: dict,
                   clamps: list[dict], it: int) -> None:
    """Freeze knots whose measurement refuses to follow their parameter.

    A target the engine structurally cannot reach (a gamut-limited arch probe, a
    high-chroma column that does not exist at that hue) otherwise makes its knot
    run away until the chroma budget trips, and :func:`_repair` then drags every
    *other* knot back with it.  Freezing is how "keep the gate, report the gap"
    is implemented.
    """
    for g in _STALL_GROUPS:
        t = np.asarray(tgt[g], dtype=np.float64)
        b = np.asarray(before[g], dtype=np.float64)
        a = np.asarray(after[g], dtype=np.float64)
        for ix in np.ndindex(t.shape):
            if freeze[g][ix] or not np.isfinite(t[ix]) or not np.isfinite(a[ix]) \
                    or not np.isfinite(b[ix]):
                continue
            want = t[ix] - b[ix]
            moved = a[ix] - b[ix]
            # only a residual that actually matters is worth freezing: a knot
            # sitting 0.02 codes off its target has nothing left to give and its
            # "gain" is numerical noise.
            if abs(want) / NORM[_STALL_KIND[g]] < STALL_MIN_NORM:
                continue
            gain = moved / want
            if gain < STALL_GAIN:
                stall[(g, ix)] = stall.get((g, ix), 0) + 1
                if stall[(g, ix)] >= STALL_MAX:
                    freeze[g][ix] = True
                    _note(clamps, it, "frozen (unreachable)",
                          f"{_STALL_LABEL[g](ix)}: target {t[ix]:.4f}, stuck at "
                          f"{a[ix]:.4f}; the knot no longer moves the measurement")
            else:
                stall[(g, ix)] = 0


#: PLAN §验证 numeric gates this module reports against (it does not re-implement
#: tools/qc.py; these three are the ones a *fit* can trade strength for).
SMOOTH_GATES = {"d2_full_p99_9": 6.0, "d2_interior_p99_9": 4.0, "d2_full_max": 14.0,
                "folds": 0}
#: the field scales the frontier scan walks
BACKOFF_SCALES = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1)


def smoothness(doc: dict, *, size: int = 33) -> dict:
    """d2 / fold / nm / strength of a look, measured on the 33-point lattice."""
    c = pipeline.compile(LookSpec.from_dict(doc), strict=False)
    t = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(t, t, t, indexing="ij")
    raw, nm = pipeline._run(c, np.stack([r, g, b], axis=-1))
    table = np.clip(raw, 0.0, 1.0)
    d2 = M.second_diff_stats(table)
    fo = M.fold_stats(table)
    # A mono look maps the cube onto a 1-D curve: every tetrahedron has volume
    # exactly 0, so `neg_count` (ratio <= 0) is 196,608 by construction and the
    # gate is structurally unreachable.  engine.pipeline.diagnostics and
    # tools/qc.py both read `neg_strict_count` there instead.
    is_mono = doc.get("mono") is not None
    folds = int(fo["neg_strict_count"] if is_mono else fo["neg_count"])
    ident = M.identity_table(17).reshape(-1, 3)
    out = np.asarray(pipeline.apply(c, ident), dtype=np.float64)
    de = M.de00(M.srgb_to_lab(ident), M.srgb_to_lab(out))
    return {
        "d2_full_p99_9": float(d2["full"]["p99_9"]),
        "d2_interior_p99_9": float(d2["interior"]["p99_9"]),
        "d2_full_max": float(d2["full"]["max"]),
        "folds": folds,
        "fold_field": "neg_strict_count" if is_mono else "neg_count",
        "fold_min_ratio": float(fo["min_ratio"]),
        "nm_p99_9": None if nm is None else float(np.percentile(nm, 99.9)),
        "de00_lattice17": float(de.mean()),
        "clip_below_codes": float(max(0.0, -raw.min()) * 255.0),
        "clip_above_codes": float(max(0.0, raw.max() - 1.0) * 255.0),
    }


def _scale_field(doc: dict, s: float) -> dict:
    """The look with every [P] amplitude scaled toward the identity by *s*."""
    d = copy.deepcopy(doc)
    if d.get("mono"):
        d["mono"]["dl"] = [x * s for x in d["mono"]["dl"]]
        return d
    fl, sk = d["field"], d["skin"]
    for k in ("dh10", "dh18", "dl"):
        fl[k] = [x * s for x in fl[k]]
    fl["cr"] = [1.0 + (x - 1.0) * s for x in fl["cr"]]
    for fam in ("warm", "green", "blue"):
        fl["arch"][fam] = [1.0 + (x - 1.0) * s for x in fl["arch"][fam]]
    fl["sat"] = 1.0 + (fl["sat"] - 1.0) * s
    fl["vibrance"] = dict(fl["vibrance"])
    fl["vibrance"]["gain"] = 1.0 + (fl["vibrance"]["gain"] - 1.0) * s
    sk["chroma"] = [1.0 + (x - 1.0) * s for x in sk["chroma"]]
    sk["l_lift"] = sk["l_lift"] * s
    sk["hue_offset"] = sk["hue_offset"] * s
    sk["pull"] = sk["pull"] * s
    for o in d.get("ops", ()):
        o["dh"] = [x * s for x in o.get("dh", (0.0, 0.0))]
        o["gain_c"] = 1.0 + (o.get("gain_c", 1.0) - 1.0) * s
        o["dl"] = o.get("dl", 0.0) * s
    for cp in d.get("caps", ()):
        # a cap has no identity: scale the ceiling away from `start` instead
        cp["cap"] = cp["start"] + (cp["cap"] - cp["start"]) / max(s, 1e-3)
    return d


def backoff_frontier(doc: dict, scales: Sequence[float] = BACKOFF_SCALES) -> list[dict]:
    """What each PLAN smoothness gate would cost in look strength.

    The 33-point lattice is the artifact, and [P]'s amplitudes are what the
    lattice cannot carry.  This walks the whole look toward the identity and
    reports, for each scale, the gates and the dE00 that is left — so "the look
    cannot reach its target without violating the fold gate" is a number, not
    an opinion.
    """
    out = []
    for s in scales:
        row = smoothness(_scale_field(doc, s))
        row["scale"] = float(s)
        row["passes"] = {k: (row[k] <= v) for k, v in SMOOTH_GATES.items()}
        out.append(row)
    return out


def _fit_mono_filter(doc: dict, tgt: dict) -> tuple[list[float], dict]:
    """Choose the B&W filter so the per-hue ΔL target needs the least ``mono.dl``.

    ``Lm = cbrt(w . linear(rgb))`` is what the channel mix decides; everything
    after it (tone curve, tint) is common to every hue.  So the filter is fitted
    to the *shape* of the target and ``mono.dl`` only carries the remainder,
    which is what keeps it inside ENGINE_SPEC §3.7's |DL| <= 0.06.
    """
    from scipy.optimize import least_squares
    from engine import color, xfer

    want = tgt["dl"]
    ok = np.isfinite(want)
    code, _ = M.code_from_oklch(np.full(12, FP.HUE_L), np.full(12, FP.C10), KNOT_H)
    code = np.clip(code, 0.0, 1.0)
    lin = xfer.signed_srgb_decode(code)
    l_in = M.oklch_from_code(code)[0]

    # the neutral-only reference: what Lm must be so that l_out - l_in = want.
    # l_out = f(Lm) with f the tone curve read in OKLab L; measure f once.
    ramp = np.linspace(0.0, 1.0, 1025)
    spec = LookSpec.from_dict({**doc, "mono": {"filter": [0.2126, 0.7152, 0.0722],
                                               "dl": [0.0] * 12}})
    c = pipeline.compile(spec)
    out = pipeline.apply(c, np.stack([ramp] * 3, axis=-1))
    L_out = color.linear_srgb_to_oklab(color.srgb_decode(np.clip(out, 0, 1)))[:, 0]
    Lm_ramp = np.cbrt(color.srgb_decode(ramp))
    want_Lm = np.interp(l_in + np.nan_to_num(want), L_out, Lm_ramp)

    def resid(p):
        w = np.array([p[0], p[1], 1.0 - p[0] - p[1]])
        Y = lin @ w
        return (np.cbrt(np.maximum(Y, 0.0)) - want_Lm)[ok]

    sol = least_squares(resid, [0.33, 0.62], bounds=([0.05, 0.25], [0.60, 0.80]),
                        max_nfev=2000)
    w = [float(sol.x[0]), float(sol.x[1]), float(1.0 - sol.x[0] - sol.x[1])]
    info = {"filter": w, "residual_Lm_rms": float(np.sqrt(np.mean(resid(sol.x) ** 2)))}
    return w, info


def fit_look(target: dict, *, max_iter: int = MAX_ITER, relax: float = RELAX,
             tol: float = TOL_RMS, verbose: bool = False) -> dict:
    """Fit one look to its 附录 A profile.  Returns the fit record."""
    name = target["name"]
    tgt = target_arrays(target)
    doc = initial_look(target)
    clamps: list[dict] = []
    t_start = time.perf_counter()

    # --- the tint sub-fit (analytic; refined by the loop's correction) -----
    corr_rg = np.zeros(7)
    corr_bg = np.zeros(7)

    def refit_tint(doc: dict) -> dict:
        d = copy.deepcopy(doc)
        if np.isfinite(tgt["tint_rg"]).any():
            d["neutral"]["tint_rg"] = fit_tint_bumps(d["neutral"], tgt["tint_rg"] + corr_rg)
        if np.isfinite(tgt["tint_bg"]).any():
            d["neutral"]["tint_bg"] = fit_tint_bumps(d["neutral"], tgt["tint_bg"] + corr_bg)
        d["neutral"], shrunk = _shrink_tint(d["neutral"])
        if shrunk:
            clamps.append({"iter": len(history), "what": "neutral tint bumps",
                           "detail": f"amplitudes shrunk {shrunk}x0.8 to satisfy "
                                     "validate_neutral (min slope 0.04 / T in [0,1])"})
        return d

    history: list[dict] = []
    doc = refit_tint(doc)

    if doc.get("mono"):
        w, minfo = _fit_mono_filter(doc, tgt)
        doc["mono"]["filter"] = w
        clamps.append({"iter": 0, "what": "mono.filter", "detail": json.dumps(minfo)})

    ok, msg = _validates(doc)
    if not ok:
        doc, ok = _repair(doc, clamps, 0)
        if not ok:
            raise SpecError(f"{name}: the starting point cannot be repaired: {_validates(doc)[1]}")

    # --- measure the ΔL leverage once (NPG puts [N] before [P]) ------------
    s0 = _sampler(doc)
    m0 = measure(s0)
    probe = copy.deepcopy(doc)
    if probe.get("mono"):
        probe["mono"]["dl"] = [x + 0.005 for x in probe["mono"]["dl"]]
    else:
        probe["field"]["dl"] = [float(np.clip(x + 0.005, -DL_BUDGET, DL_BUDGET))
                                for x in probe["field"]["dl"]]
    try:
        m1 = measure(_sampler(probe))
        d = np.asarray(m1["dl"]) - np.asarray(m0["dl"])
        lev_dl = float(np.nanmean(d) / 0.005)
    except Exception:  # pragma: no cover - defensive
        lev_dl = 1.0
    if not np.isfinite(lev_dl) or lev_dl < 0.3:
        lev_dl = 1.0

    freeze, unreachable = reachability(tgt)
    if doc.get("mono"):
        freeze, unreachable = new_freeze(), []
    if not doc.get("mono"):
        # the high-chroma column at these hues IS the C = 0.10 column: PLAN's own
        # definition chi = min(0.18, 0.80*cmax) collapses onto 0.10 where the
        # gamut is narrow at L = 0.65 (yellow-green, cyan, azure), so dh18 has no
        # authority there and its target cannot be separated from dh10's.
        freeze["dh18"] |= _g18(doc) < G18_MIN
    stall: dict[tuple, int] = {}

    meas = m0
    rows = residual_rows(meas, tgt)
    rms0 = residual_rms(rows)
    history.append({"iter": 0, "rms": rms0})
    if verbose:
        print(f"[{name}] iter 0  rms {rms0:.3f}  (lev_dl {lev_dl:.3f})")

    best = (rms0, copy.deepcopy(doc), rows, meas)
    since_best = 0
    a_base = relax
    for it in range(1, int(max_iter) + 1):
        if best[0] < tol or since_best >= PATIENCE:
            break
        a = a_base
        applied = None
        for _ in range(LINE_SEARCH):
            cand = _step(doc, meas, tgt, a=a, lev_dl=lev_dl, freeze=freeze)
            cand, budget_hit = _dl_budget(cand)
            cand = refit_tint(cand)
            cand, ok = _repair(cand, clamps, it)
            if ok:
                if budget_hit:
                    _note(clamps, it, "ΔL budget",
                          "max|field.dl| + sum|ops.dl| + |skin.l_lift| "
                          f"> {DL_BUDGET}; terms scaled back")
                applied = cand
                break
            a *= 0.5
        if applied is None:
            _note(clamps, it, "step abandoned",
                  f"no step factor down to {a:.4f} validates: {_validates(cand)[1][:300]}")
            break
        prev_meas, doc = meas, applied
        meas = measure(_sampler(doc))
        _update_stalls(stall, freeze, tgt, prev_meas, meas, clamps, it)
        # tint correction for the next refit
        if np.isfinite(tgt["tint_rg"]).any():
            corr_rg = corr_rg + relax * (tgt["tint_rg"] - meas["tint_rg"])
        if np.isfinite(tgt["tint_bg"]).any():
            corr_bg = corr_bg + relax * (tgt["tint_bg"] - meas["tint_bg"])
        rows = residual_rows(meas, tgt)
        rms = residual_rms(rows)
        history.append({"iter": it, "rms": rms, "step": a})
        if verbose:
            print(f"[{name}] iter {it}  rms {rms:.3f}  step {a:.3f}")
        if rms < best[0] - 1e-6:
            best = (rms, copy.deepcopy(doc), rows, meas)
            since_best = 0
        else:
            since_best += 1
            # the fixed point is not a contraction everywhere (the repair ladder
            # and the frozen knots make it piecewise); damp instead of bouncing.
            a_base = max(0.05, a_base * 0.5)

    rms, doc, rows, meas = best
    smooth = smoothness(doc)
    frontier = backoff_frontier(doc)
    gap = {}
    for k, lim in SMOOTH_GATES.items():
        if smooth[k] <= lim:
            gap[k] = {"status": "pass", "value": smooth[k], "gate": lim}
            continue
        hit = next((r for r in frontier if r[k] <= lim), None)
        gap[k] = {"status": "fail", "value": smooth[k], "gate": lim,
                  "passes_at_scale": None if hit is None else hit["scale"],
                  "de00_there": None if hit is None else hit["de00_lattice17"],
                  "de00_now": smooth["de00_lattice17"],
                  "de00_target": target.get("de00_target")}
    frozen_labels = sorted(_STALL_LABEL[g](ix) for g in _STALL_GROUPS
                           for ix in np.ndindex(freeze[g].shape) if freeze[g][ix])
    if not doc.get("mono"):
        for i in range(12):
            if _g18(doc)[i] < G18_MIN:
                unreachable.append({
                    "item": f"hue.chi.dh h{int(KNOT_H[i])}",
                    "target": float(tgt["dh18"][i]) if np.isfinite(tgt["dh18"][i]) else None,
                    "reason": f"chi = min(0.18, 0.80*cmax) = {CHI[i]:.4f} at this hue, so the "
                              f"rot18 gate S5(0.10,0.18) is only {_g18(doc)[i]:.3f} open — the "
                              "high-chroma column collapses onto the C = 0.10 column"})
    reach_rows = [r for r in rows
                  if not any(u["item"].startswith(r["item"]) and u["item"].endswith(r["index"])
                             for u in unreachable)]
    spec = LookSpec.from_dict(doc)
    warnings = list(validate(spec))
    return {
        "name": name,
        "look": doc,
        "rms_before": rms0,
        "rms_after": rms,
        "iterations": len(history) - 1,
        "history": history,
        "worst": rows[:5],
        "worst_reachable": reach_rows[:5],
        "rows": rows,
        "measured": {k: _json(v) for k, v in meas.items()},
        "rms_reachable": residual_rms(reach_rows),
        "n_items": len(rows),
        "n_unreachable": len(unreachable),
        "clamps": clamps,
        "frozen": frozen_labels,
        "unreachable": unreachable,
        "g18": _json(_g18(doc)) if not doc.get("mono") else None,
        "lev_dl": lev_dl,
        "compile_warnings": warnings,
        "fit_time_s": time.perf_counter() - t_start,
        "qualitative": QUALITATIVE.get(name, {}).get("why", ""),
        "appendix_consistency": appendix_consistency(target),
        "grey18_out_code": meas["grey18"],
        "black_out_code": meas["black"],
        "smoothness": smooth,
        "backoff_frontier": frontier,
        "gate_gap": gap,
    }


def _json(v):
    """numpy -> JSON, NaN -> null (a NaN is 'the probe could not measure it')."""
    if isinstance(v, np.ndarray):
        return [_json(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


# ---------------------------------------------------------------------------
# all twelve
# ---------------------------------------------------------------------------


def fit_all(names: Sequence[str] | None = None, *, looks_dir: Path | None = None,
            targets_dir: Path | None = None, max_iter: int = MAX_ITER,
            write: bool = True, verbose: bool = True) -> dict:
    """Fit every target profile and write ``looks/<name>.json``."""
    tdir = Path(targets_dir or (_root() / "looks" / "targets"))
    ldir = Path(looks_dir or (_root() / "looks"))
    if names is None:
        names = [p.stem for p in sorted(tdir.glob("*.json")) if not p.stem.startswith("_")]
    out: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "probe": FP.PROBE_ID, "relax": RELAX, "max_iter": max_iter,
                 "tol_rms": TOL_RMS, "gamut": GAMUT, "looks": []}
    for nm in names:
        target = load_target(nm, tdir)
        rec = fit_look(target, max_iter=max_iter, verbose=verbose)
        if write:
            p = ldir / f"{nm}.json"
            p.write_text(json.dumps(rec["look"], indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
            rec["file"] = str(p)
        out["looks"].append(rec)
        if verbose:
            print(f"[{nm}] rms {rec['rms_before']:.3f} -> {rec['rms_after']:.3f} "
                  f"in {rec['iterations']} iters, {len(rec['clamps'])} clamp events")
    out["summary"] = [{
        "name": r["name"],
        "rms_before": round(r["rms_before"], 3),
        "rms_after": round(r["rms_after"], 3),
        "rms_reachable_only": round(r["rms_reachable"], 3),
        "items": r["n_items"], "unreachable": r["n_unreachable"],
        "worst": [f"{w['item']} {w['index']} tgt {w['target']:.4g} got "
                  f"{w['measured']:.4g} ({w['norm']:.1f})" for w in r["worst"]],
        "worst_reachable": [f"{w['item']} {w['index']} tgt {w['target']:.4g} got "
                            f"{w['measured']:.4g} ({w['norm']:.1f})"
                            for w in r["worst_reachable"]],
        "de00_lattice17": round(r["smoothness"]["de00_lattice17"], 2),
        "de00_target": r["gate_gap"].get("folds", {}).get("de00_target"),
        "d2_full_p99_9": round(r["smoothness"]["d2_full_p99_9"], 2),
        "d2_interior_p99_9": round(r["smoothness"]["d2_interior_p99_9"], 2),
        "folds": r["smoothness"]["folds"],
        "grey_out_at_code31": round(r["grey18_out_code"], 1),
        "black_out_code": round(r["black_out_code"], 2),
        "clamp_events": len(r["clamps"]),
    } for r in out["looks"]]
    return out


def verify_cubes(names: Sequence[str] | None = None, *, out_dir: Path | None = None,
                 targets_dir: Path | None = None) -> dict:
    """Re-measure the **shipped .cube files** against their target profiles.

    ``fit_look`` optimises against the continuous engine; the camera runs the
    33-point table.  This closes the loop with the full
    :func:`tools.fingerprint.fingerprint` on the artifact read back from disk —
    the same code path ``tools/build.py`` uses — and scores it with
    :func:`tools.fingerprint.compare`.

    It also works around a defect in ``tools/qc.py``: its ``fingerprint.residual``
    gate reads ``row["norm"]``, but ``compare()`` returns ``norm_residual``, so
    that gate silently evaluates to NaN and reports ``skip`` even when a target
    file exists.  (``tools/qc.py`` is not this track's file — reported, not
    edited.)
    """
    out_dir = Path(out_dir or (_root() / "out" / "LUTs"))
    tdir = Path(targets_dir or (_root() / "looks" / "targets"))
    if names is None:
        names = [p.stem for p in sorted(tdir.glob("*.json")) if not p.stem.startswith("_")]
    rows = []
    for nm in names:
        cube = out_dir / f"{nm}.cube"
        if not cube.exists():
            rows.append({"name": nm, "error": f"no cube at {cube}"})
            continue
        target = load_target(nm, tdir)
        qcp = _root() / "work.nosync" / "review" / f"qc_{nm}.json"
        qc = None
        if qcp.exists():
            doc = json.loads(qcp.read_text(encoding="utf-8"))
            qc = {"summary": doc.get("summary"),
                  "fails": doc.get("fails"), "warns": doc.get("warns"),
                  "report": str(qcp)}
        fp = FP.fingerprint(M.sampler_from_cube(cube), None, name=nm)
        cmp = FP.compare(fp, target["fingerprint"])
        v = [r["norm_residual"] for r in cmp
             if r["norm_residual"] == r["norm_residual"]]
        rows.append({
            "name": nm, "cube": str(cube),
            "n_items": len(v),
            "rms_on_cube": float(np.sqrt(np.mean(np.square(v)))) if v else None,
            "worst": [{"path": r["path"], "index": r["index"], "target": r["target"],
                       "measured": r["measured"], "norm": r["norm_residual"]}
                      for r in cmp if r["norm_residual"] == r["norm_residual"]][:5],
            "de00_photo_mean": (fp["global"]["photo"] or {}).get("de00_mean"),
            "de00_lattice17_mean": fp["global"]["lattice17"]["de00_mean"],
            "de00_target": target.get("de00_target"),
            "grey_out_at_code31": fp["neutral"]["ramp8"][2],
            "grey_out_at_t018": fp["neutral"]["out_t"][1],
            "black_out_code": float(np.mean(fp["neutral"]["black"])),
            "white_out_code": float(np.mean(fp["neutral"]["white"])),
            "toe_slope": fp["neutral"]["toe"],
            "qc": qc,
        })
    return {"probe": FP.PROBE_ID, "looks": rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all", help="fit every looks/targets/*.json")
    a.add_argument("--iters", type=int, default=MAX_ITER)
    a.add_argument("--out", default=str(_root() / "work.nosync" / "review" / "fit_report.json"))
    a.add_argument("--no-write", action="store_true")
    f = sub.add_parser("fit", help="fit one look")
    f.add_argument("name")
    f.add_argument("--iters", type=int, default=MAX_ITER)
    f.add_argument("--no-write", action="store_true")
    s = sub.add_parser("show", help="residual table of the look already on disk")
    s.add_argument("name")
    v = sub.add_parser("verify", help="re-measure the built .cube files vs their targets")
    v.add_argument("--report", default=str(_root() / "work.nosync" / "review"
                                           / "fit_report.json"))
    args = ap.parse_args(argv)

    if args.cmd == "verify":
        doc = verify_cubes()
        p = Path(args.report)
        if p.exists():
            full = json.loads(p.read_text())
            full["verify_on_cube"] = doc
            p.write_text(json.dumps(full, indent=1, ensure_ascii=False, default=str),
                         encoding="utf-8")
            print(f"updated {p}")
        print(f"{'look':<12}{'n':>5}{'rms(cube)':>11}{'dE00 photo':>12}{'target':>8}"
              f"{'grey18':>8}{'black':>7}")
        for r in doc["looks"]:
            if "error" in r:
                print(f"{r['name']:<12} {r['error']}")
                continue
            print(f"{r['name']:<12}{r['n_items']:>5}{r['rms_on_cube']:>11.3f}"
                  f"{(r['de00_photo_mean'] or float('nan')):>12.2f}"
                  f"{(r['de00_target'] or float('nan')):>8.1f}"
                  f"{r['grey_out_at_code31']:>8.1f}{r['black_out_code']:>7.1f}")
        return 0

    if args.cmd == "show":
        target = load_target(args.name)
        doc = json.loads((_root() / "looks" / f"{args.name}.json").read_text())
        rows = residual_rows(measure(_sampler(doc)), target_arrays(target))
        print(f"{args.name}: normalised residual RMS {residual_rms(rows):.3f}")
        print(format_residuals(rows, limit=20))
        return 0

    names = None if args.cmd == "all" else [args.name]
    doc = fit_all(names, max_iter=args.iters, write=not args.no_write)
    if args.cmd == "all":
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str),
                     encoding="utf-8")
        print(f"wrote {p}")
    else:
        rec = doc["looks"][0]
        print(format_residuals(rec["rows"], limit=20))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
