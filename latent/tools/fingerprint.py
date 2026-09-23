"""The Latent-2026 measurement space — probe ``latent-probe-1`` (pinned).

A *fingerprint* is the numeric portrait of a colour transform: what it does to
the grey axis, to 24 hues at two chroma levels, to chroma as a function of
input chroma and of luminance, to three skin patches, and globally.  It is the
common language between ``docs/PLAN.md`` 附录 A (target profiles), the
colourist agents, and QC.

Everything goes through ONE code path: :func:`fingerprint` takes a *sampler*
(any callable ``rgb -> rgb`` on sRGB code values), so a ``.cube`` file and a
compiled look are measured identically.  Lattice-only metrics (fold /
second-difference / clip) are filled in when the sampler exposes a ``.table``.

The pinned probe
----------------
``probe = "latent-probe-1"``.  Two fingerprints with different probe ids are
NOT comparable and :func:`distance` raises rather than pretend otherwise
(docs/PLAN.md: *"加载器拒绝比较探针不同的两份指纹"*).

Probe details that were genuinely ambiguous, and how they are pinned here
(the exploration agent's convention — verified to reproduce R02 §A1/A3/A4/A5
exactly; see ``fingerprint.py accept``):

1. **Input is always re-measured after clipping.**  A probe point is built with
   :func:`tools.metrics.code_from_oklch` (no clipping), clipped to ``[0, 1]``,
   and the *clipped* colour's OKLCh is what ``dh`` / ``cr`` / ``dl`` are
   measured against.  Reason: the sampler clips anyway (tetrahedral
   interpolation clamps its input), so the nominal (L, C, h) is not what was
   sampled; comparing the output against the nominal would charge the LUT for
   the clip.  This matters only for out-of-gamut probe points.
2. **Out-of-gamut handling differs per block, as TOOLS_SPEC §T2 states.**
   ``hue`` and ``chroma_vs_c`` null out-of-gamut points; ``arch`` does not (the
   spec gives no null rule there, and L = 0.25 / 0.85 at C = 0.10 is out of
   gamut for warm and blue — nulling would delete the chroma-arch target that
   附录 A is written in).  ``chroma_vs_c`` additionally carries ``cr_clip``,
   the un-nulled clip-and-re-measure value, because that is the convention R02
   §A4 published.
3. **The high-chroma hue column is ``chi = min(0.18, 0.80 * cmax(L, h))``**
   (TOOLS_SPEC §T2), which is in gamut by construction.  R02 §A3's "C = .18"
   column is a *fixed* C = 0.18 with clipping, which is a different probe; it
   is kept as the diagnostic block ``hue.c18`` so the R02 numbers stay
   checkable, and it is NOT part of :func:`distance`.
4. ``de00`` vs identity is reported on the pinned uniform 17³ lattice
   (TOOLS_SPEC §T2) and, additionally, on the 33³ lattice that R02 used.

Public API:
    fingerprint(sampler, photo_sample=None) -> dict
    distance(fp_a, fp_b) -> float
    compare(fp, target) -> list[dict]
    load_sampler(path), fingerprint_path(path), format_show(fp)
CLI:
    fingerprint.py show <cube|fingerprint.json|look.json>
    fingerprint.py dump <cube...> --out <dir>
    fingerprint.py dist <a> <b>
    fingerprint.py accept                 # R02 acceptance side-by-side (P0 gate)
    fingerprint.py pairs <dir|cube...> -n 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from tools import metrics as M

__all__ = [
    "PROBE_ID",
    "fingerprint",
    "distance",
    "compare",
    "load_sampler",
    "fingerprint_path",
    "format_show",
    "format_compare",
    "DISTANCE_ITEMS",
    "NORM",
]

FloatArray = np.ndarray

# ---------------------------------------------------------------------------
# The pinned probe constants.  Changing ANY of these means a new probe id.
# ---------------------------------------------------------------------------

PROBE_ID = "latent-probe-1"

#: grey ramp: output G code at these 8-bit input codes (PLAN 附录 A 约定)
RAMP_CODES: tuple[int, ...] = (0, 11, 31, 64, 115, 166, 209, 240, 255)
#: tint / slope sample points (fractions of full scale)
NEUTRAL_T: tuple[float, ...] = M.NEUTRAL_T  # (.05,.18,.35,.50,.65,.80,.95)
#: toe / mid / shoulder secant windows
TONE_WINDOWS: tuple[tuple[float, float], ...] = ((0.05, 0.18), (0.18, 0.50), (0.50, 0.90))

#: hue sweep
HUE_L: float = 0.65
HUES: tuple[float, ...] = tuple(float(h) for h in range(0, 360, 15))
C10: float = 0.10
CHI_CAP: float = 0.18
CHI_FRAC: float = 0.80
C18_LEGACY: float = 0.18  # R02 §A3's second column (diagnostic only)

#: chroma-vs-chroma
CHROMA_L: float = 0.65
CHROMA_C: tuple[float, ...] = (0.03, 0.06, 0.10, 0.14, 0.18, 0.22)
CHROMA_H: tuple[float, ...] = (25.0, 55.0, 100.0, 145.0, 200.0, 250.0, 300.0)

#: saturation-vs-luminance arch
ARCH_C: float = 0.10
ARCH_L: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85)
ARCH_H: tuple[float, ...] = (55.0, 145.0, 250.0)

#: skin patches, sRGB 8-bit
SKIN_PATCHES: tuple[tuple[int, int, int], ...] = ((200, 150, 125), (150, 100, 80), (235, 195, 170))

#: global dE00 lattices
LATTICE_N: int = 17          # pinned by TOOLS_SPEC §T2
LATTICE_N_R02: int = 33      # the lattice R02 published its dE00 means on

_WORK = Path(os.environ.get("LATENT_WORK", Path(__file__).resolve().parent.parent / "work.nosync"))
PHOTO_SAMPLE_PATH = _WORK / "cal" / "photo_sample.npy"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _wrap180(d: FloatArray) -> FloatArray:
    """Signed angular difference folded into (-180, 180]."""
    return (np.asarray(d, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _nan_to_none(a) -> list:
    """numpy array -> nested lists with NaN replaced by ``None`` (JSON null)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        v = float(a)
        return None if math.isnan(v) else v
    return [_nan_to_none(x) for x in a]


def _f(a) -> list:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return float(a)
    return [_f(x) for x in a]


def _none_to_nan(x):
    if x is None:
        return float("nan")
    if isinstance(x, (list, tuple)):
        return [_none_to_nan(v) for v in x]
    return x


def _arr(x) -> FloatArray:
    """JSON value (nested lists possibly containing None) -> float array w/ NaN."""
    return np.asarray(_none_to_nan(x), dtype=np.float64)


def _probe_points(
    sampler: Callable[[FloatArray], FloatArray],
    L: FloatArray,
    C: FloatArray,
    h: FloatArray,
) -> dict:
    """Measure a sampler at OKLCh probe points.

    Builds the codes with no clipping, clips to ``[0, 1]`` (what the sampler
    would do anyway), and reports everything against the *clipped* input:

    ``in_gamut`` — was the unclipped code inside the sRGB gamut (linear eps
    5e-4, :func:`tools.metrics.code_from_oklch`'s mask);
    ``dh`` (out - in, signed degrees), ``cr`` (C_out / C_in), ``dl``
    (L_out - L_in), plus the realised input ``h_in`` / ``c_in`` / ``l_in`` and
    the output codes.
    """
    L, C, h = np.broadcast_arrays(
        np.asarray(L, dtype=np.float64),
        np.asarray(C, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
    )
    code, in_gamut = M.code_from_oklch(L, C, h)
    clipped = np.clip(code, 0.0, 1.0)
    l_in, c_in, h_in = M.oklch_from_code(clipped)
    out = np.asarray(sampler(clipped), dtype=np.float64)
    if out.shape != clipped.shape:
        raise ValueError(f"sampler returned shape {out.shape}, expected {clipped.shape}")
    l_out, c_out, h_out = M.oklch_from_code(out)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = np.where(c_in > 1e-9, c_out / np.where(c_in > 1e-9, c_in, 1.0), np.nan)
    return {
        "in_gamut": in_gamut,
        "dh": _wrap180(h_out - h_in),
        "cr": cr,
        "dl": l_out - l_in,
        "h_in": h_in,
        "c_in": c_in,
        "l_in": l_in,
        "code_in": clipped,
        "code_out": out,
    }


def _masked(values: FloatArray, keep: FloatArray) -> FloatArray:
    out = np.array(values, dtype=np.float64, copy=True)
    out[~np.asarray(keep, dtype=bool)] = np.nan
    return out


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------


def _neutral_block(sampler: Callable[[FloatArray], FloatArray]) -> dict:
    ns = M.neutral_stats(sampler)  # 4097-pt ramp: tint, slope, monotonicity...

    # One extra sampler call covers the 9 pinned ramp codes, the 7 tint/slope
    # sample points and the toe/mid/shoulder secant edges, evaluated exactly
    # (no interpolation of the interpolation).
    codes = np.asarray(RAMP_CODES, dtype=np.float64) / 255.0
    edges = np.array(sorted({t for w in TONE_WINDOWS for t in w}))
    ts = np.asarray(NEUTRAL_T, dtype=np.float64)
    xs = np.concatenate([codes, ts, edges])
    ev = np.asarray(sampler(np.stack([xs] * 3, axis=-1)), dtype=np.float64)
    ramp_out = ev[: codes.size]
    at_t = ev[codes.size : codes.size + ts.size]
    at = {float(t): float(v) for t, v in zip(edges, ev[codes.size + ts.size :, 1])}
    toe, mid, shoulder = ((at[b] - at[a]) / (b - a) for a, b in TONE_WINDOWS)

    return {
        "codes": list(RAMP_CODES),
        "ramp8": _f(ramp_out[:, 1] * 255.0),
        "ramp8_rgb": _f(ramp_out * 255.0),
        "t": list(NEUTRAL_T),
        "out_t": _f(at_t[:, 1] * 255.0),
        "out_t_rgb": _f(at_t * 255.0),
        "tint_rg": _f(ns["tint_rg"]),
        "tint_bg": _f(ns["tint_bg"]),
        "slope": _f(ns["slope"]),
        "toe": float(toe),
        "mid": float(mid),
        "shoulder": float(shoulder),
        "black": _f(np.asarray(ns["black"]) * 255.0),
        "white": _f(np.asarray(ns["white"]) * 255.0),
        "monotone": bool(ns["monotone"]),
        "monotone_per_channel": [bool(x) for x in ns["monotone_per_channel"]],
        "monotone_L": bool(ns["monotone_L"]),
        "min_step": float(ns["min_step"] * 255.0),
        "min_step_L": float(ns["min_step_L"]),
    }


def _hue_column(sampler, L: float, C: FloatArray, hues: FloatArray, *, strict: bool) -> dict:
    hues = np.asarray(hues, dtype=np.float64)
    C = np.broadcast_to(np.asarray(C, dtype=np.float64), hues.shape)
    p = _probe_points(sampler, np.full(hues.shape, L), C, hues)
    keep = p["in_gamut"] if strict else np.ones(hues.shape, dtype=bool)
    return {
        "c": _f(C),
        "c_in": _f(p["c_in"]),
        "h_in": _f(p["h_in"]),
        "in_gamut": [bool(x) for x in p["in_gamut"]],
        "dh": _nan_to_none(_masked(p["dh"], keep)),
        "cr": _nan_to_none(_masked(p["cr"], keep)),
        "dl": _nan_to_none(_masked(p["dl"], keep)),
    }


def _hue_block(sampler) -> dict:
    hues = np.asarray(HUES, dtype=np.float64)
    cm = M.cmax(np.full(hues.shape, HUE_L), hues)
    chi = np.minimum(CHI_CAP, CHI_FRAC * cm)
    return {
        "L": HUE_L,
        "h": list(HUES),
        "cmax": _f(cm),
        "c10": _hue_column(sampler, HUE_L, np.full(hues.shape, C10), hues, strict=True),
        "chi": _hue_column(sampler, HUE_L, chi, hues, strict=True),
        # diagnostic: R02 §A3's fixed-C column (clipped, re-measured); not in distance()
        "c18": _hue_column(sampler, HUE_L, np.full(hues.shape, C18_LEGACY), hues, strict=False),
    }


def _chroma_block(sampler) -> dict:
    C = np.asarray(CHROMA_C, dtype=np.float64)[None, :]
    h = np.asarray(CHROMA_H, dtype=np.float64)[:, None]
    C, h = np.broadcast_arrays(C, h)
    p = _probe_points(sampler, np.full(C.shape, CHROMA_L), C, h)
    return {
        "L": CHROMA_L,
        "C": list(CHROMA_C),
        "h": list(CHROMA_H),
        "cmax": _f(M.cmax(np.full(len(CHROMA_H), CHROMA_L), np.asarray(CHROMA_H))),
        "in_gamut": [[bool(x) for x in row] for row in p["in_gamut"]],
        "c_in": _f(p["c_in"]),
        "cr": _nan_to_none(_masked(p["cr"], p["in_gamut"])),
        "cr_clip": _f(p["cr"]),  # R02 §A4 convention (no nulling)
        "dl": _nan_to_none(_masked(p["dl"], p["in_gamut"])),
    }


def _arch_block(sampler) -> dict:
    L = np.asarray(ARCH_L, dtype=np.float64)[None, :]
    h = np.asarray(ARCH_H, dtype=np.float64)[:, None]
    L, h = np.broadcast_arrays(L, h)
    p = _probe_points(sampler, L, np.full(L.shape, ARCH_C), h)
    return {
        "C": ARCH_C,
        "L": list(ARCH_L),
        "h": list(ARCH_H),
        "in_gamut": [[bool(x) for x in row] for row in p["in_gamut"]],
        "c_in": _f(p["c_in"]),
        "cr": _f(p["cr"]),
        "dl": _f(p["dl"]),
    }


def _skin_block(sampler) -> dict:
    patches = np.asarray(SKIN_PATCHES, dtype=np.float64) / 255.0
    out = np.asarray(sampler(patches), dtype=np.float64)
    l_in, c_in, h_in = M.oklch_from_code(patches)
    l_out, c_out, h_out = M.oklch_from_code(out)
    de = M.de00(M.srgb_to_lab(patches), M.srgb_to_lab(out))
    return {
        "patches": [list(p) for p in SKIN_PATCHES],
        "out": _f(out * 255.0),
        "h_in": _f(h_in),
        "dh": _f(_wrap180(h_out - h_in)),
        "cr": _f(c_out / c_in),
        "dl": _f(l_out - l_in),
        "de00": _f(de),
    }


_LATTICE_CACHE: dict[int, FloatArray] = {}


def _lattice(n: int) -> FloatArray:
    grid = _LATTICE_CACHE.get(n)
    if grid is None:
        t = np.linspace(0.0, 1.0, n)
        b, g, r = np.meshgrid(t, t, t, indexing="ij")
        grid = np.stack([r, g, b], axis=-1).reshape(-1, 3)
        _LATTICE_CACHE[n] = grid
    return grid


def _de00_vs_identity(sampler, points: FloatArray) -> dict:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.asarray(sampler(points), dtype=np.float64)
    de = M.de00(M.srgb_to_lab(points), M.srgb_to_lab(out))
    return {
        "de00_mean": float(de.mean()),
        "de00_p95": float(np.percentile(de, 95.0)),
        "de00_p99": float(np.percentile(de, 99.0)),
        "de00_max": float(de.max()),
        "n": int(de.size),
    }


def _load_photo_sample(photo_sample) -> FloatArray | None:
    if photo_sample is False:
        return None
    if photo_sample is None:
        if not PHOTO_SAMPLE_PATH.exists():
            return None
        photo_sample = np.load(PHOTO_SAMPLE_PATH)
    arr = np.asarray(photo_sample, dtype=np.float64).reshape(-1, 3)
    if arr.size and arr.max() > 1.5:  # stored as 8-bit codes
        arr = arr / 255.0
    return arr


def _global_block(sampler, photo_sample) -> dict:
    out: dict = {
        "photo": None,
        "photo_source": None,
        f"lattice{LATTICE_N}": _de00_vs_identity(sampler, _lattice(LATTICE_N)),
        f"lattice{LATTICE_N_R02}": _de00_vs_identity(sampler, _lattice(LATTICE_N_R02)),
        "d2": None,
        "clip": None,
        "fold": None,
    }
    photo = _load_photo_sample(photo_sample)
    if photo is not None:
        out["photo"] = _de00_vs_identity(sampler, photo)
        out["photo_source"] = str(PHOTO_SAMPLE_PATH) if photo_sample is None else "provided"

    table = getattr(sampler, "table", None)
    if table is not None:
        table = np.asarray(table, dtype=np.float64)
        out["d2"] = M.second_diff_stats(table)
        out["clip"] = M.clip_stats(table)
        out["fold"] = M.fold_stats(table)
        out["size"] = int(table.shape[0])
    return out


# ---------------------------------------------------------------------------
# the fingerprint
# ---------------------------------------------------------------------------


def fingerprint(
    sampler: Callable[[FloatArray], FloatArray],
    photo_sample=None,
    *,
    name: str | None = None,
) -> dict:
    """Measure *sampler* with probe ``latent-probe-1``.

    ``sampler`` is any callable mapping sRGB code arrays ``(..., 3)`` to code
    arrays of the same shape.  Table-backed samplers (``.table``) additionally
    get the lattice metrics (second differences, clipping, folding).

    ``photo_sample``: ``None`` loads ``$W/cal/photo_sample.npy`` when it exists,
    ``False`` skips it, or pass an ``(N, 3)`` array of code values.
    """
    if not callable(sampler):
        raise TypeError("fingerprint() needs a callable sampler rgb -> rgb")
    fp = {
        "probe": PROBE_ID,
        "name": name or getattr(sampler, "title", None) or "unnamed",
        "source": str(getattr(sampler, "source", "") or "") or None,
        "neutral": _neutral_block(sampler),
        "hue": _hue_block(sampler),
        "chroma_vs_c": _chroma_block(sampler),
        "arch": _arch_block(sampler),
        "skin": _skin_block(sampler),
        "global": _global_block(sampler, photo_sample),
    }
    return fp


# ---------------------------------------------------------------------------
# distance / compare
# ---------------------------------------------------------------------------

#: normalisation scales (TOOLS_SPEC §T2): one unit of each is "one JND" for the
#: purposes of the collision metric.
NORM: dict[str, float] = {
    "deg": 3.0,      # hue degrees
    "ratio": 0.05,   # chroma ratio
    "code": 2.0,     # 8-bit code value
    "slope": 0.05,   # local tone slope
    "dl": 0.01,      # OKLab L difference
}

#: (dotted path, kind) pairs that enter :func:`distance`, exactly as TOOLS_SPEC
#: §T2 lists them: neutral.ramp8, neutral.tint, hue.c10.*, hue.chi.*, arch.*.
DISTANCE_ITEMS: tuple[tuple[str, str], ...] = (
    ("neutral.ramp8", "code"),
    ("neutral.tint_rg", "code"),
    ("neutral.tint_bg", "code"),
    ("hue.c10.dh", "deg"),
    ("hue.c10.cr", "ratio"),
    ("hue.c10.dl", "dl"),
    ("hue.chi.dh", "deg"),
    ("hue.chi.cr", "ratio"),
    ("hue.chi.dl", "dl"),
    ("arch.cr", "ratio"),
    ("arch.dl", "dl"),
)

#: everything :func:`compare` will line up against a target profile.  A superset
#: of DISTANCE_ITEMS — the extra rows are diagnostics, not collision terms.
COMPARE_ITEMS: tuple[tuple[str, str], ...] = DISTANCE_ITEMS + (
    ("neutral.slope", "slope"),
    ("neutral.toe", "slope"),
    ("neutral.mid", "slope"),
    ("neutral.shoulder", "slope"),
    ("neutral.black", "code"),
    ("neutral.white", "code"),
    ("chroma_vs_c.cr", "ratio"),
    ("skin.dh", "deg"),
    ("skin.cr", "ratio"),
    ("skin.dl", "dl"),
    ("skin.de00", "code"),
)


def _dig(d, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _check_probe(fp_a: dict, fp_b: dict) -> None:
    pa, pb = fp_a.get("probe"), fp_b.get("probe")
    if pa != pb:
        raise ValueError(
            f"probe mismatch: {pa!r} vs {pb!r} — fingerprints from different probes "
            "are not comparable (docs/PLAN.md: 加载器拒绝比较探针不同的两份指纹)"
        )
    if pa != PROBE_ID:
        raise ValueError(f"unknown probe {pa!r}; this module implements {PROBE_ID!r}")


def distance(fp_a: dict, fp_b: dict) -> float:
    """RMS normalised difference between two fingerprints.

    Terms: ``neutral.ramp8``, ``neutral.tint_{rg,bg}``, ``hue.c10.{dh,cr,dl}``,
    ``hue.chi.{dh,cr,dl}``, ``arch.{cr,dl}`` — each difference divided by its
    scale in :data:`NORM` (hue°/3, chroma-ratio/0.05, 8-bit code/2, ΔL/0.01).
    197 terms for a complete pair.  Items that are ``null`` in either
    fingerprint (out-of-gamut probe points) are skipped.

    Raises ``ValueError`` if the two fingerprints carry different probe ids.
    """
    _check_probe(fp_a, fp_b)
    total = 0.0
    n = 0
    for path, kind in DISTANCE_ITEMS:
        a = _arr(_dig(fp_a, path))
        b = _arr(_dig(fp_b, path))
        if a.shape != b.shape:
            raise ValueError(f"{path}: shape {a.shape} vs {b.shape}")
        d = (a - b) / NORM[kind]
        ok = np.isfinite(d)
        total += float(np.sum(d[ok] ** 2))
        n += int(ok.sum())
    if n == 0:
        raise ValueError("no comparable terms between the two fingerprints")
    return math.sqrt(total / n)


def compare(fp: dict, target) -> list[dict]:
    """Per-item residual table of *fp* against a target profile.

    ``target`` may be a fingerprint dict, a path to one, or a partial profile
    that carries only some blocks (a 附录 A target).  Only the items present in
    the target are reported.  Each row is
    ``{path, index, target, measured, residual, kind, norm_residual}`` with the
    residual normalised by :data:`NORM`; rows are sorted worst-first.
    """
    if isinstance(target, (str, Path)):
        target = json.loads(Path(target).read_text())
    if not isinstance(target, dict):
        raise TypeError("target must be a dict or a path to a JSON profile")
    if "probe" in target:
        _check_probe(fp, target)

    rows: list[dict] = []
    for path, kind in COMPARE_ITEMS:
        tv = _dig(target, path)
        if tv is None:
            continue
        mv = _dig(fp, path)
        if mv is None:
            rows.append(
                {"path": path, "index": None, "target": None, "measured": None,
                 "residual": None, "kind": kind, "norm_residual": float("nan")}
            )
            continue
        t = _arr(tv)
        m = _arr(mv)
        if t.shape != m.shape:
            raise ValueError(f"{path}: target shape {t.shape} vs measured {m.shape}")
        flat_t, flat_m = t.reshape(-1), m.reshape(-1)
        idx = list(np.ndindex(t.shape)) if t.ndim else [()]
        for k, ix in enumerate(idx):
            resid = float(flat_m[k] - flat_t[k])
            rows.append(
                {
                    "path": path,
                    "index": tuple(int(i) for i in ix) if ix else None,
                    "target": float(flat_t[k]),
                    "measured": float(flat_m[k]),
                    "residual": resid,
                    "kind": kind,
                    "norm_residual": abs(resid) / NORM[kind],
                }
            )
    rows.sort(key=lambda r: (-(r["norm_residual"] if r["norm_residual"] == r["norm_residual"] else 1e9)))
    return rows


def format_compare(rows: Sequence[dict], limit: int = 40) -> str:
    out = [f"{'item':<24s}{'idx':>10s}{'target':>10s}{'meas':>10s}{'resid':>10s}{'norm':>8s}"]
    for r in rows[:limit]:
        ix = "" if r["index"] is None else ",".join(str(i) for i in r["index"])
        if r["measured"] is None:
            out.append(f"{r['path']:<24s}{ix:>10s}{'-':>10s}{'MISSING':>10s}{'-':>10s}{'-':>8s}")
            continue
        out.append(
            f"{r['path']:<24s}{ix:>10s}{r['target']:>10.4f}{r['measured']:>10.4f}"
            f"{r['residual']:>+10.4f}{r['norm_residual']:>8.2f}"
        )
    if len(rows) > limit:
        out.append(f"... {len(rows) - limit} more rows")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_sampler(path) -> Callable[[FloatArray], FloatArray]:
    """``.cube`` / ``.npy`` / ``.npz`` / look ``.json`` -> sampler.

    ``.cube`` and raw tables go straight through
    :func:`tools.metrics.sampler_from_cube` / ``sampler_from_table``.  A look
    ``.json`` is compiled by the engine track's ``looks.compile`` /
    ``engine.compile`` if it exists (it does not yet); a JSON that simply
    carries a ``"table"`` is used directly.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".cube":
        return M.sampler_from_cube(path)
    if suffix == ".npy":
        return M.sampler_from_table(np.load(path), title=path.stem)
    if suffix == ".npz":
        with np.load(path) as z:
            key = "table" if "table" in z else list(z.keys())[0]
            return M.sampler_from_table(z[key], title=path.stem)
    if suffix == ".json":
        spec = json.loads(path.read_text())
        if "table" in spec:
            return M.sampler_from_table(np.asarray(spec["table"], dtype=np.float64), title=path.stem)
        for mod_name, fn_name in (("looks.compile", "compile_look"), ("engine.compile", "compile_look")):
            try:
                mod = __import__(mod_name, fromlist=[fn_name])
            except Exception:
                continue
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                built = fn(spec)
                table = getattr(built, "table", built)
                return M.sampler_from_table(np.asarray(table, dtype=np.float64), title=path.stem)
        raise NotImplementedError(
            f"{path.name}: no look compiler available yet (looks.compile.compile_look). "
            "Pass a .cube, or a JSON carrying a 'table'."
        )
    raise ValueError(f"unsupported input {path.name!r}")


def fingerprint_path(path, photo_sample=None) -> dict:
    """Fingerprint whatever is at *path* (a fingerprint JSON is returned as-is)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        try:
            doc = json.loads(path.read_text())
        except Exception:
            doc = None
        if isinstance(doc, dict) and doc.get("probe"):
            return doc
    sampler = load_sampler(path)
    fp = fingerprint(sampler, photo_sample, name=path.stem)
    fp["source"] = str(path)
    return fp


# ---------------------------------------------------------------------------
# human-readable view
# ---------------------------------------------------------------------------


def _n(x, fmt="{:.2f}", dash="  --"):
    return dash if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def format_show(fp: dict) -> str:
    """Compact table of a fingerprint — what a colourist needs to see."""
    L: list[str] = []
    ne, hu, ch, ar, sk, gl = (fp[k] for k in ("neutral", "hue", "chroma_vs_c", "arch", "skin", "global"))
    L.append(f"{fp.get('name', '?')}   probe={fp.get('probe')}   {fp.get('source') or ''}")

    L.append("")
    L.append("TONE  in-code  " + " ".join(f"{c:>6d}" for c in ne["codes"]))
    L.append("      out G    " + " ".join(f"{v:>6.1f}" for v in ne["ramp8"]))
    L.append(
        f"      toe(5-18%) {ne['toe']:.2f}   mid(18-50%) {ne['mid']:.2f}   "
        f"shoulder(50-90%) {ne['shoulder']:.2f}"
    )
    L.append(
        f"      black {ne['black'][0]:.1f},{ne['black'][1]:.1f},{ne['black'][2]:.1f}   "
        f"white {ne['white'][0]:.1f},{ne['white'][1]:.1f},{ne['white'][2]:.1f}   "
        f"mono={'yes' if ne['monotone'] else 'NO'}/L={'yes' if ne['monotone_L'] else 'NO'}  "
        f"min_step {ne['min_step']:+.4f} code"
    )
    L.append("TINT  t%       " + " ".join(f"{t*100:>6.0f}" for t in ne["t"]))
    L.append("      R-G      " + " ".join(f"{v:>+6.2f}" for v in ne["tint_rg"]))
    L.append("      B-G      " + " ".join(f"{v:>+6.2f}" for v in ne["tint_bg"]))
    L.append("      slope    " + " ".join(f"{v:>6.2f}" for v in ne["slope"]))

    L.append("")
    L.append(f"HUE @ L={hu['L']}          C=0.10                 C=chi=min(.18,.80*cmax)      [C=0.18 clipped]")
    L.append("   h   cmax |    dh     cr     dL |  chi     dh     cr     dL |    dh     cr")
    for i, h in enumerate(hu["h"]):
        c10, chi, c18 = hu["c10"], hu["chi"], hu["c18"]
        L.append(
            f"{h:4.0f}  {hu['cmax'][i]:.3f} | {_n(c10['dh'][i], '{:+6.2f}'):>6s} "
            f"{_n(c10['cr'][i], '{:6.3f}'):>6s} {_n(c10['dl'][i], '{:+6.3f}'):>6s} | "
            f"{chi['c'][i]:.3f} {_n(chi['dh'][i], '{:+6.2f}'):>6s} "
            f"{_n(chi['cr'][i], '{:6.3f}'):>6s} {_n(chi['dl'][i], '{:+6.3f}'):>6s} | "
            f"{_n(c18['dh'][i], '{:+6.2f}'):>6s} {_n(c18['cr'][i], '{:6.3f}'):>6s}"
        )

    L.append("")
    L.append(f"CHROMA vs C  (L={ch['L']})   C = " + " ".join(f"{c:>6.2f}" for c in ch["C"]))
    for j, h in enumerate(ch["h"]):
        L.append(
            f"   h={h:5.0f} cmax={ch['cmax'][j]:.3f} cr  "
            + " ".join(_n(v, "{:6.3f}").rjust(6) for v in ch["cr"][j])
        )
        L.append(
            "                    (clip) "
            + " ".join(f"{v:>6.3f}" for v in ch["cr_clip"][j])
        )

    L.append("")
    L.append(f"ARCH  C={ar['C']:.2f}  L = " + " ".join(f"{v:>6.2f}" for v in ar["L"]))
    for j, h in enumerate(ar["h"]):
        tag = {55.0: "warm", 145.0: "green", 250.0: "blue"}.get(float(h), "")
        L.append(
            f"   h={h:5.0f} {tag:<5s} cr " + " ".join(f"{v:>6.3f}" for v in ar["cr"][j])
            + "   dL " + " ".join(f"{v:>+6.3f}" for v in ar["dl"][j])
            + "   ig " + "".join("." if g else "x" for g in ar["in_gamut"][j])
        )

    L.append("")
    L.append("SKIN        in -> out                 dh      cr      dL    dE00")
    for i, p in enumerate(sk["patches"]):
        o = sk["out"][i]
        L.append(
            f"  {p[0]:3d},{p[1]:3d},{p[2]:3d} -> {o[0]:5.1f},{o[1]:5.1f},{o[2]:5.1f}   "
            f"{sk['dh'][i]:+6.2f}  {sk['cr'][i]:6.3f}  {sk['dl'][i]:+6.3f}  {sk['de00'][i]:6.2f}"
        )

    L.append("")
    lat = gl[f"lattice{LATTICE_N}"]
    lat33 = gl[f"lattice{LATTICE_N_R02}"]
    L.append(
        f"GLOBAL  dE00 vs identity  17^3 mean {lat['de00_mean']:.3f} p95 {lat['de00_p95']:.2f}"
        f"   | 33^3 mean {lat33['de00_mean']:.3f} p95 {lat33['de00_p95']:.2f}"
    )
    if gl.get("photo"):
        p = gl["photo"]
        L.append(f"        photo sample (n={p['n']})  mean {p['de00_mean']:.3f} p95 {p['de00_p95']:.2f}")
    else:
        L.append("        photo sample: absent ($W/cal/photo_sample.npy)")
    if gl.get("d2"):
        d2, cl, fd = gl["d2"], gl["clip"], gl["fold"]
        pa = d2["per_axis"]
        L.append(
            f"        d2 codes  full p99.9 {d2['full']['p99_9']:.2f} max {d2['full']['max']:.2f}"
            f"   interior p99.9 {d2['interior']['p99_9']:.2f} max {d2['interior']['max']:.2f}"
            f"   per-axis B/G/R p99.9 {pa['axis0']['p99_9']:.2f}/{pa['axis1']['p99_9']:.2f}/{pa['axis2']['p99_9']:.2f}"
        )
        L.append(
            f"        clip =0 {cl['zero']*100:.2f}%  =1 {cl['one']*100:.2f}%"
            f"   (nodes any: {cl['zero_any']*100:.2f}% / {cl['one_any']*100:.2f}%)"
        )
        L.append(
            f"        fold min_ratio {fd['min_ratio']:+.4f}  ratio<=0 {fd['neg_count']}/{fd['n_tetra']}"
            f"  (strict<0 {fd['neg_strict_count']}, degenerate==0 {fd['zero_count']})"
        )
    else:
        L.append("        lattice metrics: n/a (sampler is not table-backed)")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# R02 acceptance data (docs/research/02_reference_lut_fingerprints.md)
# ---------------------------------------------------------------------------
# Transcribed from R02 §A0/A1/A3/A4/A5/A7.  These are the numbers the P0 gate
# has to reproduce.  Hue rows are keyed by the PROBE hue; R02's condensed §A3
# table labels its rows by the *realised* (post-clip) input hue, which is why
# the C=0.18 rows R02 prints as "60 (64°)" / "90 (82°)" / "120 (121°)" /
# "240 (235°)" are probe hues 75 / 90 / 120 / 225 here.

# TOOLS_SPEC §T2's acceptance tolerances, plus "pct" for the clipping shares:
# R02 prints those to one decimal, so half the last printed digit is the only
# honest tolerance for them.
_R02_TOL = {"deg": 0.6, "ratio": 0.02, "dl": 0.006, "code": 1.0, "pct": 0.05}

#: R02 rows that this probe deliberately does NOT match, with the reason.  They
#: are reported and counted separately instead of being silently tolerated.
_R02_KNOWN_MISMATCH: dict[tuple[str, str], str] = {
    ("Leica_X2_VIVID_33_sRGB", "toe slope 5-18%"): (
        "R02 derived 0.85 from a grey-5% value it printed as '~2 (crushed)'. R02's own "
        "§A1 lattice row for this LUT is 0 / 0.1 / 0.2 / 0.3 / 4.1 at idx 0..4, which "
        "interpolates to ~0.16 codes at 5%, i.e. toe ~0.90 (measured 0.902). R02 flags "
        "its own 0.85 as meaningless; the lattice wins."
    ),
}

_R02: dict[str, dict] = {
    "Contax_ND_STD_33_sRGB": {
        "hue_c10": {
            0: (-2.7, 1.06, -0.00), 30: (1.0, 1.05, -0.00), 45: (4.1, 1.09, -0.00),
            60: (4.7, 1.13, -0.00), 90: (3.7, 1.19, 0.01), 120: (0.0, 1.21, 0.01),
            135: (-0.6, 1.18, 0.01), 150: (0.9, 1.14, 0.01), 195: (-0.1, 1.08, 0.01),
            240: (-6.4, 1.08, 0.00), 255: (-5.6, 1.03, 0.01), 270: (-3.2, 1.00, 0.01),
            300: (2.2, 1.04, 0.00), 330: (-0.3, 1.08, 0.00),
        },
        "hue_c18": {},
        "max_dh10": 6.4, "max_dh18": 3.6, "median_cr10": 1.085,
        "chroma_vs_c": {
            55: (1.15, 1.09, 1.12, 1.06, 1.02, 1.05),
            145: (1.22, 1.17, 1.15, 1.17, 1.13, 1.03),
            250: (1.10, 1.04, 1.05, 1.04, 0.98, 0.98),
        },
        "arch": {55: (0.76, 1.15, 1.18, 1.06, 0.96),
                 145: (0.75, 0.99, 1.27, 1.08, 0.96),
                 250: (0.94, 1.22, 1.15, 1.01, 0.96)},
        "skin": (((202, 154, 124), 6.0, 1.02, 0.009, 2.5),
                 ((154, 94, 65), 3.1, 1.25, -0.010, 3.9),
                 ((236, 197, 173), -0.2, 0.96, None, 0.6)),
        "neutral_pct": {5: (8, 8, 8), 18: (28, 28, 28), 50: (126, 127, 127),
                        80: (206, 206, 206), 100: (255, 255, 255)},
        "toe": 0.61, "mid": 1.21, "shoulder": 1.02,
        "black": 0.0, "white": 254.6,
        "de00": (2.14, 5.98), "clip": (1.0, 0.0),
        "d2_p999": (4.5, 4.5, 6.0), "d2_max": 6.0,
    },
    "Pentax_K5_Reversal_Film_33_sRGB": {
        "hue_c10": {
            0: (-1.4, 1.18, 0.03), 30: (13.0, 1.17, 0.03), 45: (15.4, 1.26, 0.04),
            60: (13.0, 1.32, 0.04), 90: (6.0, 1.36, 0.04), 120: (2.1, 1.50, 0.04),
            135: (-0.1, 1.57, 0.03), 150: (-2.1, 1.52, 0.03), 195: (2.4, 1.14, 0.03),
            240: (3.0, 1.23, 0.02), 255: (4.5, 1.23, 0.01), 270: (7.7, 1.26, 0.01),
            300: (6.4, 1.38, 0.02), 330: (-1.0, 1.37, 0.03),
        },
        # R02 §A3 prose: "At C=0.18 Pentax escalates: h=270 +15.6, h=285 +15.6, h=300 +10.9"
        "hue_c18": {270: (15.6, None, None), 285: (15.6, None, None), 300: (10.9, None, None)},
        "max_dh10": 15.4, "max_dh18": 15.6, "median_cr10": 1.285,
        "chroma_vs_c": {
            55: (1.40, 1.42, 1.31, 1.12, 1.01, 0.98),
            145: (1.47, 1.53, 1.55, 1.46, 1.28, 1.12),
            250: (1.37, 1.31, 1.23, 1.17, 1.15, 1.14),
        },
        "arch": {55: (0.84, 1.09, 1.29, 1.30, 1.08),
                 145: (0.70, 1.08, 1.52, 1.47, 1.10),
                 250: (1.15, 1.50, 1.46, 1.11, 0.81)},
        "skin": (((216, 167, 121), 17.3, 1.20, 0.049, 8.7),
                 ((163, 95, 56), 7.0, 1.44, None, 6.7),
                 ((243, 214, 181), 16.7, 0.96, 0.046, 7.0)),
        "neutral_pct": {5: (2, 2, 2), 18: (23, 23, 23), 50: (133, 133, 133),
                        80: (219, 219, 219), 100: (255, 255, 255)},
        "toe": 0.64, "mid": 1.34, "shoulder": 1.05,
        "black": 0.0, "white": 255.0,
        "de00": (5.37, 10.33), "clip": (9.7, 2.7),
        "d2_p999": (3.3, 3.3, 5.9), "d2_max": 8.7,
    },
    "Leica_X2_VIVID_33_sRGB": {
        "hue_c10": {
            0: (2.0, 1.40, -0.02), 30: (0.8, 1.41, -0.02), 45: (-1.2, 1.39, -0.02),
            60: (-3.4, 1.34, -0.02), 90: (-2.8, 1.21, -0.02), 120: (1.6, 1.30, -0.02),
            135: (0.7, 1.35, -0.02), 150: (-1.5, 1.35, -0.02), 195: (-0.4, 1.12, 0.01),
            240: (2.7, 1.33, -0.01), 255: (0.5, 1.39, -0.02), 270: (-1.1, 1.41, -0.02),
            300: (-1.3, 1.39, -0.03), 330: (0.6, 1.39, -0.03),
        },
        "hue_c18": {
            0: (3.6, 1.29, 0.01), 30: (1.0, 1.28, 0.01), 45: (-8.0, 1.24, 0.02),
            75: (-13.2, 1.16, None),   # R02 row "60 (64 deg)"
            90: (-11.6, 1.04, None),   # R02 row "90 (82 deg)"
            120: (3.8, 1.04, None),    # R02 row "120 (121 deg)"
            135: (3.5, 1.14, 0.01), 150: (-4.3, 1.20, 0.04), 195: (0.4, 1.05, 0.03),
            210: (6.4, None, None),    # R02 prose "+6.4 at 219 deg"
            225: (7.1, 1.18, None),    # R02 row "240 (235 deg)"
            255: (-1.4, 1.10, -0.01), 270: (-3.0, 1.14, -0.03),
            300: (-1.0, 1.29, -0.02), 330: (0.7, 1.31, 0.00),
        },
        "max_dh10": 4.4, "max_dh18": 13.2, "median_cr10": 1.345,
        "chroma_vs_c": {
            55: (1.42, 1.40, 1.36, 1.27, 1.24, 1.17),
            145: (1.41, 1.40, 1.36, 1.30, 1.21, 1.12),
            250: (1.41, 1.40, 1.37, 1.32, 1.05, 0.98),
        },
        "arch": {55: (1.02, 1.21, 1.32, 1.37, 1.15),
                 145: (0.74, 1.23, 1.34, 1.37, 1.37),
                 250: (1.25, 1.30, 1.35, 1.37, 1.09)},
        "skin": (((208, 138, 103), -0.6, 1.41, -0.019, 5.3),
                 ((157, 87, 59), -0.5, 1.41, None, 5.7),
                 ((251, 196, 161), 0.3, 1.37, None, 4.1)),
        "neutral_pct": {18: (30, 30, 30), 50: (118, 118, 118),
                        80: (209, 209, 209), 100: (255, 255, 255)},
        "toe": 0.85, "mid": 1.08, "shoulder": 1.18,
        "black": 0.0, "white": 254.9,
        "de00": (4.42, 6.85), "clip": (14.8, 10.9),
        "d2_p999": (10.5, 10.9, 10.7), "d2_max": 11.7,
    },
}

REFS_DIR = _WORK / "refs"
_ACCEPT_FILES = {
    "Contax_ND_STD_33_sRGB": REFS_DIR / "reverse" / "Contax_ND_STD_33_sRGB.cube",
    "Pentax_K5_Reversal_Film_33_sRGB": REFS_DIR / "reverse" / "Pentax_K5_Reversal_Film_33_sRGB.cube",
    "Leica_X2_VIVID_33_sRGB": REFS_DIR / "reverse" / "Leica_X2_VIVID_33_sRGB.cube",
}


def _cell(target, measured, kind, *, fmt="{:+7.2f}"):
    """One acceptance row cell: (text, abs deviation or None, pass/fail)."""
    if target is None:
        return (f"{'':>8s}{fmt.format(measured):>9s}{'':>9s}", None, None)
    dev = abs(measured - target)
    ok = dev <= _R02_TOL[kind] + 1e-9
    return (
        f"{fmt.format(target):>8s}{fmt.format(measured):>9s}"
        f"{('%+.2f' % (measured - target)):>8s} {'ok' if ok else 'FAIL'}",
        dev,
        ok,
    )


def acceptance_report(fps: dict[str, dict] | None = None) -> tuple[str, dict]:
    """The P0 side-by-side against R02 for Pentax / Contax / Leica X2.

    Returns ``(text, summary)`` where ``summary[name][block] =
    {"max_dev", "n", "n_fail", "tol"}``.
    """
    if fps is None:
        fps = {}
        for name, path in _ACCEPT_FILES.items():
            fps[name] = fingerprint_path(path, photo_sample=False)

    out: list[str] = []
    summary: dict[str, dict] = {}
    for name, ref in _R02.items():
        fp = fps[name]
        summary[name] = {}
        hu, ch, ar, sk, ne, gl = (
            fp["hue"], fp["chroma_vs_c"], fp["arch"], fp["skin"], fp["neutral"], fp["global"]
        )
        hidx = {int(h): i for i, h in enumerate(hu["h"])}
        out.append("")
        out.append("=" * 104)
        out.append(f"{name}   (probe {fp['probe']})")
        out.append("=" * 104)

        def block(tag: str, rows: list[tuple[str, object, float, str]], extra: str = "") -> None:
            devs, fails, known = [], 0, 0
            lines = []
            for label, tv, mv, kind in rows:
                fmt = {"deg": "{:+7.2f}", "ratio": "{:7.3f}", "dl": "{:+7.4f}",
                       "code": "{:7.2f}", "pct": "{:7.3f}"}[kind]
                txt, dev, ok = _cell(tv, mv, kind, fmt=fmt)
                note = _R02_KNOWN_MISMATCH.get((name, label))
                if note is not None and not ok:
                    txt = txt.replace("FAIL", "KNOWN")
                lines.append(f"  {label:<26s}{txt}")
                if note is not None and not ok:
                    known += 1
                    lines.append(f"      ^ known R02 defect: {note}")
                elif dev is not None:
                    devs.append(dev)
                    fails += 0 if ok else 1
            tolv = {k: _R02_TOL[k] for k in {r[3] for r in rows}}
            head = (f"-- {tag}   (tol " + ", ".join(f"{k} +-{v}" for k, v in sorted(tolv.items())) + ")")
            out.append(head + (("   " + extra) if extra else ""))
            out.append(f"  {'item':<26s}{'R02':>8s}{'measured':>9s}{'delta':>8s}")
            out.extend(lines)
            mx = max(devs) if devs else float("nan")
            out.append(
                f"  -> {len(devs)} compared, {fails} outside tolerance, "
                f"{known} known-R02-defect, max deviation {mx:.4f}"
            )
            summary[name][tag] = {"max_dev": mx, "n": len(devs), "n_fail": fails, "n_known": known}

        # --- neutral -------------------------------------------------------
        rows: list[tuple[str, object, float, str]] = []
        for pct, rgb in sorted(ref["neutral_pct"].items()):
            mv = ne["white"][1] if pct == 100 else _grey_at(ne, pct / 100.0)
            rows.append((f"grey {pct}%  out G", float(rgb[1]), mv, "code"))
        rows.append(("black (G code)", ref["black"], ne["black"][1], "code"))
        rows.append(("white (min channel)", ref["white"], min(ne["white"]), "code"))
        rows.append(("toe slope 5-18%", ref["toe"], ne["toe"], "ratio"))
        rows.append(("mid slope 18-50%", ref["mid"], ne["mid"], "ratio"))
        rows.append(("shoulder slope 50-90%", ref["shoulder"], ne["shoulder"], "ratio"))
        block("NEUTRAL", rows)

        # --- hue @ C=0.10 ---------------------------------------------------
        rows = []
        for h, (dh, cr, dl) in sorted(ref["hue_c10"].items()):
            i = hidx[h]
            rows.append((f"h={h:3d} dh", dh, hu["c10"]["dh"][i], "deg"))
            rows.append((f"h={h:3d} cr", cr, hu["c10"]["cr"][i], "ratio"))
            if dl is not None:
                rows.append((f"h={h:3d} dL", dl, hu["c10"]["dl"][i], "dl"))
        med = float(np.nanmedian(_arr(hu["c10"]["cr"])))
        mx10 = float(np.nanmax(np.abs(_arr(hu["c10"]["dh"]))))
        rows.append(("median cr (24 hues)", ref["median_cr10"], med, "ratio"))
        rows.append(("max |dh| (24 hues)", ref["max_dh10"], mx10, "deg"))
        block("HUE  C=0.10  L=0.65", rows)

        # --- hue @ C=0.18 (R02's legacy column) -----------------------------
        rows = []
        for h, (dh, cr, dl) in sorted(ref["hue_c18"].items()):
            i = hidx[h]
            if dh is not None:
                rows.append((f"h={h:3d} dh", dh, hu["c18"]["dh"][i], "deg"))
            if cr is not None:
                rows.append((f"h={h:3d} cr", cr, hu["c18"]["cr"][i], "ratio"))
            if dl is not None:
                rows.append((f"h={h:3d} dL", dl, hu["c18"]["dl"][i], "dl"))
        mx18 = float(np.nanmax(np.abs(_arr(hu["c18"]["dh"]))))
        rows.append(("max |dh| (24 hues)", ref["max_dh18"], mx18, "deg"))
        block("HUE  C=0.18 fixed (R02 legacy column; clipped input re-measured)", rows,
              extra="[diagnostic block hue.c18 -- NOT the pinned hue.chi]")

        # --- pinned chi column: what changes --------------------------------
        out.append("-- HUE  chi = min(0.18, 0.80*cmax)  [the PINNED high-C column]")
        out.append(f"  {'h':>4s} {'chi':>6s} {'c18 dh':>8s} {'chi dh':>8s} {'c18 cr':>8s} {'chi cr':>8s}")
        changed = 0
        for i, h in enumerate(hu["h"]):
            chi_c = hu["chi"]["c"][i]
            if abs(chi_c - C18_LEGACY) < 1e-12:
                continue
            changed += 1
            out.append(
                f"  {h:4.0f} {chi_c:6.3f} {hu['c18']['dh'][i]:8.2f} {hu['chi']['dh'][i]:8.2f} "
                f"{hu['c18']['cr'][i]:8.3f} {hu['chi']['cr'][i]:8.3f}"
            )
        out.append(f"  -> {changed}/24 hues have 0.80*cmax < 0.18, so the pinned column differs from R02 there")

        # --- chroma vs C -----------------------------------------------------
        rows = []
        cidx = {int(h): j for j, h in enumerate(ch["h"])}
        for h, vals in sorted(ref["chroma_vs_c"].items()):
            j = cidx[h]
            for k, tv in enumerate(vals):
                rows.append((f"h={h:3d} C={ch['C'][k]:.2f} cr", tv, ch["cr_clip"][j][k], "ratio"))
        nulls = [
            (ch["h"][j], ch["C"][k])
            for j in range(len(ch["h"])) for k in range(len(ch["C"]))
            if ch["cr"][j][k] is None
        ]
        block("CHROMA vs C  (cr_clip, R02 convention)", rows,
              extra=f"[{len(nulls)}/{len(ch['h'])*len(ch['C'])} cells are null under the strict in-gamut rule]")

        # --- arch ------------------------------------------------------------
        rows = []
        aidx = {int(h): j for j, h in enumerate(ar["h"])}
        for h, vals in sorted(ref["arch"].items()):
            j = aidx[h]
            for k, tv in enumerate(vals):
                rows.append((f"h={h:3d} L={ar['L'][k]:.2f} cr", tv, ar["cr"][j][k], "ratio"))
        n_oog = sum(1 for row in ar["in_gamut"] for g in row if not g)
        block("ARCH  C=0.10", rows, extra=f"[{n_oog}/15 probe points out of gamut -> clipped]")

        # --- skin -------------------------------------------------------------
        rows = []
        for i, (rgb, dh, cr, dl, de) in enumerate(ref["skin"]):
            p = sk["patches"][i]
            tag = f"{p[0]},{p[1]},{p[2]}"
            for c in range(3):
                rows.append((f"{tag} out[{'RGB'[c]}]", float(rgb[c]), sk["out"][i][c], "code"))
            rows.append((f"{tag} dh", dh, sk["dh"][i], "deg"))
            rows.append((f"{tag} cr", cr, sk["cr"][i], "ratio"))
            if dl is not None:
                rows.append((f"{tag} dL", dl, sk["dl"][i], "dl"))
            rows.append((f"{tag} dE00", de, sk["de00"][i], "code"))
        block("SKIN", rows)

        # --- global ------------------------------------------------------------
        rows = []
        lat33 = gl[f"lattice{LATTICE_N_R02}"]
        rows.append(("dE00 mean (33^3)", ref["de00"][0], lat33["de00_mean"], "ratio"))
        rows.append(("dE00 p95 (33^3)", ref["de00"][1], lat33["de00_p95"], "code"))
        rows.append(("clip ==0  %", ref["clip"][0], gl["clip"]["zero"] * 100.0, "pct"))
        rows.append(("clip ==1  %", ref["clip"][1], gl["clip"]["one"] * 100.0, "pct"))
        pa = gl["d2"]["per_axis"]
        meas_axes = sorted(pa[f"axis{a}"]["p99_9"] for a in range(3))
        for k, tv in enumerate(sorted(ref["d2_p999"])):
            rows.append((f"d2 p99.9 axis[{k}] (sorted)", tv, meas_axes[k], "code"))
        rows.append(("d2 max (code)", ref["d2_max"], gl["d2"]["max"], "code"))
        lat17 = gl[f"lattice{LATTICE_N}"]
        block("GLOBAL", rows,
              extra=f"[pinned 17^3: mean {lat17['de00_mean']:.3f} p95 {lat17['de00_p95']:.2f}]")

    return "\n".join(out), summary


def _grey_at(ne: dict, t: float) -> float:
    """Output G code at grey input fraction *t*.

    Reads ``neutral.out_t`` (evaluated exactly at 5/18/35/50/65/80/95 %) when
    *t* is one of those, otherwise interpolates the 9 pinned ramp anchors.
    """
    for k, tv in enumerate(ne["t"]):
        if abs(tv - t) < 1e-12:
            return float(ne["out_t"][k])
    x = np.asarray(ne["codes"], dtype=np.float64) / 255.0
    y = np.asarray(ne["ramp8"], dtype=np.float64)
    return float(np.interp(t, x, y))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _iter_cubes(paths: Iterable[str], patterns: Sequence[str] = ("*.cube",)) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for pat in patterns:
                out.extend(sorted(path.rglob(pat)))
        else:
            out.append(path)
    return out


def _dump_name(path: Path, root: Path | None) -> str:
    """`refs/reverse/Contax...cube` -> `reverse__Contax...` (names collide otherwise)."""
    stem = path.stem
    parent = path.parent.name
    if root is not None:
        try:
            rel = path.relative_to(root)
            parts = list(rel.parts[:-1])
            if parts:
                return "__".join(parts + [stem])
            return stem
        except ValueError:
            pass
    return f"{parent}__{stem}" if parent else stem


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fingerprint.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="print a compact fingerprint table")
    p_show.add_argument("target")
    p_show.add_argument("--photo", action="store_true", help="include $W/cal/photo_sample.npy")

    p_dump = sub.add_parser("dump", help="write fingerprint JSON for each input")
    p_dump.add_argument("inputs", nargs="+")
    p_dump.add_argument("--out", required=True)
    p_dump.add_argument("--root", default=None, help="prefix stripped when naming outputs")
    p_dump.add_argument("--skip", action="append", default=[], help="basename to skip")
    p_dump.add_argument("--photo", action="store_true")

    p_dist = sub.add_parser("dist", help="distance between two fingerprints/cubes")
    p_dist.add_argument("a")
    p_dist.add_argument("b")

    p_acc = sub.add_parser("accept", help="R02 acceptance side-by-side (P0 gate)")

    p_pair = sub.add_parser("pairs", help="nearest pairs among fingerprints")
    p_pair.add_argument("inputs", nargs="+")
    p_pair.add_argument("-n", type=int, default=12)

    p_cmp = sub.add_parser("compare", help="residuals of a fingerprint against a target profile")
    p_cmp.add_argument("target_profile")
    p_cmp.add_argument("measured")

    args = ap.parse_args(argv)

    if args.cmd == "show":
        fp = fingerprint_path(args.target, photo_sample=None if args.photo else False)
        print(format_show(fp))
        return 0

    if args.cmd == "dump":
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        root = Path(args.root) if args.root else None
        skip = {s.lower() for s in args.skip}
        n = 0
        for path in _iter_cubes(args.inputs):
            if path.name.lower() in skip:
                print(f"skip {path}")
                continue
            fp = fingerprint_path(path, photo_sample=None if args.photo else False)
            name = _dump_name(path, root)
            fp["name"] = name
            (out_dir / f"{name}.json").write_text(json.dumps(fp, indent=1))
            n += 1
            print(f"{name:<44s} {path}")
        print(f"wrote {n} fingerprints to {out_dir}")
        return 0

    if args.cmd == "dist":
        a = fingerprint_path(args.a, photo_sample=False)
        b = fingerprint_path(args.b, photo_sample=False)
        print(f"{distance(a, b):.4f}   {a['name']} <-> {b['name']}")
        return 0

    if args.cmd == "accept":
        text, summary = acceptance_report()
        print(text)
        print()
        print("SUMMARY  max deviation per block (tolerances: dh +-0.6 deg, ratio +-0.02, dL +-0.006, code +-1)")
        for name, blocks in summary.items():
            print(f"  {name}")
            for tag, s in blocks.items():
                print(
                    f"    {tag:<58s} n={s['n']:<4d} fail={s['n_fail']:<3d} "
                    f"known={s['n_known']:<3d} max_dev={s['max_dev']:.4f}"
                )
        total_fail = sum(s["n_fail"] for b in summary.values() for s in b.values())
        total_known = sum(s["n_known"] for b in summary.values() for s in b.values())
        print(f"  TOTAL out-of-tolerance items: {total_fail}  (plus {total_known} known R02 defects)")
        return 0 if total_fail == 0 else 1

    if args.cmd == "pairs":
        paths = _iter_cubes(args.inputs, ("*.cube", "*.json"))
        fps = []
        for p in paths:
            fp = fingerprint_path(p, photo_sample=False)
            fps.append((fp.get("name") or p.stem, fp))
        rows = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                rows.append((distance(fps[i][1], fps[j][1]), fps[i][0], fps[j][0]))
        rows.sort()
        print(f"{len(fps)} fingerprints, {len(rows)} pairs; {args.n} nearest:")
        for k, (d, a, b) in enumerate(rows[: args.n], 1):
            print(f"{k:3d}. {d:8.4f}  {a}  <->  {b}")
        if rows:
            dd = np.asarray([r[0] for r in rows])
            print(
                f"     distance distribution: min {dd.min():.3f}  p05 {np.percentile(dd, 5):.3f}"
                f"  median {np.median(dd):.3f}  max {dd.max():.3f}"
            )
        return 0

    if args.cmd == "compare":
        fp = fingerprint_path(args.measured, photo_sample=False)
        rows = compare(fp, args.target_profile)
        print(format_compare(rows))
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
