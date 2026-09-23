"""Shared numerics for Latent-2026 — colour difference, gamut, LUT health metrics.

numpy only (no scipy, no colour-science, no matplotlib).  Everything is
vectorised over arrays of shape ``(..., 3)`` and float64 throughout.

Conventions used everywhere in this project:

* **code values** are display-referred sRGB in ``[0, 1]``; "8-bit codes" means
  the same number multiplied by 255.  Metrics that the specs quote "in codes"
  (second differences, tints) are returned *already multiplied by 255*.
* a **table** is a ``(N, N, N, 3)`` array in CUBE axis order ``(B, G, R, 3)``
  (see :mod:`engine.cubeio`); ``table[b, g, r]`` is the output for the input
  code triple ``(r, g, b) / (N - 1)``.
* a **sampler** is any callable ``rgb -> rgb`` mapping code values to code
  values; :func:`sampler_from_table` / :func:`sampler_from_cube` build the
  table-backed ones and expose the table as a ``.table`` attribute so callers
  can tell table-backed samplers from continuous ones.

Public API (pinned by docs/TOOLS_SPEC.md §T1 — do not rename):
    de00, srgb_to_lab, oklch_from_code, code_from_oklch, cmax,
    second_diff_stats, fold_stats, clip_stats, neutral_stats,
    sampler_from_cube, sampler_from_table, blend_table
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from engine import color
from engine.cubeio import LUT3D, read_lut, tetrahedral_interpolation

__all__ = [
    "de00",
    "srgb_to_lab",
    "oklch_from_code",
    "code_from_oklch",
    "cmax",
    "second_diff_stats",
    "fold_stats",
    "clip_stats",
    "neutral_stats",
    "sampler_from_cube",
    "sampler_from_table",
    "blend_table",
    "identity_table",
    "interior_mask",
    "TableSampler",
    "FOLD_EPS",
    "MATERIAL_EPS",
    "CRUSH_EPS",
]

FloatArray = np.ndarray

# Percentiles reported by second_diff_stats, and the dict keys they land on.
_PCTL = ((50.0, "p50"), (95.0, "p95"), (99.0, "p99"), (99.9, "p99_9"))

#: Noise floor on the sign-normalised tetrahedron volume ratio (ENGINE_SPEC
#: v1.1 R5).  A ratio inside ``[-FOLD_EPS, +FOLD_EPS]`` is **collapsed**, not
#: reversed: a mono (rank-1) table has every volume exactly 0 by construction
#: and float noise then scatters half of them below zero — the fingerprint
#: track measured 45.9 % "strict folds" with |ratio| <= 8.2e-8 on a B&W
#: reference, and 1e-9 of jitter on a mono lattice makes 98,552 tetrahedra
#: "strictly negative" with |min ratio| 3.4e-14.  Float noise is not a fold.
FOLD_EPS = 1e-6

#: ENGINE_SPEC v1.2 W6 — folds are judged by MAGNITUDE, not by sign alone.
#:
#: A tetrahedron whose volume ratio is -3e-5 is a crushed sliver on the gamut
#: shell, not a visible reversal; the v1.1 gate (any ratio < -1e-6 = FAIL)
#: cannot tell a sliver from a real fold, and every commercial reference LUT
#: fails it by thousands.  W6 therefore splits the negatives at -0.02:
#:
#: * ``material`` — ``ratio < -MATERIAL_EPS``: a real reversal.  Must be 0 at
#:   100 % and at 70 %.
#: * ``micro``    — ``-MATERIAL_EPS <= ratio < -FOLD_EPS``: a sliver.  WARN
#:   above 0.5 % of tetrahedra, FAIL above 2 %.
#: * ``crush``    — ``|ratio| < CRUSH_EPS``: the cell has (nearly) no volume
#:   left, reversed or not — how much of the map has been flattened.  WARN
#:   above 3 %, FAIL above 8 %, measured as the EXCESS over the identity
#:   lattice (which is 1.0 everywhere, so its own crush fraction is 0).
MATERIAL_EPS = 0.02
CRUSH_EPS = 0.02


# ---------------------------------------------------------------------------
# CIELAB (D65) and CIEDE2000
# ---------------------------------------------------------------------------

# IEC 61966-2-1 linear sRGB -> CIE XYZ (D65, 2 deg).
_RGB_TO_XYZ = np.array(
    [
        [0.4123907992659595, 0.3575843393838780, 0.1804807884018343],
        [0.2126390058715104, 0.7151686787677559, 0.0721923153607337],
        [0.0193308187155918, 0.1191947797946260, 0.9505321522496606],
    ]
)
# White point = the matrix row sums, so that code (1,1,1) -> L*=100, a=b=0
# exactly (no residue from a rounded 0.95047 / 1.08883 pair).
_XYZ_WHITE = _RGB_TO_XYZ.sum(axis=1)

_LAB_EPS = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0


def srgb_to_lab(rgb_code: FloatArray) -> FloatArray:
    """sRGB **code values** (0..1, D65) -> CIELAB ``(..., 3)`` = (L*, a*, b*).

    Input is not clipped; values outside [0, 1] are carried through the sRGB
    transfer function as-is (``srgb_decode`` is defined for them).
    """
    rgb = np.asarray(rgb_code, dtype=np.float64)
    if rgb.shape[-1] != 3:
        raise ValueError("srgb_to_lab expects an array ending in 3 channels")
    xyz = color.srgb_decode(rgb) @ _RGB_TO_XYZ.T
    r = xyz / _XYZ_WHITE
    f = np.where(r > _LAB_EPS, np.cbrt(np.maximum(r, 0.0)), (_LAB_KAPPA * r + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def de00(
    lab1: FloatArray,
    lab2: FloatArray,
    *,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> FloatArray:
    """CIEDE2000 colour difference between two CIELAB arrays ``(..., 3)``.

    Straight transcription of Sharma, Wu & Dalal (2005), including the two
    quadrant rules that the original CIE text left implicit:

    * ``h'`` is set to 0 when ``C'`` is 0 (the hue of a neutral is undefined),
    * the mean hue ``h_bar`` uses the sum rule only when neither chroma is 0.

    Returns an array of shape ``lab1.shape[:-1]`` (broadcast against lab2).
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    # h' in [0, 360); undefined (-> 0) when the chroma is exactly zero.
    zero1 = (a1p == 0.0) & (b1 == 0.0)
    zero2 = (a2p == 0.0) & (b2 == 0.0)
    h1p = np.where(zero1, 0.0, np.degrees(np.arctan2(b1, a1p)) % 360.0)
    h2p = np.where(zero2, 0.0, np.degrees(np.arctan2(b2, a2p)) % 360.0)

    dLp = L2 - L1
    dCp = C2p - C1p

    Cprod_zero = (C1p * C2p) == 0.0
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where(Cprod_zero, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hp_bar = np.where(
        Cprod_zero,
        hsum,
        np.where(
            hdiff <= 180.0,
            0.5 * hsum,
            np.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0)),
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar ** 7
    R_C = 2.0 * np.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7))
    Lm50 = (Lp_bar - 50.0) ** 2
    S_L = 1.0 + (0.015 * Lm50) / np.sqrt(20.0 + Lm50)
    S_C = 1.0 + 0.045 * Cp_bar
    S_H = 1.0 + 0.015 * Cp_bar * T
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    tL = dLp / (kL * S_L)
    tC = dCp / (kC * S_C)
    tH = dHp / (kH * S_H)
    return np.sqrt(tL * tL + tC * tC + tH * tH + R_T * tC * tH)


# ---------------------------------------------------------------------------
# OKLCh <-> code, gamut boundary
# ---------------------------------------------------------------------------


def oklch_from_code(rgb: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """sRGB code values ``(..., 3)`` -> ``(L, C, h_deg)`` in OKLCh, h in [0, 360)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return color.oklab_to_lch(color.linear_srgb_to_oklab(color.srgb_decode(rgb)))


def code_from_oklch(
    L: FloatArray, C: FloatArray, h: FloatArray, *, eps: float = 5e-4
) -> tuple[FloatArray, FloatArray]:
    """OKLCh -> (sRGB code ``(..., 3)``, in-gamut mask ``(...)``).

    **No clipping.**  The linear sRGB values are encoded as they come out of the
    OKLab inverse; negative linear values are mirrored through the transfer
    function (odd extension, ``-encode(-x)``) so the map stays continuous and
    invertible, and values above 1 simply encode above 1.  Use the returned
    mask — ``True`` where every linear channel lies in ``[-eps, 1 + eps]`` with
    the spec's ``eps = 5e-4`` on *linear* values — to reject out-of-gamut probe
    points rather than trusting the codes.
    """
    lin = _linear_from_oklch(L, C, h)
    in_gamut = np.all((lin >= -eps) & (lin <= 1.0 + eps), axis=-1)
    code = np.sign(lin) * color.srgb_encode(np.abs(lin))
    return code, in_gamut


def _linear_from_oklch(L: FloatArray, C: FloatArray, h: FloatArray) -> FloatArray:
    lab = color.lch_to_oklab(
        np.asarray(L, dtype=np.float64),
        np.asarray(C, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
    )
    return color.oklab_to_linear_srgb(lab)


def cmax(
    L: FloatArray, h: FloatArray, *, iters: int = 40, eps: float = 0.0
) -> FloatArray:
    """Largest in-gamut OKLab chroma at (L, h), by vectorised bisection.

    QC / probe use only — it is a boundary *query*, not a gamut mapper (that is
    :func:`engine.color.project_into_gamut`).  Returns the last chroma known to
    be inside, so ``code_from_oklch(L, cmax(L, h), h)`` is always in gamut.
    With ``iters = 40`` the bracket is 0.5 * 2**-40 wide (< 1e-12).
    """
    L = np.clip(np.asarray(L, dtype=np.float64), 0.0, 1.0)
    h = np.asarray(h, dtype=np.float64)
    L, h = np.broadcast_arrays(L, h)
    lo = np.zeros(L.shape, dtype=np.float64)
    hi = np.full(L.shape, 0.5)  # sRGB max OKLab chroma is ~0.32
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        lin = _linear_from_oklch(L, mid, h)
        ok = np.all((lin >= -eps) & (lin <= 1.0 + eps), axis=-1)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------


def identity_table(size: int = 33) -> FloatArray:
    """The identity LUT table, ``(size, size, size, 3)`` in (B, G, R, 3) order."""
    t = np.linspace(0.0, 1.0, int(size))
    b, g, r = np.meshgrid(t, t, t, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def _as_table(table) -> FloatArray:
    table = np.asarray(getattr(table, "table", table), dtype=np.float64)
    if table.ndim != 4 or table.shape[-1] != 3:
        raise ValueError("table must have shape (N, N, N, 3)")
    if not (table.shape[0] == table.shape[1] == table.shape[2]):
        raise ValueError("table axes must have equal sizes")
    return table


_INTERIOR_MASK_CACHE: dict[tuple[int, float], FloatArray] = {}


def interior_mask(size: int = 33, rel_chroma: float = 0.85) -> FloatArray:
    """Lattice points whose INPUT colour sits in the photographic range.

    Ported from ``lumix-original-looks/qc_looks.py::_interior_mask``: a point is
    interior when its relative saturation ``C / C_max <= rel_chroma`` (0.85).
    The complement — the shell hugging the sRGB gamut faces — intrinsically
    carries curvature for any grade that moves on-face colours, because the
    boundary itself is creased in OKLab.

    Returned shape is ``(size, size, size)`` in the table's (B, G, R) index
    order.  Cached per (size, rel_chroma); the legacy version cached on the
    first call only and silently reused a 33-point mask for other sizes.
    """
    key = (int(size), float(rel_chroma))
    cached = _INTERIOR_MASK_CACHE.get(key)
    if cached is None:
        grid = identity_table(size).reshape(-1, 3)
        L, C, h = oklch_from_code(grid)
        # 16 iterations matches the legacy mask (0.5 * 2**-16 = 7.6e-6 of
        # chroma — far below the 0.85 threshold's sensitivity).
        c_max = cmax(L, h, iters=16)
        cached = (C / np.maximum(c_max, 1e-6) <= rel_chroma).reshape(size, size, size)
        _INTERIOR_MASK_CACHE[key] = cached
    return cached


# Backwards-compatible alias for the ported qc_looks name.
_interior_mask = interior_mask


# ---------------------------------------------------------------------------
# Smoothness: second differences along the lattice axes
# ---------------------------------------------------------------------------


def _pctl_block(values: FloatArray) -> dict[str, float]:
    if values.size == 0:
        return {name: float("nan") for _, name in _PCTL} | {"max": float("nan"), "n": 0}
    qs = np.percentile(values, [p for p, _ in _PCTL])
    out = {name: float(q) for (_, name), q in zip(_PCTL, qs)}
    out["max"] = float(values.max())
    out["n"] = int(values.size)
    return out


def second_diff_stats(table, *, rel_chroma: float = 0.85) -> dict:
    """|second difference| statistics along the three lattice axes, in 8-bit codes.

    ``d2 = |table[i-1] - 2*table[i] + table[i+1]|`` along each axis, for every
    channel; the values from all three axes and all three channels are pooled
    and reported as p50 / p95 / p99 / p99.9 / max (multiplied by 255, i.e.
    8-bit code values).  Two populations:

    * ``full`` — the whole lattice,
    * ``interior`` — only triples whose three input points are all inside
      :func:`interior_mask` (``C/C_max <= 0.85``), the population the QC gate
      in docs/research/06 budgets.

    ``per_axis`` carries the same blocks per axis (0 = B, 1 = G, 2 = R) for the
    full lattice, because R02 §A7 quotes the three axes separately.
    ``max`` / ``interior_max`` are convenience aliases (also in codes).
    """
    table = _as_table(table)
    size = table.shape[0]
    mask = interior_mask(size, rel_chroma)

    full_parts: list[FloatArray] = []
    inter_parts: list[FloatArray] = []
    per_axis: dict[str, dict] = {}
    for axis in range(3):
        d2 = np.abs(np.diff(table, n=2, axis=axis)) * 255.0
        full_parts.append(d2.ravel())
        sl_mid, sl_lo, sl_hi = ([slice(None)] * 3 for _ in range(3))
        sl_mid[axis] = slice(1, -1)
        sl_lo[axis] = slice(0, -2)
        sl_hi[axis] = slice(2, None)
        cell = mask[tuple(sl_mid)] & mask[tuple(sl_lo)] & mask[tuple(sl_hi)]
        inter_parts.append(d2[cell].ravel())
        per_axis[f"axis{axis}"] = _pctl_block(d2.ravel())

    full = _pctl_block(np.concatenate(full_parts))
    interior = _pctl_block(np.concatenate(inter_parts))
    return {
        "full": full,
        "interior": interior,
        "per_axis": per_axis,
        "max": full["max"],
        "interior_max": interior["max"],
    }


# ---------------------------------------------------------------------------
# Folding: signed tetrahedron volumes
# ---------------------------------------------------------------------------

# The six tetrahedra of engine.cubeio.tetrahedral_interpolation, written as the
# three corners that follow c000 along each interpolation path.  Offsets are
# (r, g, b) bits, matching cubeio's c<r><g><b> naming.
_TETRA: tuple[tuple[tuple[int, int, int], ...], ...] = (
    ((1, 0, 0), (1, 1, 0), (1, 1, 1)),  # m0: r >= g >= b  (c000 c100 c110 c111)
    ((1, 0, 0), (1, 0, 1), (1, 1, 1)),  # m1: r >= b >  g  (c000 c100 c101 c111)
    ((0, 0, 1), (1, 0, 1), (1, 1, 1)),  # m2: b >  r >= g  (c000 c001 c101 c111)
    ((0, 1, 0), (1, 1, 0), (1, 1, 1)),  # m3: g >  r >= b  (c000 c010 c110 c111)
    ((0, 1, 0), (0, 1, 1), (1, 1, 1)),  # m4: g >= b >  r  (c000 c010 c011 c111)
    ((0, 0, 1), (0, 1, 1), (1, 1, 1)),  # m5: b >  g >  r  (c000 c001 c011 c111)
)


def _corner(table: FloatArray, offset: tuple[int, int, int]) -> FloatArray:
    """The (r, g, b)-offset corner of every cell: shape (N-1, N-1, N-1, 3)."""
    i, j, k = offset  # r, g, b
    n = table.shape[0] - 1
    return table[k : k + n, j : j + n, i : i + n]


_IDENT_TP_CACHE: dict[int, FloatArray] = {}


def _identity_triple_products(size: int) -> FloatArray:
    """Cached signed volumes of the identity lattice (exactly +-h**3)."""
    tp = _IDENT_TP_CACHE.get(size)
    if tp is None:
        tp = _triple_products(identity_table(size))
        _IDENT_TP_CACHE[size] = tp
    return tp


def _triple_products(table: FloatArray) -> FloatArray:
    """Signed volumes (x6) of the six tetrahedra of every cell: (6, ncell)."""
    base = _corner(table, (0, 0, 0))
    vols = np.empty((6, base.shape[0] * base.shape[1] * base.shape[2]))
    for t, corners in enumerate(_TETRA):
        e1 = (_corner(table, corners[0]) - base).reshape(-1, 3)
        e2 = (_corner(table, corners[1]) - base).reshape(-1, 3)
        e3 = (_corner(table, corners[2]) - base).reshape(-1, 3)
        # Explicit determinant expansion: for the identity lattice every factor
        # is an exact binary fraction, so the result is exactly +-h**3.
        vols[t] = (
            e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
            - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
            + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0])
        )
    return vols


def fold_stats(table, *, eps: float = FOLD_EPS,
               material_eps: float = MATERIAL_EPS,
               crush_eps: float = CRUSH_EPS) -> dict:
    """Injectivity check on the lattice: sign-normalised tetrahedron volumes.

    For every cell the SAME six tetrahedra that
    :func:`engine.cubeio.tetrahedral_interpolation` uses are measured, and each
    signed volume is divided by the identity lattice's signed volume for the
    same tetrahedron::

        ratio = vol(table) * sign(vol(identity)) / |vol(identity)|

    A 33-point table gives 32**3 * 6 = 196,608 ratios.  The identity table gives
    exactly 1.0 everywhere.  ``ratio <= 0`` means the map folds there (it is not
    locally injective) and the camera will render a crease.

    *The trap* (docs/research/06 §1.3g): three of the six tetrahedra are
    negatively oriented by construction, so raw signed volumes report ~50 %
    "negatives" on the identity.  The sign normalisation above is what makes the
    test meaningful.

    **The noise floor** (ENGINE_SPEC v1.1 R5).  ``eps`` (default
    :data:`FOLD_EPS` = 1e-6) separates the two populations the gates read:

    * ``neg_strict_count`` / ``neg_strict_frac`` — ``ratio < -eps``: a genuine
      orientation reversal, the map really folds there.  This is the number
      ENGINE_SPEC v1.1 R5 requires to be 0 at 100 % and at 70 %.
    * ``collapsed_count`` / ``collapsed_frac`` — ``|ratio| <= eps``:
      degenerate, not reversed — a clipping plateau, or a rank-deficient map
      (every mono table is collapsed everywhere by construction).  R5 warns
      above 0.05 %.

    Reported alongside, unchanged, for continuity with the milestone-1
    reports: ``neg_count`` (``ratio <= 0``, PLAN's literal wording, which a
    collapsed tetrahedron also satisfies), ``zero_count`` (``ratio == 0``
    exactly) and ``neg_strict_raw_count`` (``ratio < 0`` with no noise floor —
    the number that made ``neg_strict_count`` meaningless on mono tables).

    **The magnitude split** (ENGINE_SPEC v1.2 W6).  ``neg_strict_count`` counts
    every reversal past the noise floor, however thin; W6 splits it because a
    ratio of -3e-5 is a crushed sliver on the gamut shell and a ratio of -0.5 is
    a visible crease:

    * ``material_count`` / ``material_frac`` — ``ratio < -0.02``.  **This is the
      gate**: 0 at 100 % and at 70 %, or FAIL.
    * ``micro_count`` / ``micro_frac`` — ``-0.02 <= ratio < -1e-6``.  WARN above
      0.5 % of tetrahedra, FAIL above 2 %.
    * ``crush_count`` / ``crush_frac`` — ``|ratio| < 0.02``, reversed or not:
      how much of the lattice has had its volume flattened.  WARN above 3 %,
      FAIL above 8 %.  The identity lattice is 1.0 everywhere, so its own crush
      fraction is 0 and ``crush_frac`` *is* W6's "excess over the identity
      lattice"; ``crush_excess_frac`` is carried explicitly for that wording.

    Plus ``n_tetra``, ``p001``, ``mean_ratio`` and the ``eps`` used.
    """
    table = _as_table(table)
    if table.shape[0] < 2:
        raise ValueError("fold_stats needs a lattice of at least 2 points per axis")
    eps = abs(float(eps))
    material_eps = abs(float(material_eps))
    crush_eps = abs(float(crush_eps))
    vols = _triple_products(table)
    ident = _identity_triple_products(table.shape[0])
    ratio = vols * np.sign(ident) / np.abs(ident)
    n = ratio.size
    neg = int(np.count_nonzero(ratio <= 0.0))
    raw_strict = int(np.count_nonzero(ratio < 0.0))
    strict = int(np.count_nonzero(ratio < -eps))
    collapsed = int(np.count_nonzero(np.abs(ratio) <= eps))
    material = int(np.count_nonzero(ratio < -material_eps))
    micro = int(np.count_nonzero((ratio >= -material_eps) & (ratio < -eps)))
    crush = int(np.count_nonzero(np.abs(ratio) < crush_eps))
    # the identity lattice's own crush population (exactly 0 — every ratio is
    # 1.0), kept explicit so W6's "excess over the identity lattice" is visible
    # rather than assumed.
    ident_ratio = ident * np.sign(ident) / np.abs(ident)
    ident_crush = int(np.count_nonzero(np.abs(ident_ratio) < crush_eps))
    return {
        # + 0.0 turns a signed -0.0 (a collapsed tetrahedron) into plain 0.0.
        "min_ratio": float(ratio.min() + 0.0),
        "neg_count": neg,
        "neg_frac": float(neg / n),
        "neg_strict_count": strict,
        "neg_strict_frac": float(strict / n),
        "neg_strict_raw_count": raw_strict,
        "collapsed_count": collapsed,
        "collapsed_frac": float(collapsed / n),
        "material_count": material,
        "material_frac": float(material / n),
        "micro_count": micro,
        "micro_frac": float(micro / n),
        "crush_count": crush,
        "crush_frac": float(crush / n),
        "crush_excess_frac": float((crush - ident_crush) / n),
        "material_eps": material_eps,
        "crush_eps": crush_eps,
        "zero_count": neg - raw_strict,
        "n_tetra": int(n),
        "eps": eps,
        "p001": float(np.percentile(ratio, 0.1)),
        "mean_ratio": float(ratio.mean()),
    }


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------


def clip_stats(table) -> dict:
    """Fraction of the table sitting exactly at 0.0 or exactly at 1.0.

    ``zero`` / ``one`` count single channel values (the definition R02 quotes:
    "1.03 % of table == 0").  ``zero_per_channel`` / ``one_per_channel`` split
    that by R, G, B, and ``zero_any`` / ``one_any`` give the fraction of lattice
    *nodes* with at least one clipped channel.
    """
    table = _as_table(table)
    z = table == 0.0
    o = table == 1.0
    return {
        "zero": float(z.mean()),
        "one": float(o.mean()),
        "zero_per_channel": [float(x) for x in z.reshape(-1, 3).mean(axis=0)],
        "one_per_channel": [float(x) for x in o.reshape(-1, 3).mean(axis=0)],
        "zero_any": float(z.any(axis=-1).mean()),
        "one_any": float(o.any(axis=-1).mean()),
    }


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


class TableSampler:
    """A callable ``rgb -> rgb`` backed by a 3D LUT (tetrahedral interpolation).

    The table is exposed as ``.table`` (and the LUT3D as ``.lut``) so callers —
    the fingerprint probe in particular — can detect table-backed samplers and
    run the lattice metrics (fold / second-difference / clip) on them.
    """

    __slots__ = ("lut", "table", "title", "source")

    def __init__(self, lut: LUT3D):
        self.lut = lut
        self.table = lut.table
        self.title = lut.title
        self.source = lut.source

    def __call__(self, rgb: FloatArray) -> FloatArray:
        return tetrahedral_interpolation(self.lut, rgb)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TableSampler(title={self.title!r}, size={self.lut.size})"


def sampler_from_table(table, *, title: str = "table") -> TableSampler:
    """Wrap a ``(N, N, N, 3)`` table (or a LUT3D) in a sampler."""
    if isinstance(table, LUT3D):
        return TableSampler(table)
    return TableSampler(LUT3D(table=_as_table(table), title=title))


def sampler_from_cube(path) -> TableSampler:
    """Read a ``.cube`` file and return a sampler over it."""
    return TableSampler(read_lut(Path(path)))


def blend_table(table, s: float) -> FloatArray:
    """``s * table + (1 - s) * identity`` — the in-camera strength model.

    ``s = 1`` returns the table unchanged, ``s = 0`` the identity.  Blending is
    provably non-worsening for range and smoothness (a convex combination), so
    only the fold check is genuinely informative on a blended table.
    """
    table = _as_table(table)
    s = float(s)
    return s * table + (1.0 - s) * identity_table(table.shape[0])


# ---------------------------------------------------------------------------
# Neutral axis
# ---------------------------------------------------------------------------

NEUTRAL_T = (0.05, 0.18, 0.35, 0.50, 0.65, 0.80, 0.95)
_RAMP_N = 4097


def neutral_stats(sampler: Callable[[FloatArray], FloatArray], *, n: int = _RAMP_N) -> dict:
    """Behaviour of a sampler on the grey axis, measured on a 4097-point ramp.

    Reports, for input codes ``t`` from 0 to 1:

    * ``monotone`` / ``monotone_per_channel`` — output codes never decrease
      (tolerance 1e-9); ``monotone_L`` — the same for OKLab L.
    * ``min_step`` — the smallest per-channel step between consecutive ramp
      samples, in code units (0 means a plateau: the clipping trap).
      ``min_step_L`` is the same for OKLab L.
    * ``black`` / ``white`` — the output at t = 0 and t = 1 (3 codes each),
      plus ``black_mean`` / ``white_err`` (max |white - 1|).
    * ``tint_rg`` / ``tint_bg`` — (R-G) and (B-G) in **8-bit codes** at
      t = 5/18/35/50/65/80/95 %.
    * ``slope`` — local slope d(out G)/d(in) at the same t (np.gradient over the
      ramp in the code domain), and ``slope_rgb`` per channel.
    * ``out`` / ``out_L`` — the output codes (G) and OKLab L at the same t.

    Values at the seven t are read with ``np.interp`` on the dense ramp, so the
    exact t is honoured (t * 4096 is not an integer for 5 %, 35 %, 65 %, 95 %).
    """
    t = np.linspace(0.0, 1.0, int(n))
    grey = np.stack([t, t, t], axis=-1)
    out = np.asarray(sampler(grey), dtype=np.float64)
    if out.shape != grey.shape:
        raise ValueError(f"sampler returned shape {out.shape}, expected {grey.shape}")

    diffs = np.diff(out, axis=0)
    tol = 1e-9
    mono_ch = [bool(np.all(diffs[:, c] >= -tol)) for c in range(3)]
    L = color.linear_srgb_to_oklab(color.srgb_decode(np.clip(out, 0.0, 1.0)))[:, 0]

    grad = np.gradient(out, t[1] - t[0], axis=0)
    ts = np.asarray(NEUTRAL_T, dtype=np.float64)

    def at(col: FloatArray) -> list[float]:
        return [float(v) for v in np.interp(ts, t, col)]

    rg = at((out[:, 0] - out[:, 1]) * 255.0)
    bg = at((out[:, 2] - out[:, 1]) * 255.0)
    return {
        "t": [float(x) for x in ts],
        "monotone": bool(all(mono_ch)),
        "monotone_per_channel": mono_ch,
        "monotone_L": bool(np.all(np.diff(L) >= -tol)),
        "min_step": float(diffs.min()),
        "min_step_per_channel": [float(diffs[:, c].min()) for c in range(3)],
        "min_step_L": float(np.diff(L).min()),
        "black": [float(v) for v in out[0]],
        "white": [float(v) for v in out[-1]],
        "black_mean": float(out[0].mean()),
        "white_err": float(np.max(np.abs(out[-1] - 1.0))),
        "tint_rg": rg,
        "tint_bg": bg,
        "slope": at(grad[:, 1]),
        "slope_rgb": [at(grad[:, c]) for c in range(3)],
        "out": at(out[:, 1]),
        "out_L": at(L),
    }


# ---------------------------------------------------------------------------
# CLI: quick look at a .cube
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    for arg in sys.argv[1:]:
        s = sampler_from_cube(arg)
        sd = second_diff_stats(s.table)
        fd = fold_stats(s.table)
        cl = clip_stats(s.table)
        ns = neutral_stats(s)
        print(f"{Path(arg).name}")
        print(
            f"  d2 full     p50 {sd['full']['p50']:.2f}  p95 {sd['full']['p95']:.2f}  "
            f"p99 {sd['full']['p99']:.2f}  p99.9 {sd['full']['p99_9']:.2f}  "
            f"max {sd['full']['max']:.2f}"
        )
        print(
            f"  d2 interior p50 {sd['interior']['p50']:.2f}  p95 {sd['interior']['p95']:.2f}  "
            f"p99 {sd['interior']['p99']:.2f}  p99.9 {sd['interior']['p99_9']:.2f}  "
            f"max {sd['interior']['max']:.2f}"
        )
        print(
            f"  fold min_ratio {fd['min_ratio']:+.4f}  material<-{fd['material_eps']:.2f} "
            f"{fd['material_count']}/{fd['n_tetra']}  micro {fd['micro_count']} "
            f"({fd['micro_frac']*100:.3f} %)  crush {fd['crush_count']} "
            f"({fd['crush_frac']*100:.3f} %)"
        )
        print(f"  clip  =0 {cl['zero']*100:.2f} %   =1 {cl['one']*100:.2f} %")
        print(
            f"  neutral mono={ns['monotone']} minstep={ns['min_step']:.2e} "
            f"black={ns['black_mean']*255:.2f} white_err={ns['white_err']:.2e}"
        )
