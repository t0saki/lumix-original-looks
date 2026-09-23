"""[G] gamut compression in the CODE domain (ENGINE_SPEC §4, v1.2 W5).

::

    n   = srgb_encode(clip(L_cur, 0, 1)**3)                # anchor (see W5.1 below)
    d   = code - n
    u   = where(d >= 0, d/(1-n+1e-12), -d/max(n, 1e-12))
    nm  = (sum_i max(u_i,0)**p)**(1/p)

    lim = max(1.04 * max(nm over the 33**3 lattice), knee + 0.05)   # measured in compile()
    q   = (lim - knee)/(1 - knee) ;  s1 = gamut.end_slope ;  q2 = (q - s1)/(1 - s1)
    x   = (nm - knee)/(lim - knee)
    R   = s1*x + (1-s1)*(1 - (1-x)**q2)        for 0 <= x <= 1
    R   = 1 + s1*(x - 1)                       for x > 1          (linear tail)
    mp  = knee + (1-knee)*R                    for x > 0 ;  mp = nm otherwise
    out = n + d*where(nm > knee, mp/max(nm,1e-9), 1)

The anchor (W5.1) — measured, and the OKLab one kept
----------------------------------------------------
W5.1 proposed replacing the equal-lightness grey by the CURRENT code's luma,
because a *linear* anchor functional ``w`` (``sum w = 1``) gives ``w.d = 0``
exactly: every point of a ray then has the same ``n``, the rays foliate the
cube, each plane ``n = const`` maps into itself radially, and a monotone
``mp(nm)`` makes [G] injective.  Both anchors were measured (the table is in
``work.nosync/review/m2_engine_report.json``, key ``w5_anchor``) and the
**OKLab anchor is kept**: on W5.1's own population — identity + sat 1.05 to
1.40 — the two are equal at 0 folds of every kind, the luma anchor is worse on
interior d2 and on ``lim``, and on the three real looks W5.3 names it explodes
(``lim`` 171,640 on 04Viride).  :func:`anchor` carries the diagnosis.  The 678
folds W5.1 cites were v1.1's, and v1.2 removes their two causes (W1's
withdrawal of R1, W5.2's slope floor) without touching the anchor.

The slope floor (W5.2)
----------------------
``R(0) = 0``, ``R'(0) = q`` (so ``mp`` is C1 at the knee), ``R(1) = 1`` and
``R'(1) = s1 > 0``.  The radial gain ``d mp/d nm = R'(x)/q`` therefore never
drops below ``s1/q`` — the flat tail of v1.1's ``1-(1-x)**q`` (measured down to
1.6e-12, 14.8 % of a lattice under a gain of 0.05) cannot happen, and the map
stays strictly increasing all the way out.  Beyond ``lim`` the curve continues
LINEARLY with slope ``s1`` instead of saturating at 1, so an off-lattice input
past ``lim`` lands marginally above 1 and meets the final clip; the shipped
artefact is the 33**3 table, where ``lim`` is by construction not exceeded.

``q <= 1`` (nothing left the gamut) makes the stage the identity.
``gamut.soft`` and ``gamut.eps_dark`` are gone from the look format (v1.1 R2);
they are accepted from an old look file, recorded and ignored, and
``spec.validate`` warns.
"""

from __future__ import annotations

import numpy as np

from .color import FloatArray, srgb_encode
from .spec import DEFAULT_P, Gamut

__all__ = ["apply_gamut", "limit_curve", "radial_gain", "measured_limit", "norm",
           "anchor", "effective_end_slope", "DEFAULT_DIAG", "LIM_MARGIN",
           "MIN_RADIAL_GAIN", "ANCHORS", "DEFAULT_ANCHOR", "LUMA"]

#: parameters used only to report ``nm`` for a look that declares ``gamut: null``
DEFAULT_DIAG = Gamut()

#: W5.2: the limit sits 4 % beyond the worst lattice node, and at least 0.05
#: above the knee (so ``lim > knee`` even for a look that never leaves the gamut).
LIM_MARGIN = 1.04
LIM_MIN_SPAN = 0.05

#: Rec.709 luma weights — the anchor functional.  They sum to exactly 1, which
#: is what makes ``w . d = 0`` and the rays a foliation.
LUMA = np.array([0.2126, 0.7152, 0.0722])
ANCHORS = ("luma", "oklab")
#: **The W5.1 study says: keep the OKLab anchor.**  Measured (table in
#: ``work.nosync/review/m2_engine_report.json``, key ``w5_anchor``):
#:
#: * identity + sat 1.00/1.05/1.10/1.20/1.40, the population W5.1 names — BOTH
#:   anchors give 0 material folds, 0 micro folds, 0 crush and 0 strict
#:   negatives.  The 678-fold failure W5.1 cites was v1.1's, and it came from
#:   R1's headroom (withdrawn by W1) and the flat tail (replaced by W5.2), not
#:   from the anchor.  On the numbers that do differ the LUMA anchor is worse:
#:   interior d2 p99.9 0.95/2.12/1.72/2.90/5.66 vs 0.85/1.30/1.42/2.55/5.22,
#:   and ``lim`` 1.96 vs 1.28 at sat 1.05.
#: * on the three real looks W5.3 names it is unusable: ``lim`` 19.59 (03Gilt)
#:   and **171,640** (04Viride) against 1.55 / 1.73 for OKLab.  The cause is
#:   structural, not numerical — see :func:`anchor`.
DEFAULT_ANCHOR = "oklab"

#: W5.1's clip on the LUMA anchor.  It is not applied to the OKLab anchor:
#: there ``n = srgb_encode(clip(L,0,1)**3)`` already lies in [0,1], and clipping
#: it to ``1 - 1e-6`` would make WHITE out of gamut — ``d = 1e-6`` against a
#: denominator of ``1e-6`` gives ``u = 1``, ``nm = 3**(1/p) = 1.147``, and the
#: compressor then moves white by 2.2e-07 (ENGINE_SPEC §8 gates it at 1e-9).
#: The denominators below carry the 1e-12 guards instead, which is what v1.0 and
#: v1.1 used and what keeps ``nm = 0`` at white and at black.
ANCHOR_EPS = 1e-6
DENOM_EPS = 1e-12


def anchor(code: FloatArray, L_cur: FloatArray | None = None,
           kind: str = DEFAULT_ANCHOR) -> FloatArray:
    """The grey the compression pushes toward, shape ``(..., 1)``.

    ``"oklab"`` (the default, v1.0/v1.1 and what the W5.1 study kept):
    ``srgb_encode(clip(L_cur,0,1)**3)`` — the grey of equal OKLab lightness.
    ``"luma"`` (W5.1's proposal): ``clip(w . code, 1e-6, 1-1e-6)``.

    **Why the luma anchor is not the default.**  Its foliation argument is
    sound *while the anchor is inside (0,1)*, and that is exactly what a real
    look breaks.  [P] legitimately produces out-of-gamut codes with a negative
    channel; a saturated blue carries only 0.0722 of the luma, so a post-[P]
    code such as ``(0.188, -0.165, 0.962)`` (04Viride's own blue corner, input
    ``(0,0,1)``) has luma ``-0.0085``.  The ruling's ``clip(..., 1e-6, ...)``
    then divides ``|d| = 0.165`` by ``1e-6``: ``nm = 165,038``, and since
    ``lim = 1.04*max nm`` that one node sets the whole compressor
    (q = 686,556, 20.5 % of the lattice crushed).  118 of the 35,937 nodes of
    that look have luma < 1e-6.  It is not a numerical detail: the plane
    ``n = const`` for ``n <= 0`` does not intersect the sRGB cube at all, so
    there is no radial compression in that plane and ``nm`` is genuinely
    unbounded — the clip only decides how large a finite number it reports.
    Making the luma anchor shippable needs a ruling (a dark/bright floor like
    v1.0's ``eps_dark``, or a constraint that no stage may take the luma
    outside [0,1]).
    """
    code = np.asarray(code, dtype=np.float64)
    if kind == "luma":
        # W5.1's own clip.  It is what makes the anchor unusable on a real look
        # (see above): where the luma leaves [0,1] it turns an unbounded `nm`
        # into a merely enormous one.
        n = np.clip(code @ LUMA, ANCHOR_EPS, 1.0 - ANCHOR_EPS)
    elif kind == "oklab":
        if L_cur is None:
            raise ValueError("the 'oklab' anchor needs the current OKLab L")
        n = srgb_encode(np.clip(np.asarray(L_cur, dtype=np.float64), 0.0, 1.0) ** 3)
    else:
        raise ValueError(f"unknown anchor {kind!r}; expected one of {ANCHORS}")
    return n[..., None]


def measured_limit(knee: float, nm_max: float) -> float:
    """``lim = max(1.04*nm_max, knee + 0.05)`` (W5.2)."""
    return float(max(LIM_MARGIN * float(nm_max), float(knee) + LIM_MIN_SPAN))


def _q(knee: float, lim: float) -> float:
    return (float(lim) - float(knee)) / (1.0 - float(knee))


def effective_end_slope(knee: float, lim: float, s1: float) -> float:
    """``max(s1, 2 - q)`` — the end slope actually used by the curve.

    W5.2 writes ``q2 = (q - s1)/(1 - s1)`` and ``R(x) = s1*x + (1-s1)*(1-(1-x)**q2)``.
    ``R'' = -(1-s1)*q2*(q2-1)*(1-x)**(q2-2)``, so whenever ``q2 < 2`` the second
    derivative DIVERGES as ``x -> 1`` (``nm -> lim``): the curve is C1 there as
    designed but its curvature is unbounded, against ENGINE_SPEC §0's C2 rule.
    ``q2 < 2`` happens exactly when ``lim < knee + (2-s1)*(1-knee)`` — 1.2275 at
    the v1.2 defaults (knee 0.65, s1 0.35), i.e. on weakly-pushing looks.
    Measured at knee 0.65 / lim 1.1931 (identity + [G], and the whole W5.1
    anchor population at sat 1.00-1.10): q = 1.5517, q2 = 1.8488, and
    ``d2 mp/d nm2`` reads -1.21 just above the knee, -1.97 at 0.96 of the way to
    ``lim`` and -8.40 at ``lim`` itself and still growing as the stencil
    shrinks.  No look in ``looks/lead`` is in that regime (min lim 1.5090,
    02Burin -> q2 = 3.24).

    Raising ``s1`` to ``2 - q`` puts ``q2`` at exactly 2, which is the smallest
    change that bounds the curvature while keeping every property W5.2 asks for:
    ``R'(0) = s1 + (1-s1)*q2 = q`` still (C1 at the knee), ``R(1) = 1``, and
    ``R'(1) = s1_eff > s1 > 0`` (a HIGHER floor on the radial gain, never a
    lower one).  It engages only where the defect is, and ``2 - q < 1`` for
    every ``q > 1``, so ``q2`` never divides by zero.
    """
    q = _q(knee, lim)
    if q <= 1.0:
        return float(s1)
    return float(max(float(s1), 2.0 - q))


def limit_curve(nm: FloatArray, knee: float, lim: float,
                s1: float = 0.25) -> FloatArray:
    """``mp(nm)`` — W5.2's limit curve with a slope floor.  Identity when q <= 1.

    C1 at the knee (``R'(0) = q``), ``R(1) = 1``, ``R'(1) = s1``, and linear
    with slope ``s1`` beyond ``lim`` — so ``mp`` is strictly increasing on the
    whole half-line and [G] never erases a radial direction.  ``s1`` is raised
    to :func:`effective_end_slope` when the literal value would leave
    ``q2 < 2``, where the curvature at ``lim`` is unbounded.
    """
    nm = np.asarray(nm, dtype=np.float64)
    knee = float(knee)
    lim = float(lim)
    q = _q(knee, lim)
    if q <= 1.0:
        return nm
    s1 = effective_end_slope(knee, lim, s1)
    x = (nm - knee) / (lim - knee)
    q2 = (q - s1) / (1.0 - s1)
    xc = np.clip(x, 0.0, 1.0)
    R = s1 * xc + (1.0 - s1) * (1.0 - (1.0 - xc) ** q2)
    R = np.where(x > 1.0, 1.0 + s1 * (x - 1.0), R)
    mp = knee + (1.0 - knee) * R
    return np.where(x <= 0.0, nm, mp)


#: radial gain below which the compressor is collapsing rather than compressing.
#: Same calibration as ``spec.CAP_MIN_RADIAL_GAIN``: a chroma map whose radial
#: derivative drops under 0.05 was measured to invert the lattice's tetrahedra.
#: W5.2's slope floor makes the compressor's own minimum ``s1/q``, which is a
#: *design* number now rather than an emergent one.
MIN_RADIAL_GAIN = 0.05


def radial_gain(nm: FloatArray, knee: float, lim: float,
                s1: float = 0.25) -> FloatArray:
    """``d mp/d nm`` — how much of a radial step survives the compressor.

    ``R'(x)/q`` with ``R'(x) = s1 + (1-s1)*q2*(1-x)**(q2-1)`` inside the knee
    and ``s1`` beyond it, so the minimum over the whole half-line is exactly
    ``s1/q`` and it is reached at ``nm = lim``.  ``s1`` is
    :func:`effective_end_slope`, as in :func:`limit_curve`.
    """
    nm = np.asarray(nm, dtype=np.float64)
    knee = float(knee)
    lim = float(lim)
    q = _q(knee, lim)
    if q <= 1.0:
        return np.ones_like(nm)
    s1 = effective_end_slope(knee, lim, s1)
    x = (nm - knee) / (lim - knee)
    q2 = (q - s1) / (1.0 - s1)
    xc = np.clip(x, 0.0, 1.0)
    dR = s1 + (1.0 - s1) * q2 * (1.0 - xc) ** (q2 - 1.0)
    dR = np.where(x > 1.0, s1, dR)
    return np.where(nm > knee, dR / q, 1.0)


def norm(code: FloatArray, L_cur: FloatArray | None = None, p: float = DEFAULT_P,
         kind: str = DEFAULT_ANCHOR) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(nm, n, d)`` of the current colour."""
    code = np.asarray(code, dtype=np.float64)
    n = anchor(code, L_cur, kind)
    d = code - n
    u = np.where(d >= 0.0, d / (1.0 - n + DENOM_EPS),
                 -d / np.maximum(n, DENOM_EPS))
    nm = np.sum(np.maximum(u, 0.0) ** p, axis=-1) ** (1.0 / p)
    return nm, n, d


def apply_gamut(g: Gamut | None, code: FloatArray, L_cur: FloatArray | None = None,
                lim: float | None = None, *, kind: str = DEFAULT_ANCHOR
                ) -> tuple[FloatArray, FloatArray]:
    """Return ``(out, nm)``.

    ``g is None`` -> pass-through, ``nm`` still measured with the default ``p``.
    ``lim is None`` -> the MEASURING pass ``pipeline.compile`` runs before it
    knows the limit: also pass-through, also measured.  Everything that ships
    goes through :func:`pipeline.compile`, which always supplies a limit.
    """
    p = DEFAULT_DIAG.p if g is None else g.p
    nm, n, d = norm(code, L_cur, p, kind)
    code = np.asarray(code, dtype=np.float64)
    if g is None or lim is None:
        return code, nm
    if _q(g.knee, lim) <= 1.0:
        return code, nm              # q <= 1: nothing left the gamut, [G] is the identity
    mp = limit_curve(nm, g.knee, lim, g.end_slope)
    s = np.where(nm > g.knee, mp / np.maximum(nm, 1e-9), 1.0)
    return n + d * s[..., None], nm
