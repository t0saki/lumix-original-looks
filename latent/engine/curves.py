"""1D curve and window primitives: monotone Hermite tone curves, wrapped hue
windows, smoothstep. Vectorized, float64."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

FloatArray = np.ndarray


def monotone_hermite(
    points: Sequence[tuple[float, float]],
    slopes: dict[int, float] | None = None,
) -> Callable[[FloatArray], FloatArray]:
    """Monotone cubic through `points` (must be strictly increasing in x and y).

    `slopes` optionally pins the derivative at point indices; pinned values are
    clamped to the Fritsch-Carlson monotonicity region so the result stays
    strictly monotone. Unpinned derivatives come from PCHIP.
    """
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0):
        raise ValueError("control points must be increasing in x and non-decreasing in y")
    d = PchipInterpolator(x, y).derivative()(x)
    if slopes:
        secants = np.diff(y) / np.diff(x)
        for idx, s in slopes.items():
            s = float(s)
            # Fritsch-Carlson: derivative must stay within 3x the adjacent secants.
            bounds = []
            if idx > 0:
                bounds.append(secants[idx - 1])
            if idx < len(x) - 1:
                bounds.append(secants[idx])
            hi = 3.0 * min(b for b in bounds) if bounds else s
            d[idx] = float(np.clip(s, 0.0, max(hi, 0.0)))
    spline = CubicHermiteSpline(x, y, d)

    def f(v: FloatArray) -> FloatArray:
        v = np.asarray(v, dtype=np.float64)
        return np.clip(spline(np.clip(v, x[0], x[-1])), y[0], y[-1])

    return f


def smoothstep(edge0: float, edge1: float, v: FloatArray) -> FloatArray:
    t = np.clip((np.asarray(v, dtype=np.float64) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hue_delta(h: FloatArray, center: float) -> FloatArray:
    """Signed wrapped hue distance in degrees, in [-180, 180)."""
    return (np.asarray(h, dtype=np.float64) - center + 180.0) % 360.0 - 180.0


def raised_cosine_window(h_deg: FloatArray, center: float, halfwidth: float) -> FloatArray:
    """1 at center, cosine falloff to 0 at +/- halfwidth (wrapped)."""
    d = np.abs(hue_delta(h_deg, center))
    w = 0.5 * (1.0 + np.cos(np.pi * np.minimum(d / halfwidth, 1.0)))
    return np.where(d < halfwidth, w, 0.0)


def gauss(v: FloatArray, center: float, sigma: float) -> FloatArray:
    v = np.asarray(v, dtype=np.float64)
    return np.exp(-0.5 * ((v - center) / sigma) ** 2)


def band_window(
    v: FloatArray, lo: float, hi: float, feather_lo: float, feather_hi: float
) -> FloatArray:
    """1 inside [lo, hi], smoothstep feather of the given widths outside."""
    up = smoothstep(lo - feather_lo, lo, v)
    down = 1.0 - smoothstep(hi, hi + feather_hi, v)
    return up * down


# ---------------------------------------------------------------------------
# Latent-2026 additions (ENGINE_SPEC §0).  Everything above is vendored and
# must stay byte-identical; only new functions are added below.
# ---------------------------------------------------------------------------


def S5(edge0: float, edge1: float, v: FloatArray) -> FloatArray:
    """Smootherstep ``u**3 * (6u**2 - 15u + 10)`` on ``u = clip((v-a)/(b-a),0,1)``.

    C^2 (first *and* second derivative vanish at both edges), unlike
    :func:`smoothstep` whose second derivative jumps by ``6/(b-a)**2`` at each
    edge — that jump lands straight in the 33-point lattice's second-difference
    budget.  Every ramp in the engine uses this.

    ``edge1 <= edge0`` **raises**.  Returning a Heaviside step for coincident
    edges (which is what the degenerate limit is) would put a genuine C0
    discontinuity inside the path — measured: an op with
    ``l_band = (0.2, 0.2, 0.8, 0.8)`` moves two probes 7.2e-07 codes apart to
    outputs 28.64 codes apart — and ENGINE_SPEC §0 / R06 §1.3(a),(j) exist to
    prevent exactly that.  Nothing in the engine wants a step.
    """
    edge0 = float(edge0)
    edge1 = float(edge1)
    v = np.asarray(v, dtype=np.float64)
    if edge1 <= edge0:
        raise ValueError(
            f"S5: edge1 = {edge1!r} must be > edge0 = {edge0!r}; coincident or "
            "reversed edges would make this a step function (a C0 crease in the path)"
        )
    u = np.clip((v - edge0) / (edge1 - edge0), 0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def gauss_hue(h_deg: FloatArray, center: float, sigma: float) -> FloatArray:
    """Wrapped Gaussian hue window: ``exp(-0.5*(hue_delta(h,c)/sigma)**2)``.

    Periodic to machine precision and C^infinity — unlike
    :func:`raised_cosine_window`, whose second derivative jumps at the window
    edge.  ``sigma`` is in degrees.
    """
    d = hue_delta(h_deg, center)
    return np.exp(-0.5 * (d / float(sigma)) ** 2)


def sp(z: FloatArray, k: float) -> FloatArray:
    """Scaled softplus ``k*log(1+exp(z/k))``, written overflow-free.

    ``max(z,0) + k*log1p(exp(-|z|/k))`` — exact to ``max(z,0)`` at ``k = 0`` and
    finite at ``|z/k| = 1e6`` (the naive form overflows at ``z/k ~ 710``).
    Used by the milestone-2 film stage; kept here because ENGINE_SPEC §0 puts
    all three primitives in this module.
    """
    z = np.asarray(z, dtype=np.float64)
    k = float(k)
    if k <= 0.0:
        return np.maximum(z, 0.0)
    return np.maximum(z, 0.0) + k * np.log1p(np.exp(-np.abs(z) / k))


def soft_relu(z: FloatArray, width: float) -> FloatArray:
    """``z * S5(0, width, z)`` — a C^2 rectifier that is **exactly** 0 at z <= 0.

    ENGINE_SPEC v1.2 W2 writes its chroma fade as ``kappa*sp(g-1, 0.02)*r0``
    with :func:`sp`, the scaled softplus.  ``sp`` is C^infinity but it is not 0
    at 0: ``sp(0, k) = k*ln 2 = 0.013863`` at the ruling's own ``k = 0.02``.  On
    an identity look every chroma gain is exactly 1, so that offset fades the
    chroma of every colour by 1.3863 % — **measured: 22.56 codes** on the sRGB
    primaries (their OKLab chroma drops by 0.0036, and near the gamut corner
    that is a large move in code space), against ENGINE_SPEC §8's 1e-6 identity
    gate.  W2 also asks, in the same breath, that ``g < 1`` be untouched and
    that a shell colour keep its chroma exactly; ``sp`` gives neither.

    ``z*S5(0, w, z)`` is the smoothing of ``max(z, 0)`` that does: 0 for z <= 0
    (identity exact, desaturation never faded), ``z`` for z >= w (a shell colour
    at kappa = 1 comes out at exactly its input chroma), C^2 in between, and
    monotone non-decreasing, which is what W2's injectivity algebra needs.  The
    price is curvature at the crossing: ``max|d2/dz2| = 4.96/w`` against
    ``1/(4k) = 12.5`` for ``sp``.  ``w`` is :data:`engine.field.RELFADE_WIDTH`.
    """
    z = np.asarray(z, dtype=np.float64)
    return z * S5(0.0, float(width), z)


def soft_ramp(z: FloatArray, width: float) -> FloatArray:
    """``int_0^z S5(0, width, t) dt`` — a rectifier whose SLOPE stays in [0, 1].

    Closed form (``u = z/width``)::

        0                         z <= 0
        width*(u**6 - 3u**5 + 2.5u**4)   0 < z < width
        z - width/2               z >= width

    It is C^3 (its derivative is the C^2 :func:`S5`), non-decreasing, exactly 0
    for ``z <= 0`` and exactly ``z - width/2`` for ``z >= width``.

    **Why this and not** :func:`soft_relu`.  Both smooth ``max(z, 0)``, but
    ``soft_relu`` overshoots in slope: ``d/dz [z*S5(0,w,z)] = 36u**5 - 75u**4 +
    40u**3`` peaks at ``u = 2/3`` with the value ``16/9 = 1.7778``, where this
    one peaks at exactly 1.  Wherever the rectifier sits inside a monotonicity
    budget the slope is what matters — see
    :func:`engine.cmax.soft_saturate`, whose saturation must keep
    ``d(s*r0)/ds <= 2`` for W2's ``d(C*g_eff)/dC >= 2 - g`` to survive.  With
    ``soft_relu`` there that derivative measures 2.576 (w = 0.12), and a gain of
    1.9 would then run the composite chroma map backwards.

    The price of the bounded slope is the ``width/2`` offset in the saturated
    branch; the caller normalises it away (``cmax._SAT_SCALE``).
    """
    z = np.asarray(z, dtype=np.float64)
    w = float(width)
    if w <= 0.0:
        return np.maximum(z, 0.0)
    u = np.clip(z / w, 0.0, 1.0)
    inner = w * (u ** 4) * (u * (u - 3.0) + 2.5)
    return np.where(z >= w, z - 0.5 * w, np.where(z <= 0.0, 0.0, inner))


if __name__ == "__main__":
    f = monotone_hermite([(0, 0), (0.3, 0.28), (0.5, 0.51), (1, 1)], slopes={0: 0.5, 3: 0.55})
    t = np.linspace(0, 1, 4097)
    out = f(t)
    assert np.all(np.diff(out) > -1e-12), "tone curve must be monotone"
    assert abs(out[0]) < 1e-12 and abs(out[-1] - 1) < 1e-12
    w = raised_cosine_window(np.array([140.0, 140 + 44.9, 140 - 45.1, 320.0]), 140.0, 45.0)
    assert w[0] == 1.0 and 0 < w[1] < 0.02 and w[2] == 0.0 and w[3] == 0.0
    d = hue_delta(np.array([350.0, 10.0]), 5.0)
    assert np.allclose(d, [-15.0, 5.0])
    print("curves.py self-check OK")
