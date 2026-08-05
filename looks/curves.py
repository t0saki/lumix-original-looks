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
