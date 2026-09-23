"""Color space primitives: sRGB <-> linear <-> OKLab/OKLCh, CIE L*, gamut compression.

All functions are vectorized over arrays of shape (..., 3) (or scalar/array for
the 1D transfer functions). float64 throughout.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray

# ---------------------------------------------------------------------------
# sRGB transfer
# ---------------------------------------------------------------------------


def srgb_decode(code: FloatArray) -> FloatArray:
    code = np.asarray(code, dtype=np.float64)
    return np.where(code <= 0.04045, code / 12.92, ((code + 0.055) / 1.055) ** 2.4)


def srgb_encode(linear: FloatArray) -> FloatArray:
    linear = np.asarray(linear, dtype=np.float64)
    linear = np.maximum(linear, 0.0)
    return np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1.0 / 2.4) - 0.055
    )


# ---------------------------------------------------------------------------
# OKLab (Björn Ottosson). Neutral axis property: gray with linear value y has
# L_ok = y**(1/3), a = b = 0 exactly (first-row coefficients sum to 1).
# ---------------------------------------------------------------------------

_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
# Renormalize (~1e-8 relative) so the neutral axis is EXACTLY neutral:
# gray (y,y,y) -> lms (y,y,y) -> L = cbrt(y), a = b = 0 with no rounding residue.
_M1 /= _M1.sum(axis=1, keepdims=True)
_M2[0] /= _M2[0].sum()
_M2[1] -= _M2[1].sum() / 3.0
_M2[2] -= _M2[2].sum() / 3.0
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def linear_srgb_to_oklab(rgb: FloatArray) -> FloatArray:
    rgb = np.asarray(rgb, dtype=np.float64)
    lms = rgb @ _M1.T
    lms_p = np.cbrt(lms)
    return lms_p @ _M2.T


def oklab_to_linear_srgb(lab: FloatArray) -> FloatArray:
    lab = np.asarray(lab, dtype=np.float64)
    lms_p = lab @ _M2_INV.T
    lms = lms_p ** 3
    return lms @ _M1_INV.T


def oklab_to_lch(lab: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return (L, C, h_deg) with h in [0, 360)."""
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return L, C, h


def lch_to_oklab(L: FloatArray, C: FloatArray, h_deg: FloatArray) -> FloatArray:
    hr = np.radians(h_deg)
    return np.stack([L, C * np.cos(hr), C * np.sin(hr)], axis=-1)


# ---------------------------------------------------------------------------
# CIE L* <-> relative luminance Y (D65, 2 deg) — used only on the neutral axis
# to translate recipe control points given in L*.
# ---------------------------------------------------------------------------

_CIE_EPS = 216.0 / 24389.0
_CIE_KAPPA = 24389.0 / 27.0


def lstar_from_y(y: FloatArray) -> FloatArray:
    y = np.asarray(y, dtype=np.float64)
    return np.where(y > _CIE_EPS, 116.0 * np.cbrt(y) - 16.0, _CIE_KAPPA * y)


def y_from_lstar(lstar: FloatArray) -> FloatArray:
    lstar = np.asarray(lstar, dtype=np.float64)
    fy = (lstar + 16.0) / 116.0
    return np.where(lstar > _CIE_KAPPA * _CIE_EPS, fy ** 3, lstar / _CIE_KAPPA)


# Neutral-axis converters between representations of the same gray:
#   sRGB code t  <->  linear y  <->  OKLab L = y**(1/3)  <->  CIE L*


def oklabL_from_code(t: FloatArray) -> FloatArray:
    return np.cbrt(srgb_decode(t))


def code_from_oklabL(L: FloatArray) -> FloatArray:
    return srgb_encode(np.asarray(L, dtype=np.float64) ** 3)


def oklabL_from_lstar(lstar: FloatArray) -> FloatArray:
    return np.cbrt(y_from_lstar(lstar))


def lstar_from_oklabL(L: FloatArray) -> FloatArray:
    return lstar_from_y(np.asarray(L, dtype=np.float64) ** 3)


# ---------------------------------------------------------------------------
# Gamut compression: constant-L, constant-hue soft-knee chroma compression
# toward the sRGB gamut boundary in OKLCh.
# ---------------------------------------------------------------------------


def _max_chroma(L: FloatArray, h_deg: FloatArray, iters: int = 16) -> FloatArray:
    """Vectorized bisection for the largest in-gamut chroma at (L, h)."""
    L = np.clip(np.asarray(L, dtype=np.float64), 0.0, 1.0)
    h_deg = np.asarray(h_deg, dtype=np.float64)
    lo = np.zeros_like(L)
    hi = np.full_like(L, 0.5)  # sRGB max OKLab chroma is ~0.32

    def in_gamut(c: FloatArray) -> FloatArray:
        rgb = oklab_to_linear_srgb(lch_to_oklab(L, c, h_deg))
        eps = 1e-7
        return np.all((rgb >= -eps) & (rgb <= 1.0 + eps), axis=-1)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = in_gamut(mid)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


def project_into_gamut(lab: FloatArray, knee: float | None = 0.90) -> FloatArray:
    """Map colors into sRGB along constant-L, constant-hue chroma.

    knee=None: exact projection — in-gamut colors pass through EXACTLY, only
    out-of-gamut pixels are clamped to the boundary (C0 at the boundary; used
    by the identity check).

    knee=k (default 0.90): C1-continuous soft knee — chroma below k*C_max is
    untouched, everything above rolls off asymptotically toward C_max. Only
    ultra-saturated colors (C > k*C_max, essentially absent from photographs)
    lose a few percent chroma, and the lattice carries no crease at the gamut
    boundary (the second-difference / banding win)."""
    lab = np.asarray(lab, dtype=np.float64)
    if knee is None:
        rgb = oklab_to_linear_srgb(lab)
        eps = 1e-9
        oog = np.any((rgb < -eps) | (rgb > 1.0 + eps), axis=-1)
        if not np.any(oog):
            return lab
        L, C, h = oklab_to_lch(lab[oog])
        c_max = _max_chroma(np.clip(L, 0.0, 1.0), h, iters=24)
        out = lab.copy()
        out[oog] = lch_to_oklab(np.clip(L, 0.0, 1.0), np.minimum(C, c_max), h)
        return out
    L, C, h = oklab_to_lch(lab)
    L_cl = np.clip(L, 0.0, 1.0)
    c_max = _max_chroma(L_cl, h, iters=24)
    c_knee = knee * c_max
    span = np.maximum((1.0 - knee) * c_max, 1e-9)
    d = np.maximum(C - c_knee, 0.0) / span
    c_out = np.where(C <= c_knee, C, c_knee + span * d / (1.0 + d))
    c_out = np.where(c_max < 1e-6, 0.0, c_out)
    return lch_to_oklab(L_cl, c_out, h)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    rgb = rng.random((4096, 3))
    # sRGB round trip
    err1 = np.max(np.abs(srgb_encode(srgb_decode(rgb)) - rgb))
    # OKLab round trip
    lin = srgb_decode(rgb)
    err2 = np.max(np.abs(oklab_to_linear_srgb(linear_srgb_to_oklab(lin)) - lin))
    # Neutral axis: gray must map to a=b=0 and L=y^(1/3)
    t = np.linspace(0, 1, 257)
    gray = np.stack([t, t, t], axis=-1)
    lab = linear_srgb_to_oklab(srgb_decode(gray))
    err3 = np.max(np.abs(lab[:, 1:]))
    err4 = np.max(np.abs(lab[:, 0] - np.cbrt(srgb_decode(t))))
    # CIE round trip
    ls = np.linspace(0, 100, 1001)
    err5 = np.max(np.abs(lstar_from_y(y_from_lstar(ls)) - ls))
    print(f"srgb roundtrip      {err1:.3e}")
    print(f"oklab roundtrip     {err2:.3e}")
    print(f"neutral a,b         {err3:.3e}")
    print(f"neutral L=y^(1/3)   {err4:.3e}")
    print(f"cie roundtrip       {err5:.3e}")
    assert max(err1, err2, err3, err4, err5) < 1e-9
    # Gamut projection sanity: out-of-gamut lands inside; ALL in-gamut inputs
    # (including saturated cube corners) pass through exactly.
    lab_oog = lch_to_oklab(np.array([0.6]), np.array([0.5]), np.array([30.0]))
    rgb_out = oklab_to_linear_srgb(project_into_gamut(lab_oog))
    assert np.all(rgb_out > -1e-6) and np.all(rgb_out < 1.0 + 1e-6)
    lab_all = linear_srgb_to_oklab(srgb_decode(rng.random((4096, 3))))
    assert np.max(np.abs(project_into_gamut(lab_all) - lab_all)) == 0.0
    print("color.py self-check OK")
