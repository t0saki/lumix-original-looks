"""Signed sRGB transfer functions (ENGINE_SPEC §0).

The perceptual stage [P] legitimately produces out-of-gamut colours — negative
linear values — which the gamut stage [G] then folds back smoothly *in the code
domain*.  Clipping them on the way into the code domain would put a crease
exactly where [G] is supposed to be smooth, so the transfer curve is extended
to negative arguments as an odd function::

    signed_encode(x) = sign(x) * srgb_encode(|x|)

That extension is C^1 through 0 (both branches are linear with slope 12.92 /
1/12.92 near the origin), strictly increasing on the whole real line, and its
inverse is the identically-extended decode.  ``engine/color.py`` is vendored and
must not be edited, so these live here.
"""

from __future__ import annotations

import numpy as np

from .color import FloatArray, srgb_decode, srgb_encode

__all__ = ["signed_srgb_encode", "signed_srgb_decode"]


def signed_srgb_encode(linear: FloatArray) -> FloatArray:
    """sRGB encode extended oddly to negative linear values."""
    linear = np.asarray(linear, dtype=np.float64)
    return np.sign(linear) * srgb_encode(np.abs(linear))


def signed_srgb_decode(code: FloatArray) -> FloatArray:
    """sRGB decode extended oddly to negative code values."""
    code = np.asarray(code, dtype=np.float64)
    return np.sign(code) * srgb_decode(np.abs(code))
