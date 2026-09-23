"""Mono branch (ENGINE_SPEC §6).

::

    Y  = w . linear(rgb_in)                       # filter weights sum to 1, each >= -0.15
    Lm = cbrt(Y)                                  # odd (signed); no clamp in the path
    Lm += DL(h0) * gch * 4*Lm*(1-Lm)/0.91         # per-hue tonal separation
    code = srgb_encode(Lm**3)
    out  = N(code) with w == 1                    # the image is grey: the tint applies to all of it

The whole look lives in [N] plus the channel mix; [P]'s chromatic operators and
[G] have nothing to act on, so they are skipped (which also makes the stage
order irrelevant for a mono look).  The filter is what makes a red and a blue of
equal luminance land on different greys — the §6 unit test.
"""

from __future__ import annotations

import numpy as np

from .color import FloatArray
from .field import CompiledField, L_SHAPE_NORM, Stage0, _periodic
from .neutral import CompiledNeutral, apply_neutral_grey
from .xfer import signed_srgb_encode

__all__ = ["compile_mono", "apply_mono"]


def compile_mono(spec):
    """The mono DL table (defaults to the look's ``field.dl``)."""
    values = spec.mono.dl if spec.mono.dl is not None else spec.field.dl
    return _periodic(values)


def apply_mono(mono_dl, cf: CompiledField, cn: CompiledNeutral,
               linear_in: FloatArray, st: Stage0) -> FloatArray:
    w = np.asarray(cf.spec.mono.filter, dtype=np.float64)
    Y = np.asarray(linear_in, dtype=np.float64) @ w
    # ``cbrt`` is odd, so this is the signed extension of the cube root — NOT
    # ``cbrt(max(Y, 0))``.  A filter with a negative weight (§6 allows down to
    # -0.15) makes Y negative on part of the cube, and clamping there is a C0
    # crease *inside* the path plus a dead plateau: measured with
    # filter (1.15, 0, -0.15), a 20,001-point blue ramp had all 20,000
    # consecutive differences exactly 0 over 76.5 codes — R06 §1.3(a)/(j)'s
    # "min step = 0.00e+00" signature, and ENGINE_SPEC §0 forbids the clamp.
    # Signed, the whole path stays strictly monotone and those colours arrive at
    # the final clip below 0, where diagnostics() measures them.  For every
    # non-negative filter (all of R06 §3.9's presets) Y >= 0 and nothing changes.
    Lm = np.cbrt(Y)
    Lm = Lm + mono_dl(st.h0) * st.gch * (4.0 * Lm * (1.0 - Lm) / L_SHAPE_NORM)
    code = signed_srgb_encode(Lm ** 3)
    return apply_neutral_grey(cn, code)
