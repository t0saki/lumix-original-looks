"""[N] neutral stage — per-channel tone + designed tint (ENGINE_SPEC §2).

The look designs the grey response directly::

    T_G(t) = monotone_hermite(tone_points)(t)                    # the only PCHIP in the engine
    tint_X(t) = sum_j  A_j/255 * exp(-0.5*((t - mu_j)/sigma_j)**2)      X in {rg, bg}
    tint_X_eff = tint_X * (1 - S5(wg_lo, wg_hi, t)) * T_G/(T_G + k_b)
    T_R = T_G + tint_rg_eff ;  T_B = T_G + tint_bg_eff

The white guard drives the tint to exactly 0 at white (so ``T(1) = (1,1,1)``
to machine precision) and the black guard drives it to 0 as the *output* goes
to black (so no channel is pushed below 0).  Gaussian bumps are C^infinity: a
piecewise-linear or PCHIP tint would put kinks straight into the shipped table.

Without the milestone-2 film stage the correction curves are the target itself
(``C_i = T_i``); with F they would be ``T_i o G_i^-1``.

Off-axis, the three curves are split into a luminance-weighted **common** part
(the same curve on all three channels, so it cannot shift hue) and the
**differential** tint, and only the differential is faded out with input
chroma::

    C_L   = 0.2126*C_R + 0.7152*C_G + 0.0722*C_B
    w     = (1 - S5(fade_lo, fade_hi, C0)) * (1 - (1 - tint_skin_residual)*w_skin)
    out_i = (1-w)*C_L(v_i) + w*C_i(v_i)

A convex combination of monotone [0,1] -> [0,1] maps: bounded without a clip,
exact on the grey axis (w = 1 there), white -> white.

Tone is applied per channel **in the code domain** on purpose: that is what
couples contrast and saturation the way film does (measured law
``C_out/C_in ~ local_slope**0.57``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import curves
from .color import FloatArray
from .spec import CHANNEL, Neutral

__all__ = [
    "RAMP_N", "CompiledNeutral", "build_target", "compile_neutral", "apply_neutral",
    "validate_neutral", "eval_curve",
]

#: the one evaluation grid.  The *same* dense table is used to build T and to
#: apply the curves, so "grey axis == T" is exact rather than approximate.
RAMP_N = 4097

_LUMA = np.array([0.2126, 0.7152, 0.0722])


@dataclass(frozen=True)
class CompiledNeutral:
    t: FloatArray          # (RAMP_N,) input code grid, 0..1
    T: FloatArray          # (RAMP_N, 3) designed grey response
    C: FloatArray          # (RAMP_N, 3) correction curves (== T without [F])
    C_L: FloatArray        # (RAMP_N,) luminance-weighted common curve
    fade: tuple[float, float]
    tint_skin_residual: float

    @property
    def slope_ends(self) -> tuple[FloatArray, FloatArray]:
        n = self.t.size - 1
        return (self.C[1] - self.C[0]) * n, (self.C[-1] - self.C[-2]) * n


def _bumps(t: FloatArray, bumps) -> FloatArray:
    out = np.zeros_like(t)
    for b in bumps:
        out = out + (b.a / 255.0) * np.exp(-0.5 * ((t - b.mu) / b.sigma) ** 2)
    return out


def build_target(ns: Neutral, n: int = RAMP_N) -> tuple[FloatArray, FloatArray]:
    """Return ``(t, T)`` — the grid and the (n, 3) designed grey response."""
    t = np.linspace(0.0, 1.0, int(n))
    pts = [(float(a) / 255.0, float(b) / 255.0) for a, b in ns.tone]
    TG = curves.monotone_hermite(pts)(t)
    guard_w = 1.0 - curves.S5(ns.white_guard[0], ns.white_guard[1], t)
    k_b = ns.tint_black_k / 255.0
    guard_b = TG / (TG + k_b) if k_b > 0 else np.ones_like(TG)
    rg = _bumps(t, ns.tint_rg) * guard_w * guard_b
    bg = _bumps(t, ns.tint_bg) * guard_w * guard_b
    return t, np.stack([TG + rg, TG, TG + bg], axis=-1)


def compile_neutral(ns: Neutral, n: int = RAMP_N) -> CompiledNeutral:
    t, T = build_target(ns, n)
    C = T  # no [F] in milestone 1  =>  C_i = T_i
    C_L = C @ _LUMA
    return CompiledNeutral(
        t=t, T=T, C=C, C_L=C_L,
        fade=(float(ns.fade[0]), float(ns.fade[1])),
        tint_skin_residual=float(ns.tint_skin_residual),
    )


def eval_curve(t: FloatArray, table: FloatArray, v: FloatArray) -> FloatArray:
    """Evaluate a dense monotone curve at ``v``, extended LINEARLY outside [0,1].

    ``np.interp`` alone would clamp, and a clamp inside the path is exactly the
    plateau that ENGINE_SPEC forbids (it is a division by zero for the [F]
    correction solve and a fold in the lattice).  Out-of-range inputs only occur
    in the PGN order, where [G]'s dark-side ``eps_dark`` floor can push a code
    slightly below 0; they stay out of range and the final clip reports them.
    """
    v = np.asarray(v, dtype=np.float64)
    y = np.interp(v, t, table)
    n = t.size - 1
    s0 = (table[1] - table[0]) * n
    s1 = (table[-1] - table[-2]) * n
    y = np.where(v < 0.0, table[0] + s0 * v, y)
    y = np.where(v > 1.0, table[-1] + s1 * (v - 1.0), y)
    return y


def apply_neutral(cn: CompiledNeutral, v: FloatArray, C0: FloatArray,
                  w_skin: FloatArray) -> FloatArray:
    """Apply the correction curves to ``v`` (code values, shape (..., 3))."""
    v = np.asarray(v, dtype=np.float64)
    w = (1.0 - curves.S5(cn.fade[0], cn.fade[1], C0)) * (
        1.0 - (1.0 - cn.tint_skin_residual) * w_skin
    )
    out = np.empty_like(v)
    for i in range(3):
        vi = v[..., i]
        ci = eval_curve(cn.t, cn.C[:, i], vi)
        cl = eval_curve(cn.t, cn.C_L, vi)
        out[..., i] = cl + w * (ci - cl)
    return out


def apply_neutral_grey(cn: CompiledNeutral, code: FloatArray) -> FloatArray:
    """The mono branch's N: ``w == 1`` (the image is grey, the tint applies to
    everything).  ``code`` is a scalar grey per pixel, shape (...)."""
    code = np.asarray(code, dtype=np.float64)
    return np.stack([eval_curve(cn.t, cn.C[:, i], code) for i in range(3)], axis=-1)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

MIN_SLOPE = 0.04
MAX_SLOPE = 3.5
MAX_TONE_F2 = 24.0
MIN_TINT_SIGMA = 0.15


def validate_neutral(ns: Neutral) -> list[str]:
    """Return a list of error strings (empty = fine).  ENGINE_SPEC §2."""
    errors: list[str] = []

    tone = np.asarray(ns.tone, dtype=np.float64)
    if tone.ndim != 2 or tone.shape[1] != 2 or tone.shape[0] < 2:
        return [f"neutral.tone: expected >= 2 [in, out] pairs, got shape {tone.shape}"]
    if tone[0, 0] != 0.0:
        errors.append(f"neutral.tone: first point in = {tone[0, 0]}, must be 0")
    if tuple(tone[-1]) != (255.0, 255.0):
        errors.append(f"neutral.tone: last point = {list(tone[-1])}, must be [255, 255]")
    if np.any(np.diff(tone[:, 0]) <= 0):
        errors.append("neutral.tone: input codes must be strictly increasing")
    if np.any(np.diff(tone[:, 1]) < 0):
        errors.append("neutral.tone: output codes must be non-decreasing")
    for j, b in enumerate(tuple(ns.tint_rg) + tuple(ns.tint_bg)):
        if b.sigma < MIN_TINT_SIGMA - 1e-12:
            errors.append(f"neutral tint bump #{j}: sigma = {b.sigma:.3f} < {MIN_TINT_SIGMA} "
                          "(narrower bumps put a kink in the shipped table)")
    if ns.white_guard[1] <= ns.white_guard[0]:
        errors.append(f"neutral.white_guard = {list(ns.white_guard)} must be increasing")
    if ns.fade[1] <= ns.fade[0]:
        errors.append(f"neutral.fade = {list(ns.fade)} must be increasing")
    if ns.tint_black_k < 0:
        errors.append(f"neutral.tint_black_k = {ns.tint_black_k} must be >= 0")
    if errors:
        return errors

    t, T = build_target(ns)
    n = t.size - 1
    slope = np.diff(T, axis=0) * n
    for ch in range(3):
        k = int(np.argmin(slope[:, ch]))
        if slope[k, ch] < MIN_SLOPE:
            errors.append(
                f"neutral: channel {CHANNEL[ch]} slope = {slope[k, ch]:.4f} < {MIN_SLOPE} "
                f"at t = {t[k]:.4f} (code {t[k] * 255:.1f}) — grey response not strictly "
                "increasing enough; a plateau divides by zero in the [F] correction"
            )
        m = int(np.argmax(slope[:, ch]))
        if slope[m, ch] > MAX_SLOPE:
            errors.append(
                f"neutral: channel {CHANNEL[ch]} slope = {slope[m, ch]:.4f} > {MAX_SLOPE} "
                f"at t = {t[m]:.4f} — 8-bit input will posterise"
            )
    if T.min() < 0.0 or T.max() > 1.0:
        lo, hi = float(T.min()), float(T.max())
        errors.append(f"neutral: T out of [0,1] (min {lo:.6f}, max {hi:.6f})")
    werr = float(np.max(np.abs(T[-1] - 1.0)))
    if werr > 1e-12:
        errors.append(f"neutral: T(1) = {[float(x) for x in T[-1]]}, error {werr:.3e} > 1e-12")

    f2 = np.abs(np.diff(T[:, 1], n=2)) * n * n
    k = int(np.argmax(f2))
    if f2[k] > MAX_TONE_F2:
        errors.append(
            f"neutral.tone: |f''| = {f2[k]:.2f} > {MAX_TONE_F2} at t = {t[k + 1]:.4f} "
            f"(code {t[k + 1] * 255:.1f}) — d2 budget of the 33-point lattice is "
            f"|f''| <= 24 (= 6 codes)"
        )
    return errors
