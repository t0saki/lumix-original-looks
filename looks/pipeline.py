"""Shared creative-look pipeline.

A look is a plain dict (see params.py). apply_look() runs a fixed stage order;
each stage reads its parameters from the dict if present, so every look uses
only the operators its recipe calls for. Input/output are display sRGB codes
in [0, 1], shape (..., 3) — the exact quantity the S9 feeds a Standard-base
Real Time LUT.

Luminance windows in the params are expressed in CIE L* (as the recipes are
written); hue windows in OKLCh degrees. Hue/sat windows are evaluated on the
pre-shift hue so object identity, not the graded color, selects the ops.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import color, curves
from .color import FloatArray


# ---------------------------------------------------------------------------
# Tone-curve construction
# ---------------------------------------------------------------------------


def build_tone_fn(tone: dict) -> Callable[[FloatArray], FloatArray]:
    """Compile a tone spec into a monotone function on OKLab L.

    tone = {"domain": "lstar" | "code", "points": [(x, y), ...],
            "slopes": {index: slope}}  — points in the native domain.
    The native-domain curve is densely sampled and re-expressed in OKLab L, so
    monotonicity survives the (monotone) domain change exactly.
    """
    domain = tone["domain"]
    f_native = curves.monotone_hermite(tone["points"], tone.get("slopes"))
    xs = np.asarray(tone["points"], dtype=np.float64)
    x_dense = np.linspace(xs[0, 0], xs[-1, 0], 2049)
    y_dense = f_native(x_dense)
    if domain == "lstar":
        xo = color.oklabL_from_lstar(x_dense)
        yo = color.oklabL_from_lstar(y_dense)
    elif domain == "code":
        xo = color.oklabL_from_code(x_dense)
        yo = color.oklabL_from_code(y_dense)
    else:
        raise ValueError(f"unknown tone domain {domain!r}")

    from scipy.interpolate import PchipInterpolator

    spline = PchipInterpolator(xo, yo)

    def f(L: FloatArray) -> FloatArray:
        L = np.asarray(L, dtype=np.float64)
        return np.clip(spline(np.clip(L, xo[0], xo[-1])), yo[0], yo[-1])

    return f


def _norm_shoulder_fn(spec: dict) -> Callable[[FloatArray], FloatArray]:
    """Concave highlight shoulder on max(R,G,B) in sRGB code domain."""
    start = spec["start"]
    end_slope = spec["end_slope"]
    return curves.monotone_hermite(
        [(0.0, 0.0), (start, start), (1.0, 1.0)],
        slopes={0: 1.0, 1: 1.0, 2: end_slope},
    )


# ---------------------------------------------------------------------------
# Window helpers (L windows take CIE L*)
# ---------------------------------------------------------------------------


def _l_gauss(lstar: FloatArray, spec: dict) -> FloatArray:
    w = curves.gauss(lstar, spec["l_center"], spec["l_sigma"])
    if "l_zero_lo" in spec:
        w = w * curves.smoothstep(spec["l_zero_lo"] - 6.0, spec["l_zero_lo"], lstar)
    if "l_zero_hi" in spec:
        w = w * (1.0 - curves.smoothstep(spec["l_zero_hi"], spec["l_zero_hi"] + 6.0, lstar))
    return w


def _skin_weight(h: FloatArray, C: FloatArray, skin: dict) -> FloatArray:
    """Suite-standard skin window: full weight h in [42,58], raised-cosine
    feather to 0 at 30/70, gated by chroma so the neutral axis is never
    'skin'."""
    lo, hi = skin.get("lo", 30.0), skin.get("hi", 70.0)
    feather = skin.get("feather", 14.0)
    d_lo = (h - lo) / feather
    d_hi = (hi - h) / feather
    ramp_lo = 0.5 * (1.0 - np.cos(np.pi * np.clip(d_lo, 0.0, 1.0)))
    ramp_hi = 0.5 * (1.0 - np.cos(np.pi * np.clip(d_hi, 0.0, 1.0)))
    w = np.where((h >= lo) & (h <= hi), np.minimum(ramp_lo, ramp_hi), 0.0)
    return w * curves.smoothstep(0.02, 0.06, C)


def _hue_op_weight(
    h0: FloatArray, lstar: FloatArray, op: dict, sat_rel: FloatArray | None = None
) -> FloatArray:
    """Common weight for hue-windowed ops: raised-cosine hue window, optional
    L* band/gate, optional hard hue cut, optional gamut-surface fade.

    "gamut_fade": (lo, hi) fades the op with RELATIVE saturation C/C_max(L,h)
    — creative moves live in the photographic range; fading them toward the
    gamut faces removes lattice curvature where no real subject sits. (sRGB
    C_max varies strongly by hue — cyan faces sit at C~0.15 — so an absolute
    chroma threshold cannot do this job.)"""
    w = curves.raised_cosine_window(h0, op["center"], op["halfwidth"])
    if "gamut_fade" in op and sat_rel is not None:
        w = w * (1.0 - curves.smoothstep(op["gamut_fade"][0], op["gamut_fade"][1], sat_rel))
    if "l_band" in op:  # (lo, hi, feather_lo, feather_hi)
        lo, hi, flo, fhi = op["l_band"]
        w = w * curves.band_window(lstar, lo, hi, flo, fhi)
    if "l_gate" in op:  # {"lo", "hi", "floor"}: weight >= floor, ramps up with L*
        g = op["l_gate"]
        w = w * (g["floor"] + (1.0 - g["floor"]) * curves.smoothstep(g["lo"], g["hi"], lstar))
    if "hard_cut_below" in op:  # zero for hue below the cut, smooth 12-deg transition
        w = w * curves.smoothstep(op["hard_cut_below"], op["hard_cut_below"] + 12.0, h0)
    return w


def _unit_ab(hue_deg: float) -> tuple[float, float]:
    hr = np.radians(hue_deg)
    return float(np.cos(hr)), float(np.sin(hr))


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def apply_look(rgb_code: FloatArray, look: dict) -> FloatArray:
    rgb = np.clip(np.asarray(rgb_code, dtype=np.float64), 0.0, 1.0)

    # 1. Optional shoulder on max(R,G,B) in code domain (NightMarket): keeps
    #    high-chroma highlights from hue-twisting under per-channel rolloff.
    if "norm_shoulder" in look:
        f_sh = _norm_shoulder_fn(look["norm_shoulder"])
        m = np.max(rgb, axis=-1, keepdims=True)
        scale = np.where(m > 1e-6, f_sh(m) / np.maximum(m, 1e-6), 1.0)
        rgb = np.clip(rgb * scale, 0.0, 1.0)

    lab = color.linear_srgb_to_oklab(color.srgb_decode(rgb))
    L0, C0, h0 = color.oklab_to_lch(lab)
    w_skin = _skin_weight(h0, C0, look.get("skin", {}))
    # relative saturation of the INPUT color vs the gamut face at its (L, h)
    c_max0 = color._max_chroma(np.clip(L0, 0.0, 1.0), h0, iters=16)
    sat_rel = C0 / np.maximum(c_max0, 1e-6)

    # 2. Tone on OKLab L (hue/chroma untouched -> neutral axis follows the
    #    1D curve exactly).
    L = build_tone_fn(look["tone"])(L0)
    if "tone_skin_relief" in look:  # skin sees only part of the tonal move
        r = look["tone_skin_relief"]
        L = L0 + (L - L0) * (1.0 - r * w_skin)

    lstar = color.lstar_from_oklabL(L)

    # 3. Declared skin-lightness move (in L* units)
    if "skin_lightening" in look:
        sl = look["skin_lightening"]
        w = w_skin * _l_gauss(lstar, sl)
        lstar = lstar + sl["delta_lstar"] * w
        L = color.oklabL_from_lstar(lstar)

    # 4. Hue moves — all deltas computed from the pre-shift hue, then summed.
    skin_hue_residual = look.get("skin_hue_residual", 0.15)
    dh = np.zeros_like(h0)
    for op in look.get("hue_shifts", []):
        w = _hue_op_weight(h0, lstar, op, sat_rel)
        if op.get("skin_exempt"):
            pass  # declared in-window move, full strength
        else:
            w = w * (1.0 - (1.0 - skin_hue_residual) * w_skin)
        dh = dh + op["delta"] * w
    for op in look.get("attractors", []):
        band = curves.band_window(h0, op["lo"], op["hi"], 8.0, 8.0)
        pull = curves.hue_delta(np.full_like(h0, op["target"]), h0)  # signed target - h
        w = band * op["strength"]
        if "l_gate" in op:
            g = op["l_gate"]
            w = w * (g["floor"] + (1.0 - g["floor"]) * curves.smoothstep(g["lo"], g["hi"], lstar))
        w = w * (1.0 - (1.0 - op.get("skin_residual", skin_hue_residual)) * w_skin)
        dh = dh + pull * w
    if "layered_hue" in look:  # Canopy: green splits by luminance
        op = look["layered_hue"]
        w = curves.raised_cosine_window(h0, op["center"], op["halfwidth"])
        if "gamut_fade" in op:
            w = w * (1.0 - curves.smoothstep(op["gamut_fade"][0], op["gamut_fade"][1], sat_rel))
        w = w * curves.smoothstep(op["hard_cut_below"], op["hard_cut_below"] + 12.0, h0)
        t = curves.smoothstep(op["l_lo"], op["l_hi"], lstar)
        delta = op["delta_lo"] + (op["delta_hi"] - op["delta_lo"]) * t
        w = w * (1.0 - (1.0 - skin_hue_residual) * w_skin)
        dh = dh + delta * w
    if "hue_c_gate" in look:  # Meridian: near-neutral pixels keep their hue
        g = look["hue_c_gate"]
        dh = dh * curves.smoothstep(g[0], g[1], C0)
    h = h0 + dh

    # 5. Chroma chain: windowed gains -> vibrance -> luminance desat -> knees.
    f = np.full_like(C0, look.get("global_sat", 1.0))
    for op in look.get("sat_ops", []):
        w = _hue_op_weight(h0, lstar, op, sat_rel)
        if "c_gate" in op:  # only rich colors take the move (NightMarket: lamps, not faces)
            w = w * curves.smoothstep(op["c_gate"][0], op["c_gate"][1], C0)
        f = f * (1.0 + (op["gain"] - 1.0) * w)
    if "vibrance" in look:
        v = look["vibrance"]
        w = 1.0 - curves.smoothstep(v["c_lo"], v["c_hi"], C0)
        f = f * (1.0 + (v["gain"] - 1.0) * w)
    if "lum_desat_shadow" in look:
        d = look["lum_desat_shadow"]
        g = d["gain_at_0"] + (1.0 - d["gain_at_0"]) * curves.smoothstep(0.0, d["l_end"], lstar)
        if "exempt_hue" in d:  # e.g. DuskTide keeps dusk-sky chroma in the dark
            lo, hi = d["exempt_hue"]
            g = 1.0 + (g - 1.0) * (1.0 - curves.band_window(h0, lo, hi, 10.0, 10.0))
        f = f * g
    if "lum_desat_high" in look:
        d = look["lum_desat_high"]
        g = 1.0 + (d["gain_end"] - 1.0) * curves.smoothstep(d["l_start"], d["l_end"], lstar)
        if "c_fade" in d:  # clean airy WHITES; saturated brights keep their color
            g = 1.0 + (g - 1.0) * (1.0 - curves.smoothstep(d["c_fade"][0], d["c_fade"][1], C0))
        f = f * g
    if "skin_sat_clamp" in look:  # clamp the combined gain inside the window
        lo, hi = look["skin_sat_clamp"]
        f = f * (1.0 - w_skin) + np.clip(f, lo, hi) * w_skin
    C = C0 * f
    if "sat_compress_high" in look:  # gain-type compression of already-rich colors
        sc = look["sat_compress_high"]
        C = C * (1.0 + (sc["gain"] - 1.0) * curves.smoothstep(sc["c_start"], sc["c_start"] + 0.15, C))
    if "chroma_knee" in look:
        k = look["chroma_knee"]
        s, m = k["start"], k["slope"]
        if "cap" in k:  # exponential knee with asymptotic cap, C2-blended in
            span = k["cap"] - s
            over = np.maximum(C - s, 0.0)
            kneed = s + span * (1.0 - np.exp(-m * over / span))
            t2 = curves.smoothstep(s - 0.02, s + 0.02, C)
            C = C * (1.0 - t2) + kneed * t2
        else:  # linear reduced slope past the knee, C1-blended over +/-0.02
            lin = s + m * (C - s)
            t = curves.smoothstep(s - 0.02, s + 0.02, C)
            C = C * (1.0 - t) + lin * t
    for op in look.get("chroma_caps", []):  # per-hue soft chroma ceiling
        w = curves.raised_cosine_window(h0, op["center"], op["halfwidth"])
        over = np.maximum(C - op["cap"], 0.0)
        C = C - w * over * (1.0 - 1.0 / (1.0 + over / 0.04))

    # 6. Hue-windowed / chroma-driven lightness gains
    for op in look.get("hue_lum", []):
        w = _hue_op_weight(h0, lstar, op, sat_rel)
        # chroma gate: neutral pixels have noise-hue, and L moves (unlike C
        # multiplies / hue rotations) would otherwise hit the gray axis
        w = w * curves.smoothstep(0.01, 0.03, C0)
        w = w * (1.0 - (1.0 - op.get("skin_residual", 0.0)) * w_skin) if op.get("skin_protect") else w
        L = L * (1.0 + (op["gain"] - 1.0) * w)
    if "micro_density" in look:  # Postcard: richer color sits a touch denser
        md = look["micro_density"]
        w = curves.smoothstep(md["c_lo"], md["c_hi"], C)
        w = w * curves.band_window(lstar, md["l_lo"], md["l_hi"], 10.0, 12.0)
        w = w * (1.0 - (1.0 - md.get("skin_residual", 0.4)) * w_skin)
        L = L * (1.0 - md["strength"] * w)
    lstar = color.lstar_from_oklabL(L)

    # 7. Midtone bias + split toning as (a, b) offsets, guarded so the white
    #    point stays exactly white and blacks only carry declared toe tints.
    da = np.zeros_like(L)
    db = np.zeros_like(L)
    skin_tone_residual = look.get("skin_tone_residual", 0.3)
    if "midtone_bias" in look:
        mb = look["midtone_bias"]
        ua, ub = _unit_ab(mb["hue"])
        w = mb["chroma"] * _l_gauss(lstar, mb)
        w = w * (1.0 - (1.0 - mb.get("skin_residual", skin_tone_residual)) * w_skin)
        da, db = da + ua * w, db + ub * w
    for sp in look.get("splits", []):
        ua, ub = _unit_ab(sp["hue"])
        shape = sp["shape"]
        if shape == "low":  # full below l_peak, raised-cosine to 0 at l_zero
            t = np.clip((lstar - sp["l_peak"]) / max(sp["l_zero"] - sp["l_peak"], 1e-6), 0.0, 1.0)
            w = 0.5 * (1.0 + np.cos(np.pi * t))
        elif shape == "bump":  # raised-cosine bump centered at l_peak
            t = np.clip(np.abs(lstar - sp["l_peak"]) / sp["halfwidth"], 0.0, 1.0)
            w = 0.5 * (1.0 + np.cos(np.pi * t))
        elif shape == "rise":  # smoothstep rise into the highlights
            w = curves.smoothstep(sp["l_rise_from"], sp["l_peak"], lstar)
        else:
            raise ValueError(f"unknown split shape {shape!r}")
        if sp.get("black_guard"):
            w = w * curves.smoothstep(0.0, sp["black_guard"], lstar)
        amt = sp["c_peak"] * w
        amt = amt * (1.0 - (1.0 - sp.get("skin_residual", skin_tone_residual)) * w_skin)
        da, db = da + ua * amt, db + ub * amt
    if "toe_tint" in look:  # declared tinted black lift (full at L*=0)
        tt = look["toe_tint"]
        ua, ub = _unit_ab(tt["hue"])
        w = tt["chroma"] * (1.0 - curves.smoothstep(0.0, tt["l_end"], lstar))
        da, db = da + ua * w, db + ub * w
    wg = look.get("white_guard", (92.0, 98.0))
    guard = 1.0 - curves.smoothstep(wg[0], wg[1], lstar)
    da, db = da * guard, db * guard

    hr = np.radians(h)
    a_out = C * np.cos(hr) + da
    b_out = C * np.sin(hr) + db

    # 8. Gray guard: pixels that came in neutral leave with at most the
    #    declared bias chroma.
    if "gray_guard" in look:
        gg = look["gray_guard"]
        c_out = np.hypot(a_out, b_out)
        near = 1.0 - curves.smoothstep(gg["c_in"], gg["c_in"] * 2.0, C0)
        limit = np.maximum(gg["limit"], C0)
        scale = np.where(c_out > 1e-12, np.minimum(c_out, limit) / np.maximum(c_out, 1e-12), 1.0)
        scale = 1.0 + (scale - 1.0) * near
        a_out, b_out = a_out * scale, b_out * scale

    lab_out = np.stack([L, a_out, b_out], axis=-1)
    lab_out = color.project_into_gamut(lab_out, knee=look.get("gamut_knee", 0.90))
    out = color.srgb_encode(np.clip(color.oklab_to_linear_srgb(lab_out), 0.0, 1.0))
    return np.clip(out, 0.0, 1.0)


def generate_look_table(look: dict, size: int = 33) -> FloatArray:
    """Full lattice in CUBE row order, shape (size**3, 3)."""
    t = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(t, t, t, indexing="ij")  # blue slowest, red fastest
    grid = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)
    out = apply_look(grid, look)
    # Pin the corners: white->white structurally guaranteed, this removes
    # float residue only; black keeps its declared (possibly tinted) lift.
    out[-1] = 1.0
    return out


if __name__ == "__main__":
    identity = {
        "name": "Identity",
        "tone": {"domain": "code", "points": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]},
        "gamut_knee": None,  # exact projection path for the identity check
    }
    t = np.linspace(0, 1, 33)
    b, g, r = np.meshgrid(t, t, t, indexing="ij")
    grid = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)
    out = generate_look_table(identity, 33)
    err = np.max(np.abs(out - grid))
    print(f"identity max abs err {err:.3e}")
    assert err < 1e-7
    print("pipeline.py identity check OK")
