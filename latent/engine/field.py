"""[S] global chroma + [P] perceptual field (ENGINE_SPEC §3, amended by v1.2).

Four 12-knot hue tables (`DH10`, `DH18`, `CR`, `DL`), interpolated by a periodic
cubic spline, plus a chroma-vs-lightness arch, a skin protocol and a list of
Gaussian-window local ops — and, since v1.2 W3, a separate **stage [S]** that
carries the look's global saturation / vibrance and runs on the INPUT colour
before [N].

**Stage-0 invariant (load-bearing).**  Every weight, window and gate here is
computed from the ORIGINAL input colour's OKLCh ``(L0, C0, h0)``, never from the
current colour.  Input grey has ``C0 = 0`` exactly, so every chromatic weight is
exactly 0 and the grey axis is structurally invariant through [S] and [P];
nothing feeds back.

Hue rotation is applied as a 2x2 rotation of ``(a, b)``, never by adding to an
angle and re-wrapping.

What v1.2 changed here
----------------------
* **W1** — v1.1 R1's headroom (``nm0``-driven S5 fades on chroma / rotation /
  lightness) is withdrawn and gone.  ``field.headroom`` is accepted from an old
  look file, recorded and ignored, and :func:`engine.spec.validate` warns.
* **W2** — the one gamut-aware fade that remains is *linear in gamut-relative
  chroma*: ``g_eff = g - kappa*soft_relu(g-1)*r0``, ``r0 = soft_saturate(C0/CMAX(L0,h0))``
  (W2 writes ``min(., 1)``; :data:`engine.cmax.R0_WIDTH` says why the clamp has
  a C2 shoulder, and ``d(s*r0)/ds <= 2`` still holds exactly).  Along a ray of
  constant ``(L0, h0)`` the map is ``C*g - kappa*soft_relu(g-1)*CMAX*s*r0(s)``,
  whose derivative ``g - kappa*soft_relu(g-1)*d(s*r0)/ds >= 2 - g`` is positive
  for every ``g < 2`` and for ANY shape of CMAX — monotone by construction — and
  at ``s >= 1`` (kappa = 1) the output chroma equals the input chroma, so a
  colour on the shell stays on the shell and no chroma gain can push anything
  out.  That last claim needs ``CMAX <= the true boundary``, which is what
  :func:`engine.cmax._build`'s cap guarantees.
* **W3** — ``field.sat`` / ``field.vibrance`` are still the look-file keys but
  are executed in [S] (ungated, on the input colour).  The ``iso`` gate is gone.
* **W4** — the DL table and the ops' ``dl`` have their own gate ``lgt``; they
  used to share ``chr``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from . import cmax as _cmax, curves
from .color import FloatArray
from .spec import LookSpec, Op

__all__ = [
    "CompiledField", "Stage0", "compile_field", "stage0", "apply_field", "apply_sat",
    "delta_h", "delta_l", "chroma_gain", "sat_gain", "relative_fade",
    "arch_range", "total_gain_range", "sat_gain_range", "fold_bound",
    "chroma_radial_bound", "probe_code",
    "ARCH_CENTERS", "ARCH_SIGMA", "ARCH_L", "RELFADE_WIDTH",
]

#: hue families of the chroma arch and their window width (ENGINE_SPEC §3.4)
ARCH_CENTERS = (55.0, 145.0, 250.0)
ARCH_SIGMA = 45.0
#: the look gives 5 values here; outside [0.25, 0.85] the curve is flat
ARCH_L = (0.25, 0.40, 0.55, 0.70, 0.85)
ARCH_NORM_L = 0.65
#: skin chroma knots (dark / mid / bright)
SKINC_L = (0.50, 0.68, 0.86)
#: lightness-shape normaliser of §3.5
L_SHAPE_NORM = 0.91
#: lightness ramp of the rotation's exposure scaling (§3.3)
ROT_L_RAMP = (0.35, 0.80)

#: width of the C^2 rectifier that replaces W2's ``sp(g-1, 0.02)``.
#:
#: W2 fades ``kappa*sp(g-1, 0.02)*r0``, but ``sp(0, 0.02) = 0.013863`` is not 0,
#: so an identity look (every gain exactly 1) would lose 1.3863 % of its chroma
#: — measured 22.56 codes on the sRGB primaries, against a 1e-6 gate — and a
#: desaturating gain would be faded too, which W2 forbids in the same
#: paragraph.  :func:`engine.curves.soft_relu` is the rectifier that is exactly
#: 0 for ``g <= 1`` and exactly ``g-1`` for ``g >= 1 + RELFADE_WIDTH``, so all
#: four of W2's properties hold exactly.  0.08 keeps its curvature
#: (4.96/w = 62) an order of magnitude under what the lattice's d2 budget
#: notices at the gains this set carries (measured in the W5/W7 tables of
#: ``work.nosync/review/m2_engine_report.json``), while leaving every gain from
#: 1 + w up faded at full strength.
#:
#: Chosen by measurement (report key ``w5_relfade_width``): a gain inside
#: ``(1, 1+w)`` is only *partly* faded, so it still pushes a shell colour out.
#: At the engine defaults, identity + sat 1.05 measures ``lim`` / interior d2
#: 1.1931 / 2.25 at w = 0.02 and w = 0.04 (bit-identical to each other in every
#: column measured, on identity+sat and on 03Gilt), 1.2825 / 2.82 at w = 0.08
#: and 1.3789 / 3.45 at w = 0.16.  0.04 is the larger of the two that costs
#: nothing, i.e. half the curvature (4.96/w) of 0.02 for the same result.
RELFADE_WIDTH = 0.04

_HUE_KNOTS = np.arange(12) * 30.0


def _periodic(values) -> CubicSpline:
    """C^2 periodic cubic spline through 12 knots at h = 0, 30, ..., 330."""
    y = np.asarray(values, dtype=np.float64)
    x = np.concatenate([_HUE_KNOTS, [360.0]])
    return CubicSpline(x, np.concatenate([y, y[:1]]), bc_type="periodic")


def _arch_spline(values) -> CubicSpline:
    """C^2 cubic through the 5 look values, clamped (zero end slope).

    Outside ``[0.25, 0.85]`` the curve is flat, which :func:`_arch` gets by
    clipping ``L0`` into that range.  The first draft instead *added* knots at
    0.08 and 0.97 carrying the end values and asked for zero end slope there;
    with C^2 continuity at 0.25 that made the [0.08, 0.25] segment dip below the
    declared value (measured: Glaze warm declared 0.920, actual range
    0.9113..0.9200; Arcade warm declared 0.800, actual 0.7861), i.e. exactly not
    flat.  With zero end slope at 0.25/0.85 the join to the constant is C^1.
    """
    v = np.asarray(values, dtype=np.float64)
    return CubicSpline(np.asarray(ARCH_L), v, bc_type=((1, 0.0), (1, 0.0)))


@dataclass(frozen=True)
class CompiledField:
    spec: LookSpec
    dh10: CubicSpline
    dh18: CubicSpline
    cr: CubicSpline
    dl: CubicSpline
    arch: tuple[CubicSpline, CubicSpline, CubicSpline]
    arch_at_norm: tuple[float, float, float]


def compile_field(spec: LookSpec) -> CompiledField:
    f = spec.field
    arch = (_arch_spline(f.arch.warm), _arch_spline(f.arch.green), _arch_spline(f.arch.blue))
    return CompiledField(
        spec=spec,
        dh10=_periodic(f.dh10),
        dh18=_periodic(f.dh18),
        cr=_periodic(f.cr),
        dl=_periodic(f.dl),
        arch=arch,
        arch_at_norm=tuple(float(a(ARCH_NORM_L)) for a in arch),
    )


# ---------------------------------------------------------------------------
# stage 0
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage0:
    """Every weight of every stage, computed once from the input colour.

    ``r0`` is W2's gamut-relative chroma ``soft_saturate(C0/CMAX(L0, h0))`` — the only
    gamut-aware quantity left in [S] and [P] after W1 withdrew the headroom.  It
    needs nothing but ``(L0, h0)``, so the validation probes of §3.7 see exactly
    the same fade the lattice does (v1.1's ``nm0`` needed the input *code* and
    the probes therefore measured an unfaded operator).
    """

    L0: FloatArray
    C0: FloatArray
    h0: FloatArray
    g10: FloatArray
    g18: FloatArray
    gch: FloatArray
    glgt: FloatArray
    grd: FloatArray
    w_skin: FloatArray
    wins: tuple[FloatArray, ...]
    r0: FloatArray


def _skin_weight(sk, L0, C0, h0) -> FloatArray:
    f_lo, c_lo, c_hi, f_hi = sk.window
    # evaluate on a branch cut opposite the window so a window straddling 0 deg
    # still works (and the result is identical for a non-wrapping window)
    centre = 0.5 * (c_lo + c_hi)
    hh = centre + curves.hue_delta(h0, centre)
    w_h = curves.S5(f_lo, c_lo, hh) * (1.0 - curves.S5(c_hi, f_hi, hh))
    w_c = curves.S5(sk.c_gate[0], sk.c_gate[1], C0) * (
        1.0 - curves.S5(sk.c_fade[0], sk.c_fade[1], C0)
    )
    lg = sk.l_gate
    w_l = curves.S5(lg[0], lg[1], L0) * (1.0 - curves.S5(lg[2], lg[3], L0))
    return w_h * w_c * w_l


def _op_window(op: Op, L0, C0, h0, w_skin) -> FloatArray:
    w = curves.gauss_hue(h0, op.center, op.sigma)
    w = w * curves.S5(op.c_gate[0], op.c_gate[1], C0)
    if op.c_fade is not None:
        w = w * (1.0 - curves.S5(op.c_fade[0], op.c_fade[1], C0))
    if op.l_band is not None:
        lo0, lo1, hi0, hi1 = op.l_band
        w = w * curves.S5(lo0, lo1, L0) * (1.0 - curves.S5(hi0, hi1, L0))
    return w * (1.0 - (1.0 - op.skin_residual) * w_skin)


def relative_fade(g: FloatArray, r0: FloatArray, kappa: float) -> FloatArray:
    """W2's fade: ``g_eff = g - kappa*soft_relu(g-1, RELFADE_WIDTH)*r0``.

    * ``g <= 1`` -> untouched (a desaturation is never faded);
    * ``g >= 1 + RELFADE_WIDTH`` and ``C0 >= CMAX`` (on the shell), ``kappa = 1``
      -> ``r0 = 1`` and ``g_eff = 1`` exactly: the colour keeps its chroma;
    * along a ray of constant ``(L0, h0)``, with ``s = C/CMAX``,
      ``d(C*g_eff)/dC = g - kappa*soft_relu(g-1)*d(s*r0(s))/ds`` (for ``g``
      constant) and ``d(s*r0)/ds <= 2`` by construction
      (:func:`engine.cmax.soft_saturate`), so it is ``>= 2 - g > 0`` for every
      ``g < 2`` — exactly W2's bound, shoulder or no shoulder.
    """
    return g - float(kappa) * curves.soft_relu(g - 1.0, RELFADE_WIDTH) * r0


def stage0(cf: CompiledField, L0, C0, h0) -> Stage0:
    spec = cf.spec
    gt = spec.field.gates
    w_skin = _skin_weight(spec.skin, L0, C0, h0)
    wins = tuple(_op_window(op, L0, C0, h0, w_skin) for op in spec.ops)
    return Stage0(
        L0=L0, C0=C0, h0=h0,
        g10=curves.S5(gt.rot10[0], gt.rot10[1], C0),
        g18=curves.S5(gt.rot18[0], gt.rot18[1], C0),
        gch=curves.S5(gt.chr[0], gt.chr[1], C0),
        glgt=curves.S5(gt.lgt[0], gt.lgt[1], C0),
        grd=curves.S5(gt.shadow[0], gt.shadow[1], L0),
        w_skin=w_skin,
        wins=wins,
        r0=_cmax.relative_chroma(L0, C0, h0),
    )


# ---------------------------------------------------------------------------
# [S] — global saturation / vibrance on the INPUT colour (v1.2 W3)
# ---------------------------------------------------------------------------


def sat_gain(cf: CompiledField, st: Stage0, *, fade: bool = True) -> FloatArray:
    """``g_S = sat * vib(C0)``, with W2's relative fade (W3).

    No gate: a chroma scale about the grey axis is linear near the axis, so it
    adds no curvature, and an input grey (``C0 = 0``) is untouched — the grey
    axis is still exactly ``T``.  v1.1 measured the narrow ``S5(0, 0.06)``
    ``iso`` gate on a gain of 1.3-1.4 as one of the largest curvature sources in
    the whole engine; that gate is deleted.
    """
    fl = cf.spec.field
    vc_lo, vc_hi = fl.vibrance.c
    vib = 1.0 + (fl.vibrance.gain - 1.0) * (1.0 - curves.S5(vc_lo, vc_hi, st.C0))
    g = fl.sat * vib
    if fade:
        g = relative_fade(g, st.r0, fl.relfade)
    return g


def is_identity_sat(cf: CompiledField) -> bool:
    """``sat == 1`` and ``vibrance.gain == 1`` -> [S] is exactly the identity."""
    fl = cf.spec.field
    return fl.sat == 1.0 and fl.vibrance.gain == 1.0


def apply_sat(cf: CompiledField, lab: FloatArray, st: Stage0) -> FloatArray:
    """[S]: scale ``(a, b)`` by ``g_S``; ``L`` and the hue are untouched."""
    if is_identity_sat(cf):
        return lab
    g = sat_gain(cf, st)[..., None]
    return np.concatenate([lab[..., :1], lab[..., 1:] * g], axis=-1)


# ---------------------------------------------------------------------------
# [P] — the operators
# ---------------------------------------------------------------------------


def delta_h(cf: CompiledField, st: Stage0) -> FloatArray:
    """Total hue rotation in degrees, incl. the shadow guard.

    v1.2 W1: rotation is no longer faded by anything gamut-related.  v1.1's
    ``hr_r`` was a hue-DEPENDENT rotation amplitude (``nm0`` depends on how wide
    the gamut is at that hue), and a flat 4 deg rotation folded once multiplied
    by it.
    """
    spec = cf.spec
    sk = spec.skin
    rl = spec.field.rot_l_scale
    rot_l = rl[0] + (rl[1] - rl[0]) * curves.S5(ROT_L_RAMP[0], ROT_L_RAMP[1], st.L0)
    d10 = cf.dh10(st.h0)
    d18 = cf.dh18(st.h0)
    dh = (d10 * st.g10 + (d18 - d10) * st.g18) * rot_l * (
        1.0 - (1.0 - sk.hue_residual) * st.w_skin
    )
    for op, win in zip(spec.ops, st.wins):
        amt = op.dh[0] + (op.dh[1] - op.dh[0]) * curves.S5(op.dh_l[0], op.dh_l[1], st.L0)
        dh = dh + amt * win
    dh = dh + (-sk.pull * curves.hue_delta(st.h0, sk.center) + sk.hue_offset) * st.w_skin
    return dh * st.grd


def _arch(cf: CompiledField, st: Stage0) -> FloatArray:
    wf = np.stack(
        [curves.gauss_hue(st.h0, c, ARCH_SIGMA) for c in ARCH_CENTERS], axis=0
    )
    wf = wf / np.sum(wf, axis=0, keepdims=True)
    Lc = np.clip(st.L0, ARCH_L[0], ARCH_L[-1])   # flat outside the declared range
    num = sum(wf[i] * cf.arch[i](Lc) for i in range(3))
    den = sum(wf[i] * cf.arch_at_norm[i] for i in range(3))
    return num / den


def _skinc(sk, L0) -> FloatArray:
    c_d, c_m, c_b = sk.chroma
    return (
        c_d
        + (c_m - c_d) * curves.S5(SKINC_L[0], SKINC_L[1], L0)
        + (c_b - c_m) * curves.S5(SKINC_L[1], SKINC_L[2], L0)
    )


def chroma_gain(cf: CompiledField, st: Stage0, *, fade: bool = True) -> FloatArray:
    """[P]'s hue-dependent chroma gain ``g`` before the shadow guard (§3.4).

    ``sat`` and ``vibrance`` are NOT here any more (W3 moved them to [S]); what
    is left is ``CR*ARCH`` x the ops x the skin curve.  W2's relative fade is
    applied to that total, exactly as it is applied to [S]'s own total.
    ``fade=False`` returns the raw §3.4 product — what §3.7's budget scan checks.
    """
    spec = cf.spec
    sk = spec.skin
    cra = cf.cr(st.h0) * _arch(cf, st)
    g = 1.0 + (cra - 1.0) * st.gch * (1.0 - (1.0 - sk.chroma_residual) * st.w_skin)
    for op, win in zip(spec.ops, st.wins):
        if op.gain_c != 1.0:
            g = g * (1.0 + (op.gain_c - 1.0) * win)
    g = g * (1.0 + (_skinc(sk, st.L0) - 1.0) * st.w_skin)
    if fade:
        g = relative_fade(g, st.r0, spec.field.relfade)
    return g


def delta_l(cf: CompiledField, st: Stage0) -> FloatArray:
    """Total OKLab lightness offset amplitude (before the 4L(1-L) shape, §3.5).

    W4 gives the DL table and the ops' ``dl`` their own gate ``lgt``: sharing
    ``chr`` meant a ΔL of 0.04 switched on across 0.06 of chroma, which measured
    3-10 codes of second difference by itself.
    """
    spec = cf.spec
    sk = spec.skin
    d = cf.dl(st.h0) * st.glgt * (1.0 - (1.0 - sk.l_residual) * st.w_skin)
    for op, win in zip(spec.ops, st.wins):
        if op.dl != 0.0:
            d = d + op.dl * win
    return d + sk.l_lift * st.w_skin


def apply_field(cf: CompiledField, lab: FloatArray, st: Stage0) -> FloatArray:
    """Run [P] on OKLab ``lab`` (shape (..., 3)); weights come from ``st``."""
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # 3.3 rotation -- as a 2x2 rotation of (a, b)
    ang = np.radians(delta_h(cf, st))
    ca = np.cos(ang)
    sa = np.sin(ang)
    a2 = ca * a - sa * b
    b2 = sa * a + ca * b

    # 3.4 chroma
    g = chroma_gain(cf, st)
    f = 1.0 + (g - 1.0) * st.grd
    a3 = a2 * f
    b3 = b2 * f

    # 3.4 chroma caps: soft tanh ceiling inside a hue window.
    #
    # The *trigger* is the current chroma (a ceiling has to be), but the WINDOW
    # is stage-0 like every other op in [P]: hue window x stage-0 chroma gate x
    # shadow guard.  Without the chroma gate the cap fired on the tinted grey
    # that [N] hands to [P] in the NPG order and broke `grey == T` by 2.39 codes
    # (gate: 0.02); without `grd` it cut 21.7 % of the chroma at L0 = 0.05,
    # where every other ab-modifying op is identically zero.
    for cp in cf.spec.caps:
        span = cp.cap - cp.start
        C = np.hypot(a3, b3)
        # v1.1 R3's `center: null` is the neon protection — one chroma knee for
        # every hue — so it drops the hue window and keeps the chroma gate and
        # the shadow guard alone.  (v1.1 used the `iso` gate there; W3 deleted
        # `iso`, and the cap's own validated `c_gate` replaces it.  Both are
        # exactly 0 on the grey axis, which is what the NPG order's tinted grey
        # needs.)
        win = curves.S5(cp.c_gate[0], cp.c_gate[1], st.C0) * st.grd
        if not cp.hue_independent:
            win = win * curves.gauss_hue(st.h0, cp.center, cp.sigma)
        kneed = cp.start + span * np.tanh(np.maximum(C - cp.start, 0.0) / span)
        Cn = np.where(C > cp.start, C + (kneed - C) * win, C)
        scale = np.where(C > 1e-12, Cn / np.maximum(C, 1e-12), 1.0)
        a3 = a3 * scale
        b3 = b3 * scale

    # 3.5 lightness -- shape from the CURRENT L so the result stays in (0,1)
    L2 = L + delta_l(cf, st) * (4.0 * L * (1.0 - L) / L_SHAPE_NORM)
    return np.stack([L2, a3, b3], axis=-1)


# ---------------------------------------------------------------------------
# validation helpers (ENGINE_SPEC §3.7)
# ---------------------------------------------------------------------------


def _probe_stage0(cf: CompiledField, L, C, h) -> Stage0:
    return stage0(cf, np.asarray(L, dtype=np.float64),
                  np.asarray(C, dtype=np.float64),
                  np.asarray(h, dtype=np.float64))


def arch_range(cf: CompiledField) -> tuple[float, float, tuple[float, float]]:
    """min / max of ``CR(h0) * ARCH(L0, h0)`` over the lattice's (h0, L0) range."""
    h = np.arange(0.0, 360.0, 1.0)
    Lg = np.linspace(0.08, 0.97, 91)
    H, L = np.meshgrid(h, Lg, indexing="ij")
    st = _probe_stage0(cf, L, np.full_like(L, 0.18), H)
    val = cf.cr(H) * _arch(cf, st)
    i = np.unravel_index(int(np.argmin(val)), val.shape)
    j = np.unravel_index(int(np.argmax(val)), val.shape)
    worst = i if abs(val[i] - 1.0) > abs(val[j] - 1.0) else j
    return float(val.min()), float(val.max()), (float(H[worst]), float(L[worst]))


#: (h0, L0, C0) scan of :func:`total_gain_range`
GAIN_SCAN_H = np.arange(0.0, 360.0, 4.0)
GAIN_SCAN_L = np.linspace(0.05, 0.97, 24)
GAIN_SCAN_C = np.linspace(0.0, 0.33, 34)


def total_gain_range(cf: CompiledField) -> tuple[float, float, tuple[float, float, float]]:
    """min / max of [P]'s FULL chroma gain over ``(h0, L0, C0)``.

    ``CR*ARCH * op gains * SKINC``, each with its own gate — the quantity W2's
    ``total g < 1.9`` budget is about.  Measured WITHOUT the relative fade: the
    fade can only ever shrink a boost, so the raw product is the conservative
    number to budget, and it is the number a colourist reads off the tables.
    """
    H, L, C = np.meshgrid(GAIN_SCAN_H, GAIN_SCAN_L, GAIN_SCAN_C, indexing="ij")
    st = _probe_stage0(cf, L, C, H)
    g = chroma_gain(cf, st, fade=False)
    i = np.unravel_index(int(np.argmax(np.abs(g - 1.0))), g.shape)
    return float(g.min()), float(g.max()), (float(H[i]), float(L[i]), float(C[i]))


def sat_gain_range(cf: CompiledField) -> tuple[float, float]:
    """min / max of [S]'s ``sat * vib(C0)`` over the reachable chroma range."""
    C = np.linspace(0.0, 0.33, 331)
    st = _probe_stage0(cf, np.full_like(C, 0.5), C, np.zeros_like(C))
    g = sat_gain(cf, st, fade=False)
    return float(np.min(g)), float(np.max(g))


def probe_code(L, C, h) -> tuple[FloatArray, FloatArray]:
    """OKLCh -> (sRGB code, in-gamut mask).  Engine-local, so the validation
    probes can build a real colour without importing ``tools.metrics``
    (engine must not depend on tools)."""
    from . import color as _color, xfer as _xfer   # noqa: PLC0415 - avoid a cycle

    L = np.asarray(L, dtype=np.float64)
    a = C * np.cos(np.radians(h))
    b = C * np.sin(np.radians(h))
    lin = _color.oklab_to_linear_srgb(np.stack([L, a, b], axis=-1))
    ok = np.all((lin >= -5e-4) & (lin <= 1.0 + 5e-4), axis=-1)
    return _xfer.signed_srgb_encode(lin), ok


def _in_gamut(L, C, h) -> FloatArray:
    """The mask half of :func:`probe_code`, without building the code array.

    :func:`chroma_radial_bound` evaluates this on ~1.2e6 probe points and never
    looks at the code; skipping the signed encode there is ~40 % of the probe.
    """
    from . import color as _color   # noqa: PLC0415 - avoid a cycle

    L = np.asarray(L, dtype=np.float64)
    lin = _color.oklab_to_linear_srgb(np.stack(
        [L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))], axis=-1))
    return np.all((lin >= -5e-4) & (lin <= 1.0 + 5e-4), axis=-1)


def fold_bound(cf: CompiledField, c_probe, l_probe, step: float = 1.0):
    """min over the probe grid of ``1 + dDh/dh0`` — the closed-form fold test.

    ``Dh(h0)`` is sampled on a ``step``-degree grid at every (C0, L0) probe (the
    probe grid spans both inside and outside the skin gates, because ``w_c``
    fades skin out by C0 = 0.24), and the discrete forward difference of the map
    ``h0 -> h0 + Dh`` must stay positive with margin.

    v1.2 W1 removed the one term that made this probe optimistic (v1.1's
    ``hr_r``, whose ``nm0`` needed the actual colour): the rotation is now a
    function of ``(h0, L0, C0)`` alone, so this probe sees exactly what the
    lattice sees.
    """
    h = np.arange(0.0, 360.0, step)
    best = np.inf
    where = (0.0, 0.0, 0.0)
    half = 0.5 * step
    for C0 in c_probe:
        for L0 in l_probe:
            hs = np.concatenate([h - half, h + half]) % 360.0
            LL = np.full_like(hs, L0)
            CC = np.full_like(hs, C0)
            st = _probe_stage0(cf, LL, CC, hs)
            d = delta_h(cf, st)
            lo, hi = d[: h.size], d[h.size:]
            jac = 1.0 + (hi - lo) / step
            k = int(np.argmin(jac))
            if jac[k] < best:
                best = float(jac[k])
                where = (float(h[k]), float(C0), float(L0))
    return best, where


#: grid of :func:`chroma_radial_bound`.
#:
#: **Densified after a verifier escape.**  The first draft sampled hue every
#: 15 deg at 6 fixed lightnesses, and the same failure mode the codebase already
#: documents for the ROTATION probe (see ``spec.FOLD_PROBE_C``: "the sparse
#: probe would have shipped a look that still folds") applied here too:
#: ``LookSpec(field=Field(sat=1.0, cr=(0.72,)*12))`` passed ``compile(strict=True)``
#: with a reported bound of **+0.1780** while the real composite chroma map runs
#: BACKWARDS at -0.1787 (h0 = 38 deg, L0 = 0.635, C = 0.204) and its shipped
#: 33**3 table carries 332 material folds, 266 micro folds, min ratio -0.0679.
#: Two more of the same shape: sat 1.15 / cr 0.78 -> accepted, real -0.0531,
#: 3 material folds; sat 1.45 / cr 0.85 -> accepted, real -0.0347, 10 folds.
#:
#: Attribution, by refining one axis at a time on sat 1.0 / cr 0.72: shipped
#: grid +0.1780; 10x finer in C +0.1766 (no effect); finer in L -0.1278; 1 deg
#: in H -0.1784.  The violation sits at h0 = 38-40 deg, inside the skin window's
#: rising feather (20 -> 38), which a 15 deg hue grid steps straight over
#: between its 30 and 45 samples.  2 deg x 20 lightnesses finds -0.1784, the
#: same number the 1 deg x 30 grid does.
RADIAL_H = np.arange(0.0, 360.0, 2.0)
RADIAL_L = tuple(np.linspace(0.12, 0.95, 20))
RADIAL_C = np.linspace(1e-4, 0.34, 341)


def chroma_radial_bound(cf: CompiledField):
    """min of ``d(C_out)/dC`` along in-gamut rays — the chroma fold test.

    Every chroma gain in [S] and [P] is a function of the stage-0 chroma, so the
    composite chroma map along a ray of constant ``(L0, h0)`` is

        ``C_out(C) = C * g_S_eff(C) * (1 + (g_eff(C) - 1) * grd)``

    and it stays injective only while ``dC_out/dC > 0``.  W2 keeps this probe as
    an ERROR below 0.15 — the fade it measures is now the v1.2 one.

    Vectorised over (h0, C) one lightness at a time: the grid is 25x denser than
    the first draft's (see :data:`RADIAL_H`) and costs ~0.09 s, against 0.03 s
    for the sparse grid evaluated ray by ray.

    Returns ``(min slope, (h0, L0, C0))``.
    """
    C = RADIAL_C
    H, CC = np.meshgrid(RADIAL_H, C, indexing="ij")
    best = np.inf
    where = (0.0, 0.0, 0.0)
    for L0 in RADIAL_L:
        LL = np.full_like(CC, float(L0))
        ok = _in_gamut(LL, CC, H)
        st = _probe_stage0(cf, LL, CC, H)
        gs = sat_gain(cf, st)
        g = chroma_gain(cf, st)
        Cout = CC * gs * (1.0 + (g - 1.0) * st.grd)
        keep = ok[:, :-1] & ok[:, 1:]
        if not keep.any():
            continue
        slope = np.where(keep, np.diff(Cout, axis=1) / np.diff(C), np.inf)
        k = np.unravel_index(int(np.argmin(slope)), slope.shape)
        if slope[k] < best:
            best = float(slope[k])
            where = (float(RADIAL_H[k[0]]), float(L0), float(C[k[1]]))
    return best, where
