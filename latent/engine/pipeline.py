"""The engine: compile a look, apply it, sample it to a 33-point .cube.

::

    rgb_in --> stage 0: (L0, C0, h0, r0), all weights
           --> [F] film front stage        (milestone 2; default OFF = identity)
           --> [S] global chroma           sat * vibrance on the INPUT colour (v1.2 W3)
           --> [N] neutral stage           per-channel tone + designed tint
           --> [P] perceptual field        OKLCh, table-driven
           --> [G] gamut compression       code domain, luma-anchored p-norm + limit curve
           --> final clip                  (must be ~no-op; diagnostics report how much it did)

``order`` is a look-level option: ``"NPG"`` (default, above, i.e. S->N->P->G) or
``"PGN"`` (S->P->G->N, the order R06 prototyped).  Both compose the same stage
functions.  Mono looks take the §6 branch, which skips [S], [P] and [G].

There is exactly ONE ``np.clip`` on colour values in the whole path — the last
line of :func:`apply`.  Everything before it is structurally bounded, and
:func:`diagnostics` reports how far outside [0,1] the pre-clip values went.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import color, field as _field, gamut as _gamut, mono as _mono, neutral as _neutral, xfer
from .color import FloatArray
from .cubeio import LUT3D, write_cube
from .spec import LookSpec, SpecError, load_look, validate

__all__ = [
    "Compiled", "compile", "apply", "lattice", "write_cube_file", "blend",
    "diagnostics", "load_look", "grey_response", "target_grey", "grid_error",
    "grey_axis_error", "grey_axis_error_of", "push_stats", "gamut_overshoot",
    "BYPASS_TAG",
]

#: prefix of the warning ``compile(strict=False)`` leaves behind; downstream
#: refuses to ship anything carrying it.
BYPASS_TAG = "VALIDATION BYPASSED"


#: the lattice the [G] limit is measured on (v1.1 R2) — the shipped one
LIM_SIZE = 33


@dataclass(frozen=True)
class Compiled:
    spec: LookSpec
    neutral: _neutral.CompiledNeutral
    field: _field.CompiledField
    mono_dl: object | None
    warnings: tuple[str, ...]
    #: [G]'s measured limit (v1.1 R2).  ``None`` = the measuring pass itself, in
    #: which [G] is a pass-through; every compiled look has a number here.
    gamut_lim: float | None = None
    #: max pre-gamut ``nm`` over the 33**3 lattice, i.e. what ``gamut_lim`` is from
    nm_pre_max: float | None = None
    #: fraction of the 33**3 lattice whose [G] radial gain is below
    #: ``gamut.MIN_RADIAL_GAIN``.  W5.2's slope floor bounds that gain below by
    #: ``s1/q``, so this is 0 or 100 % of the compressed population, not a tail.
    gamut_flat_frac: float | None = None
    #: which grey [G] compresses toward — "luma" (v1.2 W5.1) or "oklab"
    #: (v1.0/v1.1).  A look does not choose this; the W5.1 study does.
    anchor: str = _gamut.DEFAULT_ANCHOR

    @property
    def gamut_min_radial_gain(self) -> float | None:
        """``s1_eff/q`` — the compressor's floor on the radial gain (W5.2).

        ``s1_eff`` is :func:`engine.gamut.effective_end_slope`, which raises the
        declared ``end_slope`` on a weakly-pushing look so the curve's curvature
        at ``lim`` stays bounded.
        """
        q = self.gamut_q
        if q is None or q <= 1.0:
            return 1.0 if q is not None else None
        s1 = _gamut.effective_end_slope(self.spec.gamut.knee, self.gamut_lim,
                                        self.spec.gamut.end_slope)
        return float(s1 / q)

    @property
    def order(self) -> str:
        return self.spec.order

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def gamut_q(self) -> float | None:
        """``q = (lim - knee)/(1 - knee)``; ``<= 1`` means [G] is the identity."""
        if self.spec.gamut is None or self.gamut_lim is None:
            return None
        return (self.gamut_lim - self.spec.gamut.knee) / (1.0 - self.spec.gamut.knee)


def _measure_limit(c: Compiled) -> Compiled:
    """The second pass of v1.1 R2: run the 33**3 lattice with [G] switched off,
    take the largest pre-gamut ``nm``, and pin the compressor's limit to it.

    The pre-gamut colour does not depend on the limit, so one extra lattice
    evaluation (~20 ms) is all this costs, and it is what makes the compressor's
    tail land exactly on the gamut shell instead of on an arbitrary tanh
    asymptote.

    The same pass measures ``gamut_flat_frac``: the fraction of the lattice the
    compressor puts on its FLAT TAIL (radial gain < ``gamut.MIN_RADIAL_GAIN``).
    R2 says the curve has "no collapse except at the single extreme node"; that
    is true of ``q`` near 1 and false of a look that pushes hard, so the number
    is measured here instead of assumed (``spec.validate`` warns on it).
    """
    if c.spec.gamut is None or c.spec.mono is not None:
        # a mono look takes the §6 branch, which never calls [G]; validate()
        # warns that a gamut block on a mono look is inert.
        return c
    t = np.linspace(0.0, 1.0, LIM_SIZE)
    b, g, r = np.meshgrid(t, t, t, indexing="ij")
    _, nm = _run(c, np.stack([r, g, b], axis=-1))      # c.gamut_lim is None here
    nm_max = float(np.max(nm))
    lim = _gamut.measured_limit(c.spec.gamut.knee, nm_max)
    rg = _gamut.radial_gain(nm, c.spec.gamut.knee, lim, c.spec.gamut.end_slope)
    flat = float(np.count_nonzero(rg < _gamut.MIN_RADIAL_GAIN) / rg.size)
    return replace(c, gamut_lim=lim, nm_pre_max=nm_max, gamut_flat_frac=flat)


def _compile_unchecked(spec: LookSpec) -> Compiled:
    """Compile without validating — for ``spec.validate`` itself, which needs to
    run the pipeline on the grey ramp before it can pass judgement."""
    return _measure_limit(Compiled(
        spec=spec,
        neutral=_neutral.compile_neutral(spec.neutral),
        field=_field.compile_field(spec),
        mono_dl=_mono.compile_mono(spec) if spec.mono is not None else None,
        warnings=(),
    ))


def grey_axis_error(spec: LookSpec) -> tuple[float, float, str]:
    """``(max |out - T| in codes, the t where it happens, the channel)``.

    ENGINE_SPEC §8 gates this at 0.02 codes.  It is *measured*, not inferred:
    [P]'s stage-0 invariant makes the grey axis exact through [P], but [G] in
    the NPG order sees the grey that [N] has already tinted, and a chroma cap or
    a low knee can compress that tint away (see ``engine/gamut.py``).
    """
    return grey_axis_error_of(_compile_unchecked(spec))


def grey_axis_error_of(c: Compiled) -> tuple[float, float, str]:
    """:func:`grey_axis_error` on an already-compiled look — so ``validate`` can
    take the grey axis and the [G] flatness off ONE compile."""
    t, out = grey_response(c)
    err = np.abs(out - c.neutral.T) * 255.0
    k = int(np.argmax(err.max(axis=1)))
    return float(err.max()), float(t[k]), "RGB"[int(np.argmax(err[k]))]


def compile(spec: LookSpec, *, strict: bool = True) -> Compiled:  # noqa: A001
    """Validate and compile a look.

    ``strict=False`` downgrades validation errors to warnings — used only to
    *measure* a look that a gate rejects, so the milestone report can quote real
    numbers next to the diagnosis.  Never use it to ship a LUT.
    """
    try:
        warnings = list(validate(spec))
    except SpecError as exc:
        if strict:
            raise
        warnings = [f"{BYPASS_TAG} (strict=False): {exc}"]
    return _measure_limit(Compiled(
        spec=spec,
        neutral=_neutral.compile_neutral(spec.neutral),
        field=_field.compile_field(spec),
        mono_dl=_mono.compile_mono(spec) if spec.mono is not None else None,
        warnings=tuple(warnings),
    ))


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def _run(c: Compiled, rgb: FloatArray) -> tuple[FloatArray, FloatArray | None]:
    """Everything except the final clip.  Returns (pre-clip code, pre-gamut nm)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lin0 = xfer.signed_srgb_decode(rgb)
    lab0 = color.linear_srgb_to_oklab(lin0)
    L0, C0, h0 = color.oklab_to_lch(lab0)
    st = _field.stage0(c.field, L0, C0, h0)

    if c.spec.mono is not None:
        return _mono.apply_mono(c.mono_dl, c.field, c.neutral, lin0, st), None

    # [S] v1.2 W3: the look's global saturation / vibrance, ungated, on the
    # stage-0 colour.  `apply_sat` returns `lab0` itself when sat == vib == 1,
    # so an identity look never pays the round trip (and stays bit-exact).
    lab_s = _field.apply_sat(c.field, lab0, st)
    code_s = rgb if lab_s is lab0 else xfer.signed_srgb_encode(
        color.oklab_to_linear_srgb(lab_s))

    # [F] milestone-2 slot: with film == None this is the identity.
    if c.order == "NPG":
        v = _neutral.apply_neutral(c.neutral, code_s, st.C0, st.w_skin)
        lab = color.linear_srgb_to_oklab(xfer.signed_srgb_decode(v))
        lab = _field.apply_field(c.field, lab, st)
        code = xfer.signed_srgb_encode(color.oklab_to_linear_srgb(lab))
        code, nm = _gamut.apply_gamut(c.spec.gamut, code, lab[..., 0], c.gamut_lim,
                                      kind=c.anchor)
    else:  # "PGN"
        lab = _field.apply_field(c.field, lab_s, st)
        code = xfer.signed_srgb_encode(color.oklab_to_linear_srgb(lab))
        code, nm = _gamut.apply_gamut(c.spec.gamut, code, lab[..., 0], c.gamut_lim,
                                      kind=c.anchor)
        code = _neutral.apply_neutral(c.neutral, code, st.C0, st.w_skin)
    return code, nm


def apply(c: Compiled, rgb: FloatArray) -> FloatArray:
    """Apply a compiled look to sRGB code values, shape (..., 3)."""
    out, _ = _run(c, rgb)
    return np.clip(out, 0.0, 1.0)


def lattice(c: Compiled, n: int = 33) -> FloatArray:
    """The full lattice as ``(n, n, n, 3)`` in cubeio's (B, G, R, 3) order."""
    t = np.linspace(0.0, 1.0, int(n))
    b, g, r = np.meshgrid(t, t, t, indexing="ij")  # blue slowest, red fastest
    return apply(c, np.stack([r, g, b], axis=-1))


def blend(out: FloatArray, inp: FloatArray, s: float) -> FloatArray:
    """The in-camera strength model: ``s*out + (1-s)*inp``."""
    s = float(s)
    return s * np.asarray(out, dtype=np.float64) + (1.0 - s) * np.asarray(inp, dtype=np.float64)


def write_cube_file(c: Compiled, path: str | Path, *, size: int = 33,
                    comments: tuple[str, ...] = ()) -> Path:
    """Sample the look to a 33-point .cube with the house LUMIX header.

    Refuses a ``Compiled`` that only exists because ``compile(strict=False)``
    swallowed a validation error: §7 makes ``validate`` the gate, and a
    docstring saying "never ship this" is not a gate.
    """
    path = Path(path)
    from .spec import _NAME_RE  # same rule as LookSpec.name

    bypassed = [w for w in c.warnings if w.startswith(BYPASS_TAG)]
    if bypassed:
        raise SpecError(
            f"refusing to write {path.name}: this look was compiled with "
            f"strict=False and does not validate.\n  {bypassed[0]}"
        )
    if not _NAME_RE.match(path.stem):
        raise SpecError(
            f"cube filename stem {path.stem!r}: must be 1-8 ASCII alphanumerics "
            "(the S9 rejects longer stems)"
        )
    lut = LUT3D(table=lattice(c, size), title=c.spec.title or c.spec.name)
    write_cube(path, lut, photo_style="STD", comments=comments)
    return path


# ---------------------------------------------------------------------------
# grey-axis helpers
# ---------------------------------------------------------------------------


def target_grey(c: Compiled) -> tuple[FloatArray, FloatArray]:
    """``(t, T)`` — the designed grey response on the 4097-point ramp."""
    return c.neutral.t, c.neutral.T


def grey_response(c: Compiled, n: int = _neutral.RAMP_N) -> tuple[FloatArray, FloatArray]:
    """``(t, out)`` — what the whole pipeline actually does to the grey axis."""
    t = np.linspace(0.0, 1.0, int(n))
    return t, apply(c, np.stack([t, t, t], axis=-1))


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def diagnostics(c: Compiled, *, size: int = 33, blend_s: float = 0.7,
                grid_n: int = 200_000) -> dict:
    """Every number ENGINE_SPEC §8 asks a milestone-1 report to carry.

    Fold / second-difference / neutral statistics all come from
    ``tools.metrics`` (imported lazily, so ``import engine.pipeline`` stays
    cheap) — they are defined once, there.
    """
    from tools import metrics  # noqa: PLC0415 - deliberate lazy import

    t = np.linspace(0.0, 1.0, int(size))
    b, g, r = np.meshgrid(t, t, t, indexing="ij")
    grid = np.stack([r, g, b], axis=-1)

    t0 = time.perf_counter()
    raw, nm = _run(c, grid)
    table = np.clip(raw, 0.0, 1.0)
    lattice_time = time.perf_counter() - t0

    below = float(max(0.0, -raw.min()) * 255.0)
    above = float(max(0.0, raw.max() - 1.0) * 255.0)
    n_all = raw.size

    # grey axis vs the designed target T
    tt, out = grey_response(c)
    T = c.neutral.T
    grey_err = np.abs(out - T) * 255.0
    k = int(np.argmax(grey_err.max(axis=1)))

    d2 = metrics.second_diff_stats(table)
    fold = metrics.fold_stats(table)
    fold70 = metrics.fold_stats(metrics.blend_table(table, blend_s))
    clip = metrics.clip_stats(table)
    neu = metrics.neutral_stats(metrics.sampler_from_table(table, title=c.name))

    out_d = {
        "name": c.name,
        "order": c.order,
        "size": int(size),
        "lattice_time_s": float(lattice_time),
        "warnings": list(c.warnings),
        "grey": {
            "max_err_codes": float(grey_err.max()),
            "max_err_at_t": float(tt[k]),
            "max_err_at_code": float(tt[k] * 255.0),
            "max_err_channel": "RGB"[int(np.argmax(grey_err[k]))],
            "white_err": float(np.max(np.abs(out[-1] - 1.0))),
            "black": [float(x) * 255.0 for x in out[0]],
            "monotone_per_channel": neu["monotone_per_channel"],
            "monotone_L": neu["monotone_L"],
            "min_step_codes": float(neu["min_step"] * 255.0),
            "tint_rg": neu["tint_rg"],
            "tint_bg": neu["tint_bg"],
            "slope": neu["slope"],
        },
        "d2": {
            "full": d2["full"],
            "interior": d2["interior"],
        },
        "fold": {
            "min_ratio": fold["min_ratio"],
            "neg_count": fold["neg_count"],
            "neg_strict_count": fold["neg_strict_count"],
            # v1.2 W6: folds are judged by MAGNITUDE
            "material_count": fold["material_count"],
            "micro_count": fold["micro_count"],
            "micro_frac": fold["micro_frac"],
            "crush_count": fold["crush_count"],
            "crush_frac": fold["crush_frac"],
            "zero_count": fold["zero_count"],
            "n_tetra": fold["n_tetra"],
            # ENGINE_SPEC §8's "zero negative tetrahedra" reads `neg_count`
            # (ratio <= 0) for a colour look.  A MONO look maps the whole cube
            # onto a 1-D curve, so every tetrahedron has volume exactly 0 and
            # `neg_count` is 196,608 by construction: the gate is structurally
            # unreachable there and must be read as `material_count == 0`
            # plus the 1-D monotonicity below.  v1.2 W6 makes `material_count`
            # the gate for every look, mono or not.
            "gate_field": "material_count",
            "degenerate_by_construction": c.spec.mono is not None,
        },
        "fold_blend70": {
            "min_ratio": fold70["min_ratio"],
            "neg_count": fold70["neg_count"],
            "neg_strict_count": fold70["neg_strict_count"],
            "material_count": fold70["material_count"],
            "micro_count": fold70["micro_count"],
            "micro_frac": fold70["micro_frac"],
            "crush_frac": fold70["crush_frac"],
            "zero_count": fold70["zero_count"],
        },
        "clip": {
            "excursion_below_codes": below,
            "excursion_above_codes": above,
            "frac_below": float(np.count_nonzero(raw < 0.0) / n_all),
            "frac_above": float(np.count_nonzero(raw > 1.0) / n_all),
            "at_zero": clip["zero"],
            "at_one": clip["one"],
        },
    }
    out_d["grid_error"] = grid_error(c, size=size, n=grid_n, table=table)

    if c.spec.mono is not None:
        # the injectivity statement that IS reachable for a rank-1 map: the grey
        # ramp strictly increasing per channel, and the mono lightness curve
        # strictly increasing in Y.  [G] is skipped, so there is no `nm` block.
        t_m = np.linspace(0.0, 1.0, 4097)
        ramp = apply(c, np.stack([t_m, t_m, t_m], axis=-1))
        steps = np.diff(ramp, axis=0)
        out_d["mono"] = {
            "filter": list(c.spec.mono.filter),
            "grey_min_step_codes": float(steps.min() * 255.0),
            "grey_strictly_increasing": bool(steps.min() > 0.0),
            "note": "fold gate reads neg_strict_count; a mono lattice is rank 1, "
                    "so neg_count == n_tetra by construction and [G] is skipped "
                    "(no nm block)",
        }

    if nm is not None:
        gg = _gamut.DEFAULT_DIAG if c.spec.gamut is None else c.spec.gamut
        out_d["nm"] = {
            "p99": float(np.percentile(nm, 99.0)),
            "p99_9": float(np.percentile(nm, 99.9)),
            "max": float(nm.max()),
            "frac_touched": float(np.count_nonzero(nm > gg.knee) / nm.size),
            "knee": float(gg.knee),
            "p": float(gg.p),
            "end_slope": float(gg.end_slope),
            "anchor": c.anchor,
            "lim": c.gamut_lim,
            "q": c.gamut_q,
            "min_radial_gain": c.gamut_min_radial_gain,
            "flat_frac": c.gamut_flat_frac,
            "flat_gain": _gamut.MIN_RADIAL_GAIN,
            "gamut": None if c.spec.gamut is None else c.spec.gamut.to_dict(),
        }
        if c.gamut_lim is not None:
            out_d["nm"]["overshoot"] = gamut_overshoot(c)
    return out_d


#: v1.1 R5 gates on the out-of-gamut pressure
PUSH_WARN, PUSH_FAIL = 0.35, 0.60
LIM_WARN, LIM_FAIL = 1.6, 1.9


def push_stats(c: Compiled, px: FloatArray) -> dict:
    """v1.1 R5's ``push`` — how far [P] moves a colour OUT of the gamut.

    ``push = nm_pre - nm_in`` per pixel, where ``nm_in`` is the stage-0 norm of
    the input colour and ``nm_pre`` the norm of the colour [G] receives.  It
    replaces the v1.0 ``nm p99.9 <= 1.15`` gate, which the identity look already
    failed (it reads 1.27 at p = 4: ``nm`` measures saturation, not excess).

    **Both ends use [G]'s own norm** (``gamut.norm``, the same anchor and the
    same ``p``).  v1.1 took the two ends from two different formulas (R1's
    ``input_nm`` had a 1e-6 dark floor, [G] had 1e-12) and read a push of 1.09
    on near-black where [P] had moved nothing; R1's formula is gone with W1 and
    the anchor is now shared by construction.
    """
    px = np.asarray(px, dtype=np.float64)
    _, nm_pre = _run(c, px)
    if nm_pre is None:
        return {"n": int(px.shape[0]), "p99_9": None, "max": None, "mean": None,
                "note": "mono look: [G] is skipped"}
    p = _gamut.DEFAULT_DIAG.p if c.spec.gamut is None else c.spec.gamut.p
    L0 = color.linear_srgb_to_oklab(xfer.signed_srgb_decode(px))[..., 0]
    nm_in, _, _ = _gamut.norm(px, L0, p, c.anchor)
    d = nm_pre - nm_in
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "p99": float(np.percentile(d, 99.0)),
        "p99_9": float(np.percentile(d, 99.9)),
        "max": float(d.max()),
        "p": float(p),
        "warn": PUSH_WARN, "fail": PUSH_FAIL,
    }


#: sample size / seed of :func:`gamut_overshoot`
OVERSHOOT_N = 500_000
OVERSHOOT_SEED = 20260922


def gamut_overshoot(c: Compiled, *, n: int = OVERSHOOT_N,
                    seed: int = OVERSHOOT_SEED) -> dict:
    """How often a CONTINUOUS input lands beyond the 33-lattice-measured ``lim``.

    ``lim`` is measured on the 33**3 lattice, but the engine is a continuous map
    and the camera's table is not its only consumer: ``grid_error`` and
    ``tools.qc``'s 65**3 Jacobian run :func:`apply` itself, and
    ``lattice(c, n > 33)`` reuses the 33-measured limit.

    **v1.2 W5.2 defuses this.**  v1.1's curve was constant at 1 above ``lim``, so
    a whole outward ray collapsed onto one point (two pre-gamut colours at
    ``nm`` 1.89 and 3.79 came out bit-identical and the continuous Jacobian went
    negative).  The W5 curve continues LINEARLY with slope ``s1``, so a sample
    past ``lim`` is still injective — it simply lands marginally above 1 and
    meets the final clip.  The count is still reported, because it says how much
    of the map the final clip is now responsible for.
    """
    if c.gamut_lim is None:
        return {"n": int(n), "lim": None, "note": "no [G] (or a mono look)"}
    rng = np.random.default_rng(int(seed))
    px = rng.uniform(0.0, 1.0, size=(int(n), 3))
    _, nm = _run(c, px)
    over = int(np.count_nonzero(nm > c.gamut_lim))
    return {
        "n": int(n), "lim": float(c.gamut_lim),
        "nm_max_lattice33": c.nm_pre_max,
        "nm_max_continuous": float(nm.max()),
        "over_lim_count": over,
        "over_lim_frac": float(over / nm.size),
        "margin_needed": float(nm.max() / c.nm_pre_max) if c.nm_pre_max else None,
        "note": f"samples above lim continue linearly with slope s1 "
                f"(W5.2) and meet the final clip; LIM_MARGIN is {_gamut.LIM_MARGIN}",
    }


#: R06 §4 grid-error gates, in 8-bit codes (mean / p99 / max)
GRID_ERROR_FAIL = {"mean": 0.35, "p99": 1.6, "max": 5.0}
GRID_ERROR_WARN = {"mean": 0.20}


def grid_error(c: Compiled, *, size: int = 33, n: int = 200_000,
               seed: int = 20260922, table: FloatArray | None = None) -> dict:
    """How far the shipped 33-point table is from the continuous engine.

    The artifact the camera runs is the tetrahedrally-interpolated table, not
    :func:`apply`; R06 §4 gates the difference (mean > 0.35, p99 > 1.6,
    max > 5.0 codes = FAIL) and nothing in the engine measured it.  ``n``
    uniform samples in ``[0.002, 0.998]**3``, in 8-bit codes, max over channels.
    """
    if table is None:
        table = lattice(c, size)
    rng = np.random.default_rng(seed)
    px = rng.uniform(0.002, 0.998, size=(int(n), 3))
    cont = apply(c, px)
    from .cubeio import tetrahedral_interpolation  # noqa: PLC0415 - lazy
    samp = tetrahedral_interpolation(LUT3D(table=np.asarray(table), title=c.name), px)
    e = np.abs(cont - samp).max(axis=-1) * 255.0
    out = {
        "n": int(n), "size": int(size),
        "mean": float(e.mean()),
        "p99": float(np.percentile(e, 99.0)),
        "p99_9": float(np.percentile(e, 99.9)),
        "max": float(e.max()),
    }
    out["pass"] = bool(out["mean"] <= GRID_ERROR_FAIL["mean"]
                       and out["p99"] <= GRID_ERROR_FAIL["p99"]
                       and out["max"] <= GRID_ERROR_FAIL["max"])
    out["gates"] = dict(GRID_ERROR_FAIL)
    return out


def identity_error(c: Compiled, size: int = 33) -> float:
    """max |lattice - identity| for an identity look."""
    t = np.linspace(0.0, 1.0, int(size))
    b, g, r = np.meshgrid(t, t, t, indexing="ij")
    grid = np.stack([r, g, b], axis=-1)
    return float(np.max(np.abs(apply(c, grid) - grid)))


def with_order(c: Compiled, order: str) -> Compiled:
    """The same look compiled in the other stage order."""
    return compile(replace(c.spec, order=order), strict=False)
