"""Look specification (ENGINE_SPEC §7) — frozen dataclasses whose defaults are
the identity map, plus ``from_dict`` / ``to_dict`` / ``validate``.

A look file is a small, human-readable JSON document; most of a look is four
12-entry hue tables saying, *in measurement space*, what happens to a colour of
input hue h.  Every field is optional except ``name``; anything missing is the
identity.  See ENGINE_SPEC §7 for the canonical example.

``validate(spec)`` returns a list of warnings and **raises** :class:`SpecError`
on anything that would ship a broken LUT — a non-monotone grey target, a hue
window narrower than the 33-point lattice can carry, a rotation field that
folds, a chroma gate that reaches into the noisy near-neutral hues.  Every
message names the offending number.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "SpecError",
    "TintBump",
    "Neutral",
    "Gates",
    "Vibrance",
    "Arch",
    "Field",
    "Skin",
    "Op",
    "Cap",
    "Gamut",
    "Mono",
    "LookSpec",
    "validate",
    "IDENTITY",
    "DEFAULT_P",
    "DEFAULT_KNEE",
    "DEFAULT_END_SLOPE",
    "LEGACY_FIELD_KEYS",
    "LEGACY_GATE_KEYS",
]


class SpecError(ValueError):
    """A look that must not be compiled: the message names the number."""


_NAME_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")

_Z12 = (0.0,) * 12
_O12 = (1.0,) * 12
_O5 = (1.0,) * 5

CHANNEL = ("R", "G", "B")


# ---------------------------------------------------------------------------
# small helpers for dict <-> dataclass
# ---------------------------------------------------------------------------


def _check_keys(d: dict, allowed: Sequence[str], where: str) -> None:
    extra = sorted(set(d) - set(allowed))
    if extra:
        raise SpecError(f"{where}: unknown key(s) {extra}; allowed {sorted(allowed)}")


def _tup(v, n: int | None, where: str) -> tuple[float, ...]:
    arr = tuple(float(x) for x in v)
    if n is not None and len(arr) != n:
        raise SpecError(f"{where}: expected {n} numbers, got {len(arr)}")
    return arr


def _opt_tup(v, n: int, where: str):
    return None if v is None else _tup(v, n, where)


# ---------------------------------------------------------------------------
# [N] neutral stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TintBump:
    """One Gaussian tint bump: amplitude in 8-bit codes, centre and width in
    input code fraction [0,1].  ``sigma >= 0.15`` (ENGINE_SPEC §2)."""

    a: float = 0.0
    mu: float = 0.5
    sigma: float = 0.25

    @staticmethod
    def from_list(v, where: str) -> "TintBump":
        a, mu, sigma = _tup(v, 3, where)
        return TintBump(a=a, mu=mu, sigma=sigma)

    def to_list(self) -> list[float]:
        return [self.a, self.mu, self.sigma]


@dataclass(frozen=True)
class Neutral:
    """Target grey response ``T(t)`` and the off-axis extension rule."""

    #: 8-bit [in, out] pairs; first in = 0, last = [255, 255].
    tone: tuple[tuple[float, float], ...] = ((0.0, 0.0), (255.0, 255.0))
    tint_rg: tuple[TintBump, ...] = ()
    tint_bg: tuple[TintBump, ...] = ()
    white_guard: tuple[float, float] = (0.88, 1.0)
    tint_black_k: float = 12.0
    fade: tuple[float, float] = (0.05, 0.16)
    tint_skin_residual: float = 0.5

    @staticmethod
    def from_dict(d: dict) -> "Neutral":
        _check_keys(
            d,
            ("tone", "tint_rg", "tint_bg", "white_guard", "tint_black_k", "fade",
             "tint_skin_residual"),
            "neutral",
        )
        base = Neutral()
        tone = d.get("tone")
        return Neutral(
            tone=tuple(tuple(_tup(p, 2, "neutral.tone")) for p in tone) if tone else base.tone,
            tint_rg=tuple(TintBump.from_list(b, "neutral.tint_rg") for b in d.get("tint_rg", ())),
            tint_bg=tuple(TintBump.from_list(b, "neutral.tint_bg") for b in d.get("tint_bg", ())),
            white_guard=_tup(d.get("white_guard", base.white_guard), 2, "neutral.white_guard"),
            tint_black_k=float(d.get("tint_black_k", base.tint_black_k)),
            fade=_tup(d.get("fade", base.fade), 2, "neutral.fade"),
            tint_skin_residual=float(d.get("tint_skin_residual", base.tint_skin_residual)),
        )

    def to_dict(self) -> dict:
        return {
            "tone": [list(p) for p in self.tone],
            "tint_rg": [b.to_list() for b in self.tint_rg],
            "tint_bg": [b.to_list() for b in self.tint_bg],
            "white_guard": list(self.white_guard),
            "tint_black_k": self.tint_black_k,
            "fade": list(self.fade),
            "tint_skin_residual": self.tint_skin_residual,
        }


# ---------------------------------------------------------------------------
# [P] perceptual field
# ---------------------------------------------------------------------------


#: gate keys of the v1.0/v1.1 look format that v1.2 W3 deleted.  Old look files
#: keep loading (the values are ignored) and ``validate`` warns.
LEGACY_GATE_KEYS = ("iso",)


@dataclass(frozen=True)
class Gates:
    """Chroma / lightness gates of ENGINE_SPEC §3, defaults per v1.2 W4.

    Curvature goes as amplitude / width**2, so W4 widens every default and
    gives the DL table its own gate::

        rot10  (0.03, 0.12)    rot18 (0.10, 0.20)    chr (0.02, 0.14)
        lgt    (0.02, 0.16)    shadow (0.04, 0.16)

    ``lgt`` is new: the DL table and the ops' ``dl`` used to share ``chr``, and
    a ΔL of 0.04 switched on across 0.06 of chroma measured 3-10 codes of
    second difference by itself.  ``iso`` is gone — W3 moved ``sat`` /
    ``vibrance`` into the ungated stage [S].
    """

    rot10: tuple[float, float] = (0.03, 0.12)
    rot18: tuple[float, float] = (0.10, 0.20)
    chr: tuple[float, float] = (0.02, 0.14)
    lgt: tuple[float, float] = (0.02, 0.16)
    shadow: tuple[float, float] = (0.04, 0.16)
    #: keys accepted from a v1.0/v1.1 look file and ignored (warned about)
    legacy: tuple[str, ...] = ()

    @staticmethod
    def from_dict(d: dict) -> "Gates":
        _check_keys(d, ("rot10", "rot18", "chr", "lgt", "shadow") + LEGACY_GATE_KEYS,
                    "field.gates")
        b = Gates()
        return Gates(
            rot10=_tup(d.get("rot10", b.rot10), 2, "field.gates.rot10"),
            rot18=_tup(d.get("rot18", b.rot18), 2, "field.gates.rot18"),
            chr=_tup(d.get("chr", b.chr), 2, "field.gates.chr"),
            lgt=_tup(d.get("lgt", b.lgt), 2, "field.gates.lgt"),
            shadow=_tup(d.get("shadow", b.shadow), 2, "field.gates.shadow"),
            legacy=tuple(k for k in LEGACY_GATE_KEYS if k in d),
        )

    def to_dict(self) -> dict:
        return {
            "rot10": list(self.rot10), "rot18": list(self.rot18), "chr": list(self.chr),
            "lgt": list(self.lgt), "shadow": list(self.shadow),
        }


@dataclass(frozen=True)
class Vibrance:
    gain: float = 1.0
    c: tuple[float, float] = (0.02, 0.16)

    @staticmethod
    def from_dict(d: dict) -> "Vibrance":
        _check_keys(d, ("gain", "c"), "field.vibrance")
        b = Vibrance()
        return Vibrance(gain=float(d.get("gain", b.gain)),
                        c=_tup(d.get("c", b.c), 2, "field.vibrance.c"))

    def to_dict(self) -> dict:
        return {"gain": self.gain, "c": list(self.c)}


@dataclass(frozen=True)
class Arch:
    """Chroma-vs-lightness arch: 5 chroma ratios at L0 = .25/.40/.55/.70/.85 for
    each of the three hue families (warm 55 deg, green 145 deg, blue 250 deg),
    normalised to 1 at L0 = 0.65 by the engine."""

    warm: tuple[float, ...] = _O5
    green: tuple[float, ...] = _O5
    blue: tuple[float, ...] = _O5

    @staticmethod
    def from_dict(d: dict) -> "Arch":
        _check_keys(d, ("warm", "green", "blue"), "field.arch")
        b = Arch()
        return Arch(
            warm=_tup(d.get("warm", b.warm), 5, "field.arch.warm"),
            green=_tup(d.get("green", b.green), 5, "field.arch.green"),
            blue=_tup(d.get("blue", b.blue), 5, "field.arch.blue"),
        )

    def to_dict(self) -> dict:
        return {"warm": list(self.warm), "green": list(self.green), "blue": list(self.blue)}


#: keys of the v1.1 look format that v1.2 W1 withdrew.  Old look files keep
#: loading (the values are ignored) and ``validate`` warns.
LEGACY_FIELD_KEYS = ("headroom",)

#: W2's ``kappa`` — how much of a chroma BOOST a colour already on the gamut
#: shell gives back.  1.0 = the boost is cancelled exactly at ``r0 = 1``, so no
#: chroma gain can push anything out of gamut; 0 = no fade at all.
DEFAULT_RELFADE = 1.0


@dataclass(frozen=True)
class Field:
    """The four hue tables (12 knots at h0 = 0,30,...,330) plus the global ops.

    ``sat`` and ``vibrance`` are still the look-file keys a colourist writes,
    but since v1.2 W3 they are executed in **stage [S]**, ungated, on the input
    colour before [N] — not inside [P].  W4's authoring guidance follows: put
    the MEAN saturation change of a look into ``sat`` (free of curvature) and
    keep the ``cr`` table for the hue ANISOTROPY around 1.0.
    """

    dh10: tuple[float, ...] = _Z12
    dh18: tuple[float, ...] = _Z12
    cr: tuple[float, ...] = _O12
    dl: tuple[float, ...] = _Z12
    rot_l_scale: tuple[float, float] = (1.0, 1.0)
    arch: Arch = Arch()
    sat: float = 1.0
    vibrance: Vibrance = Vibrance()
    #: W2's relative-fade strength ``kappa``
    relfade: float = DEFAULT_RELFADE
    gates: Gates = Gates()
    #: keys accepted from a v1.1 look file and ignored (warned about)
    legacy: tuple[str, ...] = ()

    @staticmethod
    def from_dict(d: dict) -> "Field":
        _check_keys(
            d, ("dh10", "dh18", "cr", "dl", "rot_l_scale", "arch", "sat", "vibrance",
                "relfade", "gates") + LEGACY_FIELD_KEYS,
            "field",
        )
        b = Field()
        return Field(
            dh10=_tup(d.get("dh10", b.dh10), 12, "field.dh10"),
            dh18=_tup(d.get("dh18", b.dh18), 12, "field.dh18"),
            cr=_tup(d.get("cr", b.cr), 12, "field.cr"),
            dl=_tup(d.get("dl", b.dl), 12, "field.dl"),
            rot_l_scale=_tup(d.get("rot_l_scale", b.rot_l_scale), 2, "field.rot_l_scale"),
            arch=Arch.from_dict(d.get("arch", {})),
            sat=float(d.get("sat", b.sat)),
            vibrance=Vibrance.from_dict(d.get("vibrance", {})),
            relfade=float(d.get("relfade", b.relfade)),
            gates=Gates.from_dict(d.get("gates", {})),
            legacy=tuple(k for k in LEGACY_FIELD_KEYS if k in d),
        )

    def to_dict(self) -> dict:
        return {
            "dh10": list(self.dh10), "dh18": list(self.dh18), "cr": list(self.cr),
            "dl": list(self.dl), "rot_l_scale": list(self.rot_l_scale),
            "arch": self.arch.to_dict(), "sat": self.sat,
            "vibrance": self.vibrance.to_dict(), "relfade": self.relfade,
            "gates": self.gates.to_dict(),
        }


@dataclass(frozen=True)
class Skin:
    """Skin window and the skin protocol (ENGINE_SPEC §3.2).

    The default feather is **20/38/62/80** (v1.1 R3): §3.2's 26/38/62/74 cannot
    pass the fold bound at the rotation amplitudes this set actually uses — the
    12-degree feather makes ``1 + dDh/dh0`` dip to 0.098 on 01Glaze (bound 0.3)
    and to 0.28-0.30 on _demo_arcade.
    """

    window: tuple[float, float, float, float] = (20.0, 38.0, 62.0, 80.0)
    c_gate: tuple[float, float] = (0.03, 0.06)
    c_fade: tuple[float, float] = (0.16, 0.24)
    l_gate: tuple[float, float, float, float] = (0.20, 0.35, 0.90, 0.97)
    center: float = 50.0
    pull: float = 0.0
    hue_offset: float = 0.0
    hue_residual: float = 0.25
    chroma_residual: float = 0.25
    l_residual: float = 0.25
    #: chroma gains at L0 = .50 / .68 / .86 (dark / mid / bright skin)
    chroma: tuple[float, float, float] = (1.0, 1.0, 1.0)
    l_lift: float = 0.0

    @staticmethod
    def from_dict(d: dict) -> "Skin":
        _check_keys(
            d,
            ("window", "c_gate", "c_fade", "l_gate", "center", "pull", "hue_offset",
             "hue_residual", "chroma_residual", "l_residual", "chroma", "l_lift"),
            "skin",
        )
        b = Skin()
        return Skin(
            window=_tup(d.get("window", b.window), 4, "skin.window"),
            c_gate=_tup(d.get("c_gate", b.c_gate), 2, "skin.c_gate"),
            c_fade=_tup(d.get("c_fade", b.c_fade), 2, "skin.c_fade"),
            l_gate=_tup(d.get("l_gate", b.l_gate), 4, "skin.l_gate"),
            center=float(d.get("center", b.center)),
            pull=float(d.get("pull", b.pull)),
            hue_offset=float(d.get("hue_offset", b.hue_offset)),
            hue_residual=float(d.get("hue_residual", b.hue_residual)),
            chroma_residual=float(d.get("chroma_residual", b.chroma_residual)),
            l_residual=float(d.get("l_residual", b.l_residual)),
            chroma=_tup(d.get("chroma", b.chroma), 3, "skin.chroma"),
            l_lift=float(d.get("l_lift", b.l_lift)),
        )

    def to_dict(self) -> dict:
        return {
            "window": list(self.window), "c_gate": list(self.c_gate),
            "c_fade": list(self.c_fade), "l_gate": list(self.l_gate),
            "center": self.center, "pull": self.pull, "hue_offset": self.hue_offset,
            "hue_residual": self.hue_residual, "chroma_residual": self.chroma_residual,
            "l_residual": self.l_residual, "chroma": list(self.chroma), "l_lift": self.l_lift,
        }


@dataclass(frozen=True)
class Op:
    """A local, hue-windowed operator (ENGINE_SPEC §3.6)."""

    id: str = ""
    center: float = 0.0
    sigma: float = 25.0
    #: two-level rotation by lightness: (dh_lo, dh_hi) blended by S5(dh_l)
    dh: tuple[float, float] = (0.0, 0.0)
    dh_l: tuple[float, float] = (0.40, 0.60)
    gain_c: float = 1.0
    dl: float = 0.0
    c_gate: tuple[float, float] = (0.03, 0.10)
    c_fade: tuple[float, float] | None = None
    l_band: tuple[float, float, float, float] | None = None
    skin_residual: float = 0.15

    @staticmethod
    def from_dict(d: dict) -> "Op":
        _check_keys(
            d,
            ("id", "center", "sigma", "dh", "dh_l", "gain_c", "dl", "c_gate", "c_fade",
             "l_band", "skin_residual"),
            "ops[]",
        )
        b = Op()
        if "center" not in d:
            raise SpecError("ops[]: 'center' is required")
        dh = d.get("dh", b.dh)
        if np.isscalar(dh):
            dh = (float(dh), float(dh))
        return Op(
            id=str(d.get("id", b.id)),
            center=float(d["center"]),
            sigma=float(d.get("sigma", b.sigma)),
            dh=_tup(dh, 2, "ops[].dh"),
            dh_l=_tup(d.get("dh_l", b.dh_l), 2, "ops[].dh_l"),
            gain_c=float(d.get("gain_c", b.gain_c)),
            dl=float(d.get("dl", b.dl)),
            c_gate=_tup(d.get("c_gate", b.c_gate), 2, "ops[].c_gate"),
            c_fade=_opt_tup(d.get("c_fade", b.c_fade), 2, "ops[].c_fade"),
            l_band=_opt_tup(d.get("l_band", b.l_band), 4, "ops[].l_band"),
            skin_residual=float(d.get("skin_residual", b.skin_residual)),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "center": self.center, "sigma": self.sigma, "dh": list(self.dh),
            "dh_l": list(self.dh_l), "gain_c": self.gain_c, "dl": self.dl,
            "c_gate": list(self.c_gate),
            "c_fade": None if self.c_fade is None else list(self.c_fade),
            "l_band": None if self.l_band is None else list(self.l_band),
            "skin_residual": self.skin_residual,
        }


@dataclass(frozen=True)
class Cap:
    """Soft chroma ceiling inside a hue window (ENGINE_SPEC §3.4).

    The one block of the look format that has **no identity default**: a cap is
    a tanh saturation whose radial gain is ``sech²((C-start)/(cap-start))``, so
    there is no ``(start, cap)`` pair that means "do nothing".  ``center``,
    ``start`` and ``cap`` are therefore *required* keys in a look file — writing
    ``{"center": 140}`` used to inherit ``start .10 / cap .15``, i.e. a real
    chroma compression where §7's "missing = identity" rule promises a no-op.

    ``c_gate`` is the stage-0 chroma gate every other ab-modifying op has
    (§3 hard rule: ``lo >= 0.03``, width ``>= 0.06``); without it the cap fires
    on the *tinted grey* of the NPG order and destroys ``grey == T``.

    ``center = None`` (v1.1 R3) makes the cap **hue-independent** — a neon
    protection that applies to every hue.  Its window is then ``gis * grd``
    exactly (the hue-independent gate of §3 plus the shadow guard), so
    ``sigma``, ``c_gate`` and the hue-window curvature budget do not apply:
    there is no hue window to be too narrow.
    """

    center: float | None = 0.0
    sigma: float = 35.0
    start: float = 0.10
    cap: float = 0.15
    c_gate: tuple[float, float] = (0.03, 0.10)

    @property
    def hue_independent(self) -> bool:
        return self.center is None

    @staticmethod
    def from_dict(d: dict) -> "Cap":
        _check_keys(d, ("center", "sigma", "start", "cap", "c_gate"), "caps[]")
        b = Cap()
        missing = [k for k in ("center", "start", "cap") if k not in d]
        if missing:
            raise SpecError(
                f"caps[]: {missing} required — a cap has no identity default "
                "(ENGINE_SPEC §7's 'missing = identity' cannot be honoured by a "
                "tanh ceiling); write the numbers out or drop the cap.  "
                '"center": null is the hue-independent cap (v1.1 R3)'
            )
        ctr = d["center"]
        return Cap(center=None if ctr is None else float(ctr),
                   sigma=float(d.get("sigma", b.sigma)),
                   start=float(d["start"]), cap=float(d["cap"]),
                   c_gate=_tup(d.get("c_gate", b.c_gate), 2, "caps[].c_gate"))

    def to_dict(self) -> dict:
        return {"center": self.center, "sigma": self.sigma, "start": self.start,
                "cap": self.cap, "c_gate": list(self.c_gate)}


# ---------------------------------------------------------------------------
# [G] gamut, mono, the look
# ---------------------------------------------------------------------------


#: [G] norm exponent, knee and end slope.  ``p = 8`` is W5.3's; ``knee`` and
#: ``end_slope`` come from W5.3's own knee x s1 sweep on 03Gilt / 04Viride /
#: 10Splice (table in ``work.nosync/review/m2_engine_report.json``, key
#: ``w5_sweep``), NOT from W5.3's provisional 0.75 / 0.25.
#:
#: W5.3 asks for "no material folds, smallest crush, then smallest interior
#: d2".  **No cell of the 4 x 3 grid reaches zero material folds** — the three
#: looks measure 3,440 to 4,154 over the grid, and they measure 3,983 with [G]
#: switched off entirely, so the residual folds are [P]'s and the compressor is
#: reducing them, not causing them.  With the first criterion infeasible the
#: choice falls to fewest material folds, then crush, then d2: knee 0.65 /
#: s1 0.35 (3,440 material @100 %, **0** @70 %, crush 14.50 %, interior d2
#: 11.24 — the joint-best d2 in the grid).  Both optima sit on the edge of the
#: swept range: extending it to knee 0.50 / s1 0.60 keeps improving (2,668
#: material, d2 9.88), i.e. the grid says "compress less".
#:
#: The one column W5.3 does not ask for and the lead should see: the tax on
#: in-gamut colours.  identity + [G] alone moves the 33**3 lattice by a mean of
#: 3.31 codes at knee 0.65 against 1.93 at knee 0.80, and touches 63.6 % of it
#: against 44.5 % — v1.1 R2 chose 0.80 on exactly that criterion.
DEFAULT_P = 8.0
DEFAULT_KNEE = 0.65
DEFAULT_END_SLOPE = 0.35

#: keys of the v1.0 look format that v1.1 R2 deleted.  Old look files keep
#: loading (the values are ignored) and ``validate`` warns, once per key.
LEGACY_GAMUT_KEYS = ("soft", "eps_dark")


@dataclass(frozen=True)
class Gamut:
    """[G] code-domain compression with a MEASURED limit and a slope floor.

    ``p`` is the exponent of the out-of-gamut norm ``nm``.  ``lim`` is not a
    look parameter: ``pipeline.compile`` measures it on the 33**3 lattice of the
    pre-gamut colour (``lim = 1.04 * max nm``, W5.2), so the compressor's range
    lands exactly on what the look actually produces.

    ``end_slope`` (``s1``) is W5.2's slope floor: the radial gain never drops
    below ``s1/q``, and beyond ``lim`` the curve continues linearly instead of
    flattening onto one shell.

    ``soft`` and ``eps_dark`` were deleted by v1.1 R2.  A look file that still
    carries them loads — the keys are recorded in ``legacy`` and
    :func:`validate` warns — but they change nothing.
    """

    knee: float = DEFAULT_KNEE
    p: float = DEFAULT_P
    end_slope: float = DEFAULT_END_SLOPE
    #: keys accepted from a v1.0 look file and ignored (warned about)
    legacy: tuple[str, ...] = ()

    @staticmethod
    def from_dict(d: dict) -> "Gamut":
        _check_keys(d, ("knee", "p", "end_slope") + LEGACY_GAMUT_KEYS, "gamut")
        b = Gamut()
        return Gamut(knee=float(d.get("knee", b.knee)),
                     p=float(d.get("p", b.p)),
                     end_slope=float(d.get("end_slope", b.end_slope)),
                     legacy=tuple(k for k in LEGACY_GAMUT_KEYS if k in d))

    def to_dict(self) -> dict:
        return {"knee": self.knee, "p": self.p, "end_slope": self.end_slope}


@dataclass(frozen=True)
class Mono:
    """B&W branch (ENGINE_SPEC §6).  ``filter`` are LINEAR-light channel weights
    summing to 1; ``dl`` (12 knots) adds per-hue tonal separation beyond the
    channel mix and defaults to the look's ``field.dl``."""

    filter: tuple[float, float, float] = (0.2126, 0.7152, 0.0722)
    dl: tuple[float, ...] | None = None

    @staticmethod
    def from_dict(d: dict) -> "Mono":
        _check_keys(d, ("filter", "dl"), "mono")
        b = Mono()
        dl = d.get("dl")
        return Mono(filter=_tup(d.get("filter", b.filter), 3, "mono.filter"),
                    dl=None if dl is None else _tup(dl, 12, "mono.dl"))

    def to_dict(self) -> dict:
        return {"filter": list(self.filter), "dl": None if self.dl is None else list(self.dl)}


@dataclass(frozen=True)
class LookSpec:
    name: str = "Ident"
    cn: str = ""
    title: str = ""
    order: str = "NPG"
    neutral: Neutral = Neutral()
    field: Field = Field()
    skin: Skin = Skin()
    ops: tuple[Op, ...] = ()
    caps: tuple[Cap, ...] = ()
    gamut: Gamut | None = None
    #: milestone-2 film front stage.  Must be null in milestone 1.
    film: None = None
    mono: Mono | None = None

    # -- serialisation ----------------------------------------------------
    @staticmethod
    def from_dict(d: dict) -> "LookSpec":
        _check_keys(
            d,
            ("name", "cn", "title", "order", "neutral", "field", "skin", "ops", "caps",
             "gamut", "film", "mono"),
            "look",
        )
        if "name" not in d:
            raise SpecError("look: 'name' is required")
        if d.get("film") is not None:
            raise NotImplementedError(
                "the film front stage [F] is milestone 2; a milestone-1 look must have "
                '"film": null'
            )
        gam = d.get("gamut", None)
        mono = d.get("mono", None)
        return LookSpec(
            name=str(d["name"]),
            cn=str(d.get("cn", "")),
            title=str(d.get("title", "")),
            order=str(d.get("order", "NPG")),
            neutral=Neutral.from_dict(d.get("neutral", {})),
            field=Field.from_dict(d.get("field", {})),
            skin=Skin.from_dict(d.get("skin", {})),
            ops=tuple(Op.from_dict(o) for o in d.get("ops", ())),
            caps=tuple(Cap.from_dict(c) for c in d.get("caps", ())),
            gamut=None if gam is None else Gamut.from_dict(gam),
            film=None,
            mono=None if mono is None else Mono.from_dict(mono),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name, "cn": self.cn, "title": self.title, "order": self.order,
            "neutral": self.neutral.to_dict(), "field": self.field.to_dict(),
            "skin": self.skin.to_dict(), "ops": [o.to_dict() for o in self.ops],
            "caps": [c.to_dict() for c in self.caps],
            "gamut": None if self.gamut is None else self.gamut.to_dict(),
            "film": None,
            "mono": None if self.mono is None else self.mono.to_dict(),
        }

    @staticmethod
    def from_json(text: str) -> "LookSpec":
        return LookSpec.from_dict(json.loads(text))

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


IDENTITY = LookSpec(name="Ident")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

#: (C0, L0) probes for the fold bound of ENGINE_SPEC §3.7.
#:
#: The three-chroma probe of the first draft missed the true minimum, which for
#: a skin-window feather lands at C0 ~ 0.17 — between the pinned 0.10 and 0.18.
#: Measured on ``_demo_arcade``: pinned min 0.1346 vs dense (0.04:0.005:0.33)
#: 0.0924, and two candidate fixes (skin.hue_residual 0.43, window 21/39/61/79)
#: pass the sparse probe at 0.310/0.327 while the dense grid says 0.284/0.299 —
#: i.e. the sparse probe would have shipped a look that still folds.
FOLD_PROBE_C = (0.06, 0.10, 0.14, 0.17, 0.18, 0.22, 0.25, 0.30)
FOLD_PROBE_L = (0.20, 0.35, 0.50, 0.65, 0.85, 0.94)
FOLD_MIN_JACOBIAN = 0.3

#: curvature budget reference chroma (R06 §1.3i)
BUDGET_C = 0.12

#: §3.7 chroma budget.  The design budget [0.4, 1.8] is what §3.7 writes for
#: ``CR*ARCH``; it is applied here to every *factor* of the chroma gain and,
#: as a warning, to the measured total.  ``GAIN_HARD_HI`` is the hard ceiling on
#: the measured total product: doubling the input chroma cannot be rescued by
#: any window width (R06 §1.3i gives sigma >= 13.6*sqrt(|g-1|/C) = 41 deg at
#: g = 2, C = 0.12, and a global gain has no window at all).
GAIN_LO, GAIN_HI = 0.4, 1.8
#: v1.2 W2: "validate: total g < 1.9".  Applied to [P]'s measured total
#: (CR*ARCH x ops x SKINC) and, separately, to [S]'s ``sat * vib``.
GAIN_HARD_HI = 1.9

#: §3.5 lightness budget: |DL| <= 0.06 applies to the SUM of the three terms
#: (table + local ops + skin lift), not to the table alone.
DL_MAX = 0.06

#: v1.2 W4: curvature goes as amplitude / width**2, so every look-level gate in
#: [P] needs this much width.  Skin's own ``c_gate`` keeps the 0.03 floor (its
#: amplitudes are small); the ops' and caps' ``c_gate`` keep ENGINE_SPEC §3's
#: 0.06 hard floor as the ERROR line and take this one as a WARNING, because
#: raising it to an error would reject look files another track owns (the twelve
#: ``looks/*.json`` written by ``tools/fit.py`` all declare 0.06-0.07 widths).
GATE_MIN_WIDTH = 0.08

#: v1.2 W2: the measured radial chroma bound is an ERROR below this.
RADIAL_MIN_SLOPE = 0.15

#: ENGINE_SPEC §8 grey-axis gate, in 8-bit codes
GREY_ERR_MAX = 0.02

#: v1.1 R2 puts no bound on q = (lim-knee)/(1-knee).  ``validate`` warns when the
#: compiled compressor is flatter than ``gamut.MIN_RADIAL_GAIN`` over more than
#: this fraction of the 33**3 lattice.  Calibrated so identity + [G] is silent at
#: every knee of the R2 study (0.70-0.85 all measure 0.0000 %) while a look that
#: really has a flat tail is not (_demo_glaze 14.8 %, {"knee": 0.999} 9.4 %).
GAMUT_FLAT_FRAC_WARN = 0.01

#: chroma the 33-lattice really reaches (the sRGB primaries sit near OKLab
#: C = 0.30) and the radial-gain floor a chroma cap must keep there.
#: Calibrated on the identity lattice + one cap at centre 140, sigma 35:
#:   gain@0.30  0.0003 -> 20 folded tetrahedra   (the §7 example, start .11/cap .15)
#:              0.0174 -> 0 folds, min ratio +1.3e-04   (razor thin)
#:              0.0570 -> 0 folds, min ratio +4.9e-04
#:              0.1937 -> 0 folds, min ratio +2.5e-03
#: 0.05 is the first value with room for the rest of a look underneath it.
CAP_PROBE_C = 0.30
CAP_MIN_RADIAL_GAIN = 0.05


def _gate_rule(name: str, gate: Sequence[float], errors: list[str],
               *, min_lo: float = 0.03, min_width: float = 0.06,
               warn_width: float | None = None,
               warnings: list[str] | None = None) -> None:
    lo, hi = float(gate[0]), float(gate[1])
    if warn_width is not None and warnings is not None and \
            min_width - 1e-12 <= hi - lo < warn_width - 1e-12:
        warnings.append(
            f"{name}: width = {hi - lo:.4f} < {warn_width:.2f} (gate "
            f"{lo:.3f}->{hi:.3f}) — v1.2 W4's width budget; curvature goes as "
            "amplitude / width**2"
        )
    if lo < min_lo - 1e-12:
        errors.append(f"{name}: lo = {lo:.4f} < {min_lo:.2f} — hue is unusable below "
                      f"C0 = 0.03 (adjacent 33-lattice nodes differ by ~154 deg there)")
    if hi - lo < min_width - 1e-12:
        errors.append(f"{name}: width = {hi - lo:.4f} < {min_width:.2f} "
                      f"(gate {lo:.3f}->{hi:.3f}); a chroma step of the lattice is ~0.014")


def _ramp_rule(name: str, lo: float, hi: float, errors: list[str],
               *, min_width: float) -> None:
    """Every S5 ramp in the path needs a real width: coincident edges are a
    Heaviside step (``curves.S5`` now raises on them) and a reversed pair
    silently installs the window on the wrong side."""
    lo, hi = float(lo), float(hi)
    if hi - lo < min_width - 1e-12:
        errors.append(
            f"{name} = [{lo:.4f}, {hi:.4f}]: must be increasing with width "
            f">= {min_width:.2f} (a coincident pair is a step discontinuity; "
            "a reversed pair inverts the window)"
        )


def validate(spec: LookSpec) -> list[str]:
    """Check a look against every rule of ENGINE_SPEC §2-§4, §6.

    Returns a list of warnings.  Raises :class:`SpecError` listing every error
    found (all of them, not just the first), each naming the offending number.
    """
    # imported here: neutral/field import spec, so keep the cycle out of import time
    from . import field as _field
    from . import neutral as _neutral

    errors: list[str] = []
    warnings: list[str] = []

    # -- name / order -----------------------------------------------------
    if not _NAME_RE.match(spec.name):
        errors.append(
            f"name {spec.name!r}: must be 1-8 ASCII alphanumerics (the S9 rejects longer stems)"
        )
    if spec.order not in ("NPG", "PGN"):
        errors.append(f"order {spec.order!r}: must be 'NPG' or 'PGN'")
    if spec.film is not None:
        errors.append("film: the [F] stage is milestone 2; must be null")

    # -- [N] --------------------------------------------------------------
    errors += _neutral.validate_neutral(spec.neutral)

    # -- [G] --------------------------------------------------------------
    if spec.gamut is not None:
        g = spec.gamut
        if not (0.0 < g.knee < 1.0):
            errors.append(f"gamut: knee = {g.knee} must be in (0,1) — the limit curve needs "
                          "(1 - knee) > 0 for its exponent q = (lim-knee)/(1-knee)")
        if g.p < 1.0:
            errors.append(f"gamut: p = {g.p} must be >= 1 (it is an L^p norm exponent; "
                          "p < 1 is not a norm and the nm level sets stop being convex)")
        if not (0.0 < g.end_slope < 1.0):
            errors.append(
                f"gamut: end_slope = {g.end_slope} must be in (0,1) — it is W5.2's "
                "slope floor s1: at 0 the compressor's radial gain reaches 0 at "
                "`lim` (v1.1's flat tail) and at >= 1 the exponent "
                "q2 = (q-s1)/(1-s1) is not defined"
            )
        if g.legacy:
            warnings.append(
                f"gamut: {list(g.legacy)} accepted and IGNORED — v1.1 R2 deleted them from "
                "the look format (the tanh tail they parametrised is gone; the limit is "
                "measured per look at compile time).  Drop the key(s)"
            )
        if g.p != DEFAULT_P:
            warnings.append(
                f"gamut.p = {g.p}: v1.2 W5.3 pins p = {DEFAULT_P} (the v1.1 R2 study "
                "measured p in {4, 6, 8} and W5 fixes 8); the knee x s1 sweep was "
                "run at p = 8 only"
            )
        if spec.mono is not None:
            # §6 replaces the whole colour path, and pipeline._measure_limit
            # skips a mono look, so `gamut_lim` stays None and apply_gamut is
            # never called.  Measured on {"mono": {...}, "gamut": {...}}:
            # gamut_lim = gamut_q = None, and the declared block changes nothing.
            warnings.append(
                "gamut: this look is MONO (§6), and the mono branch never runs "
                "[G] — the gamut block is inert (compile leaves gamut_lim = None). "
                "Drop it, or drop `mono`"
            )

    # -- [P] gates (v1.2 W4) ----------------------------------------------
    gt = spec.field.gates
    # W4 pins the widths; the `lo` floor stays ENGINE_SPEC §3's hue-noise rule
    # for the two rotation gates and drops to 0.02 for `chr` / `lgt`, which is
    # where W4's own defaults put them.
    _gate_rule("field.gates.rot10", gt.rot10, errors,
               min_lo=0.03, min_width=GATE_MIN_WIDTH)
    _gate_rule("field.gates.rot18", gt.rot18, errors,
               min_lo=0.03, min_width=GATE_MIN_WIDTH)
    _gate_rule("field.gates.chr", gt.chr, errors,
               min_lo=0.02, min_width=GATE_MIN_WIDTH)
    _gate_rule("field.gates.lgt", gt.lgt, errors,
               min_lo=0.02, min_width=GATE_MIN_WIDTH)
    _ramp_rule("field.gates.shadow", gt.shadow[0], gt.shadow[1], errors,
               min_width=GATE_MIN_WIDTH)
    if gt.legacy:
        warnings.append(
            f"field.gates: {list(gt.legacy)} accepted and IGNORED — v1.2 W3 "
            "deleted the `iso` gate: `sat` / `vibrance` now run ungated in stage "
            "[S] on the input colour (a chroma scale about the grey axis is "
            "linear near the axis, so it needs no gate, and C0 = 0 is untouched)"
        )
    if spec.field.legacy:
        warnings.append(
            f"field: {list(spec.field.legacy)} accepted and IGNORED — v1.2 W1 "
            "WITHDREW the v1.1 headroom entirely (an S5 fade of a gain over a "
            "narrow band of the code-domain norm is itself non-injective).  W2's "
            "`field.relfade` is the replacement: a LINEAR fade in gamut-relative "
            "chroma.  Drop the key"
        )
    if spec.field.relfade < 0.0:
        errors.append(f"field.relfade = {spec.field.relfade} must be >= 0 "
                      "(it is kappa of W2's fade; 0 = no fade, 1 = a shell colour "
                      "keeps its chroma exactly)")
    elif spec.field.relfade > 1.0 + 1e-12:
        warnings.append(
            f"field.relfade = {spec.field.relfade:.3f} > 1: a shell colour would "
            "be DESATURATED by a chroma boost, and the radial slope bound "
            "g - 2*kappa*(g-1)*r0 drops below 2 - g"
        )

    # skin's own gate may be 0.03->0.06 (§3: its amplitudes are small)
    sk = spec.skin
    _gate_rule("skin.c_gate", sk.c_gate, errors, min_lo=0.03, min_width=0.03)
    # c_fade / l_gate are S5 ramps like every other gate and were unchecked:
    # a reversed pair installs the skin window on the wrong side, silently.
    _ramp_rule("skin.c_fade", sk.c_fade[0], sk.c_fade[1], errors, min_width=0.06)
    if sk.c_fade[0] < sk.c_gate[1] - 1e-12:
        errors.append(
            f"skin.c_fade = {list(sk.c_fade)} starts below skin.c_gate hi = "
            f"{sk.c_gate[1]:.3f}: the window would fade out before it fades in"
        )
    _ramp_rule("skin.l_gate[0:2]", sk.l_gate[0], sk.l_gate[1], errors, min_width=0.03)
    _ramp_rule("skin.l_gate[2:4]", sk.l_gate[2], sk.l_gate[3], errors, min_width=0.03)
    if sk.l_gate[1] > sk.l_gate[2] + 1e-12:
        errors.append(f"skin.l_gate = {list(sk.l_gate)} must be non-decreasing "
                      f"(l1 = {sk.l_gate[1]:.3f} > l2 = {sk.l_gate[2]:.3f})")
    if sk.pull > 0.35 + 1e-12:
        errors.append(f"skin.pull = {sk.pull:.3f} > 0.35 (fold bound, R06 §1.3f)")
    if sk.pull < 0.0:
        errors.append(f"skin.pull = {sk.pull} must be >= 0")
    fw = sk.window
    if not (fw[0] < fw[1] <= fw[2] < fw[3]):
        errors.append(f"skin.window = {list(fw)} must satisfy f_lo < c_lo <= c_hi < f_hi")

    # -- [P] tables -------------------------------------------------------
    # §3.5 adds DL(h0), the local ops' dl and skin.l_lift; the |DL| <= 0.06
    # ceiling of §3.7 therefore belongs on their sum, not on the table alone.
    # (Measured: ops[0].dl = 0.30 alone moves OKLab L by +0.3297 at
    # (L0 .5, C0 .15, h 140) and folds 37,721 tetrahedra; skin.l_lift = 0.5
    # folds 11,964 with d2 p99.9 = 142 codes.  Both used to validate clean.)
    dl_table = max(abs(x) for x in spec.field.dl)
    dl_ops = sum(abs(o.dl) for o in spec.ops)
    dl_skin = abs(sk.l_lift)
    dl_sum = dl_table + dl_ops + dl_skin
    if dl_sum > DL_MAX + 1e-12:
        errors.append(
            f"lightness budget: max|field.dl| {dl_table:.4f} + sum|ops[].dl| {dl_ops:.4f} "
            f"+ |skin.l_lift| {dl_skin:.4f} = {dl_sum:.4f} > {DL_MAX} (ENGINE_SPEC §3.5/§3.7: "
            "the three terms add before the 4L(1-L) shape)"
        )
    elif dl_sum > 0.05:
        warnings.append(f"lightness budget: total |DL| = {dl_sum:.4f} is close to {DL_MAX}")

    v = spec.field.vibrance
    vc_lo, vc_hi = v.c
    need = 1.25 * (v.gain - 1.0) * vc_hi
    if v.gain > 1.0 and (vc_hi - vc_lo) < need - 1e-12:
        errors.append(
            f"field.vibrance: width {vc_hi - vc_lo:.4f} < 1.25*(v-1)*c_hi = {need:.4f} "
            f"(gain {v.gain}, c {list(v.c)}) — the chroma map folds (R06 §1.3f)"
        )
    if vc_hi <= vc_lo:
        errors.append(f"field.vibrance.c = {list(v.c)} must be increasing")

    # every FACTOR of the chroma gain carries the §3.7 budget, not just CR*ARCH
    # (field.sat = 4.0 folded 185,513 tetrahedra and field.sat = -1.0 flipped the
    #  chroma vector on 106,316 of them, both with zero warnings)
    if not (GAIN_LO - 1e-12 <= spec.field.sat <= GAIN_HI + 1e-12):
        errors.append(f"field.sat = {spec.field.sat:.3f} outside [{GAIN_LO}, {GAIN_HI}] "
                      "(ENGINE_SPEC §3.7 chroma budget; sat is stage [S]'s gain)")
    if not (GAIN_LO - 1e-12 <= v.gain <= GAIN_HI + 1e-12):
        errors.append(f"field.vibrance.gain = {v.gain:.3f} outside [{GAIN_LO}, {GAIN_HI}]")
    # v1.2 W2/W3: [S]'s own total is `sat * vib(C0)`, maximised at C0 = 0.
    g_s = spec.field.sat * v.gain
    if g_s > GAIN_HARD_HI + 1e-9:
        errors.append(
            f"field: stage [S] total gain sat*vibrance.gain = {g_s:.3f} >= "
            f"{GAIN_HARD_HI} (v1.2 W2's 'total g < 1.9'); [S] is ungated, so this "
            "multiplies the chroma of every colour in the frame"
        )
    for j, x in enumerate(sk.chroma):
        if not (GAIN_LO - 1e-12 <= x <= GAIN_HI + 1e-12):
            errors.append(
                f"skin.chroma[{j}] = {x:.3f} outside [{GAIN_LO}, {GAIN_HI}] "
                "(§3.4 SKINC multiplies the same chroma vector; skin.chroma = 3 folded "
                "13,711 tetrahedra, = -1 flipped the sign of the chroma vector)"
            )

    # -- [P] local ops ----------------------------------------------------
    for i, op in enumerate(spec.ops):
        tag = f"ops[{i}]{(' ' + op.id) if op.id else ''}"
        if op.sigma < 15.0 - 1e-12:
            errors.append(f"{tag}: sigma = {op.sigma:.2f} deg < 15 deg floor (33-lattice)")
        amp = max(abs(op.dh[0]), abs(op.dh[1]))
        need_rot = 1.8 * np.sqrt(amp / BUDGET_C)
        if amp > 0 and op.sigma < need_rot - 1e-9:
            errors.append(
                f"{tag}: sigma = {op.sigma:.2f} deg < 1.8*sqrt(A/C) = {need_rot:.2f} deg "
                f"for A = {amp:.2f} deg at C = {BUDGET_C}"
            )
        need_gain = 13.6 * np.sqrt(abs(op.gain_c - 1.0) / BUDGET_C)
        if abs(op.gain_c - 1.0) > 0 and op.sigma < need_gain - 1e-9:
            errors.append(
                f"{tag}: sigma = {op.sigma:.2f} deg < 13.6*sqrt(|g-1|/C) = {need_gain:.2f} deg "
                f"for g = {op.gain_c:.3f} at C = {BUDGET_C}"
            )
        if not (GAIN_LO - 1e-12 <= op.gain_c <= GAIN_HI + 1e-12):
            errors.append(f"{tag}: gain_c = {op.gain_c:.3f} outside [{GAIN_LO}, {GAIN_HI}]")
        if abs(op.dl) > DL_MAX + 1e-12:
            errors.append(f"{tag}: |dl| = {abs(op.dl):.4f} > {DL_MAX} (ENGINE_SPEC §3.7)")
        _gate_rule(f"{tag}.c_gate", op.c_gate, errors, min_lo=0.03, min_width=0.06,
                   warn_width=GATE_MIN_WIDTH, warnings=warnings)
        if op.c_fade is not None:
            _ramp_rule(f"{tag}.c_fade", op.c_fade[0], op.c_fade[1], errors, min_width=0.06)
        if op.l_band is not None:
            lb = op.l_band
            _ramp_rule(f"{tag}.l_band[0:2]", lb[0], lb[1], errors, min_width=0.03)
            _ramp_rule(f"{tag}.l_band[2:4]", lb[2], lb[3], errors, min_width=0.03)
            if lb[1] > lb[2] + 1e-12:
                errors.append(f"{tag}.l_band = {list(lb)} must be non-decreasing")
        _ramp_rule(f"{tag}.dh_l", op.dh_l[0], op.dh_l[1], errors, min_width=0.03)

    for i, cp in enumerate(spec.caps):
        ctag = f"caps[{i}]"
        if cp.cap <= cp.start:
            errors.append(f"{ctag}: cap = {cp.cap} must be > start = {cp.start}")
        if cp.start < 0.0:
            errors.append(f"{ctag}: start = {cp.start} must be >= 0")
        # v1.1 R3's `center: null` cap has no hue window, so the 15 deg floor and
        # the hue-curvature budget below do not apply.  Its `c_gate` DOES: W3
        # deleted the `iso` gate that used to keep it off the grey axis, and the
        # cap's own chroma gate is what replaces it (``engine/field.py``).
        _gate_rule(f"{ctag}.c_gate", cp.c_gate, errors, min_lo=0.03, min_width=0.06,
                   warn_width=GATE_MIN_WIDTH, warnings=warnings)
        if not cp.hue_independent:
            if cp.sigma < 15.0 - 1e-12:
                errors.append(f"{ctag}: sigma = {cp.sigma:.2f} deg < 15 deg floor")
        if cp.cap > cp.start:
            span = cp.cap - cp.start
            # radial injectivity: dC_out/dC = sech^2((C-start)/span).  The
            # lattice really reaches C = 0.30 (the sRGB primaries), and at
            # start .11 / cap .15 the gain there is 3.0e-04 -- the map collapses
            # onto the cap surface and the tetrahedra invert (measured: the §7
            # example cap alone puts 20 negative tetrahedra in an identity look).
            gain = 1.0 / np.cosh((CAP_PROBE_C - cp.start) / span) ** 2
            if gain < CAP_MIN_RADIAL_GAIN:
                need = (CAP_PROBE_C - cp.start) / np.arccosh(
                    1.0 / np.sqrt(CAP_MIN_RADIAL_GAIN))
                errors.append(
                    f"{ctag}: radial gain sech^2((C-start)/span) = {gain:.2e} at "
                    f"C = {CAP_PROBE_C} < {CAP_MIN_RADIAL_GAIN} — the chroma map is "
                    f"not injective there (start {cp.start}, cap {cp.cap}); need "
                    f"cap - start >= {need:.4f}"
                )
            # R06 §1.3i chroma-window curvature budget, with the cap's own |g-1|
            g_eff = (cp.start + span * np.tanh((BUDGET_C - cp.start) / span)) / BUDGET_C \
                if BUDGET_C > cp.start else 1.0
            need_sig = 13.6 * np.sqrt(abs(g_eff - 1.0) / BUDGET_C)
            if not cp.hue_independent and cp.sigma < need_sig - 1e-9:
                errors.append(
                    f"{ctag}: sigma = {cp.sigma:.2f} deg < 13.6*sqrt(|g-1|/C) = "
                    f"{need_sig:.2f} deg for the cap's own gain g = {g_eff:.3f} at "
                    f"C = {BUDGET_C}"
                )

    # -- [P] numeric checks that need the compiled field -------------------
    # Only when the structure is sound: every one of these runs the gates, and
    # a malformed ramp now raises out of ``curves.S5`` instead of quietly
    # becoming a step, which would mask the specific error already collected.
    if not errors:
        try:
            cf = _field.compile_field(spec)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"field: could not compile ({exc})")
            cf = None
        if cf is not None:
            lo, hi, where = _field.arch_range(cf)
            if lo < 0.4 - 1e-9 or hi > 1.8 + 1e-9:
                errors.append(
                    f"field: CR*ARCH ranges [{lo:.3f}, {hi:.3f}], outside [0.4, 1.8] "
                    f"(worst at h0 = {where[0]:.0f} deg, L0 = {where[1]:.2f})"
                )
            # v1.2 W2's "total g < 1.9" on the gain the lattice actually sees in
            # [P]: CR*ARCH * op gains * SKINC, scanned over (h0, L0, C0), before
            # the relative fade (which can only shrink a boost).  [S]'s own
            # sat*vib is checked separately above; their product is warned on,
            # because a colour passes through both.
            glo, ghi, gwhere = _field.total_gain_range(cf)
            if ghi > GAIN_HARD_HI + 1e-9 or glo < GAIN_LO - 1e-9:
                errors.append(
                    f"field: total chroma gain ranges [{glo:.3f}, {ghi:.3f}], outside the "
                    f"hard bound [{GAIN_LO}, {GAIN_HARD_HI}] (worst at h0 = {gwhere[0]:.0f} deg, "
                    f"L0 = {gwhere[1]:.2f}, C0 = {gwhere[2]:.3f}) — no hue window can carry "
                    "twice the input chroma (R06 §1.3i)"
                )
            elif ghi > GAIN_HI + 1e-9:
                warnings.append(
                    f"field: total chroma gain reaches {ghi:.3f} > {GAIN_HI} at h0 = "
                    f"{gwhere[0]:.0f} deg, L0 = {gwhere[1]:.2f}, C0 = {gwhere[2]:.3f} — "
                    "CR*ARCH*ops*skin multiply; §3.7's budget is on the product"
                )
            if g_s * ghi > GAIN_HARD_HI + 1e-9:
                warnings.append(
                    f"field: [S] x [P] chroma gain reaches {g_s * ghi:.3f} "
                    f"(sat*vibrance {g_s:.3f} x [P] {ghi:.3f}) > {GAIN_HARD_HI} — the "
                    "two stages multiply on a colour that passes through both.  W4's "
                    "guidance: put the MEAN saturation into `sat` and keep `cr` for "
                    "the anisotropy around 1.0"
                )
            jmin, jwhere = _field.fold_bound(cf, FOLD_PROBE_C, FOLD_PROBE_L)
            if jmin < FOLD_MIN_JACOBIAN - 1e-9:
                errors.append(
                    f"field: hue map folds — min(1 + dDh/dh0) = {jmin:.3f} < "
                    f"{FOLD_MIN_JACOBIAN} at h0 = {jwhere[0]:.0f} deg, C0 = {jwhere[1]:.2f}, "
                    f"L0 = {jwhere[2]:.2f}"
                )
            elif jmin < 0.5:
                warnings.append(
                    f"field: min(1 + dDh/dh0) = {jmin:.3f} at h0 = {jwhere[0]:.0f} deg, "
                    f"C0 = {jwhere[1]:.2f} — close to the 0.3 fold bound"
                )
            # v1.2 W2 keeps the measured radial-chroma probe — now WITH the
            # relative fade and with [S]'s gain in the product — as an ERROR
            # below 0.15.  It is the one check that sees the *gradient* of every
            # chroma gain rather than its magnitude, i.e. the thing that decides
            # whether C -> C_out(C) is still injective.
            cmin, cwhere = _field.chroma_radial_bound(cf)
            if cmin < RADIAL_MIN_SLOPE - 1e-9:
                errors.append(
                    f"field: radial chroma slope min d(C_out)/dC = {cmin:.3f} < "
                    f"{RADIAL_MIN_SLOPE} at h0 = {cwhere[0]:.0f} deg, "
                    f"L0 = {cwhere[1]:.2f}, C0 = {cwhere[2]:.3f} (v1.2 W2).  The "
                    "composite chroma map C*g_S(C)*(1+(g(C)-1)*grd) is flat or "
                    "reversed there — lower the gain, or widen the gate whose ramp "
                    "carries it (chr / vibrance.c / an op c_gate)"
                )
            elif cmin < 0.30:
                warnings.append(
                    f"field: radial chroma slope min d(C_out)/dC = {cmin:.3f} at "
                    f"h0 = {cwhere[0]:.0f} deg, L0 = {cwhere[1]:.2f}, "
                    f"C0 = {cwhere[2]:.3f} — over the {RADIAL_MIN_SLOPE} error line "
                    "but the chroma map is getting flat"
                )

    # -- mono -------------------------------------------------------------
    if spec.mono is not None:
        w = spec.mono.filter
        if abs(sum(w) - 1.0) > 1e-9:
            errors.append(f"mono.filter = {list(w)} sums to {sum(w):.6f}, must be 1")
        for j, x in enumerate(w):
            if x < -0.15 - 1e-12:
                errors.append(f"mono.filter[{j}] = {x:.3f} < -0.15")
        # §6 reuses the §3.5 DL shape, so it reuses its ceiling.  Unchecked,
        # mono.dl = 0.40 drove the pre-clip table to 1.1089 (+27.7 codes above
        # white) and pinned 1.6 % of it to exactly 1.0 — R06 §1.3(j)'s plateau.
        mdl = spec.mono.dl if spec.mono.dl is not None else spec.field.dl
        mdl_max = max(abs(x) for x in mdl)
        if mdl_max > DL_MAX + 1e-12:
            errors.append(f"mono.dl: max |DL| = {mdl_max:.4f} > {DL_MAX} (ENGINE_SPEC "
                          "§3.7; §6 reuses the same DL shape and the same ceiling)")
        neg = min(w)
        if neg < 0.0:
            # Y = w . linear(rgb) then goes negative on part of the cube; the
            # engine keeps the path clamp-free (signed cube root), so those
            # colours arrive at the final clip below 0 instead of plateauing
            # inside the path.  It is still a clip, and it is the look's choice.
            worst = -neg  # min Y over the cube: the channel with the negative weight at 1
            warnings.append(
                f"mono.filter = {list(w)} has a negative weight ({neg:.3f}): Y goes down to "
                f"{-worst:.4f} on the sRGB cube, so those colours reach the final clip "
                "below black (the path itself stays clamp-free).  Check "
                "diagnostics()['clip']['excursion_below_codes']"
            )

    # -- the grey axis, measured on the compiled pipeline ------------------
    # ENGINE_SPEC §8 gates this at 0.02 codes.  It is the one gate that cannot
    # be read off the parameters: [P] is exact on the grey axis by the stage-0
    # invariant, but in the NPG order [G] sees the grey that [N] has tinted
    # (d != 0, nm = tint/max(n, eps_dark)) and a low knee or a chroma cap can
    # compress the designed tint away.  Measured examples that used to validate
    # clean: a cap with start 0.005 -> 2.39 codes; tint +-40 with knee 0.30 ->
    # 0.85 codes.  Only run it when nothing structural is wrong, because it
    # needs the tone curve to be buildable.
    if not errors:
        from . import gamut as _gamut_mod  # lazy: same cycle as pipeline
        from . import pipeline as _pipeline  # lazy: pipeline imports spec

        try:
            c_ = _pipeline._compile_unchecked(spec)
            gerr, gt, gch_ = _pipeline.grey_axis_error_of(c_)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"grey axis: could not be measured ({exc})")
        else:
            # -- [G]: how flat the MEASURED compressor really is ---------
            # The only [G] rules above are knee in (0,1) and p >= 1: nothing
            # bounds q = (lim-knee)/(1-knee), and q is what decides whether the
            # curve compresses or erases.  Measured: {"knee": 0.999, "p": 8}
            # validates clean, compiles to lim 1.1701 / q 171.1, puts 9.43 % of
            # the lattice under a radial gain of 0.05 and moves the IDENTITY
            # lattice by 31.3 codes.  Both numbers come free from compile.
            if c_.gamut_lim is not None:
                if c_.gamut_flat_frac > GAMUT_FLAT_FRAC_WARN:
                    warnings.append(
                        f"gamut: the compressor is flat over "
                        f"{c_.gamut_flat_frac * 100:.2f} % of the 33**3 lattice "
                        f"(radial gain < {_gamut_mod.MIN_RADIAL_GAIN}, q = "
                        f"{c_.gamut_q:.2f}, lim = {c_.gamut_lim:.3f}, s1 = "
                        f"{g.end_slope:.2f}).  W5.2's slope floor bounds the gain "
                        f"below by s1/q = {g.end_slope / c_.gamut_q:.4f}, so a large "
                        "q still means a soft tail even though it can no longer "
                        "reach 0.  Lower the [P]/[S] gains (q follows lim follows push)"
                    )
                if c_.gamut_lim > _pipeline.LIM_WARN:
                    warnings.append(
                        f"gamut: lim = {c_.gamut_lim:.3f} > {_pipeline.LIM_WARN} — "
                        "v1.2 W6 keeps R5's lim line but WARN-only"
                    )
            if gerr > GREY_ERR_MAX:
                errors.append(
                    f"grey axis: max |out - T| = {gerr:.4f} codes > {GREY_ERR_MAX} at "
                    f"t = {gt:.4f} (input code {gt * 255:.1f}), channel {gch_} — the "
                    f"{spec.order} pipeline does not reproduce the designed grey response "
                    "(ENGINE_SPEC §8).  In NPG this is usually [G] compressing the tint: "
                    "raise gamut.knee, or drop a chroma cap whose start sits below the "
                    "tint's own chroma"
                )
            elif gerr > 0.5 * GREY_ERR_MAX:
                warnings.append(
                    f"grey axis: max |out - T| = {gerr:.4f} codes at t = {gt:.4f} "
                    f"— over half the {GREY_ERR_MAX}-code gate"
                )

    if errors:
        raise SpecError(
            f"look {spec.name!r}: {len(errors)} error(s)\n  - " + "\n  - ".join(errors)
        )
    return warnings


def load_look(path_or_name: str | Path, looks_dir: Path | None = None) -> LookSpec:
    """Load a look JSON by path, or by bare name from ``$R/looks/<name>.json``."""
    p = Path(path_or_name)
    if p.suffix == ".json" and p.exists():
        return LookSpec.from_json(p.read_text(encoding="utf-8"))
    base = looks_dir or (Path(__file__).resolve().parent.parent / "looks")
    cand = base / f"{p.stem if p.suffix else p.name}.json"
    if not cand.exists():
        raise FileNotFoundError(f"no look file at {p} or {cand}")
    return LookSpec.from_json(cand.read_text(encoding="utf-8"))
