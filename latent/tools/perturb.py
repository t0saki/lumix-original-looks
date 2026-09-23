"""P5 capture-side perturbations of a base (exposure / white balance / noise).

A LUT is judged on the camera's *Standard* rendering.  The question P5 asks is:
if the photographer had been half a stop off, or had left the white balance on
the wrong preset, or had shot the same frame at ISO 12800 — does the look fall
apart?  So every perturbation here is applied to the BASE, in LINEAR light,
BEFORE the LUT, and the perturbed base must itself stay a plausible camera JPEG.

Everything in this module speaks *sRGB code values in [0, 1]* on the outside
(the same convention as ``tools.render`` and the base ``.npz`` files) and
linear light on the inside.

Perturbations
-------------
``exposure(ev)``
    ``lin * 2**ev``.  On the POSITIVE side the result goes through a
    hue-preserving soft shoulder: the shoulder is computed on ``max(R, G, B)``
    and applied as a single scale factor to all three channels, so a highlight
    rolls off without ever changing hue or turning into a per-channel clip.
    ``ev = 0`` is the exact identity (not "identity to 1e-16" — the same array).
``white_balance(cct, tint)``
    Bradford adaptation of the linear image from the shot's implicit D65 white
    to a daylight-locus white at *cct* kelvin, optionally shifted +-0.010 in
    CIE *y* for tint (+ = green, - = magenta; "tint +-10" in the spec is
    +-0.010 in xy).  The gains actually used are recorded: the Bradford cone
    ratios, the full 3x3 linear-sRGB matrix, and the per-channel gain a neutral
    white receives.
``noise(iso)``
    Signal-dependent Gaussian noise in linear light, ``sigma**2 = a*y + b``,
    per channel and independent, so it shows up as chroma noise as well as
    luma noise.  ``a`` and ``b`` are pinned in :data:`NOISE_MODEL`; seed 20260922.

Public API
----------
``perturb(base, ev=, cct=, tint=, iso=, seed=)`` -> ``(img, meta)``
``PERTURBATIONS`` — the 12 named perturbations P5 measures.
``apply_named(base, name)`` -> ``(img, meta)``
``shoulder(lin)``, ``exposure(lin, ev)``, ``white_balance(lin, cct, tint)``,
``add_noise(lin, iso, rng)``, ``bradford_matrix(src_xy, dst_xy)``,
``daylight_xy(cct)``, ``noise_sigma(y, iso)``

CLI::

    py tools/perturb.py --base PANA0116 --ev +1.0 --out pert.npz --preview pert.jpg
    py tools/perturb.py --base PANA0005 --iso 12800 --out n.npz --preview n.jpg
    py tools/perturb.py --base PANA0116 --name wb6000 --out wb.npz
    py tools/perturb.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from engine.color import srgb_decode, srgb_encode
from tools import render

__all__ = [
    "SEED",
    "KNEE",
    "D65_XY",
    "NOISE_MODEL",
    "PERTURBATIONS",
    "EV_STEPS",
    "shoulder",
    "shoulder_curve",
    "exposure",
    "daylight_xy",
    "bradford_matrix",
    "wb_gains",
    "white_balance",
    "noise_sigma",
    "add_noise",
    "perturb",
    "apply_named",
    "save_npz",
    "load_npz",
]

#: the one seed this project's noise is allowed to use (P5 spec).
SEED = 20260922

#: linear-light knee of the highlight shoulder.  Below it the shoulder is the
#: exact identity; 0.72 linear is sRGB code 0.874, i.e. only real highlights.
KNEE = 0.72

#: sRGB's white point (the "implicit D65" of every base).
D65_XY = (0.3127, 0.3290)

#: sigma**2 = a * y + b in LINEAR light, y in [0, 1], per channel.
#: b is the read-noise floor, a the photon term.  Calibrated so that at 1600 px
#: ISO 12800 puts ~3 codes (8-bit) of noise into a linear-0.01 shadow — visibly
#: grainy with visible chroma speckle — and ISO 3200 ~1 code (a/4, b/16, the
#: way output-referred sensor noise scales with two stops of gain).
NOISE_MODEL: dict[int, tuple[float, float]] = {
    3200: (3.0e-5, 1.5625e-7),
    12800: (1.2e-4, 2.5e-6),
}

#: exposure offsets P5 measures.
EV_STEPS: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)

# --------------------------------------------------------------------------- #
# the 12 named perturbations
# --------------------------------------------------------------------------- #
def _ev_name(ev: float) -> str:
    return f"ev{ev:+.1f}"


PERTURBATIONS: dict[str, dict] = {
    **{_ev_name(ev): {"ev": ev} for ev in EV_STEPS},
    "wb6000": {"cct": 6000.0},
    "wb7000": {"cct": 7000.0},
    "tint+10": {"tint": +10.0},
    "tint-10": {"tint": -10.0},
    "iso3200": {"iso": 3200},
    "iso12800": {"iso": 12800},
}

#: perturbations that legitimately push the scene into the highlight shoulder;
#: the clip-creep gate excepts the strongest of them (P5 spec).
CLIP_EXEMPT = ("ev+1.5",)

_FAMILY = {
    **{_ev_name(ev): "exposure" for ev in EV_STEPS},
    "wb6000": "wb",
    "wb7000": "wb",
    "tint+10": "wb",
    "tint-10": "wb",
    "iso3200": "noise",
    "iso12800": "noise",
}


def family(name: str) -> str:
    """``"exposure"`` / ``"wb"`` / ``"noise"`` for a named perturbation."""
    return _FAMILY[name]


# --------------------------------------------------------------------------- #
# highlight shoulder (hue preserving, max-RGB)
# --------------------------------------------------------------------------- #
def shoulder_curve(m: np.ndarray | float, white: float, knee: float = KNEE) -> np.ndarray:
    """1-D shoulder mapping ``[knee, white] -> [knee, 1]``.

    ``f(m) = 1 - (1-knee) * ((white - m) / (white - knee)) ** p`` with
    ``p = (white - knee) / (1 - knee)``, which gives all four properties the
    spec needs at once:

    * ``f(knee) = knee`` and ``f'(knee) = 1`` — no crease where it starts;
    * ``f(white) = 1`` and ``f'(white) = 0`` — the gain's new white lands
      exactly on white and lands flat, so a highlight that WAS clipped before
      the perturbation is still clipped after it (this is what makes the clip
      metrics mean anything);
    * strictly monotone on ``[knee, white]``;
    * at ``white = 1`` the exponent is 1 and the whole thing collapses to
      ``f(m) = m`` — the EXACT identity, which is why an unperturbed base and
      a noise-only perturbation never go near a shoulder.
    """
    m = np.asarray(m, dtype=np.float64)
    if not 0.0 < knee < 1.0:
        raise ValueError(f"knee must be in (0, 1), got {knee!r}")
    if white < 1.0:
        raise ValueError(f"shoulder white point must be >= 1, got {white!r}")
    if white == 1.0:
        return np.minimum(m, 1.0)
    p = (white - knee) / (1.0 - knee)
    t = np.clip((white - m) / (white - knee), 0.0, 1.0)
    return np.where(m <= knee, m, 1.0 - (1.0 - knee) * t**p)


def shoulder(lin: np.ndarray, white: float = 1.0, knee: float = KNEE) -> np.ndarray:
    """Hue-preserving soft shoulder on ``max(R, G, B)`` in linear light.

    ``m = max(R, G, B)`` is put through :func:`shoulder_curve` and all three
    channels are multiplied by the SAME factor ``f(m)/m``, so the ratios
    between channels — and therefore hue and saturation — are untouched: the
    highlight rolls off toward white only as far as the shoulder pushes the
    whole triple down, never by clipping one channel before the others.

    *white* is the value the perturbation's gain puts the old white at
    (``2**ev`` for exposure, the largest white-balance gain for WB, the product
    when both apply).  Values with ``m <= knee`` pass through EXACTLY.
    """
    lin = np.asarray(lin, dtype=np.float64)
    m = lin.max(axis=-1)
    over = m > knee
    if not np.any(over):
        return lin
    compressed = shoulder_curve(m, white, knee)
    scale = np.where(over, compressed / np.where(m > 0.0, m, 1.0), 1.0)
    return lin * scale[..., None]


def exposure(lin: np.ndarray, ev: float, knee: float = KNEE) -> np.ndarray:
    """``lin * 2**ev``; the positive side goes through :func:`shoulder`.

    ``ev == 0`` returns the input array itself (exact identity).
    """
    ev = float(ev)
    if ev == 0.0:
        return np.asarray(lin, dtype=np.float64)
    gain = 2.0**ev
    out = np.asarray(lin, dtype=np.float64) * gain
    return shoulder(out, gain, knee) if ev > 0.0 else out


# --------------------------------------------------------------------------- #
# white balance (Bradford)
# --------------------------------------------------------------------------- #
_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)
_BRADFORD_INV = np.linalg.inv(_BRADFORD)

# IEC 61966-2-1 sRGB (D65) primaries.
_XYZ_FROM_RGB = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)


def daylight_xy(cct: float) -> tuple[float, float]:
    """CIE D-series daylight chromaticity for *cct* kelvin (4000..25000 K)."""
    t = float(cct)
    if not 4000.0 <= t <= 25000.0:
        raise ValueError(f"the CIE daylight locus is defined for 4000..25000 K, got {t}")
    if t <= 7000.0:
        x = -4.6070e9 / t**3 + 2.9678e6 / t**2 + 0.09911e3 / t + 0.244063
    else:
        x = -2.0064e9 / t**3 + 1.9018e6 / t**2 + 0.24748e3 / t + 0.237040
    y = -3.000 * x * x + 2.870 * x - 0.275
    return float(x), float(y)


def _xyz_from_xy(xy: Sequence[float]) -> np.ndarray:
    x, y = float(xy[0]), float(xy[1])
    if y <= 0.0:
        raise ValueError(f"illegal chromaticity {xy!r}")
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def bradford_matrix(src_xy: Sequence[float], dst_xy: Sequence[float]) -> np.ndarray:
    """3x3 matrix taking LINEAR sRGB under *src_xy* to linear sRGB under *dst_xy*.

    Bradford cone response domain (the CAT the spec names), wrapped in the
    sRGB <-> XYZ matrices so the caller only ever sees linear RGB.
    """
    rho_s = _BRADFORD @ _xyz_from_xy(src_xy)
    rho_d = _BRADFORD @ _xyz_from_xy(dst_xy)
    cat = _BRADFORD_INV @ np.diag(rho_d / rho_s) @ _BRADFORD
    return _RGB_FROM_XYZ @ cat @ _XYZ_FROM_RGB


def _target_white(cct: float | None, tint: float) -> tuple[float, float]:
    """Destination white: the daylight locus at *cct*, shifted by *tint*.

    ``tint`` is in the spec's units of ten-per-0.010: ``+10`` moves CIE *y* up
    by 0.010 (a greener illuminant, therefore a greener picture), ``-10`` down
    (magenta).  ``cct=None`` keeps D65.
    """
    x, y = daylight_xy(cct) if cct is not None else D65_XY
    return float(x), float(y + 0.001 * float(tint))


def wb_gains(cct: float | None = None, tint: float = 0.0) -> dict:
    """The white-balance transform and the gains it actually applies."""
    dst = _target_white(cct, tint)
    matrix = bradford_matrix(D65_XY, dst)
    white = matrix @ np.ones(3)
    rho_s = _BRADFORD @ _xyz_from_xy(D65_XY)
    rho_d = _BRADFORD @ _xyz_from_xy(dst)
    return {
        "cct": None if cct is None else float(cct),
        "tint": float(tint),
        "src_white_xy": [float(v) for v in D65_XY],
        "dst_white_xy": [float(v) for v in dst],
        "cat": "bradford",
        # the gains: what a neutral white receives, the matrix diagonal, and the
        # underlying Bradford cone ratios.
        "white_gains": [float(v) for v in white],
        "diag_gains": [float(v) for v in np.diag(matrix)],
        "lms_gains": [float(v) for v in (rho_d / rho_s)],
        "matrix": [[float(v) for v in row] for row in matrix],
    }


def white_balance(lin: np.ndarray, cct: float | None = None, tint: float = 0.0) -> tuple[np.ndarray, dict]:
    """Bradford-adapt linear sRGB from D65 to *cct*/*tint*; returns (img, gains)."""
    info = wb_gains(cct, tint)
    matrix = np.asarray(info["matrix"], dtype=np.float64)
    out = np.asarray(lin, dtype=np.float64) @ matrix.T
    return out, info


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #
def noise_sigma(y: np.ndarray | float, iso: int) -> np.ndarray:
    """``sqrt(a*y + b)`` — the linear-light noise sigma of the ISO model."""
    if int(iso) not in NOISE_MODEL:
        raise KeyError(f"no noise model for ISO {iso}; have {sorted(NOISE_MODEL)}")
    a, b = NOISE_MODEL[int(iso)]
    return np.sqrt(np.maximum(a * np.maximum(np.asarray(y, dtype=np.float64), 0.0) + b, 0.0))


def add_noise(lin: np.ndarray, iso: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add signal-dependent Gaussian noise to a LINEAR image, per channel."""
    lin = np.asarray(lin, dtype=np.float64)
    rng = rng if rng is not None else np.random.default_rng(SEED)
    sigma = noise_sigma(lin, iso)
    return lin + sigma * rng.standard_normal(lin.shape)


# --------------------------------------------------------------------------- #
# the perturbation itself
# --------------------------------------------------------------------------- #
def perturb(
    base: np.ndarray,
    *,
    ev: float = 0.0,
    cct: float | None = None,
    tint: float = 0.0,
    iso: int | None = None,
    seed: int = SEED,
    knee: float = KNEE,
) -> tuple[np.ndarray, dict]:
    """Perturb a base (sRGB code values) and return ``(image, metadata)``.

    Order: exposure -> white balance -> noise, all in linear light; then one
    shoulder/clip pass and back to code values.  The shoulder is used when a
    stage can push values ABOVE 1 for a photographic reason (positive exposure,
    or a WB gain > 1) — that is the "still a plausible camera JPEG" clause.
    Sensor noise is not such a reason, so a pure-ISO perturbation clips hard,
    exactly as a sensor does.

    With no perturbation requested the input array is returned untouched (an
    exact copy, not a round trip through the transfer function).
    """
    src = np.asarray(base, dtype=np.float64)
    if src.ndim < 1 or src.shape[-1] != 3:
        raise ValueError(f"base must end in a 3-channel axis, got {src.shape}")
    ev = float(ev)
    tint = float(tint)
    iso = None if iso in (None, 0, "") else int(iso)

    meta: dict = {
        "ev": ev,
        "cct": None if cct is None else float(cct),
        "tint": tint,
        "iso": iso,
        "seed": int(seed),
        "knee": float(knee),
        "shoulder": False,
        "shoulder_white": 1.0,
        "wb": None,
        "noise": None,
    }

    if ev == 0.0 and cct is None and tint == 0.0 and iso is None:
        meta["identity"] = True
        return src.copy(), meta
    meta["identity"] = False

    lin = srgb_decode(src)
    white = 1.0  # what the stages' gains do to the old white

    if ev != 0.0:
        gain = 2.0**ev
        lin = lin * gain
        white *= max(1.0, gain)

    if cct is not None or tint != 0.0:
        lin, wb_info = white_balance(lin, cct, tint)
        meta["wb"] = wb_info
        white *= max(1.0, max(wb_info["white_gains"]))

    if iso is not None:
        rng = np.random.default_rng(int(seed))
        a, b = NOISE_MODEL[iso]
        lin = add_noise(lin, iso, rng)
        meta["noise"] = {
            "iso": iso,
            "a": a,
            "b": b,
            "model": "sigma^2 = a*y + b (linear light, per channel, independent)",
            "seed": int(seed),
            "sigma_at_linear_0.01": float(noise_sigma(0.01, iso)),
            "sigma_at_linear_0.18": float(noise_sigma(0.18, iso)),
        }

    if white > 1.0:
        lin = shoulder(lin, white, knee)
        meta["shoulder"] = True
        meta["shoulder_white"] = float(white)
    lin = np.clip(lin, 0.0, 1.0)
    return srgb_encode(lin), meta


def apply_named(base: np.ndarray, name: str, *, seed: int = SEED) -> tuple[np.ndarray, dict]:
    """Apply one of :data:`PERTURBATIONS` by name."""
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; have {sorted(PERTURBATIONS)}")
    img, meta = perturb(base, seed=seed, **PERTURBATIONS[name])
    meta["name"] = name
    meta["family"] = family(name)
    return img, meta


def perturb_many(
    base: np.ndarray, names: Iterable[str] | None = None, *, seed: int = SEED
) -> dict[str, tuple[np.ndarray, dict]]:
    """``{name: (image, meta)}`` for every requested perturbation."""
    names = list(names) if names is not None else list(PERTURBATIONS)
    return {n: apply_named(base, n, seed=seed) for n in names}


# --------------------------------------------------------------------------- #
# i/o
# --------------------------------------------------------------------------- #
def save_npz(path: str | Path, img: np.ndarray, meta: dict) -> Path:
    """Write a perturbed base the same way a base is written: float16 key ``rgb``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rgb=np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0).astype(np.float16),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    return path


def load_npz(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read back :func:`save_npz` -> ``(float64 code values, meta)``."""
    with np.load(Path(path), allow_pickle=False) as handle:
        img = np.asarray(handle["rgb"], dtype=np.float64)
        meta = json.loads(str(handle["meta"])) if "meta" in handle else {}
    return img, meta


def save_preview(path: str | Path, img: np.ndarray, quality: int = 92) -> Path:
    """Write a JPEG preview (sRGB ICC embedded, 4:4:4)."""
    from PIL import ImageCms

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    render.to_pil(img).save(path, "JPEG", quality=int(quality), subsampling=0, icc_profile=icc)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perturb.py", description=__doc__.split("\n")[0])
    ap.add_argument("--base", help="base id (PANA0116) or path to an .npz")
    ap.add_argument("--name", help=f"named perturbation: {', '.join(PERTURBATIONS)}")
    ap.add_argument("--ev", type=float, default=0.0)
    ap.add_argument("--wb", type=float, default=None, help="target CCT in kelvin (6000 / 7000)")
    ap.add_argument("--tint", type=float, default=0.0, help="+10 = green, -10 = magenta (+-0.010 in xy)")
    ap.add_argument("--iso", type=int, default=None, help=f"noise ISO: {sorted(NOISE_MODEL)}")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", help="output .npz")
    ap.add_argument("--preview", help="output .jpg")
    ap.add_argument("--list", action="store_true", help="print the named perturbations and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name, kw in PERTURBATIONS.items():
            print(f"{name:10s} {family(name):8s} {kw}")
        return 0
    if not args.base:
        ap.error("--base is required (or use --list)")

    img = render.load_base(args.base)
    if args.name:
        out, meta = apply_named(img, args.name, seed=args.seed)
    else:
        out, meta = perturb(img, ev=args.ev, cct=args.wb, tint=args.tint, iso=args.iso, seed=args.seed)
        meta["name"] = args.name or "custom"
    meta["base"] = str(args.base)

    delta = np.abs(out - img)
    print(json.dumps({k: v for k, v in meta.items() if k != "wb"}, ensure_ascii=False))
    if meta.get("wb"):
        g = meta["wb"]["white_gains"]
        print(f"wb white gains  R {g[0]:.4f}  G {g[1]:.4f}  B {g[2]:.4f}  -> {meta['wb']['dst_white_xy']}")
    print(
        f"delta codes8: mean {delta.mean() * 255:.3f}  p99 {np.percentile(delta, 99) * 255:.3f}  "
        f"max {delta.max() * 255:.3f}   clipped0 {(out <= 0).mean() * 100:.3f}%  "
        f"clipped1 {(out >= 1).mean() * 100:.3f}%"
    )
    if args.out:
        print("wrote", save_npz(args.out, out, meta))
    if args.preview:
        print("wrote", save_preview(args.preview, out))
    if not args.out and not args.preview:
        print("(nothing written: pass --out and/or --preview)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
