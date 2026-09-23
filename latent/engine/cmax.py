"""``CMAX(L, h)`` — the smoothed sRGB gamut boundary in OKLCh (v1.2 W2).

W2 needs one number per colour: how much of the gamut's chroma the INPUT colour
already uses, ``r0 = min(C0 / CMAX(L0, h0), 1)``.  That is what makes the chroma
fade *gamut-relative* instead of code-domain, and therefore monotone by
construction (see :func:`engine.field.relative_fade`).  Two things here are not
W2's literal text, and both are defects of the ruling rather than of the
implementation: the ``min`` has a C^2 shoulder (:data:`R0_WIDTH`), and the
table is capped by the raw boundary (below).

The table, exactly as W2 specifies it:

* grid ``L = 0 … 1`` step 0.01 (101 rows) x ``h = 0 … 359 deg`` step 1 (360 cols);
* each cell = largest chroma whose **linear** sRGB triple is inside ``[0, 1]``,
  by vectorised bisection (48 halvings: the bracket ends 0.5*2**-48 wide);
* Gaussian-smoothed with ``sigma_L = 0.02`` (2 cells, edge-clamped) and
  ``sigma_h = 6 deg`` (6 cells, **periodic** in h);
* **then capped by the raw boundary** — see below;
* floored at 0.004;
* read back by bilinear interpolation, h wrapping, L clamped to [0, 1].

The cap (``np.minimum(smoothed, raw)``) is not decoration
--------------------------------------------------------
W2 says "because it is smoothed it may sit a little inside the true boundary
near the cusps — that is fine (r0 is clipped at 1)", and the whole safety claim
of W2 rests on that direction of the inequality: *at r0 = 1 the output chroma
equals the input chroma, so a colour on the shell stays on the shell*.  A table
that sits OUTSIDE the true shell breaks it — a shell colour then reads r0 < 1,
keeps part of its boost and leaves the gamut.

A Gaussian raises a function wherever it is locally convex, and the boundary
surface is convex nearly everywhere (it is a tent in L with a single ridge at
the cusp), so the plain smoothing was **above** the true boundary on 83.8 % of
the (L, h) grid: max excess +0.0302 at L = 1.00, h = 111 deg (table 0.0302
against a true boundary of 0), p99.9 +0.0180, and on the 33**3 lattice's own
(L0, h0) positive at 56.9 % of the nodes, max +0.0222.  Measured consequence,
same code path, only ``r0``'s denominator swapped: pure [S] at sat 1.4 drove the
pre-clip lattice 12.333 codes below 0 (2.85 % of nodes outside [0, 1]) with the
uncapped table and 0.675 codes / 0.006 % with the true boundary — i.e. ~95 % of
the excursion was table error, not the fade.

Capping costs nothing: the cap removes only the cells the Gaussian *raised*, so
the mean deficit against the raw boundary is unchanged at 0.00106.  What it does
change is where the table is smooth.  Near a cusp the raw surface has a ridge (a
local maximum), the Gaussian lowers it, and the cap keeps the SMOOTHED value —
so the ridge, which is the only real roughness, is still smoothed away.  Away
from the ridge the raw surface is analytic and the cap simply returns it.

What is left after the cap is bilinear-interpolation noise BETWEEN grid nodes:
on a 199 x 720 off-node (L, h) grid the excess over the true boundary is at
most +0.0035 with p99.9 +0.0020 (against +0.0302 / +0.0180 uncapped), and on
the table's own nodes the inequality is exact.

The table costs ~95 ms to build.  It is built **once per process** (module-level
cache) and persisted to ``~/.cache/latent-2026/cmax_v2.npz`` (1 ms to read) so
the second process does not pay for it either.  A cache that cannot be read or written is
not an error — the table is just rebuilt (the directory may be read-only, and
``$R/py`` keeps caches out of iCloud on purpose).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import color, curves
from .color import FloatArray

__all__ = [
    "cmax", "cmax_table", "relative_chroma", "soft_saturate", "cache_path", "clear_cache",
    "L_STEP", "H_STEP", "N_L", "N_H", "SIGMA_L", "SIGMA_H", "CMAX_FLOOR", "VERSION",
    "R0_WIDTH",
]

#: v2 = the raw-boundary cap of :func:`_build` (v1 shipped the uncapped
#: Gaussian).  The name is the cache key, so an old ``cmax_v1.npz`` is simply
#: ignored rather than silently reused.
VERSION = "cmax_v2"

#: grid of the table (W2)
L_STEP = 0.01
H_STEP = 1.0
N_L = 101          # L = 0, 0.01, ..., 1.00
N_H = 360          # h = 0, 1, ..., 359 (periodic)

#: Gaussian smoothing (W2), in the same units as the axes
SIGMA_L = 0.02
SIGMA_H = 6.0

#: W2's floor.  Keeps ``r0 = C0/CMAX`` finite at black and at white, where the
#: true boundary chroma is 0 and every colour would otherwise read as "on the
#: shell" (or divide by zero).
CMAX_FLOOR = 0.004

#: bisection depth; 48 halvings of [0, 0.5] is 1.8e-15, i.e. float64 exact
BISECT_ITERS = 48
#: sRGB's largest OKLab chroma is ~0.32, so 0.5 brackets every hue
BISECT_HI = 0.5

_TABLE: FloatArray | None = None


def cache_path() -> Path:
    return Path.home() / ".cache" / "latent-2026" / f"{VERSION}.npz"


def _bisect() -> FloatArray:
    """Raw boundary chroma on the (L, h) grid — the largest in-gamut chroma."""
    L = np.linspace(0.0, 1.0, N_L)
    h = np.arange(N_H, dtype=np.float64) * H_STEP
    LL, HH = np.meshgrid(L, h, indexing="ij")
    lo = np.zeros_like(LL)
    hi = np.full_like(LL, BISECT_HI)
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        lin = color.oklab_to_linear_srgb(color.lch_to_oklab(LL, mid, HH))
        ok = np.all((lin >= 0.0) & (lin <= 1.0), axis=-1)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


def _gauss_kernel(sigma_cells: float) -> FloatArray:
    r = max(1, int(np.ceil(4.0 * float(sigma_cells))))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma_cells)) ** 2)
    return k / k.sum()


def _smooth(tab: FloatArray) -> FloatArray:
    """Separable Gaussian: periodic along h, edge-clamped along L."""
    kh = _gauss_kernel(SIGMA_H / H_STEP)
    rh = (kh.size - 1) // 2
    acc = np.zeros_like(tab)
    for i, w in enumerate(kh):
        acc += w * np.roll(tab, i - rh, axis=1)
    tab = acc

    kl = _gauss_kernel(SIGMA_L / L_STEP)
    rl = (kl.size - 1) // 2
    idx = np.clip(np.arange(N_L)[:, None] + (np.arange(kl.size) - rl)[None, :],
                  0, N_L - 1)
    return np.einsum("ikh,k->ih", tab[idx], kl)


def _build() -> FloatArray:
    """Bisect, smooth, **cap by the raw boundary**, floor.

    The cap is what makes ``CMAX <= true boundary`` hold, which is the
    inequality W2's gamut-closure claim is built on (module docstring).
    """
    raw = _bisect()
    return np.maximum(np.minimum(_smooth(raw), raw), CMAX_FLOOR)


def cmax_table() -> FloatArray:
    """The ``(101, 360)`` table, built once per process and cached on disk."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    p = cache_path()
    try:
        if p.exists():
            with np.load(p) as z:
                tab = np.asarray(z["cmax"], dtype=np.float64)
            if tab.shape == (N_L, N_H):
                _TABLE = tab
                return _TABLE
    except Exception:  # pragma: no cover - a corrupt / unreadable cache
        pass
    tab = _build()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, cmax=tab)
    except Exception:  # pragma: no cover - read-only cache dir
        pass
    _TABLE = tab
    return _TABLE


def clear_cache(*, on_disk: bool = False) -> None:
    """Forget the in-process table (tests); optionally delete the npz too."""
    global _TABLE
    _TABLE = None
    if on_disk:
        try:
            cache_path().unlink(missing_ok=True)
        except Exception:  # pragma: no cover
            pass


def cmax(L: FloatArray, h: FloatArray) -> FloatArray:
    """Bilinear lookup of the boundary chroma at ``(L, h)``.

    ``L`` is clamped to [0, 1]; ``h`` wraps (359 -> 0 is one cell wide).
    """
    tab = cmax_table()
    L = np.clip(np.asarray(L, dtype=np.float64), 0.0, 1.0)
    h = np.asarray(h, dtype=np.float64) % 360.0

    x = L / L_STEP
    i0 = np.clip(np.floor(x).astype(np.intp), 0, N_L - 2)
    fx = x - i0
    y = h / H_STEP
    j0 = np.floor(y).astype(np.intp) % N_H
    fy = y - np.floor(y)
    j1 = (j0 + 1) % N_H

    c00 = tab[i0, j0]
    c01 = tab[i0, j1]
    c10 = tab[i0 + 1, j0]
    c11 = tab[i0 + 1, j1]
    return ((1.0 - fx) * ((1.0 - fy) * c00 + fy * c01)
            + fx * ((1.0 - fy) * c10 + fy * c11))


#: width of the C^2 shoulder that replaces W2's hard ``min(., 1)`` on ``r0``.
#:
#: W2 writes ``r0 = min(C0/CMAX, 1)``.  That ``min`` is a hard clamp in the
#: MIDDLE of the reachable input set — 15.03 % of the 33**3 lattice sits at
#: ``r0 = 1`` exactly and 36.03 % above 0.95 — so the shipped colour map had a
#: genuine C1 crease on the whole gamut shell, against ENGINE_SPEC §0's C2 rule
#: (the same rule that makes :func:`engine.curves.S5` raise on coincident
#: edges).  Measured on the full pipeline (pure [S], sat 1.4, gamut null), the
#: second difference refined at the crease read D(h) = 798 / 1987 / 18382 /
#: 182387 / 1822438 for h = 1e-2 … 1e-6, i.e. ``D ~ 1/h`` over four decades —
#: the signature of a slope discontinuity — while the same stencil with
#: ``field.relfade = 0`` converged (2.600 / 2.072 / 2.028 / 2.024).  The radial
#: chroma gain jumps from ``2 - g`` to 1 across the shell: 0.600 -> 1.000 at
#: g = 1.4.
#:
#: Chosen by measurement (report key ``w2_r0_width``); see
#: :func:`soft_saturate` for why the shoulder cannot simply reuse
#: :func:`engine.curves.soft_relu`.  The same refined stencil, and material
#: folds / interior d2 on the three looks W5.3 names:
#:
#: ===== ========================== ======== ========== ==========
#: w     D(h) 1e-2 / 1e-3 / 1e-5     03Gilt   04Viride   10Splice
#: ===== ========================== ======== ========== ==========
#: 0.00  798 / 1987 / 182387 (1/h)  192/8.42 1637/12.57 1833/12.80
#: 0.04  798 /  473 /    249          5/8.33  497/12.47 1751/12.77
#: 0.08  798 /  359 /    249          0/8.28  266/12.13 1692/12.71
#: 0.12  798 /  359 /    249          0/8.29  185/11.86 1637/12.72
#: 0.16  798 /  359 /    249          0/8.23  171/11.75 1558/12.82
#: 0.24  798 /  359 /    249          0/8.10  155/11.35 1429/12.81
#: ===== ========================== ======== ========== ==========
#:
#: The stencil stops diverging at w = 0.04 and is fully resolved from 0.08 up.
#: 0.12 is where the fold gains have essentially arrived (04Viride is 89 % of
#: the way from the clamp to w = 0.24) while the shoulder's price — the
#: ``1/(1-w/2) = 1.064`` stronger fade in the linear region, i.e. a *more*
#: conservative r0 than the ruling's, never a weaker one — is still small.
#: Above 0.12 the interior d2 of 10Splice turns back up.
R0_WIDTH = 0.12

#: ``soft_saturate`` divides by this so that ``rho(s) = 1`` exactly at s = 1.
def _sat_scale(w: float) -> float:
    return 1.0 - 0.5 * float(w)


def soft_saturate(s: FloatArray, width: float | None = None) -> FloatArray:
    """C^2 stand-in for ``min(s, 1)``: ``(s - soft_ramp(s-(1-w), w)) / (1-w/2)``.

    Properties, all exact:

    * ``rho(0) = 0``;  ``rho(s) = s/(1-w/2)`` for ``s <= 1-w`` (still *linear*
      in gamut-relative chroma, which is W2's point, 1/(1-w/2) = 1.064 at
      w = 0.12 — the fade is slightly STRONGER than the ruling's, never weaker);
    * ``rho(s) = 1`` for every ``s >= 1`` — a colour on the shell still reads
      "fully used", so W2's "a colour on the shell stays on the shell" holds
      exactly;
    * non-decreasing (``rho' = (1 - S5(0,w,s-(1-w)))/(1-w/2) >= 0``);
    * and, the reason for :func:`engine.curves.soft_ramp` rather than
      :func:`engine.curves.soft_relu`, ``d(s*rho)/ds <= 2`` — so W2's
      ``d(C*g_eff)/dC = g - kappa*soft_relu(g-1)*d(s*rho)/ds >= 2 - g``
      survives unchanged (measured max 1.890 at w = 0.12, against the hard
      clamp's exact 2.000).  ``soft_relu`` has slope up to ``16/9 = 1.7778``
      and would take that derivative to 2.576.
    """
    s = np.asarray(s, dtype=np.float64)
    w = R0_WIDTH if width is None else float(width)
    if w <= 0.0:
        return np.minimum(s, 1.0)
    return (s - curves.soft_ramp(s - (1.0 - w), w)) / _sat_scale(w)


def relative_chroma(L: FloatArray, C: FloatArray, h: FloatArray,
                    *, width: float | None = None) -> FloatArray:
    """W2's gamut-relative chroma ``r0``, with a C^2 shoulder at the shell.

    ``min(C/CMAX, 1)`` in the ruling; :func:`soft_saturate` of ``C/CMAX`` here.
    """
    C = np.asarray(C, dtype=np.float64)
    return soft_saturate(C / cmax(L, h), width)
