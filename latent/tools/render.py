"""Image-domain rendering helpers for Latent-2026 review sheets (T4).

Everything in this module speaks *sRGB code values in [0, 1]* as float arrays
shaped ``(..., 3)``.  That is exactly what the base ``.npz`` files, the
synthetic charts and the 33-point ``.cube`` LUTs all use, so no gamma or
colour-space conversion ever happens implicitly here.

Public API
----------
``apply_table(img, table, strength=1.0)``
    Run an image through a LUT table (tetrahedral, float, chunked).
``load_base(name)``
    Load a base scene from ``$W/bases`` (and the other base tiers).
``load_cube(path)`` / ``table_of(path_or_table)``
    Read a ``.cube`` into an ``engine.cubeio.LUT3D`` / into a raw table.
``crop(img, box)`` / ``crop_box(base_id, crop_name)`` / ``load_crops()``
    Frozen crop rectangles from ``$W/bases_hi/crops.json``.
``resize_long_edge(img, n)`` / ``to_uint8(img)`` / ``to_pil(img)``
    Float-domain Lanczos resampling and 8-bit conversion.
``amplified_difference(before, after, gain=8.0)``
    ``0.5 + gain * (after - before)`` — the difference row of a chart sheet.
``patch_measure(before, after, box)``
    Mean OKLab/OKLCh of a rectangle before and after, plus dL / dC / dh.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from engine.cubeio import LUT3D, read_lut, sha256_file, tetrahedral_interpolation
from tools import metrics

__all__ = [
    "ROOT",
    "WORK",
    "apply_table",
    "load_base",
    "base_ids",
    "load_cube",
    "table_of",
    "identity_table",
    "crop",
    "crop_box",
    "load_crops",
    "resize_long_edge",
    "to_uint8",
    "to_pil",
    "from_pil",
    "amplified_difference",
    "oklab_mean",
    "patch_measure",
    "sha256_file",
]

ROOT = Path(os.environ.get("LATENT_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("LATENT_WORK", ROOT / "work.nosync"))

#: directories searched by :func:`load_base`, in order.
BASE_DIRS: tuple[Path, ...] = (
    WORK / "bases",
    WORK / "bases_core",
    WORK / "bases_holdout",
    WORK / "bases_hi",
    WORK / "incam",
    WORK / "bases" / "_legacy_uncal",
)

CROPS_JSON = WORK / "bases_hi" / "crops.json"

# Tetrahedral sampling materialises eight corner gathers; chunk so that a
# 3000x2000 base never allocates more than ~200 MB at once.
_CHUNK = 400_000


# --------------------------------------------------------------------------- #
# LUT tables
# --------------------------------------------------------------------------- #
def identity_table(size: int = 33) -> np.ndarray:
    """Identity LUT table, ``(size, size, size, 3)`` in (B, G, R, 3) order."""
    return metrics.identity_table(size)


def load_cube(path: str | Path) -> LUT3D:
    """Read a ``.cube`` file (thin wrapper over :func:`engine.cubeio.read_lut`)."""
    return read_lut(Path(path))


def table_of(spec: str | Path | np.ndarray | LUT3D | None) -> np.ndarray:
    """Coerce *spec* into a LUT table.

    ``None`` / ``"base"`` / ``"identity"`` / ``"none"`` give the 33-point
    identity; a path gives that ``.cube``'s table; an array is returned as a
    float64 copy.
    """
    if spec is None:
        return identity_table(33)
    if isinstance(spec, LUT3D):
        return np.asarray(spec.table, dtype=np.float64)
    if isinstance(spec, np.ndarray):
        table = np.asarray(spec, dtype=np.float64)
        if table.ndim != 4 or table.shape[-1] != 3:
            raise ValueError(f"LUT table must be (N, N, N, 3), got {table.shape}")
        return table
    text = str(spec)
    if text.lower() in {"base", "identity", "none", "id"}:
        return identity_table(33)
    return np.asarray(load_cube(text).table, dtype=np.float64)


def apply_table(
    img: np.ndarray, table: np.ndarray | LUT3D | str | Path | None, strength: float = 1.0
) -> np.ndarray:
    """Apply a LUT *table* to *img* (code values in [0, 1]).

    Tetrahedral interpolation, identical to the camera's sampling and to
    ``engine.cubeio.tetrahedral_interpolation``.  *strength* blends in code
    values exactly the way the camera's Real Time LUT strength is assumed to:
    ``s * LUT(x) + (1 - s) * x``.  Input is clipped to [0, 1] before sampling
    (the LUT domain); the result is NOT clipped beyond what the table holds.
    """
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength!r}")
    src = np.asarray(img, dtype=np.float64)
    if src.ndim < 1 or src.shape[-1] != 3:
        raise ValueError(f"image must end in a 3-channel axis, got {src.shape}")
    lut = LUT3D(table=table_of(table))
    flat = np.clip(src.reshape(-1, 3), 0.0, 1.0)
    out = np.empty_like(flat)
    for start in range(0, flat.shape[0], _CHUNK):
        stop = min(start + _CHUNK, flat.shape[0])
        out[start:stop] = tetrahedral_interpolation(lut, flat[start:stop])
    s = float(strength)
    if s != 1.0:
        out = s * out + (1.0 - s) * flat
    return out.reshape(src.shape)


# --------------------------------------------------------------------------- #
# bases
# --------------------------------------------------------------------------- #
_BASE_KEYS = ("rgb", "image", "img", "base", "arr", "data")


def _read_npz(path: Path) -> np.ndarray:
    with np.load(path) as handle:
        keys = list(handle.keys())
        key = next((k for k in _BASE_KEYS if k in keys), None)
        if key is None:
            arrays = [k for k in keys if handle[k].ndim == 3]
            if len(arrays) != 1:
                raise KeyError(f"{path}: cannot identify the image array among {keys}")
            key = arrays[0]
        arr = np.asarray(handle[key], dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{path}: expected (H, W, 3), got {arr.shape}")
    return arr


def load_base(name: str | Path, dirs: Iterable[Path] | None = None) -> np.ndarray:
    """Load a base scene as float64 sRGB code values, shape ``(H, W, 3)``.

    *name* is a base id (``"PANA9997"``), an id with extension, or a path.
    Searched: ``$W/bases``, ``bases_core``, ``bases_holdout``, ``bases_hi``,
    ``incam``, ``bases/_legacy_uncal`` — the first hit wins.
    """
    candidate = Path(name)
    if candidate.suffix == ".npz" and candidate.exists():
        return _read_npz(candidate)
    stem = candidate.stem if candidate.suffix else str(name)
    for directory in dirs if dirs is not None else BASE_DIRS:
        path = Path(directory) / f"{stem}.npz"
        if path.exists():
            return _read_npz(path)
    searched = ", ".join(str(d) for d in (dirs if dirs is not None else BASE_DIRS))
    raise FileNotFoundError(f"base {stem!r} not found in: {searched}")


def base_ids(dirs: Iterable[Path] | None = None) -> list[str]:
    """Every base id reachable by :func:`load_base`, de-duplicated, sorted."""
    found: dict[str, None] = {}
    for directory in dirs if dirs is not None else BASE_DIRS:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.npz")):
            found.setdefault(path.stem, None)
    return sorted(found)


# --------------------------------------------------------------------------- #
# crops
# --------------------------------------------------------------------------- #
def load_crops(path: Path | None = None) -> dict:
    """Load the frozen crop rectangles (``$W/bases_hi/crops.json``).

    Raises ``FileNotFoundError`` with an explicit message when the file has not
    been produced yet (it is frozen by T5, not by this module).
    """
    path = Path(path) if path is not None else CROPS_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"frozen crop boxes not found at {path} (written by tools/prepare_bases.py)"
        )
    return json.loads(path.read_text())


def crop_box(base_id: str, crop_name: str, path: Path | None = None) -> tuple[float, float, float, float]:
    """Look up one frozen box as ``(x0, y0, x1, y1)``.

    Accepts both shapes a ``crops.json`` may take: ``{id: {name: box}}`` and
    ``{id: {name: {"box": box, "tag": ...}}}``.
    """
    crops = load_crops(path)
    entry = crops.get(base_id)
    if entry is None and isinstance(crops.get("crops"), dict):
        entry = crops["crops"].get(base_id)
    if entry is None:
        raise KeyError(f"no crops for base {base_id!r} in {path or CROPS_JSON}")
    box = entry.get(crop_name)
    if box is None:
        raise KeyError(f"crop {crop_name!r} not defined for {base_id!r}")
    if isinstance(box, dict):
        box = box.get("box", box.get("rect"))
    if box is None or len(box) != 4:
        raise ValueError(f"malformed crop box for {base_id}/{crop_name}: {box!r}")
    return tuple(float(v) for v in box)  # type: ignore[return-value]


def crop(img: np.ndarray, box: Sequence[float] | None) -> np.ndarray:
    """Crop *img* to ``box = (x0, y0, x1, y1)``.

    Boxes whose values are all <= 1.0 are treated as normalised fractions of
    the image size; otherwise they are pixel coordinates.  ``None`` is a no-op.
    """
    if box is None:
        return np.asarray(img)
    arr = np.asarray(img)
    if len(box) != 4:
        raise ValueError(f"box must be (x0, y0, x1, y1), got {box!r}")
    height, width = arr.shape[0], arr.shape[1]
    x0, y0, x1, y1 = (float(v) for v in box)
    if max(x0, y0, x1, y1) <= 1.0:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    ix0, ix1 = sorted((int(round(x0)), int(round(x1))))
    iy0, iy1 = sorted((int(round(y0)), int(round(y1))))
    ix0, iy0 = max(0, ix0), max(0, iy0)
    ix1, iy1 = min(width, ix1), min(height, iy1)
    if ix1 - ix0 < 1 or iy1 - iy0 < 1:
        raise ValueError(f"empty crop {box!r} on a {width}x{height} image")
    return arr[iy0:iy1, ix0:ix1]


# --------------------------------------------------------------------------- #
# resampling / 8-bit
# --------------------------------------------------------------------------- #
def resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    """Lanczos-resample so ``max(H, W) == long_edge`` (float, per channel).

    Resampling happens in code space on float data, so the amplified-difference
    row keeps its precision (do the amplification at native resolution first).
    """
    arr = np.asarray(img, dtype=np.float64)
    height, width = arr.shape[0], arr.shape[1]
    if max(height, width) == long_edge:
        return arr
    scale = long_edge / float(max(height, width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    planes = [
        np.asarray(
            Image.fromarray(arr[..., c].astype(np.float32), mode="F").resize(
                (new_w, new_h), Image.LANCZOS
            ),
            dtype=np.float64,
        )
        for c in range(3)
    ]
    return np.stack(planes, axis=-1)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] and quantise to 8-bit."""
    clipped = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)


def to_pil(img: np.ndarray) -> Image.Image:
    """Float code values -> 8-bit RGB :class:`PIL.Image.Image`."""
    return Image.fromarray(to_uint8(img), mode="RGB")


def from_pil(image: Image.Image) -> np.ndarray:
    """8-bit :class:`PIL.Image.Image` -> float64 code values."""
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


# --------------------------------------------------------------------------- #
# difference / measurement
# --------------------------------------------------------------------------- #
def amplified_difference(before: np.ndarray, after: np.ndarray, gain: float = 8.0) -> np.ndarray:
    """``0.5 + gain * (after - before)``, clipped to [0, 1].

    Mid-grey means "no change"; warmer/cooler, lighter/darker deviations become
    visible colour at ``gain`` times their real size.  Computed at native
    resolution — downsample afterwards, never before.
    """
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return np.clip(0.5 + float(gain) * (b - a), 0.0, 1.0)


def oklab_mean(region: np.ndarray) -> dict:
    """Mean OKLab / OKLCh of a region of code values.

    The mean is taken in OKLab (a, b are linear in the opponent sense, so the
    mean is meaningful); C and h are derived from the mean a, b.
    """
    arr = np.asarray(region, dtype=np.float64).reshape(-1, 3)
    L, C, h = metrics.oklch_from_code(arr)
    a = C * np.cos(np.deg2rad(h))
    b = C * np.sin(np.deg2rad(h))
    Lm, am, bm = float(L.mean()), float(a.mean()), float(b.mean())
    Cm = float(np.hypot(am, bm))
    hm = float(np.rad2deg(np.arctan2(bm, am)) % 360.0)
    return {
        "L": Lm,
        "a": am,
        "b": bm,
        "C": Cm,
        "h": hm,
        "rgb": [float(v) for v in arr.mean(axis=0)],
        "n": int(arr.shape[0]),
    }


def patch_measure(before: np.ndarray, after: np.ndarray, box: Sequence[float] | None = None) -> dict:
    """Measure one patch before/after a LUT.

    Returns the two OKLab means plus ``dL`` (absolute), ``cr`` (chroma ratio
    after/before), ``dh`` (signed degrees, wrapped to +/-180) and the mean code
    values, which are what the patch strip prints.
    """
    b_mean = oklab_mean(crop(before, box))
    a_mean = oklab_mean(crop(after, box))
    # On a patch that is essentially neutral the hue angle is numerical noise,
    # so report nan rather than an impressive-looking made-up rotation.
    if min(a_mean["C"], b_mean["C"]) < 1e-4:
        dh = float("nan")
        cr = float("nan")
    else:
        dh = (a_mean["h"] - b_mean["h"] + 180.0) % 360.0 - 180.0
        cr = a_mean["C"] / b_mean["C"]
    return {
        "before": b_mean,
        "after": a_mean,
        "dL": a_mean["L"] - b_mean["L"],
        "dC": a_mean["C"] - b_mean["C"],
        "cr": cr,
        "dh": dh,
    }
