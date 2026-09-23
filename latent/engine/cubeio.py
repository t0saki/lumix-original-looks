"""Minimal self-contained .cube I/O and sampling.

This repo used to lean on the sibling ``lumix-lut-converter`` project for
cube reading/writing and tetrahedral sampling; these are vendored here so
the generator is fully standalone.

Conventions (must not change — the shipped .cube files depend on them):
- ``LUT3D.table`` axes are (blue, green, red, 3) because CUBE rows change
  red fastest.
- ``write_cube`` emits the house header used by our LUMIX S9 workflow:
  TITLE → #LUMIXPHOTOSTYLE → free comments → LUT_3D_SIZE → blank line →
  rows with 10 decimals, clipped to [0, 1].
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TITLE_RE = re.compile(r'^TITLE\s+["\']?(.*?)["\']?\s*$')


@dataclass(frozen=True)
class LUT3D:
    """A normalized RGB 3D LUT stored in CUBE ordering (B, G, R, 3)."""

    table: np.ndarray
    title: str = "Untitled"
    domain_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    domain_max: np.ndarray = field(default_factory=lambda: np.ones(3))
    comments: tuple[str, ...] = ()
    source: Path | None = None

    def __post_init__(self) -> None:
        table = np.asarray(self.table, dtype=np.float64)
        if table.ndim != 4 or table.shape[-1] != 3:
            raise ValueError("A 3D LUT table must have shape (N, N, N, 3)")
        if not (table.shape[0] == table.shape[1] == table.shape[2]):
            raise ValueError("3D LUT axes must have equal sizes")
        object.__setattr__(self, "table", table)
        object.__setattr__(
            self, "domain_min", np.asarray(self.domain_min, dtype=np.float64)
        )
        object.__setattr__(
            self, "domain_max", np.asarray(self.domain_max, dtype=np.float64)
        )

    @property
    def size(self) -> int:
        return int(self.table.shape[0])

    def normalise_input(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        return (rgb - self.domain_min) / (self.domain_max - self.domain_min)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lut(path: Path) -> LUT3D:
    path = Path(path)
    if path.suffix.lower() != ".cube":
        raise ValueError(f"Unsupported LUT format: {path.suffix}")
    size: int | None = None
    title = path.stem
    domain_min = np.zeros(3)
    domain_max = np.ones(3)
    comments: list[str] = []
    rows: list[list[float]] = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
            continue
        title_match = TITLE_RE.match(line)
        if title_match:
            title = title_match.group(1)
            continue
        if line.startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue
        if line.startswith("LUT_1D_SIZE"):
            raise ValueError("1D LUTs are not supported")
        if line.startswith("DOMAIN_MIN"):
            domain_min = np.asarray([float(x) for x in line.split()[1:4]])
            continue
        if line.startswith("DOMAIN_MAX"):
            domain_max = np.asarray([float(x) for x in line.split()[1:4]])
            continue
        fields = line.split()
        if len(fields) == 3:
            rows.append([float(f) for f in fields])
    if size is None:
        raise ValueError("LUT_3D_SIZE is missing")
    if len(rows) != size**3:
        raise ValueError(f"Expected {size**3} LUT rows, found {len(rows)}")
    table = np.asarray(rows, dtype=np.float64).reshape(size, size, size, 3)
    return LUT3D(
        table=table,
        title=title,
        domain_min=domain_min,
        domain_max=domain_max,
        comments=tuple(comments),
        source=path,
    )


def write_cube(
    path: Path,
    lut: LUT3D,
    *,
    photo_style: str,
    comments: tuple[str, ...] = (),
    decimals: int = 10,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'TITLE "{lut.title}"',
        f"#LUMIXPHOTOSTYLE {photo_style}",
        *comments,
        f"LUT_3D_SIZE {lut.size}",
        "",
    ]
    formatter = f"{{:.{decimals}f}} {{:.{decimals}f}} {{:.{decimals}f}}"
    for row in lut.table.reshape(-1, 3):
        # LUMIX camera LUTs are defined over normalized display code values.
        clipped = np.clip(row, 0.0, 1.0)
        lines.append(formatter.format(*clipped))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tetrahedral_interpolation(lut: LUT3D, rgb: np.ndarray) -> np.ndarray:
    """Sample *lut* with vectorised tetrahedral interpolation."""
    original_shape = np.asarray(rgb).shape
    if not original_shape or original_shape[-1] != 3:
        raise ValueError("RGB input must end with a three-channel dimension")

    points = lut.normalise_input(np.asarray(rgb, dtype=np.float64)).reshape(-1, 3)
    points = np.clip(points, 0.0, 1.0)
    q = points * (lut.size - 1)
    lower = np.floor(q).astype(np.int32)
    fraction = q - lower
    upper = np.minimum(lower + 1, lut.size - 1)

    r0, g0, b0 = lower[:, 0], lower[:, 1], lower[:, 2]
    r1, g1, b1 = upper[:, 0], upper[:, 1], upper[:, 2]
    fr, fg, fb = fraction[:, 0:1], fraction[:, 1:2], fraction[:, 2:3]
    table = lut.table

    c000 = table[b0, g0, r0]
    c100 = table[b0, g0, r1]
    c010 = table[b0, g1, r0]
    c110 = table[b0, g1, r1]
    c001 = table[b1, g0, r0]
    c101 = table[b1, g0, r1]
    c011 = table[b1, g1, r0]
    c111 = table[b1, g1, r1]

    output = np.empty_like(c000)

    # Six tetrahedra, selected by the ordering of the fractional coordinates.
    m0 = ((fr >= fg) & (fg >= fb))[:, 0]  # r >= g >= b
    m1 = ((fr >= fb) & (fb > fg))[:, 0]   # r >= b > g
    m2 = ((fb > fr) & (fr >= fg))[:, 0]   # b > r >= g
    m3 = ((fg > fr) & (fr >= fb))[:, 0]   # g > r >= b
    m4 = ((fg >= fb) & (fb > fr))[:, 0]   # g >= b > r
    m5 = ((fb > fg) & (fg > fr))[:, 0]    # b > g > r

    output[m0] = (
        c000[m0]
        + fr[m0] * (c100[m0] - c000[m0])
        + fg[m0] * (c110[m0] - c100[m0])
        + fb[m0] * (c111[m0] - c110[m0])
    )
    output[m1] = (
        c000[m1]
        + fr[m1] * (c100[m1] - c000[m1])
        + fb[m1] * (c101[m1] - c100[m1])
        + fg[m1] * (c111[m1] - c101[m1])
    )
    output[m2] = (
        c000[m2]
        + fb[m2] * (c001[m2] - c000[m2])
        + fr[m2] * (c101[m2] - c001[m2])
        + fg[m2] * (c111[m2] - c101[m2])
    )
    output[m3] = (
        c000[m3]
        + fg[m3] * (c010[m3] - c000[m3])
        + fr[m3] * (c110[m3] - c010[m3])
        + fb[m3] * (c111[m3] - c110[m3])
    )
    output[m4] = (
        c000[m4]
        + fg[m4] * (c010[m4] - c000[m4])
        + fb[m4] * (c011[m4] - c010[m4])
        + fr[m4] * (c111[m4] - c011[m4])
    )
    output[m5] = (
        c000[m5]
        + fb[m5] * (c001[m5] - c000[m5])
        + fg[m5] * (c011[m5] - c001[m5])
        + fr[m5] * (c111[m5] - c011[m5])
    )

    return output.reshape(original_shape)
