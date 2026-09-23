"""P6 delivery packaging: named copies, manifest, zip — and a byte-level re-read.

Produces
--------
``<out>/LUTs/*.cube``
    The shipped cubes (copied in when ``--cubes`` points somewhere else), each
    validated: ``LUT_3D_SIZE 33``, the ``#LUMIXPHOTOSTYLE STD`` header line,
    a stem of at most 8 ASCII alphanumerics, 35 937 rows, every value in [0, 1].
``<out>/LUTs_named/<NN>_<Name>_<中文>.cube``
    Byte-identical copies under a human name, for a desktop grading app.
``<out>/manifest.json``
    sha256 per cube (both copies), the engine version — git-less: the sha256 of
    every ``engine/*.py`` concatenated in name order — and a QC summary per
    look, measured here from the cube itself (second differences, clipping,
    folding, the neutral ramp, photo-sample dE00) and merged with the build
    manifest's own QC block when one is given with ``--qc``.
``<out>/Latent-2026_LUTs.zip``
    ``LUTs/``, ``LUTs_named/``, ``使用说明.md``, ``gallery/``.  The gallery's
    per-look download link is ``../LUTs/<stem>.cube``, so the two folders must
    stay siblings inside the archive — they do.

The zip is then VERIFIED: ``testzip()``, then every ``.cube`` member is read
back out of the archive and compared byte for byte (and by sha256) with the
file on disk.  A packaging step that is not re-read is not a packaging step.

CLI::

    py tools/package.py --cubes out/LUTs --out out --gallery out/gallery
    py tools/package.py --cubes out/LUTs_r1 --out /tmp/pkg --qc out/manifest_r1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from engine.cubeio import read_lut
from tools import metrics
from tools.gallery import look_meta
from tools.render import ROOT, WORK

__all__ = [
    "CUBE_SIZE",
    "PHOTO_STYLE",
    "MAX_STEM",
    "engine_version",
    "sha256_bytes",
    "validate_cube",
    "qc_summary",
    "named_filename",
    "build_manifest",
    "make_zip",
    "verify_zip",
    "run",
]

CUBE_SIZE = 33
PHOTO_STYLE = "STD"
MAX_STEM = 8
ZIP_NAME = "Latent-2026_LUTs.zip"
ZIP_PREFIX = "Latent-2026"
MAX_ZIP_MB = 60.0
GUIDE_NAME = "使用说明.md"

_PLACEHOLDER_GUIDE = """# Latent 2026 — 使用说明（占位）

`docs/{name}` 尚未写好，这份是打包时自动生成的占位文件。

- 相机：Panasonic Lumix S9，Photo Style 选 **Standard**，色彩空间 **sRGB**，再加载 Real Time LUT。
- 文件：33 点 `.cube`，文件名不超过 8 个 ASCII 字符（相机限制）。
- 强度：先从 **100 %** 试拍；机内强度按 `s·LUT(x) + (1−s)·x` 混合，70 % 的效果在试色册里可以直接对比。
- LUT 只改变颜色与明暗，不包含颗粒、柔焦、局部提亮或光晕。

本包内的 LUT：

{table}
"""


# --------------------------------------------------------------------------- #
# hashes
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def engine_version(engine_dir: str | Path | None = None) -> dict:
    """sha256 of every ``engine/*.py`` concatenated in name order (git-less)."""
    engine_dir = Path(engine_dir) if engine_dir is not None else ROOT / "engine"
    files = sorted(Path(engine_dir).glob("*.py"))
    if not files:
        raise FileNotFoundError(f"no engine/*.py under {engine_dir}")
    digest = hashlib.sha256()
    per_file = []
    for path in files:
        data = path.read_bytes()
        digest.update(data)
        per_file.append({"file": path.name, "bytes": len(data), "sha256": sha256_bytes(data)})
    return {
        "version": digest.hexdigest(),
        "method": "sha256 of engine/*.py concatenated in sorted filename order",
        "files": per_file,
    }


# --------------------------------------------------------------------------- #
# cube validation
# --------------------------------------------------------------------------- #
def validate_cube(path: str | Path) -> dict:
    """Check one deliverable ``.cube``; raises ``ValueError`` on any violation."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="strict")
    problems: list[str] = []
    stem = path.stem
    if not stem.isascii() or not stem.isalnum():
        problems.append(f"stem {stem!r} must be ASCII alphanumeric")
    if len(stem) > MAX_STEM:
        problems.append(f"stem {stem!r} is {len(stem)} chars, the camera allows {MAX_STEM}")
    header = [ln.strip() for ln in text.splitlines()[:6]]
    if f"#LUMIXPHOTOSTYLE {PHOTO_STYLE}" not in header:
        problems.append(f"missing '#LUMIXPHOTOSTYLE {PHOTO_STYLE}' in the first 6 lines")
    lut = read_lut(path)
    if lut.size != CUBE_SIZE:
        problems.append(f"LUT_3D_SIZE is {lut.size}, the camera needs {CUBE_SIZE}")
    table = np.asarray(lut.table, dtype=np.float64)
    if not np.isfinite(table).all():
        problems.append("non-finite values in the table")
    if table.min() < 0.0 or table.max() > 1.0:
        problems.append(f"values outside [0, 1]: [{table.min():.6f}, {table.max():.6f}]")
    data = path.read_bytes()
    if problems:
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    return {
        "file": path.name,
        "path": str(path),
        "stem": stem,
        "title": lut.title,
        "size": int(lut.size),
        "rows": int(lut.size**3),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "min": float(table.min()),
        "max": float(table.max()),
    }


# --------------------------------------------------------------------------- #
# QC summary measured from the cube itself
# --------------------------------------------------------------------------- #
_photo_sample: np.ndarray | None = None


def _load_photo_sample() -> np.ndarray | None:
    global _photo_sample
    if _photo_sample is None:
        path = WORK / "cal" / "photo_sample.npy"
        if not path.exists():
            return None
        arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1, 3)
        _photo_sample = arr / 255.0 if arr.max() > 1.5 else arr
    return _photo_sample


def qc_summary(table: np.ndarray) -> dict:
    """A self-contained QC block for the manifest, measured from the table."""
    table = np.asarray(table, dtype=np.float64)
    sampler = metrics.sampler_from_table(table)
    d2 = metrics.second_diff_stats(table)
    clip = metrics.clip_stats(table)
    fold = metrics.fold_stats(table)
    neutral = metrics.neutral_stats(sampler)
    out: dict = {
        "second_diff": d2,
        "clip": clip,
        "fold": fold,
        "neutral": {
            "black": [float(v) * 255.0 for v in np.atleast_1d(neutral["black"])],
            "white": [float(v) * 255.0 for v in np.atleast_1d(neutral["white"])],
            "monotone": bool(neutral["monotone"]),
            "monotone_L": bool(neutral.get("monotone_L", neutral["monotone"])),
            "min_step_code8": float(neutral["min_step"]) * 255.0,
        },
    }
    photo = _load_photo_sample()
    if photo is not None:
        de = metrics.de00(metrics.srgb_to_lab(photo), metrics.srgb_to_lab(sampler(photo)))
        out["photo_de00"] = {
            "n": int(de.size),
            "mean": float(de.mean()),
            "p95": float(np.percentile(de, 95)),
            "max": float(de.max()),
        }
    out["headline"] = {
        "d2_interior_p99_9": _dig(d2, ("interior", "p99_9")) or _dig(d2, ("interior", "p99.9")),
        # `material_count` is the one the build's QC gates on (folds big enough to
        # matter); `neg_count` also counts micro folds under the material eps, and
        # it counts the whole lattice for a deliberately non-injective mono look,
        # so both are reported rather than one being passed off as "the" number.
        "fold_material_count": fold.get("material_count"),
        "fold_micro_count": fold.get("micro_count"),
        "fold_neg_count": fold.get("neg_count"),
        "fold_min_ratio": fold.get("min_ratio"),
        "fold_collapsed_frac": fold.get("collapsed_frac"),
        "clip_zero": clip.get("zero"),
        "clip_one": clip.get("one"),
        "grey_monotone": out["neutral"]["monotone"],
        "photo_de00_mean": (out.get("photo_de00") or {}).get("mean"),
    }
    return out


def _dig(d: dict, keys: Sequence[str]):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
def named_filename(meta: dict) -> str:
    """``<NN>_<Name>_<中文>.cube`` from a look's metadata."""
    stem = meta["name"]
    m = re.match(r"^(\d+)(.*)$", stem)
    number = m.group(1) if m else ""
    title = meta.get("title") or (m.group(2) if m else stem)
    cn = meta.get("cn") or ""
    parts = [p for p in (number, title, cn) if p]
    return "_".join(parts) + ".cube"


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_manifest(entries: list[dict], *, cubes_dir: Path, out_dir: Path,
                   qc_source: Path | None, prefix: str) -> dict:
    return {
        "project": "Latent-2026",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "camera": {
            "body": "Panasonic Lumix DC-S9",
            "photo_style": PHOTO_STYLE,
            "colour_space": "sRGB",
            "lut": f"{CUBE_SIZE}-point .cube, display-referred sRGB -> sRGB",
            "strength_model": "s*LUT(x) + (1-s)*x in sRGB code values (assumed, unverified)",
        },
        "engine": engine_version(),
        "source_cubes": str(cubes_dir.resolve()),
        "out": str(out_dir.resolve()),
        "zip_prefix": prefix,
        "qc_source": str(qc_source) if qc_source else None,
        "n_looks": len(entries),
        "looks": entries,
    }


def make_zip(zip_path: Path, *, out_dir: Path, gallery: Path | None, guide: Path | None,
             prefix: str, guide_text: str | None) -> dict:
    """Write the delivery zip; returns its member listing."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    pre = (prefix.strip("/") + "/") if prefix else ""
    members: list[str] = []
    with ZipFile(zip_path, "w", ZIP_DEFLATED, compresslevel=9) as z:
        for sub in ("LUTs", "LUTs_named"):
            for path in sorted((out_dir / sub).glob("*.cube")):
                arc = f"{pre}{sub}/{path.name}"
                z.write(path, arc)
                members.append(arc)
        if guide and Path(guide).exists():
            z.write(guide, f"{pre}{GUIDE_NAME}")
        else:
            z.writestr(f"{pre}{GUIDE_NAME}", guide_text or "")
        members.append(f"{pre}{GUIDE_NAME}")
        manifest = out_dir / "manifest.json"
        if manifest.exists():
            z.write(manifest, f"{pre}manifest.json")
            members.append(f"{pre}manifest.json")
        if gallery and Path(gallery).is_dir():
            for path in sorted(Path(gallery).rglob("*")):
                if path.is_file():
                    arc = f"{pre}gallery/{path.relative_to(gallery).as_posix()}"
                    z.write(path, arc)
                    members.append(arc)
    return {
        "file": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "mb": round(zip_path.stat().st_size / 1e6, 2),
        "members": len(members),
        "cubes": len([m for m in members if m.endswith(".cube")]),
    }


def verify_zip(zip_path: Path, entries: list[dict], *, prefix: str) -> dict:
    """Re-read every cube out of the archive and compare bytes + sha256."""
    pre = (prefix.strip("/") + "/") if prefix else ""
    checked, problems = 0, []
    with ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            problems.append(f"corrupt member: {bad}")
        names = set(z.namelist())
        for entry in entries:
            for key, sub in (("file", "LUTs"), ("named_file", "LUTs_named")):
                arc = f"{pre}{sub}/{entry[key]}"
                if arc not in names:
                    problems.append(f"missing from zip: {arc}")
                    continue
                on_disk = Path(entry["path" if key == "file" else "named_path"]).read_bytes()
                in_zip = z.read(arc)
                if in_zip != on_disk:
                    problems.append(f"byte mismatch for {arc}")
                elif sha256_bytes(in_zip) != entry["sha256"]:
                    problems.append(f"sha256 mismatch for {arc}")
                checked += 1
        if f"{pre}{GUIDE_NAME}" not in names:
            problems.append(f"missing from zip: {pre}{GUIDE_NAME}")
    return {"cubes_reread": checked, "problems": problems, "ok": not problems}


def run(
    cubes_dir: str | Path,
    out_dir: str | Path,
    *,
    gallery: str | Path | None = None,
    guide: str | Path | None = None,
    qc: str | Path | None = None,
    zip_name: str = ZIP_NAME,
    prefix: str = ZIP_PREFIX,
    max_mb: float = MAX_ZIP_MB,
    allow_oversize: bool = False,
    looks: Sequence[str] | None = None,
    progress: bool = True,
) -> dict:
    """Validate, copy, name, manifest, zip and re-read.  Returns the report."""
    t0 = time.perf_counter()
    cubes_dir = Path(cubes_dir)
    out_dir = Path(out_dir)
    luts_dir = out_dir / "LUTs"
    named_dir = out_dir / "LUTs_named"
    luts_dir.mkdir(parents=True, exist_ok=True)
    named_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(cubes_dir.glob("*.cube"))
    if looks:
        wanted = {s.strip() for s in looks}
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise FileNotFoundError(f"no .cube files in {cubes_dir}")

    qc_manifest: dict[str, dict] = {}
    qc_path = Path(qc) if qc else None
    if qc_path and qc_path.exists():
        data = json.loads(qc_path.read_text())
        for item in data.get("looks", []):
            if isinstance(item, dict) and item.get("name"):
                qc_manifest[item["name"]] = item.get("qc", {})

    entries: list[dict] = []
    for src in paths:
        info = validate_cube(src)
        dest = luts_dir / src.name
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
        meta = look_meta(src.stem)
        named = named_dir / named_filename(meta)
        shutil.copy2(dest, named)
        if named.read_bytes() != dest.read_bytes():  # pragma: no cover - copy2 is exact
            raise RuntimeError(f"named copy differs from {dest}")
        table = np.asarray(read_lut(dest).table, dtype=np.float64)
        entry = {
            **info,
            "file": dest.name,
            "path": str(dest),
            "named_file": named.name,
            "named_path": str(named),
            "named_sha256": sha256_bytes(named.read_bytes()),
            "cn": meta.get("cn", ""),
            "display_title": meta.get("title", src.stem),
            "order": meta.get("order", ""),
            "qc": qc_summary(table),
        }
        if src.stem in qc_manifest:
            entry["build_qc"] = qc_manifest[src.stem]
        entries.append(entry)
        if progress:
            head = entry["qc"]["headline"]
            print(
                f"  {src.stem:10s} -> {named.name:26s} {entry['bytes'] / 1024:7.0f} KB  "
                f"sha {entry['sha256'][:12]}  d2int99.9 {head['d2_interior_p99_9'] or float('nan'):5.2f}  "
                f"folds {head['fold_material_count']} (micro {head['fold_micro_count']})  "
                f"photo dE00 {head['photo_de00_mean'] or float('nan'):.2f}",
                flush=True,
            )

    manifest = build_manifest(entries, cubes_dir=cubes_dir, out_dir=out_dir,
                              qc_source=qc_path, prefix=prefix)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    guide_path = Path(guide) if guide else (ROOT / "docs" / GUIDE_NAME)
    guide_generated = None
    if not Path(guide_path).exists():
        table_md = "\n".join(
            f"- `{e['file']}` — {e['display_title']} {e['cn']}".rstrip() for e in entries
        )
        guide_generated = _PLACEHOLDER_GUIDE.format(name=GUIDE_NAME, table=table_md)

    zip_path = out_dir / zip_name
    zip_info = make_zip(zip_path, out_dir=out_dir, gallery=Path(gallery) if gallery else None,
                        guide=Path(guide_path), prefix=prefix, guide_text=guide_generated)
    check = verify_zip(zip_path, entries, prefix=prefix)

    report = {
        "generated": manifest["generated"],
        "cubes": str(cubes_dir.resolve()),
        "out": str(out_dir.resolve()),
        "n_looks": len(entries),
        "engine_version": manifest["engine"]["version"],
        "manifest": str(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "zip": zip_info,
        "verify": check,
        "guide": str(guide_path) if Path(guide_path).exists() else "GENERATED PLACEHOLDER",
        "gallery": str(gallery) if gallery else None,
        "max_mb": max_mb,
        "oversize": zip_info["mb"] > max_mb,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "package_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    if progress:
        print(
            f"zip {zip_path.name}: {zip_info['mb']:.2f} MB, {zip_info['members']} members "
            f"({zip_info['cubes']} cubes)", flush=True
        )
        print(
            f"verify: re-read {check['cubes_reread']} cubes from the archive -> "
            f"{'OK, byte-identical' if check['ok'] else 'PROBLEMS: ' + '; '.join(check['problems'])}",
            flush=True,
        )
        print(f"engine version {manifest['engine']['version'][:16]}  "
              f"({len(manifest['engine']['files'])} files)", flush=True)
    if not check["ok"]:
        raise RuntimeError("zip verification failed: " + "; ".join(check["problems"]))
    if report["oversize"] and not allow_oversize:
        raise RuntimeError(
            f"{zip_path.name} is {zip_info['mb']:.1f} MB > {max_mb} MB "
            f"(shrink the gallery images or pass --allow-oversize)"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="package.py", description=__doc__.split("\n")[0])
    ap.add_argument("--cubes", default=str(ROOT / "out" / "LUTs"))
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--guide", default=None)
    ap.add_argument("--qc", default=None, help="a build manifest whose per-look qc block is merged in")
    ap.add_argument("--zip-name", default=ZIP_NAME)
    ap.add_argument("--prefix", default=ZIP_PREFIX, help="top-level folder inside the zip ('' for none)")
    ap.add_argument("--max-mb", type=float, default=MAX_ZIP_MB)
    ap.add_argument("--allow-oversize", action="store_true")
    ap.add_argument("--looks", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    run(
        args.cubes,
        args.out,
        gallery=args.gallery,
        guide=args.guide,
        qc=args.qc,
        zip_name=args.zip_name,
        prefix=args.prefix,
        max_mb=args.max_mb,
        allow_oversize=args.allow_oversize,
        looks=args.looks.split(",") if args.looks else None,
        progress=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
