"""tools/round.py — one colourist round in one Python process.

::

    ./py -m tools.round <look.json> --round r1 [--out-dir DIR] [--rival auto|none|<cube>]
                        [--baseline <look.json>] [--final]

Build one look -> QC -> fingerprint -> the standard review sheets -> a compact
text report (``report.txt``, also printed).  Everything goes through the Python
APIs (``engine.pipeline``, ``tools.qc``, ``tools.fingerprint``, ``tools.sheets``,
``tools.render``, ``tools.charts``, ``tools.metrics``) — this module never shells
out to ``tools/build.py``.

Outputs in the out-dir::

    <name>.cube  qc.json  fp.json  report.txt
    A_portrait.jpg  B_face.jpg  C_hero12.jpg  D_hero34.jpg
    E_rival.jpg (unless --rival none)  F_chart_C1.jpg  G_chart_C3.jpg
    H_strength.jpg (only with --final)        (+ one .json sidecar each)

Every sheet is re-opened after saving and re-checked against the T4 limits
(<= 1.15 MP, long edge <= 1536 px, <= 6 tiles, per-kind tile long edge) —
independently of the assertions inside ``tools.sheets.save_sheet``.

Defensive by design: ``engine/*``, ``tools/qc.py``, ``tools/build.py`` and
``tools/metrics.py`` are being upgraded to v1.1 by other engineers while this
runs, so every diagnostic/QC/fingerprint key is read through :func:`_dig` and a
missing one prints ``n/a`` instead of raising.

Scene list: ``docs/round_scenes.json``.  Any base id listed in
``$W/bases_holdout.txt`` is refused with exit code 2 — holdout scenes are never
shown to a colourist.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "HoldoutError",
    "RoundError",
    "run_round",
    "load_scenes",
    "holdout_ids",
    "scene_for",
    "patch_boxes_full_frame",
    "patch_boxes_for_crop",
    "ablations",
    "block_amplitudes",
    "build_report",
    "verify_sheet",
    "main",
]

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
ROOT = Path(os.environ.get("LATENT_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("LATENT_WORK", ROOT / "work.nosync"))
SCENES_JSON = ROOT / "docs" / "round_scenes.json"
HOLDOUT_TXT = WORK / "bases_holdout.txt"
REFS_DIR = WORK / "refs"
ROUNDS_ROOT = WORK / "rounds"

#: the LibRaw half-size develop the frozen crop boxes are measured in
#: (``$W/bases_hi/crops.json``: "LibRaw half_size pixels"), long/short edge.
HALF_LONG, HALF_SHORT = 3008, 2008

#: fraction of a frozen crop box kept when it is reduced to a measurement patch
PATCH_FRAC = 0.45
#: smallest patch we will measure, in pixels of the image being measured
PATCH_MIN_PX = 12

#: report.txt budget.  ROUND_SPEC says "compact, plain text, <= 120 lines", and
#: that was the budget until ``docs/GATES_v3.md``: the ruling's HARD list is 30
#: gate rows where the hand-kept list was 15, its WARN-only block is 9 more, and
#: section 2 now carries the measured patch numbers of sheets A and B (up to 10).
#: A full round — sheets, baseline, ablation, collisions — measures ~150 lines,
#: so holding 120 would mean truncating the lead's own hard-gate list, which is
#: the one thing the report exists to show.  Raised to 160; ROUND_SPEC's budget
#: line needs the lead's amendment (reported, not assumed).
REPORT_MAX_LINES = 160

#: The hard gates ROUND_SPEC §1 asks to be listed first, in order.
#:
#: ``docs/GATES_v3.md`` is the authority now, and it writes the HARD list out
#: in full, so this is exactly that list, in that order, read from
#: ``tools.qc.GATES_V3_HARD`` — one ruling, one place, and round.py cannot
#: drift out of step with what qc actually FAILs on.  (It used to be a hand-kept
#: copy, and twice already a hard FAIL in qc was missing from it, so a run could
#: print "hard gates: ok" while the build was blocked.)
#:
#: v3 changes visible here: ``blend70.d2_p99_9`` is gone (REMOVED), the two
#: ``skin.skin_chroma_*`` gates are now absolute lines rather than a comparison
#: against ``looks/targets``, ``skin.displacement`` has moved to the WARN-only
#: list, and ``ap.highlight_clean`` (which replaces ``ap.wb_preset``),
#: ``neutral.monotone_L``, the full-lattice d2 pair, the three photo grid-error
#: numbers, the clip excesses, the hair slope and the black lift have joined it.
def _hard_gates() -> tuple[tuple[str, str], ...]:
    try:
        from tools.qc import GATES_V3_HARD

        return tuple((str(k), str(a)) for k, a in GATES_V3_HARD)
    except Exception:  # pragma: no cover - tools/qc.py being rewritten
        return (("fold.neg_count", "material folds @100%"),
                ("blend70.neg_count", "material folds @70%"),
                ("fold.micro", "micro folds %"),
                ("fold.crush", "crushed %"),
                ("d2.interior_p99_9", "interior d2 p99.9"))


HARD_GATES: tuple[tuple[str, str], ...] = _hard_gates()

#: ``docs/GATES_v3.md``'s WARN-only list, printed under its own heading: these
#: numbers are reported for the colourist's judgement and never block shipping.
def _warn_gates() -> tuple[tuple[str, str], ...]:
    try:
        from tools.qc import GATES_V3_WARN_ONLY

        return tuple((str(k), str(a)) for k, a in GATES_V3_WARN_ONLY)
    except Exception:  # pragma: no cover
        return ()


WARN_ONLY_GATES: tuple[tuple[str, str], ...] = _warn_gates()

#: the smoothness / fold gates whose failure triggers the operator ablation
#: (``blend70.d2_p99_9`` dropped — GATES_v3 REMOVED that gate)
SMOOTH_FOLD_GATES: tuple[str, ...] = (
    "d2.interior_p99_9", "d2.full_p99_9", "d2.full_max",
    "fold.neg_count", "fold.micro", "fold.crush", "fold.min_ratio",
    "jacobian.neg_count", "blend70.neg_count", "blend70.micro",
)

#: the 9 pinned grey-ramp input codes of the fingerprint probe
RAMP_CODES = (0, 11, 31, 64, 115, 166, 209, 240, 255)

#: blandness: a block may not shrink toward identity by more than this
BLAND_RATIO = 0.92


class RoundError(RuntimeError):
    """Anything that makes the round impossible (exit code 1)."""


class HoldoutError(RoundError):
    """A holdout scene id was requested (exit code 2)."""


# --------------------------------------------------------------------------- #
# defensive readers — the engine / qc APIs are moving under us
# --------------------------------------------------------------------------- #
def _dig(obj: Any, path: str, default: Any = None) -> Any:
    """``_dig(d, "a.b.c")`` -> ``d["a"]["b"]["c"]`` or *default*.

    Never raises: a missing key, a ``None`` on the way down, a list where a dict
    was expected — all give *default*.  This is the whole defence against the
    v1.0 -> v1.1 key drift happening in ``engine/`` and ``tools/qc.py``.
    """
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return default
        else:
            cur = getattr(cur, part, None)
            if cur is None:
                return default
    return default if cur is None else cur


def _n(v: Any, fmt: str = "{:.3f}", dash: str = "n/a") -> str:
    """Format a number for the report; anything unusable prints *dash*.

    The dash is padded to the width the format would have produced, so a column
    of numbers stays a column when one of them goes missing (which is exactly
    what happens while the engine's diagnostics keys are in flux).
    """
    width = _safe(lambda: len(fmt.format(0.0)), len(dash)) or len(dash)
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if v is None:
        return dash.rjust(width)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return dash.rjust(width)
    return _safe(lambda: fmt.format(f), dash.rjust(width))


def _row(values, fmt="{:6.2f}", dash="n/a") -> str:
    """One space-separated row of numbers; ``None`` anywhere prints *dash*."""
    if values is None:
        return dash
    return " ".join(_n(v, fmt, dash) for v in values)


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap (no textwrap import for one call)."""
    out: list[str] = []
    line = ""
    for word in str(text).split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out or [""]


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Run *fn*; on any exception return *default*.  Used around third-party
    metrics that may drift (missing kwarg, renamed key, ...)."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 - the whole point is to never crash a round
        return default


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


# --------------------------------------------------------------------------- #
# scenes / holdout
# --------------------------------------------------------------------------- #
def load_scenes(path: Path | None = None) -> dict:
    """``docs/round_scenes.json``."""
    p = Path(path) if path is not None else SCENES_JSON
    if not p.exists():
        raise RoundError(f"scene list not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def holdout_ids(path: Path | None = None) -> set[str]:
    """The base ids a colourist agent must never see.

    A missing ``bases_holdout.txt`` is itself a refusal condition for anything
    that would have to guess — we return an empty set but the caller is told,
    so the round still runs on an unprepared checkout (tests).
    """
    p = Path(path) if path is not None else HOLDOUT_TXT
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")}


def _base_of(scene: str) -> str:
    """``"PANA0116:face"`` -> ``"PANA0116"``."""
    return str(scene).split(":", 1)[0]


def scene_for(name: str, scenes: dict) -> dict:
    """Resolve the scene set for look *name*.

    Missing looks are not fatal: the portrait / face / charts of ``common`` are
    always available, and the round simply has no heroes and no rival.
    """
    common = dict(scenes.get("common") or {})
    entry = dict((scenes.get("looks") or {}).get(name) or {})
    portrait = str(common.get("portrait") or "PANA0116")
    face = str(entry.get("face_crop") or common.get("face_crop") or f"{portrait}:face")
    charts = list(entry.get("charts") or common.get("charts") or ["C1", "C3"])
    return {
        "known": name in (scenes.get("looks") or {}),
        "portrait": portrait,
        "face_crop": face,
        "charts": charts,
        "heroes": [str(h) for h in (entry.get("heroes") or [])],
        "rival": entry.get("rival"),
    }


def check_holdout(scene: dict, holdout: set[str]) -> None:
    """Refuse the whole round if any scene we would show is a holdout base."""
    used: list[str] = [scene["portrait"], _base_of(scene["face_crop"])]
    used += [_base_of(h) for h in scene["heroes"]]
    bad = sorted({b for b in used if b in holdout})
    if bad:
        raise HoldoutError(
            "refusing to run: scene id(s) "
            + ", ".join(bad)
            + f" are listed in {HOLDOUT_TXT} — holdout bases are never shown to a "
              "colourist. Fix docs/round_scenes.json or pass a different look."
        )


def resolve_rival(spec: str | None, scene: dict) -> Path | None:
    """``auto`` -> the look's rival from the scene JSON; ``none`` -> None;
    anything else is a path (absolute, or relative to ``$W/refs``)."""
    text = "auto" if spec is None else str(spec)
    if text.lower() in {"none", "no", "off", ""}:
        return None
    if text.lower() == "auto":
        rel = scene.get("rival")
        if not rel:
            return None
        text = str(rel)
    p = Path(text)
    if p.is_absolute() or p.exists():
        if not p.exists():
            raise RoundError(f"rival cube not found: {p}")
        return p
    cand = REFS_DIR / text
    if not cand.exists():
        raise RoundError(f"rival cube not found: {p} nor {cand}")
    return cand


# --------------------------------------------------------------------------- #
# patch boxes
# --------------------------------------------------------------------------- #
def _crops_for(base_id: str) -> dict:
    """``{crop_name: {"box": (x0,y0,x1,y1), "tag": str}}`` in half-size pixels."""
    from tools import render

    def _load():
        crops = render.load_crops()
        entry = crops.get(base_id)
        if entry is None and isinstance(crops.get("crops"), dict):
            entry = crops["crops"].get(base_id)
        out = {}
        for k, v in (entry or {}).items():
            box = v.get("box", v.get("rect")) if isinstance(v, dict) else v
            tag = v.get("tag", "") if isinstance(v, dict) else ""
            if box is not None and len(box) == 4:
                out[str(k)] = {"box": tuple(float(x) for x in box), "tag": str(tag)}
        return out

    return _safe(_load, {}) or {}


def _half_dims(img_shape: Sequence[int]) -> tuple[float, float]:
    """``(half_w, half_h)`` of the LibRaw half-size develop this base came from.

    ``crops.json`` boxes are in half-size pixels (~3008x2008 landscape,
    2008x3008 portrait); the 1600 px base is the same full frame, scaled.
    """
    h, w = float(img_shape[0]), float(img_shape[1])
    return (HALF_LONG, HALF_SHORT) if w >= h else (HALF_SHORT, HALF_LONG)


def _box_to_frame(box: Sequence[float], img_shape: Sequence[int]) -> tuple[float, ...]:
    """Half-size pixel box -> pixel box in the (H, W) image *img_shape*."""
    hw, hh = _half_dims(img_shape)
    sx = float(img_shape[1]) / hw
    sy = float(img_shape[0]) / hh
    x0, y0, x1, y1 = (float(v) for v in box)
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def _shrink(box: Sequence[float], frac: float = PATCH_FRAC,
            min_px: float = PATCH_MIN_PX) -> tuple[float, ...]:
    """Centred sub-box covering *frac* of each edge of *box*.

    The frozen crops are generous rectangles (a whole cardigan, a whole leafy
    background); a patch measurement wants a small piece of one, so that "skin"
    is really skin and "white" is really the garment.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w = max(abs(x1 - x0) * frac, min_px)
    h = max(abs(y1 - y0) * frac, min_px)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _integral(a: np.ndarray) -> np.ndarray:
    """2-D cumulative sum padded with a zero row and column."""
    c = np.cumsum(np.cumsum(a, axis=0), axis=1)
    pad = [(1, 0), (1, 0)] + [(0, 0)] * (a.ndim - 2)
    return np.pad(c, pad)


def _window_sum(cum: np.ndarray, wh: int, ww: int) -> np.ndarray:
    return (cum[wh:, ww:] - cum[:-wh, ww:] - cum[wh:, :-ww] + cum[:-wh, :-ww])


#: how hard a clipped pixel is punished when choosing a measurement patch
_CLIP_PENALTY = 4.0


def _uniform_subbox(img: np.ndarray, box: Sequence[float], frac: float = PATCH_FRAC,
                    min_px: float = PATCH_MIN_PX,
                    search: float = 0.75) -> tuple[float, ...]:
    """The most *uniform*, least clipped sub-box of *box* in *img*.

    A frozen crop is a generous rectangle: ``skin_cheek`` on PANA0116 also
    contains the mouth, ``white_cardigan`` contains folds and a shadow edge.
    Measuring its geometric centre can therefore report lipstick as skin.  So
    the patch is the ``frac``-sized window, inside the centred ``search``
    fraction of the box, that minimises in-window variance plus a penalty for
    pixels on the 0/1 rails (a clipped "white" makes the cleanliness number
    meaningless).  Staying inside the centred ``search`` region keeps the patch
    on the content the crop was frozen for; the variance only decides where in
    it.  Falls back to :func:`_shrink` when the region is too small to search.
    """
    arr = np.asarray(img, dtype=np.float64)
    H, W = arr.shape[0], arr.shape[1]
    search = min(1.0, max(0.05, float(search)))
    x0, y0, x1, y1 = _shrink(box, search, min_px)
    win = min(0.95, float(frac) / search)      # window as a fraction of the region
    ix0, ix1 = sorted((int(round(x0)), int(round(x1))))
    iy0, iy1 = sorted((int(round(y0)), int(round(y1))))
    ix0, iy0 = max(0, ix0), max(0, iy0)
    ix1, iy1 = min(W, ix1), min(H, iy1)
    rw, rh = ix1 - ix0, iy1 - iy0
    if rw < 8 or rh < 8:
        return _shrink(box, frac, min_px)

    step = max(1, int(max(rw, rh) // 96))
    reg = arr[iy0:iy1:step, ix0:ix1:step]
    h, w = reg.shape[0], reg.shape[1]
    wh = max(3, int(round(h * win)))
    ww = max(3, int(round(w * win)))
    if wh >= h or ww >= w:
        return _shrink(box, frac, min_px)

    n = float(wh * ww)
    s = _window_sum(_integral(reg), wh, ww) / n
    s2 = _window_sum(_integral(reg ** 2), wh, ww) / n
    var = np.maximum(s2 - s ** 2, 0.0).sum(axis=-1)
    railed = ((reg <= 0.0) | (reg >= 1.0)).any(axis=-1).astype(np.float64)
    clipf = _window_sum(_integral(railed), wh, ww) / n
    k = int(np.argmin(var + _CLIP_PENALTY * clipf))
    i, j = divmod(k, var.shape[1])
    return (float(ix0 + j * step), float(iy0 + i * step),
            float(ix0 + (j + ww) * step), float(iy0 + (i + wh) * step))


def _pick(crops: dict, *needles: str, used: set[str] | None = None) -> str | None:
    """First crop whose tag (then name) contains one of *needles*."""
    used = used or set()
    for needle in needles:
        for key in sorted(crops):
            if key in used:
                continue
            if needle in crops[key]["tag"].lower():
                return key
    for needle in needles:
        for key in sorted(crops):
            if key not in used and needle in key.lower():
                return key
    return None


#: fallback boxes when a base has no tagged crops at all (ROUND_SPEC A:
#: "three fixed boxes on the face/neck/hand if none are tagged") — a vertical
#: strip down the middle of the frame, upper / centre / lower.
FALLBACK_BOXES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("upper", (0.36, 0.16, 0.56, 0.30)),
    ("centre", (0.36, 0.43, 0.56, 0.57)),
    ("lower", (0.36, 0.70, 0.56, 0.84)),
)


def _fallback(img: np.ndarray) -> list[tuple[str, tuple]]:
    h, w = float(np.shape(img)[0]), float(np.shape(img)[1])
    return [(name, (b[0] * w, b[1] * h, b[2] * w, b[3] * h)) for name, b in FALLBACK_BOXES]


def patch_boxes_full_frame(base_id: str, img: np.ndarray) -> list[tuple[str, tuple]]:
    """Up to 3 measurement boxes on the **full 1600 px frame** of *base_id*.

    Skin first (so the skin move is a number), then a near-white surface (so
    white cleanliness is a measured number, ROUND_SPEC A), then a third
    foliage / sky / gradient patch.  Boxes come from the frozen
    ``bases_hi/crops.json`` rectangles, mapped from half-size pixels into the
    1600 px frame and reduced to their most uniform, unclipped interior.
    """
    crops = _crops_for(base_id)
    out: list[tuple[str, tuple]] = []
    used: set[str] = set()
    plan = (
        ("skin", ("skin", "face")),
        ("white", ("near_white", "high_key", "white")),
        ("third", ("foliage", "green", "sky", "gradient", "shadow", "red", "magenta")),
    )
    for _role, needles in plan:
        key = _pick(crops, *needles, used=used)
        if key is None:
            continue
        used.add(key)
        frame_box = _box_to_frame(crops[key]["box"], np.shape(img))
        out.append((key, _uniform_subbox(img, frame_box)))
    return out[:4] if out else _fallback(img)


def patch_boxes_for_crop(base_id: str, crop_name: str,
                         img: np.ndarray) -> list[tuple[str, tuple]]:
    """Measurement boxes inside the hi-res crop ``base_id:crop_name``.

    Any *other* frozen crop of the same base that lies (mostly) inside this one
    is re-used as a patch — on PANA0116 the ``skin_cheek`` rectangle sits inside
    ``face``, so the face sheet measures real skin.  Otherwise three fixed
    fractional boxes down the middle of the crop.
    """
    crops = _crops_for(base_id)
    host = crops.get(crop_name)
    out: list[tuple[str, tuple]] = []
    if host is not None:
        hx0, hy0, hx1, hy1 = host["box"]
        sx = float(np.shape(img)[1]) / max(hx1 - hx0, 1e-9)
        sy = float(np.shape(img)[0]) / max(hy1 - hy0, 1e-9)
        for key in sorted(crops):
            if key == crop_name:
                continue
            x0, y0, x1, y1 = crops[key]["box"]
            ix0, iy0 = max(x0, hx0), max(y0, hy0)
            ix1, iy1 = min(x1, hx1), min(y1, hy1)
            area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if area / max((x1 - x0) * (y1 - y0), 1e-9) < 0.60:
                continue
            local = ((ix0 - hx0) * sx, (iy0 - hy0) * sy,
                     (ix1 - hx0) * sx, (iy1 - hy0) * sy)
            out.append((key, _uniform_subbox(img, local)))
    if not out:
        # no tagged sub-crop: three fixed boxes down the middle, each pulled to
        # its most uniform interior so a face crop measures skin, not an eye.
        out = [(name, _uniform_subbox(img, box, frac=0.7, search=1.0))
               for name, box in _fallback(img)]
    return out[:4]


# --------------------------------------------------------------------------- #
# operator ablation
# --------------------------------------------------------------------------- #
#: ROUND_SPEC §5: rotation / chroma tables / vibrance / arch / ops / caps / dl /
#: tint, switched off one at a time on a deep copy of the look dict.
def _ab_rotation(d: dict) -> None:
    f = d.setdefault("field", {})
    f["dh10"] = [0.0] * 12
    f["dh18"] = [0.0] * 12


def _ab_chroma(d: dict) -> None:
    d.setdefault("field", {})["cr"] = [1.0] * 12


def _ab_vibrance(d: dict) -> None:
    v = d.setdefault("field", {}).setdefault("vibrance", {})
    v["gain"] = 1.0


def _ab_sat(d: dict) -> None:
    # v1.2 W3 gave `sat` its own ungated stage [S]; it is a separate switch now.
    d.setdefault("field", {})["sat"] = 1.0


def _ab_arch(d: dict) -> None:
    d.setdefault("field", {})["arch"] = {"warm": [1.0] * 5, "green": [1.0] * 5,
                                         "blue": [1.0] * 5}


def _ab_ops(d: dict) -> None:
    d["ops"] = []


def _ab_caps(d: dict) -> None:
    d["caps"] = []


def _ab_dl(d: dict) -> None:
    d.setdefault("field", {})["dl"] = [0.0] * 12


def _ab_tint(d: dict) -> None:
    n = d.setdefault("neutral", {})
    n["tint_rg"] = []
    n["tint_bg"] = []


ABLATIONS: tuple[tuple[str, Callable[[dict], None]], ...] = (
    ("rotation", _ab_rotation),
    ("chroma", _ab_chroma),
    ("sat", _ab_sat),
    ("vibrance", _ab_vibrance),
    ("arch", _ab_arch),
    ("ops", _ab_ops),
    ("caps", _ab_caps),
    ("dl", _ab_dl),
    ("tint", _ab_tint),
)


def _smooth_fold_metrics(table) -> dict:
    """The numbers the ablation compares, all from ``tools.metrics``.

    ``fold100`` / ``fold70`` count **material** folds (ratio < -0.02) since
    ENGINE_SPEC v1.2 W6; ``micro_pct`` and ``crush_pct`` carry the two
    populations W6 gates alongside them.  The names are kept so an older stored
    round still reads.
    """
    from tools import metrics as M

    out = {"d2_interior_p99_9": None, "d2_full_p99_9": None, "d2_full_max": None,
           "fold100": None, "fold70": None, "min_ratio": None,
           "micro_pct": None, "crush_pct": None, "micro_pct70": None}
    d2 = _safe(lambda: M.second_diff_stats(table))
    if d2 is not None:
        out["d2_interior_p99_9"] = _dig(d2, "interior.p99_9")
        out["d2_full_p99_9"] = _dig(d2, "full.p99_9")
        out["d2_full_max"] = _dig(d2, "full.max")
    f1 = _safe(lambda: M.fold_stats(table))
    if f1 is not None:
        out["fold100"] = _dig(f1, "material_count",
                              _dig(f1, "neg_strict_count", _dig(f1, "neg_count")))
        out["min_ratio"] = _dig(f1, "min_ratio")
        mf = _dig(f1, "micro_frac")
        cf = _dig(f1, "crush_excess_frac", _dig(f1, "crush_frac"))
        out["micro_pct"] = None if mf is None else float(mf) * 100.0
        out["crush_pct"] = None if cf is None else float(cf) * 100.0
    f7 = _safe(lambda: M.fold_stats(M.blend_table(table, 0.70)))
    if f7 is not None:
        out["fold70"] = _dig(f7, "material_count",
                             _dig(f7, "neg_strict_count", _dig(f7, "neg_count")))
        mf7 = _dig(f7, "micro_frac")
        out["micro_pct70"] = None if mf7 is None else float(mf7) * 100.0
    return out


def _clears(before: dict, after: dict) -> list[str]:
    """Which failing gate each switch would clear (thresholds = v1.2 W6)."""
    rules = (
        ("d2.interior_p99_9", "d2_interior_p99_9", 6.0),
        ("d2.full_p99_9", "d2_full_p99_9", 12.0),
        ("d2.full_max", "d2_full_max", 30.0),
        ("fold.neg_count", "fold100", 0.0),
        ("blend70.neg_count", "fold70", 0.0),
        ("fold.micro", "micro_pct", 2.0),
        ("fold.crush", "crush_pct", 8.0),
        ("blend70.micro", "micro_pct70", 2.0),
    )
    out = []
    for gate, key, limit in rules:
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            continue
        if float(b) > limit >= float(a):
            out.append(gate)
    return out


def ablations(look_dict: dict, base_metrics: dict, *, size: int = 33) -> list[dict]:
    """Rebuild the look with one operator switched off at a time.

    Compiles with ``strict=False`` so that a look that a gate rejects can still
    be *measured* — we never write a cube from these.
    """
    from engine import pipeline
    from engine.spec import LookSpec

    rows: list[dict] = []
    for name, mutate in ABLATIONS:
        d = copy.deepcopy(look_dict)
        try:
            mutate(d)
        except Exception as exc:  # noqa: BLE001
            rows.append({"switch": name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        def _measure(dd=d):
            spec = LookSpec.from_dict(dd)
            c = pipeline.compile(spec, strict=False)
            return _smooth_fold_metrics(pipeline.lattice(c, size))

        m = _safe(_measure)
        if m is None:
            rows.append({"switch": name, "error": "compile/measure failed"})
            continue
        rows.append({"switch": name, **m, "clears": _clears(base_metrics, m)})
    return rows


# --------------------------------------------------------------------------- #
# anti-blandness block amplitudes
# --------------------------------------------------------------------------- #
def _rms(values, identity) -> float | None:
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if a.size == 0:
        return None
    b = np.asarray(identity, dtype=np.float64)
    if b.size not in (1, a.size):
        return None
    d = a - b
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    return float(np.sqrt(np.mean(d ** 2)))


def block_amplitudes(look_dict: dict, fp: dict) -> dict:
    """RMS of ``(value - identity value)`` per block, ROUND_SPEC §3.

    ``tone ramp`` and ``tint`` are read from the *measured* fingerprint (identity
    = the input code / zero tint); ``dh10 / dh18 / cr / dl`` are the look file's
    own four hue tables (identity = 0 / 0 / 1 / 0), which is where a fitter or a
    nervous colourist shrinks a look toward identity.
    """
    field = look_dict.get("field") or {}
    out: dict[str, float | None] = {}
    out["tone_ramp"] = _rms(_dig(fp, "neutral.ramp8") or [], RAMP_CODES)
    tint = list(_dig(fp, "neutral.tint_rg") or []) + list(_dig(fp, "neutral.tint_bg") or [])
    out["tint"] = _rms(tint, [0.0])
    out["dh10"] = _rms(field.get("dh10") or [0.0] * 12, [0.0])
    out["dh18"] = _rms(field.get("dh18") or [0.0] * 12, [0.0])
    out["cr"] = _rms(field.get("cr") or [1.0] * 12, [1.0])
    out["dl"] = _rms(field.get("dl") or [0.0] * 12, [0.0])
    return out


# --------------------------------------------------------------------------- #
# sheets
# --------------------------------------------------------------------------- #
def verify_sheet(path: Path) -> dict:
    """Re-open a saved sheet and assert the T4 hard limits independently.

    ``tools.sheets.save_sheet`` asserts them too; ROUND_SPEC says *the tool*
    asserts them, so we check the artifact on disk rather than trusting the
    writer (which is itself being changed under us).
    """
    from PIL import Image
    from tools import sheets as SH

    path = Path(path)
    side = path.with_suffix(".json")
    if not path.exists():
        raise AssertionError(f"{path.name}: sheet was not written")
    with Image.open(path) as im:
        w, h = im.size
        icc = bool(im.info.get("icc_profile"))
    meta = _safe(lambda: json.loads(side.read_text(encoding="utf-8")), {}) or {}
    kind = str(meta.get("kind", "?"))
    tiles = meta.get("tiles") or []
    limit = meta.get("tile_long_edge_limit")
    if limit is None:
        limit = SH.TILE_LONG.get(kind, SH.CHART_LONG)
    assert side.exists(), f"{path.name}: missing sidecar {side.name}"
    assert w * h <= SH.MAX_PIXELS, f"{path.name}: {w}x{h} = {w * h} px > {SH.MAX_PIXELS}"
    assert max(w, h) <= SH.MAX_EDGE, f"{path.name}: long edge {max(w, h)} > {SH.MAX_EDGE}"
    assert len(tiles) <= SH.MAX_TILES, f"{path.name}: {len(tiles)} tiles > {SH.MAX_TILES}"
    for t in tiles:
        tw, th = t.get("tile_px", (0, 0))
        assert max(tw, th) <= float(limit), (
            f"{path.name}: tile {max(tw, th)} px > {limit} for kind {kind}")
    assert icc, f"{path.name}: no embedded ICC profile"
    return {"file": str(path), "kind": kind, "px": [w, h],
            "mp": round(w * h / 1e6, 3), "tiles": len(tiles),
            "tile_long": max((max(t.get("tile_px", (0, 0))) for t in tiles), default=0),
            "bytes": path.stat().st_size}


class _Scenes:
    """Loads a base / crop once, applies a table once, and remembers both."""

    def __init__(self) -> None:
        from tools import render

        self._render = render
        self._img: dict[str, np.ndarray] = {}
        self._out: dict[tuple[str, str, float], np.ndarray] = {}

    def image(self, scene: str) -> np.ndarray:
        """``"PANA0116"`` -> the 1600 px frame; ``"PANA0116:face"`` -> the 1:1 crop.

        A ``BASE:crop`` scene resolves to the FROZEN hi-res crop
        ``$W/bases_hi/<base>__<crop>.npz`` (T5 stores the half-size develop's
        crop regions there, 1:1).  Only if that file is missing do we fall back
        to the 1600 px frame — and then the ``crops.json`` box has to be mapped
        from half-size pixels into the frame first, which is exactly what
        ``sheets._load_scene`` / the ``sheets.py`` CLI do NOT do (they hand a
        half-size box straight to ``render.crop``, which cuts the wrong region
        of the 1600 px base; reported, not fixed here — sheets.py is not ours).
        """
        if scene not in self._img:
            base, _, crop_name = str(scene).partition(":")
            if crop_name:
                img = _safe(lambda: self._render.load_base(f"{base}__{crop_name}"))
                if img is None:
                    full = self._render.load_base(base)
                    box = _box_to_frame(self._render.crop_box(base, crop_name), full.shape)
                    img = self._render.crop(full, box)
            else:
                img = self._render.load_base(base)
            self._img[scene] = np.asarray(img, dtype=np.float64)
        return self._img[scene]

    def through(self, scene: str, table, tag: str, strength: float = 1.0) -> np.ndarray:
        key = (scene, str(tag), float(strength))
        if key not in self._out:
            self._out[key] = self._render.apply_table(self.image(scene), table,
                                                      strength=strength)
        return self._out[key]


def _make_sheets(out_dir: Path, name: str, table, scene: dict, rival: Path | None,
                 *, final: bool, verbose: bool,
                 patches: dict | None = None) -> list[dict]:
    """Draw the round's sheets.

    *patches* — when a dict is passed in, the MEASURED patch strips of sheets A
    and B are stored in it as ``{"A": {"scene": …, "patches": [measurement…]},
    "B": {…}}`` so the report can print the numbers as text (ROUND_SPEC A asks
    for the white cleanliness and the skin move to be *numbers*; they were only
    ever drawn into the JPEG, which means they could not be read without
    opening the image).
    """
    from tools import charts as CH
    from tools import sheets as SH

    sc = _Scenes()
    made: list[dict] = []

    def _tile(img, label, **kw):
        return SH.Tile(img, label, **kw)

    def _emit(fn, path: Path, what: str) -> None:
        t0 = time.perf_counter()
        try:
            fn(path)
        except Exception as exc:  # noqa: BLE001 - one bad scene must not kill a round
            made.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            if verbose:
                print(f"  {path.name:<16} SKIPPED: {type(exc).__name__}: {exc}")
            return
        info = verify_sheet(path)
        info["seconds"] = round(time.perf_counter() - t0, 2)
        info["what"] = what
        made.append(info)
        if verbose:
            print(f"  {path.name:<16} {info['px'][0]}x{info['px'][1]} "
                  f"{info['mp']:.3f} MP  tile {info['tile_long']}  "
                  f"{info['bytes'] / 1024:.0f} KB  {info['seconds']:.2f} s")

    portrait = scene["portrait"]
    face = scene["face_crop"]
    heroes = scene["heroes"]

    def _keep(letter: str, scene_id: str, strip) -> None:
        if patches is None:
            return
        patches[letter] = {
            "scene": scene_id,
            "patches": [dict(m) for m in getattr(strip, "measurements", ()) or ()],
        }

    # --- A: full frame portrait + patch strip -----------------------------
    def _a(path: Path) -> None:
        img = sc.image(portrait)
        out = sc.through(portrait, table, name)
        boxes = patch_boxes_full_frame(portrait, img)
        strip = SH.patch_strip(img, out, boxes)
        _keep("A", portrait, strip)
        SH.pair(_tile(img, f"base {portrait}", base=portrait, look="base", strength=0.0),
                _tile(out, f"{name} s100", base=portrait, look=name, strength=1.0),
                out=path, title=f"A portrait {portrait} - base vs {name}",
                subtitle="patches: " + ", ".join(k for k, _ in boxes), strip=strip)

    _emit(_a, out_dir / "A_portrait.jpg", f"PAIR {portrait} + patch strip")

    # --- B: hi-res face crop + patch strip --------------------------------
    def _b(path: Path) -> None:
        img = sc.image(face)
        out = sc.through(face, table, name)
        base_id, _, crop_name = face.partition(":")
        boxes = (patch_boxes_for_crop(base_id, crop_name, img) if crop_name
                 else patch_boxes_full_frame(base_id, img))
        strip = SH.patch_strip(img, out, boxes)
        _keep("B", face, strip)
        SH.pair(_tile(img, f"base {face}", base=base_id, look="base", strength=0.0,
                      crop=crop_name or None),
                _tile(out, f"{name} s100", base=base_id, look=name, strength=1.0,
                      crop=crop_name or None),
                out=path, title=f"B face {face} - base vs {name}",
                subtitle="patches: " + ", ".join(k for k, _ in boxes), strip=strip)

    _emit(_b, out_dir / "B_face.jpg", f"PAIR {face} + patch strip")

    # --- C / D: the four hero scenes --------------------------------------
    def _quad(path: Path, pair_ids: Sequence[str], letter: str) -> None:
        tiles = []
        for hid in pair_ids:
            img = sc.image(hid)
            tiles.append(_tile(img, f"base {hid}", base=hid, look="base", strength=0.0))
            tiles.append(_tile(sc.through(hid, table, name), f"{name} s100 {hid}",
                               base=hid, look=name, strength=1.0))
        SH.quad(tiles, out=path, title=f"{letter} heroes {' / '.join(pair_ids)} - "
                                       f"base vs {name}")

    if len(heroes) >= 2:
        _emit(lambda p: _quad(p, heroes[0:2], "C"), out_dir / "C_hero12.jpg",
              f"QUAD {heroes[0]}, {heroes[1]}")
    if len(heroes) >= 4:
        _emit(lambda p: _quad(p, heroes[2:4], "D"), out_dir / "D_hero34.jpg",
              f"QUAD {heroes[2]}, {heroes[3]}")

    # --- E: rival reference ------------------------------------------------
    if rival is not None and heroes:
        hid = heroes[0]

        def _e(path: Path) -> None:
            img = sc.image(hid)
            SH.triad(
                _tile(img, f"base {hid}", base=hid, look="base", strength=0.0),
                _tile(sc.through(hid, table, name), f"{name} s100",
                      base=hid, look=name, strength=1.0),
                _tile(sc.through(hid, str(rival), rival.stem), f"rival {rival.stem} s100",
                      base=hid, look=rival.stem, strength=1.0, source=str(rival)),
                out=path, title=f"E rival {hid}: base / {name} / {rival.stem}")

        _emit(_e, out_dir / "E_rival.jpg", f"TRIAD {hid} vs {rival.stem}")

    # --- F / G: charts -----------------------------------------------------
    for letter, chart_name in zip("FG", scene["charts"]):
        def _chart(path: Path, cn=chart_name) -> None:
            SH.chart_sheet(CH.load_chart(cn), table, name, out=path, chart_name=cn)

        _emit(_chart, out_dir / f"{letter}_chart_{chart_name}.jpg", f"chart {chart_name}")

    # --- H: strength (only with --final) -----------------------------------
    if final and len(heroes) >= 2:
        hid = heroes[1]

        def _h(path: Path) -> None:
            img = sc.image(hid)
            SH.triad(
                _tile(img, f"base {hid}", base=hid, look="base", strength=0.0),
                _tile(sc.through(hid, table, name, 0.70), f"{name} s070",
                      base=hid, look=name, strength=0.70),
                _tile(sc.through(hid, table, name), f"{name} s100",
                      base=hid, look=name, strength=1.0),
                out=path, title=f"H strength {hid}: base / 70% / 100%")

        _emit(_h, out_dir / "H_strength.jpg", f"TRIAD {hid} strength")

    return made


# --------------------------------------------------------------------------- #
# latest / collision
# --------------------------------------------------------------------------- #
def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def update_latest(rounds_root: Path, name: str, out_dir: Path, round_name: str) -> Path:
    """``<rounds_root>/<name>/latest`` = this round's fp.json + cube (atomic)."""
    latest = Path(rounds_root) / name / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for fn in ("fp.json", f"{name}.cube", "qc.json"):
        src = Path(out_dir) / fn
        if src.exists():
            _atomic_copy(src, latest / fn)
    ptr = {"look": name, "round": round_name, "round_dir": str(out_dir),
           "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tmp = latest / f"latest.json.tmp{os.getpid()}"
    tmp.write_text(json.dumps(ptr, indent=1), encoding="utf-8")
    os.replace(tmp, latest / "latest.json")
    return latest


def collisions(rounds_root: Path, name: str, fp: dict, limit: int = 3) -> list[dict]:
    """``fingerprint.distance`` to every other look's latest fingerprint."""
    from tools import fingerprint as FP

    rows: list[dict] = []
    for path in sorted(Path(rounds_root).glob("*/latest/fp.json")):
        other = _safe(lambda p=path: json.loads(p.read_text(encoding="utf-8")))
        if not isinstance(other, dict):
            continue
        other_name = str(other.get("name") or path.parent.parent.name)
        if other_name == name:
            continue
        d = _safe(lambda o=other: FP.distance(fp, o))
        rows.append({"name": other_name, "distance": d, "path": str(path),
                     "note": "" if d is not None else "not comparable (probe/shape)"})
    rows.sort(key=lambda r: (r["distance"] is None, r["distance"] or 0.0))
    return rows[:limit]


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
_MARK = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL", "skip": "skip"}


def _gate_line(g: dict | None, key: str, alias: str = "") -> str:
    if g is None:
        return f"  skip {key:<24} {alias:<22} {'n/a':>10}  (gate absent in this qc build)"
    mark = _MARK.get(str(g.get("status")), "?   ")
    text = str(g.get("text", "--"))
    unit = str(g.get("unit", "") or "")
    thr = str(g.get("threshold", "") or "")
    label = alias or str(g.get("label", ""))
    val = (text + (" " + unit if unit else "")).strip()
    return f"  {mark} {key:<24} {label:<22} {val:>10}  [{thr}]"


#: a patch whose name says "near-white surface" gets the cleanliness line
#: (chroma before -> after and the delta) instead of the rotation line
_WHITE_NEEDLES = ("white", "cardigan", "high_key", "highkey", "shirt", "sheet")


def _patch_lines(patches: dict | None) -> list[str]:
    """Sheets A and B's MEASURED patch numbers, as text.

    ROUND_SPEC A wants the skin move and the white cleanliness as numbers; they
    were only ever drawn into the sheet JPEG, so nobody could read them without
    opening the image.  Cheek / foliage print the chroma ratio, the hue
    rotation and the lightness move; a near-white surface prints the chroma
    before, the chroma after and the delta — a white that gains chroma is
    exactly the defect the sheet exists to catch.
    """
    out: list[str] = []
    if not patches:
        return ["  patches   no sheets in this round (--no-sheets) -> no measured "
                "patch numbers"]
    for letter in ("A", "B"):
        blk = patches.get(letter) or {}
        rows = blk.get("patches") or []
        if not rows:
            continue
        out.append(f"  {letter} {str(blk.get('scene', '?')):<22} "
                   f"{len(rows)} measured patch(es)")
        for m in rows[:4]:
            nm = str(m.get("name", "?"))
            before = m.get("before") or {}
            after = m.get("after") or {}
            if any(k in nm.lower() for k in _WHITE_NEEDLES):
                out.append(f"      {nm:<18} C {_n(before.get('C'), '{:.4f}')} -> "
                           f"{_n(after.get('C'), '{:.4f}')}  dC "
                           f"{_n(m.get('dC'), '{:+.4f}')}  "
                           f"dL {_n(m.get('dL'), '{:+.4f}')}")
            else:
                out.append(f"      {nm:<18} cr {_n(m.get('cr'), '{:6.3f}')}  "
                           f"dh {_n(m.get('dh'), '{:+6.2f}')} deg  "
                           f"dL {_n(m.get('dL'), '{:+.4f}')}")
    if not out:
        out.append("  patches   sheets A/B produced no patch strip (no tagged crops)")
    return out


def build_report(ctx: dict) -> str:
    """The whole report.txt (also printed).  Compact, <= 120 lines."""
    qc = ctx.get("qc") or {}
    fp = ctx.get("fp") or {}
    L: list[str] = []
    add = L.append

    name = ctx["name"]
    add("=" * 96)
    add(f"ROUND {ctx['round']}  look {name}"
        + (f" ({ctx.get('title') or ''}{'/' + ctx['cn'] if ctx.get('cn') else ''})"
           if ctx.get("title") or ctx.get("cn") else "")
        + f"  order {ctx.get('order', '?')}"
        + ("  MONO" if ctx.get("mono") else "")
        + f"  {ctx['stamp']}")
    add(f"  look    {ctx['look_path']}")
    add(f"  cube    {ctx['cube']}  {ctx.get('cube_size', '?')}^3  "
        f"sha {str(ctx.get('sha256', ''))[:12]}")
    add(f"  scenes  portrait {ctx['scene']['portrait']}  face {ctx['scene']['face_crop']}  "
        f"heroes {','.join(ctx['scene']['heroes']) or '-'}  charts "
        f"{','.join(ctx['scene']['charts'])}")
    add(f"  rival   {ctx.get('rival') or 'none'}"
        + ("" if ctx["scene"]["known"] else
           "   [!] look not in docs/round_scenes.json: no heroes/rival"))
    for w in (ctx.get("warnings") or [])[:3]:
        add(f"  compile WARNING {str(w)[:86]}")

    # -- 1. QC -------------------------------------------------------------
    # GATES_v3 order: the HARD gates exactly as the ruling lists them, then the
    # WARN-only items under their own heading, so what blocks shipping and what
    # is only an opinion are never in the same block.
    add("-" * 96)
    gates = {str(g.get("key")): g for g in (qc.get("gates") or [])}
    hard_fail = [k for k, _ in HARD_GATES
                 if str((gates.get(k) or {}).get("status")) == "fail"]
    add("1. QC   GATES_v3 HARD gates -- a FAIL on any of these BLOCKS SHIPPING")
    for key, alias in HARD_GATES:
        add(_gate_line(gates.get(key), key, alias))
    add(f"  ==> HARD: {len(hard_fail)} FAIL"
        + (" -- BLOCKS SHIPPING: " + ", ".join(hard_fail) if hard_fail
           else " -- nothing on the HARD list blocks this look"))
    shown = {k for k, _ in HARD_GATES}
    add("  -- WARN-only (GATES_v3: reported, never blocks shipping) --")
    for key, alias in WARN_ONLY_GATES:
        shown.add(key)
        add(_gate_line(gates.get(key), key, alias))
    add("     (also WARN-only: pairwise mean dE00 < 2.5, judged on the collision "
        "page; and grey vs T > 0.2, whose HARD line is 0.5)")
    rest = [g for g in (qc.get("gates") or [])
            if str(g.get("key")) not in shown and g.get("status") in ("fail", "warn")]
    if rest:
        # not on either GATES_v3 list: a FAIL here still blocks a build (build.py
        # counts every FAIL), it is just not one of the lead's hard lines.
        add(f"  -- other gates that FAIL/WARN ({len(rest)}; a FAIL here also "
            f"blocks the build) --")
        for g in rest:
            add(_gate_line(g, str(g.get("key"))))
    s = qc.get("summary") or {}
    add(f"  SUMMARY {s.get('pass', '?')} pass / {s.get('warn', '?')} warn / "
        f"{s.get('fail', '?')} fail / {s.get('skip', '?')} skip  -> "
        f"{str(s.get('worst', 'n/a')).upper()}")

    # -- 2. fingerprint ----------------------------------------------------
    add("-" * 96)
    add(f"2. FINGERPRINT  probe {fp.get('probe', 'n/a')}")
    add("  ramp8 in   " + " ".join(f"{c:6d}" for c in RAMP_CODES))
    add("        out  " + _row(_dig(fp, "neutral.ramp8"), "{:6.1f}"))
    tt = _dig(fp, "neutral.t") or (0.05, 0.18, 0.35, 0.50, 0.65, 0.80, 0.95)
    add("  tint  t%   " + " ".join(f"{float(t) * 100:6.0f}" for t in tt))
    add("        rg   " + _row(_dig(fp, "neutral.tint_rg"), "{:+6.2f}"))
    add("        bg   " + _row(_dig(fp, "neutral.tint_bg"), "{:+6.2f}"))
    add(f"  hue rows at L={_n(_dig(fp, 'hue.L'), '{:.2f}')}, the 12 table knots:"
        "  left block C=0.10 | right block chi=min(.18,.80*cmax)")
    add("      h |" + "     dh" + "      cr" + "       dl"
        + " |" + "     dh" + "      cr" + "       dl")
    hues = _dig(fp, "hue.h") or []
    for i, h in enumerate(hues):
        if float(h) % 30.0 != 0.0:
            continue
        cells = [(_n(_dig(fp, f"hue.{col}.dh.{i}"), "{:+7.2f}")
                  + _n(_dig(fp, f"hue.{col}.cr.{i}"), "{:8.3f}")
                  + _n(_dig(fp, f"hue.{col}.dl.{i}"), "{:+9.4f}"))
                 for col in ("c10", "chi")]
        add(f"   {float(h):5.0f} |" + cells[0] + " |" + cells[1])
    arch_L = _dig(fp, "arch.L") or []
    add(f"  arch C={_n(_dig(fp, 'arch.C'), '{:.2f}')}  L="
        + ",".join(f"{float(x):.2f}" for x in arch_L))
    for j, hh in enumerate(_dig(fp, "arch.h") or []):
        add(f"    h {float(hh):3.0f}  cr " + _row(_dig(fp, f"arch.cr.{j}"), "{:6.3f}")
            + "   dl " + _row(_dig(fp, f"arch.dl.{j}"), "{:+7.4f}"))
    for j, patch in enumerate(_dig(fp, "skin.patches") or []):
        add(f"  skin {str(tuple(int(v) for v in patch)):<18}"
            f" dh {_n(_dig(fp, f'skin.dh.{j}'), '{:+6.2f}')}"
            f"  cr {_n(_dig(fp, f'skin.cr.{j}'), '{:6.3f}')}"
            f"  dl {_n(_dig(fp, f'skin.dl.{j}'), '{:+7.4f}')}"
            f"  dE00 {_n(_dig(fp, f'skin.de00.{j}'), '{:6.2f}')}")
    # the measured patch strips of sheets A and B, as text: cheek and foliage as
    # chroma ratio / hue rotation / lightness move, a near-white surface as the
    # chroma it had and the chroma it has now.
    add("  MEASURED PATCHES on sheets A and B (same numbers the strip draws)")
    for line in _patch_lines(ctx.get("patches")):
        add(line)

    # -- 3. strength / anti-blandness --------------------------------------
    add("-" * 96)
    add("3. STRENGTH  dE00 vs the untouched base")
    src = "photo" if _dig(fp, "global.photo") else "lattice17"
    got = _dig(fp, "global.photo") or _dig(fp, "global.lattice17") or {}
    add(f"  {src:<10} n={_n(got.get('n'), '{:.0f}')}   mean "
        f"{_n(got.get('de00_mean'), '{:.3f}')}   p95 {_n(got.get('de00_p95'), '{:.3f}')}"
        f"   p99 {_n(got.get('de00_p99'), '{:.3f}')}   max "
        f"{_n(got.get('de00_max'), '{:.3f}')}")
    bl = ctx.get("baseline")
    if bl is None:
        add("  baseline   none given (--baseline) -> no anti-blandness numbers")
    else:
        add(f"  baseline   {bl['path']}")
        add(f"  {'':<10} mean {_n(bl.get('de00_mean'), '{:.3f}')}"
            f"   ratio {_n(bl.get('ratio'), '{:.3f}')}"
            f"   fingerprint.distance {_n(bl.get('distance'), '{:.3f}')}")
        add("  block amplitude (RMS vs identity)      look   baseline    ratio")
        for key in ("tone_ramp", "tint", "dh10", "dh18", "cr", "dl"):
            a = (bl.get("amp_look") or {}).get(key)
            b = (bl.get("amp_base") or {}).get(key)
            r = (bl.get("amp_ratio") or {}).get(key)
            flag = "  <-- shrank > 8 %" if (r is not None and r < BLAND_RATIO) else ""
            add(f"    {key:<32} {_n(a, '{:8.4f}')} {_n(b, '{:10.4f}')} "
                f"{_n(r, '{:8.3f}')}{flag}")
        for line in bl.get("violations", []):
            add(f"  BLANDNESS VIOLATION: {line}")
        if not bl.get("violations"):
            add("  no blandness violation")

    # -- 4. collision ------------------------------------------------------
    add("-" * 96)
    add(f"4. COLLISION  fingerprint.distance to {ctx['rounds_root']}/*/latest/fp.json")
    col = ctx.get("collisions") or []
    if not col:
        add("  no other look has a latest fingerprint yet")
    for r in col:
        add(f"  {r['name']:<14} {_n(r.get('distance'), '{:8.3f}')}  {r.get('note', '')}")

    # -- 5. ablation -------------------------------------------------------
    add("-" * 96)
    ab = ctx.get("ablation")
    if not ab:
        add("5. OPERATOR ABLATION  not run (no hard smoothness/fold gate fails)")
    else:
        add("5. OPERATOR ABLATION  triggered by these failing hard gates:")
        for chunk in _wrap(", ".join(ab["failing"]), 88):
            add("     " + chunk)
        add("  switch      d2_int_p999  d2_full_p999  d2_full_max  matfold@100  @70"
            "  micro%  crush%   clears")
        base = ab["base"]
        add(f"  {'(none)':<11} {_n(base.get('d2_interior_p99_9'), '{:11.2f}')} "
            f"{_n(base.get('d2_full_p99_9'), '{:13.2f}')} "
            f"{_n(base.get('d2_full_max'), '{:12.2f}')} "
            f"{_n(base.get('fold100'), '{:12.0f}')} {_n(base.get('fold70'), '{:4.0f}')} "
            f"{_n(base.get('micro_pct'), '{:7.3f}')} "
            f"{_n(base.get('crush_pct'), '{:7.2f}')}   -")
        for r in ab["rows"]:
            if r.get("error"):
                add(f"  {r['switch']:<11} {r['error']}")
                continue
            add(f"  {r['switch']:<11} {_n(r.get('d2_interior_p99_9'), '{:11.2f}')} "
                f"{_n(r.get('d2_full_p99_9'), '{:13.2f}')} "
                f"{_n(r.get('d2_full_max'), '{:12.2f}')} "
                f"{_n(r.get('fold100'), '{:12.0f}')} {_n(r.get('fold70'), '{:4.0f}')} "
                f"{_n(r.get('micro_pct'), '{:7.3f}')} "
                f"{_n(r.get('crush_pct'), '{:7.2f}')}   "
                + (", ".join(r.get("clears") or []) or "-"))

    # -- sheets / timing ---------------------------------------------------
    add("-" * 96)
    add("SHEETS (T4 limits re-checked on the saved files)")
    for sh in ctx.get("sheets") or []:
        if sh.get("error"):
            add(f"  {Path(sh['file']).name:<16} ERROR {sh['error'][:60]}")
            continue
        add(f"  {Path(sh['file']).name:<16} {sh['kind']:<6} "
            f"{sh['px'][0]:>4}x{sh['px'][1]:<4} {sh['mp']:.3f} MP  "
            f"{sh['tiles']} tiles  tile {sh['tile_long']:>4}  "
            f"{sh['bytes'] / 1024:>5.0f} KB  {sh['seconds']:>5.2f} s  {sh.get('what', '')}")
    tm = ctx.get("timing") or {}
    add("TIMING  " + "  ".join(f"{k} {v:.2f}s" for k, v in tm.items()))

    if len(L) > REPORT_MAX_LINES:
        keep = REPORT_MAX_LINES - 1
        L = L[:keep] + [f"... {len(L) - keep} lines trimmed to the {REPORT_MAX_LINES}-line "
                        "budget; full numbers in qc.json / fp.json"]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# the round
# --------------------------------------------------------------------------- #
def run_round(
    look_path: str | Path,
    *,
    round_name: str = "r1",
    out_dir: str | Path | None = None,
    rounds_root: str | Path | None = None,
    rival: str | None = "auto",
    baseline: str | Path | None = None,
    final: bool = False,
    sheets: bool = True,
    grid_n: int = 200_000,
    jacobian: bool = True,
    scenes_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Build, QC, fingerprint, draw and report one look.  Returns the context."""
    from engine import pipeline
    from engine.cubeio import read_lut, sha256_file
    from engine.spec import LookSpec
    from tools import fingerprint as FP
    from tools import metrics as M
    from tools import qc as QC

    t_all = time.perf_counter()
    timing: dict[str, float] = {}

    look_path = Path(look_path)
    if not look_path.exists():
        raise RoundError(f"look file not found: {look_path}")
    look_dict = json.loads(look_path.read_text(encoding="utf-8"))
    spec = LookSpec.from_dict(look_dict)
    name = spec.name

    scenes = load_scenes(scenes_path)
    scene = scene_for(name, scenes)
    check_holdout(scene, holdout_ids())
    rival_path = resolve_rival(rival, scene)

    # `latest` and the collision scan live under <rounds_root>.  An explicit
    # --out-dir without an explicit --rounds-root infers the root from the
    # out-dir, so a run pointed somewhere else (a self-test, a scratch tree)
    # keeps its `latest` next to its own output instead of writing into the
    # shared $W/rounds.
    if rounds_root is not None:
        rounds_root = Path(rounds_root)
    elif out_dir is None:
        rounds_root = ROUNDS_ROOT
    else:
        od = Path(out_dir)
        rounds_root = od.parent.parent if od.parent.name == name else od.parent
    out_dir = Path(out_dir) if out_dir is not None else rounds_root / name / round_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- compile + cube ----------------------------------------------------
    t0 = time.perf_counter()
    compiled = pipeline.compile(spec)
    timing["compile"] = time.perf_counter() - t0

    cube = out_dir / f"{name}.cube"
    t0 = time.perf_counter()
    pipeline.write_cube_file(
        compiled, cube, size=33,
        comments=(f"# round {round_name}  look {name}"
                  + (f" ({spec.cn})" if spec.cn else "")
                  + f"  order {spec.order}  source {look_path.name}",))
    timing["cube"] = time.perf_counter() - t0
    lut = read_lut(cube)                       # QC reads the artifact back
    table = np.asarray(lut.table, dtype=np.float64)

    # -- QC ----------------------------------------------------------------
    t0 = time.perf_counter()
    qc = _safe(lambda: QC.qc_table(cube, compiled, name=name, grid_n=grid_n,
                                   jacobian=jacobian))
    if qc is None:                             # qc.py is being rewritten: degrade
        qc = _safe(lambda: QC.qc_table(cube, compiled, name=name)) or {
            "name": name, "gates": [], "summary": {}, "metrics": {},
            "error": "tools.qc.qc_table raised; see qc.json"}
    timing["qc"] = time.perf_counter() - t0
    (out_dir / "qc.json").write_text(json.dumps(qc, indent=1, default=_jsonable),
                                     encoding="utf-8")

    # -- fingerprint --------------------------------------------------------
    t0 = time.perf_counter()
    sampler = M.sampler_from_table(table, title=name)
    fp = FP.fingerprint(sampler, name=name)
    fp["source"] = str(cube)
    fp["round"] = round_name
    timing["fingerprint"] = time.perf_counter() - t0
    (out_dir / "fp.json").write_text(json.dumps(fp, indent=1, default=_jsonable),
                                     encoding="utf-8")

    # -- baseline / anti-blandness -----------------------------------------
    base_ctx = None
    if baseline is not None:
        t0 = time.perf_counter()
        base_ctx = _safe(lambda: _baseline_block(Path(baseline), look_dict, fp))
        if base_ctx is None:
            base_ctx = {"path": str(baseline), "violations": [
                "baseline could not be compiled/measured — see stderr"]}
        timing["baseline"] = time.perf_counter() - t0

    # -- sheets -------------------------------------------------------------
    made: list[dict] = []
    patch_meas: dict = {}
    if sheets:
        t0 = time.perf_counter()
        if verbose:
            print("sheets:")
        made = _make_sheets(out_dir, name, table, scene, rival_path,
                            final=final, verbose=verbose, patches=patch_meas)
        timing["sheets"] = time.perf_counter() - t0

    # -- latest + collision -------------------------------------------------
    t0 = time.perf_counter()
    _safe(lambda: update_latest(rounds_root, name, out_dir, round_name))
    col = _safe(lambda: collisions(rounds_root, name, fp), []) or []
    timing["collision"] = time.perf_counter() - t0

    # -- ablation -----------------------------------------------------------
    gates = {str(g.get("key")): g for g in (qc.get("gates") or [])}
    failing = [k for k in SMOOTH_FOLD_GATES
               if str((gates.get(k) or {}).get("status")) == "fail"]
    ablation = None
    if failing:
        t0 = time.perf_counter()
        base_metrics = _smooth_fold_metrics(table)
        ablation = {"failing": failing, "base": base_metrics,
                    "rows": ablations(look_dict, base_metrics)}
        timing["ablation"] = time.perf_counter() - t0

    timing["total"] = time.perf_counter() - t_all

    ctx = {
        "round": round_name, "name": name, "title": spec.title, "cn": spec.cn,
        "order": spec.order, "mono": spec.mono is not None,
        "look_path": str(look_path), "cube": str(cube),
        "cube_size": int(getattr(lut, "size", table.shape[0])),
        "sha256": _safe(lambda: sha256_file(cube), ""),
        "warnings": list(getattr(compiled, "warnings", ()) or ()),
        "scene": scene, "rival": str(rival_path) if rival_path else None,
        "rounds_root": str(rounds_root),
        "stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "qc": qc, "fp": fp, "baseline": base_ctx, "collisions": col,
        "ablation": ablation, "sheets": made, "patches": patch_meas,
        "timing": timing, "final": bool(final),
    }
    report = build_report(ctx)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    ctx["report"] = report
    ctx["out_dir"] = str(out_dir)
    if verbose:
        print(report)
    return ctx


def _baseline_block(baseline_path: Path, look_dict: dict, fp: dict) -> dict:
    """dE00 / fingerprint distance / block amplitudes of the look vs *baseline*."""
    from engine import pipeline
    from engine.spec import LookSpec
    from tools import fingerprint as FP
    from tools import metrics as M

    base_dict = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    bspec = LookSpec.from_dict(base_dict)
    bc = pipeline.compile(bspec, strict=False)
    btable = pipeline.lattice(bc, 33)
    bfp = FP.fingerprint(M.sampler_from_table(btable, title=bspec.name), name=bspec.name)

    def _mean(f):
        return _dig(f, "global.photo.de00_mean", _dig(f, "global.lattice17.de00_mean"))

    look_mean, base_mean = _mean(fp), _mean(bfp)
    ratio = (float(look_mean) / float(base_mean)
             if look_mean is not None and base_mean not in (None, 0) else None)
    amp_l = block_amplitudes(look_dict, fp)
    amp_b = block_amplitudes(base_dict, bfp)
    amp_r: dict[str, float | None] = {}
    violations: list[str] = []
    for key in amp_l:
        a, b = amp_l[key], amp_b[key]
        if a is None or b is None or b < 1e-9:
            amp_r[key] = None
            continue
        amp_r[key] = a / b
        if amp_r[key] < BLAND_RATIO:
            violations.append(
                f"block {key} amplitude {a:.4f} vs baseline {b:.4f} "
                f"= {amp_r[key] * 100:.1f} % (< {BLAND_RATIO * 100:.0f} %)")
    if ratio is not None and ratio < BLAND_RATIO:
        violations.insert(0, f"dE00 mean {look_mean:.3f} vs baseline {base_mean:.3f} "
                             f"= {ratio * 100:.1f} % (< {BLAND_RATIO * 100:.0f} %)")
    return {
        "path": str(baseline_path), "name": bspec.name,
        "de00_mean": base_mean, "look_de00_mean": look_mean, "ratio": ratio,
        "distance": _safe(lambda: FP.distance(fp, bfp)),
        "amp_look": amp_l, "amp_base": amp_b, "amp_ratio": amp_r,
        "violations": violations,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="round.py",
        description="one colourist round: build -> QC -> fingerprint -> sheets -> report")
    ap.add_argument("look", help="path to a look .json")
    ap.add_argument("--round", dest="round_name", default="r1", help="round name (r1, r2, ...)")
    ap.add_argument("--out-dir", default=None,
                    help="default: <rounds-root>/<look name>/<round>")
    ap.add_argument("--rounds-root", default=None,
                    help=f"holds <name>/latest and the collision scan; default {ROUNDS_ROOT}, "
                         "or inferred from an explicit --out-dir")
    ap.add_argument("--rival", default="auto", help="auto | none | <cube path>")
    ap.add_argument("--baseline", default=None,
                    help="an earlier version of the same look -> anti-blandness numbers")
    ap.add_argument("--final", action="store_true", help="also draw H_strength.jpg")
    ap.add_argument("--scenes", default=None,
                    help=f"scene list (default {SCENES_JSON})")
    ap.add_argument("--no-sheets", action="store_true", help="skip every sheet (numbers only)")
    ap.add_argument("--no-jacobian", action="store_true", help="skip qc's 65^3 Jacobian check")
    ap.add_argument("--grid-n", type=int, default=200_000, help="qc grid-error samples")
    ap.add_argument("--quiet", action="store_true", help="do not print the report")
    args = ap.parse_args(argv)

    try:
        run_round(
            args.look, round_name=args.round_name, out_dir=args.out_dir,
            rounds_root=args.rounds_root, rival=args.rival, baseline=args.baseline,
            final=args.final, sheets=not args.no_sheets, grid_n=args.grid_n,
            jacobian=not args.no_jacobian, scenes_path=args.scenes,
            verbose=not args.quiet,
        )
    except HoldoutError as exc:
        print(f"round.py: {exc}", file=sys.stderr)
        return 2
    except RoundError as exc:
        print(f"round.py: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"round.py: {args.look} is not valid JSON: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        # A look that does not validate (engine.spec.SpecError) lands here, as
        # does any other hard stop: say what it was on one line rather than
        # handing the orchestrator a traceback.
        print(f"round.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
