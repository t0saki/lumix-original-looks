"""tools/build.py — one process: every look json -> a shipped .cube -> QC -> manifest.

For each ``looks/<name>.json`` (files whose name starts with ``_`` are skipped
unless ``--include-demo`` / ``--include-all`` asks for them)::

    load_look -> compile (strict) -> lattice(33) -> out/LUTs/<Name>.cube
              -> read the file back from disk
              -> tools.qc.qc_table(path, compiled)        (every PLAN §验证 gate)
              -> tools.fingerprint.fingerprint(sampler)   (probe latent-probe-1)
              -> work.nosync/review/qc_<Name>.json
    ... then the cross-look gate (pairwise dE00) and out/manifest.json.

The ``.cube`` is written by ``engine.pipeline.write_cube_file``, which enforces
the two camera rules (``#LUMIXPHOTOSTYLE STD`` header, stem <= 8 ASCII
alphanumerics) and refuses a look that only compiles with ``strict=False``.

Exit code (``docs/REVIEW_r1.md``, tooling rulings) follows the ``GATES_v3``
HARD list ONLY: it is non-zero when a look FAILs one of those gates, or when a
look fails to compile at all.  A FAIL on any other gate — ``skin.displacement``
on 10Splice is the ruling's own example — is reported as "other FAIL" and does
not block shipping.  ``ap.banding`` is an INFO alias of ``d2.interior_p99_9``
and is never a second FAIL/WARN.  The build-level distinctiveness gate now has
a FAIL level (pairwise photo dE00 < 1.8, WARN < 2.5) and is printed as a FAIL,
but "HARD list only" governs the exit code, so it does not set it.

``--tag <str>`` builds an experimental variant without touching the shipped
set: cubes go to ``out/LUTs_<tag>/``, QC reports to
``work.nosync/review/qc_<tag>/`` and the manifest to
``out/manifest_<tag>.json``.  ``--looks-dir`` points the whole thing at another
directory of look files (default ``looks/``).

CLI::

    ./py -m tools.build --include-demo
    ./py -m tools.build --only 01Glaze --no-jacobian
    ./py -m tools.build --tag gates_v11
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from engine import pipeline
from engine.cubeio import read_lut, sha256_file
from engine.spec import SpecError, load_look
from tools import metrics as M
from tools import qc as QC

__all__ = ["discover_looks", "build_look", "build_all", "format_summary", "main"]


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


DEFAULT_LOOKS = _root() / "looks"
DEFAULT_OUT = _root() / "out"
DEFAULT_REVIEW = _root() / "work.nosync" / "review"


def tag_paths(tag: str | None) -> dict:
    """Where a ``--tag`` build writes: cubes, QC reports, manifest.

    ``tag = None`` is the shipped set (``out/LUTs``,
    ``work.nosync/review``, ``out/manifest.json``); any other tag is a variant
    that must not overwrite a single file of it.
    """
    if not tag:
        return {"out_dir": DEFAULT_OUT / "LUTs", "review_dir": DEFAULT_REVIEW,
                "manifest": DEFAULT_OUT / "manifest.json", "tag": None}
    t = str(tag)
    if "/" in t or "\\" in t or t in (".", ".."):
        raise ValueError(f"--tag must be a plain name, got {tag!r}")
    return {
        "out_dir": DEFAULT_OUT / f"LUTs_{t}",
        "review_dir": DEFAULT_REVIEW / f"qc_{t}",
        "manifest": DEFAULT_OUT / f"manifest_{t}.json",
        "tag": t,
    }


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def discover_looks(
    looks_dir: Path | None = None,
    *,
    include_demo: bool = False,
    include_all: bool = False,
    only: Sequence[str] = (),
) -> list[Path]:
    """The look files to build, sorted.

    Ships everything not starting with ``_``.  ``include_demo`` adds
    ``_demo_*.json``; ``include_all`` adds every underscore file (``_identity``
    included).  ``only`` filters by file stem or by the look's ``name``.
    """
    d = Path(looks_dir or DEFAULT_LOOKS)
    out: list[Path] = []
    for p in sorted(d.glob("*.json")):
        stem = p.stem
        if stem.startswith("_"):
            if include_all:
                pass
            elif include_demo and stem.startswith("_demo"):
                pass
            else:
                continue
        out.append(p)
    if only:
        want = {str(o) for o in only}
        keep = []
        for p in out:
            if p.stem in want or p.name in want:
                keep.append(p)
                continue
            try:
                if load_look(p).name in want:
                    keep.append(p)
            except Exception:
                pass
        out = keep
    return out


# ---------------------------------------------------------------------------
# one look
# ---------------------------------------------------------------------------


def build_look(
    look_path: Path,
    *,
    out_dir: Path | None = None,
    review_dir: Path | None = None,
    size: int = 33,
    run_qc: bool = True,
    run_fingerprint: bool = True,
    photo_sample=None,
    grid_n: int = 200_000,
    jacobian: bool = True,
    jac_size: int = 65,
    lim_from_diagnostics: bool = True,
) -> dict:
    """Compile, write, read back, QC and fingerprint a single look.

    Returns the manifest entry.  Raises ``SpecError`` if the look does not
    validate — a look that does not validate is not shipped, full stop.
    """
    out_dir = Path(out_dir or (DEFAULT_OUT / "LUTs"))
    review_dir = Path(review_dir or DEFAULT_REVIEW)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    spec = load_look(look_path)
    compiled = pipeline.compile(spec)
    t_compile = time.perf_counter() - t0

    cube = out_dir / f"{spec.name}.cube"
    t0 = time.perf_counter()
    comments = (
        f"# look {spec.name}"
        + (f" ({spec.cn})" if spec.cn else "")
        + f"  order={spec.order}  source={Path(look_path).name}",
    )
    pipeline.write_cube_file(compiled, cube, size=size, comments=comments)
    t_write = time.perf_counter() - t0

    # --- read the artifact back: QC runs on the file, never on memory ------
    lut = read_lut(cube)
    table = lut.table

    entry: dict = {
        "name": spec.name,
        "cn": spec.cn,
        "title": spec.title,
        "order": spec.order,
        "mono": spec.mono is not None,
        "source": str(Path(look_path).resolve().relative_to(_root())
                      if Path(look_path).resolve().is_relative_to(_root())
                      else Path(look_path)),
        "file": str(cube.relative_to(_root())) if cube.is_relative_to(_root()) else str(cube),
        "size": int(lut.size),
        "bytes": int(cube.stat().st_size),
        "sha256": sha256_file(cube),
        "compile_warnings": list(compiled.warnings),
        "timing_s": {"compile": t_compile, "write": t_write},
    }

    report = None
    if run_qc:
        t0 = time.perf_counter()
        report = QC.qc_table(
            cube, compiled, name=spec.name, photo_sample=photo_sample,
            grid_n=grid_n, jacobian=jacobian, jac_size=jac_size,
            lim_from_diagnostics=lim_from_diagnostics,
        )
        entry["timing_s"]["qc"] = time.perf_counter() - t0
        entry["qc"] = {
            "summary": report["summary"],
            "fails": report["fails"],
            # GATES_v3's own HARD list, split out: what blocks shipping, in the
            # order the ruling writes it, and what else FAILed on top of it.
            "hard_fails": report.get("hard_fails", []),
            "other_fails": report.get("other_fails", []),
            "warns": report["warns"],
            "headline": _headline(report),
        }

    if run_fingerprint:
        from tools import fingerprint as FP

        t0 = time.perf_counter()
        sampler = M.sampler_from_table(table, title=spec.name)
        fp = FP.fingerprint(sampler, photo_sample, name=spec.name)
        fp["source"] = entry["file"]
        entry["timing_s"]["fingerprint"] = time.perf_counter() - t0
        entry["probe"] = fp.get("probe")
        if report is not None:
            report["fingerprint"] = fp
        else:
            report = {"name": spec.name, "fingerprint": fp, "gates": [],
                      "summary": QC.gate_summary([])}

    if report is not None:
        review_dir.mkdir(parents=True, exist_ok=True)
        qc_path = review_dir / f"qc_{spec.name}.json"
        qc_path.write_text(json.dumps(report, indent=1, default=_jsonable),
                           encoding="utf-8")
        entry["qc_report"] = str(qc_path)
    entry["_table"] = table  # stripped before the manifest is written
    entry["_report"] = report
    return entry


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return float(o)


#: ``(headline label, qc gate key)``.  The HARD block comes first, in
#: ``docs/GATES_v3.md`` order, then the WARN-only numbers, then the few
#: informational ones.  A key that is no longer a gate (GATES_v3 REMOVED it)
#: is filled from ``report["metrics"]`` by :func:`_headline` instead.
_HEADLINE = (
    # --- GATES_v3 HARD ----------------------------------------------------
    ("grey", "neutral.vs_target"),
    ("white", "neutral.white_err"),
    ("hair", "ap.hair_black"),
    ("black_lift", "ap.black_lift"),
    ("folds", "fold.neg_count"),          # v1.2 W6: MATERIAL folds
    ("fold70", "blend70.neg_count"),
    ("micro", "fold.micro"),
    ("micro70", "blend70.micro"),
    ("crush", "fold.crush"),
    ("jac_neg", "jacobian.neg_count"),
    ("d2_int", "d2.interior_p99_9"),
    ("d2_full", "d2.full_p99_9"),
    ("d2_max", "d2.full_max"),
    ("grid_p_mean", "grid.photo_mean"),
    ("grid_p_p99", "grid.photo_p99"),
    ("grid_p_max", "grid.photo_max"),
    ("at_zero", "clip.at_zero"),
    ("at_one", "clip.at_one"),
    ("skin_h_lo", "skin.hue_band_lo"),
    ("skin_h_hi", "skin.hue_band_hi"),
    ("skin_cr_b", "skin.skin_chroma_bright"),
    ("skin_cr_d", "skin.skin_chroma_dark"),
    ("hilite", "ap.highlight_clean"),     # GATES_v3: replaces ap.wb_preset
    # --- GATES_v3 WARN-only -----------------------------------------------
    ("grid_r_mean", "grid.random_mean"),
    ("grid_r_p99", "grid.random_p99"),
    ("grid_r_max", "grid.random_max"),
    ("push", "gamut.push_p99_9"),
    ("lim", "gamut.lim"),
    ("min_gain", "gamut.min_radial_gain"),
    ("de00", "de00.vs_identity"),
    ("skin_disp", "skin.displacement"),
    # --- neither: still gates, just not on either of GATES_v3's lists ------
    ("tone_slope", "neutral.tone_slope_min"),
    ("collapsed", "fold.collapsed"),
    ("clip_noop", "clip.final_noop"),
    # REVIEW_r1: `ap.banding` is an INFO alias of `d2.interior_p99_9`.  It is
    # carried so the number is visible in the manifest, and it has no status
    # column of its own — the gate that owns it is `d2_int`, above.
    ("banding", "ap.banding"),
    # GATES_v3 REMOVED the ``blend70.d2_p99_9`` GATE ("exactly 0.7 x the
    # full-lattice d2 — redundant and contradictory"), so there is no gate row
    # to read any more.  The NUMBER is still worth a column, so it is filled
    # from metrics below and carries no status.
    ("d2_70", "blend70.d2_p99_9"),
)


def _headline(report: dict) -> dict:
    by = {g["key"]: g for g in report["gates"]}
    out = {}
    for label, key in _HEADLINE:
        g = by.get(key)
        out[label] = None if g is None else g["value"]
        if g is not None:
            out[label + "_status"] = g["status"]
    metrics = report.get("metrics", {}) or {}
    # GATES_v3 removed the 70 %-blend smoothness gate; the measurement lives on.
    if out.get("d2_70") is None:
        out["d2_70"] = ((metrics.get("blend70") or {}).get("d2") or {}).get("p99_9")
    # `lim` is the engine's own measured compressor limit (v1.1 R2).  Until the
    # engine publishes it the gate is a skip; qc still reports what R2's
    # arithmetic would give on this lattice, which is worth carrying in the
    # table so the lead can see the shape of the number.
    est = metrics.get("nm_lim", {}) or {}
    out["lim_est"] = est.get("qc_estimate")
    out["lim_source"] = est.get("source")
    # the highlight-cleanliness population size, which GATES_v3 asks for by name
    hc = metrics.get("ap_highlight_clean") or {}
    out["hilite_n"] = hc.get("n")
    out["hilite_pop"] = hc.get("population")
    out["hilite_fell_back"] = bool(hc.get("fell_back"))
    return out


# ---------------------------------------------------------------------------
# the whole set
# ---------------------------------------------------------------------------


def build_all(
    look_paths: Sequence[Path],
    *,
    out_dir: Path | None = None,
    review_dir: Path | None = None,
    manifest: Path | None = None,
    photo_sample=None,
    **kw,
) -> dict:
    """Build every look, then the cross-look gate, then write ``out/manifest.json``."""
    out_dir = Path(out_dir or (DEFAULT_OUT / "LUTs"))
    manifest = Path(manifest or (DEFAULT_OUT / "manifest.json"))

    entries: list[dict] = []
    errors: list[dict] = []
    t_start = time.perf_counter()
    for p in look_paths:
        try:
            entries.append(build_look(p, out_dir=out_dir, review_dir=review_dir,
                                      photo_sample=photo_sample, **kw))
        except (SpecError, NotImplementedError) as exc:
            errors.append({"source": str(p), "error": f"{type(exc).__name__}: {exc}"})

    tables = {e["name"]: e.pop("_table") for e in entries}
    reports = {e["name"]: e.pop("_report") for e in entries}

    pair = QC.pairwise_de00(tables, photo_sample=photo_sample) if len(tables) > 1 else {
        "status": QC.SKIP, "pairs": [], "min_mean_de00": None,
        "note": "fewer than two looks built",
    }

    fails = sum(len(e.get("qc", {}).get("fails", ())) for e in entries)
    hard = sum(len(e.get("qc", {}).get("hard_fails", ())) for e in entries)
    # docs/REVIEW_r1.md, tooling rulings: "`tools/build.py`: exit code and
    # 'NOT SHIPPABLE' must follow the GATES_v3 HARD list only (10Splice's
    # `skin.displacement` is WARN-only)."  So a FAIL on a gate that is not on
    # that list is printed under "other FAIL" and does NOT block the build.
    # Two things block it besides the HARD list, and only these two:
    #   * a look that does not compile/validate: it has no artifact at all, so
    #     there is nothing to gate and nothing to ship.
    # The build-level DISTINCTIVENESS gate gains a FAIL level in the same ruling
    # ("pairwise photo dE00 < 1.8 = FAIL") and is reported as a FAIL, but the
    # "HARD list only" sentence governs the exit code, so it does not set it.
    doc = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(_root()),
        "n_looks": len(entries),
        "build_time_s": time.perf_counter() - t_start,
        "looks": entries,
        "pairwise_de00": pair,
        "errors": errors,
        "gates": {
            "fail_count": fails,
            # GATES_v3's own HARD list, counted separately from any other gate
            # that happens to FAIL.
            "hard_fail_count": hard,
            "other_fail_count": fails - hard,
            "pairwise_status": pair.get("status"),
            # The distinctiveness gate now has a FAIL level (REVIEW_r1) and it
            # is reported as one, but the SAME ruling says the exit code follows
            # "the GATES_v3 HARD list only", and that list is per-look.  So a
            # collision is a loud FAIL in the log and in the manifest; it does
            # not change the exit code.  (On the r1 set it bites: 05Clear vs
            # 06Almond at 1.39, which is exactly the round-2 work the review
            # orders.)
            "distinctiveness_ok": pair.get("status") != QC.FAIL,
            "ok": bool(hard == 0 and not errors),
            "ok_rule": "GATES_v3 HARD fails == 0 and every look compiles "
                       "(REVIEW_r1: HARD list only)",
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(doc, indent=1, default=_jsonable), encoding="utf-8")
    doc["_tables"] = tables
    doc["_reports"] = reports
    doc["manifest"] = str(manifest)
    return doc


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

#: ``(header, width, section, headline key)``.  ``section`` is "H" for a
#: ``docs/GATES_v3.md`` HARD gate, "W" for one of its WARN-only numbers and "i"
#: for a number that is not on either list — the header block says which is
#: which, so the table itself reads as "what blocks shipping | what does not".
_COLS = (
    ("look", 10, "H", "name"),
    ("ord", 4, "H", "order"),
    ("greyT", 6, "H", "grey"),
    ("hair", 6, "H", "hair"),
    ("blkLift", 7, "H", "black_lift"),
    ("matfold", 7, "H", "folds"),
    ("f70", 4, "H", "fold70"),
    ("micro%", 6, "H", "micro"),
    ("mic70%", 6, "H", "micro70"),
    ("crush%", 6, "H", "crush"),
    ("jacneg", 6, "H", "jac_neg"),
    ("d2int", 6, "H", "d2_int"),
    ("d2full", 6, "H", "d2_full"),
    ("d2max", 6, "H", "d2_max"),
    ("phM", 5, "H", "grid_p_mean"),
    ("phP99", 6, "H", "grid_p_p99"),
    ("phMax", 6, "H", "grid_p_max"),
    ("clip0", 5, "H", "at_zero"),
    ("clip1", 5, "H", "at_one"),
    ("hLo", 5, "H", "skin_h_lo"),
    ("hHi", 5, "H", "skin_h_hi"),
    ("crBrt", 6, "H", "skin_cr_b"),
    ("crDrk", 6, "H", "skin_cr_d"),
    ("hilite", 6, "H", "hilite"),
    ("gridM", 6, "W", "grid_r_mean"),
    ("gridMax", 7, "W", "grid_r_max"),
    ("push", 6, "W", "push"),
    ("lim", 6, "W", "lim_cell"),
    ("mgain", 6, "W", "min_gain"),
    ("dE00", 6, "W", "de00"),
    ("skinD", 6, "W", "skin_disp"),
    ("slope", 6, "i", "tone_slope"),
    ("d2@70", 6, "i", "d2_70"),
)

_SECTION_NAME = {"H": "GATES_v3 HARD - a FAIL here blocks shipping",
                 "W": "WARN-only - never blocks shipping", "i": "info"}


def _cell(v) -> str:
    if v is None:
        return "--"
    v = float(v)
    if v == int(v) and abs(v) < 1e6:
        return f"{int(v)}"
    if abs(v) >= 100 or (abs(v) < 0.01 and v != 0.0):
        return f"{v:.2e}"
    return f"{v:.3f}"


def _section_ruler() -> str:
    """The banner over the table that names the three column blocks."""
    parts: list[str] = []
    run_w, run_s = 0, _COLS[0][2]
    for i, (_n, w, sec, _k) in enumerate(_COLS):
        if sec != run_s:
            parts.append((run_s, run_w))
            run_w, run_s = 0, sec
        run_w += w + (2 if i else 0)
    parts.append((run_s, run_w))
    out = []
    for sec, w in parts:
        label = f"|<- {_SECTION_NAME[sec]} "
        out.append(label[:w].ljust(w, "-") if len(label) <= w else "|<- " + sec)
    return "  ".join(out)


def format_summary(doc: dict, *, verbose: bool = False) -> str:
    """The one-row-per-look build table, plus the cross-look gate.

    Column order follows ``docs/GATES_v3.md``: every HARD gate first, then the
    WARN-only numbers, then the two informational ones.
    """
    lines: list[str] = []
    head = "  ".join(f"{name:>{w}}" if i else f"{name:<{w}}"
                     for i, (name, w, _s, _k) in enumerate(_COLS))
    lines.append(_section_ruler())
    lines.append(head)
    lines.append("-" * len(head))
    for e in doc["looks"]:
        h = e.get("qc", {}).get("headline", {})
        cells = []
        for i, (_name, w, _sec, key) in enumerate(_COLS):
            if key == "name":
                txt = e["name"]
            elif key == "order":
                txt = e["order"]
            elif key == "lim_cell":
                # the gate's value when the engine publishes `lim`; otherwise
                # qc's own R2 arithmetic, marked with a leading ~
                txt = (_cell(h.get("lim")) if h.get("lim") is not None
                       else ("~" + _cell(h.get("lim_est"))
                             if h.get("lim_est") is not None else "--"))
            else:
                txt = _cell(h.get(key))
            cells.append(f"{txt:>{w}}" if i else f"{txt:<{w}}")
        s = e.get("qc", {}).get("summary", {})
        hard = e.get("qc", {}).get("hard_fails", [])
        lines.append("  ".join(cells)
                     + f"   {s.get('pass', 0)}P/{s.get('warn', 0)}W/{s.get('fail', 0)}F"
                     + (f"  HARD:{len(hard)}" if hard else ""))
    lines.append("")
    for e in doc["looks"]:
        qc = e.get("qc", {})
        hard = qc.get("hard_fails", [])
        other = qc.get("other_fails")
        if other is None:                     # a report from before GATES_v3
            other = [k for k in qc.get("fails", []) if k not in set(hard)]
        w = qc.get("warns", [])
        h = qc.get("headline", {})
        lines.append(f"{e['name']}: {e['file']}  sha {e['sha256'][:12]}  "
                     f"{e['bytes'] / 1024:.0f} KiB")
        lines.append(f"    hard FAIL {len(hard)}"
                     + (": " + ", ".join(hard) if hard else ": none"))
        if other:
            lines.append(f"    other FAIL {len(other)} (not on the HARD list "
                         f"-> does NOT block shipping): " + ", ".join(other))
        if h.get("banding") is not None:
            lines.append(f"    info  ap.banding {h['banding']:.2f} codes "
                         f"(alias of d2.interior_p99_9, gated there)")
        if w:
            lines.append(f"    warn {len(w)}: " + ", ".join(w))
        if h.get("hilite_n") is not None:
            lines.append(f"    highlight cleanliness population n={h['hilite_n']}"
                         + (f" ({h['hilite_pop']})" if h.get("hilite_pop") else "")
                         + ("  (FELL BACK to input L >= 0.70: the L >= 0.80 "
                            "population held < 500 px)" if h.get("hilite_fell_back")
                            else ""))
        for cw in e.get("compile_warnings", []):
            lines.append(f"    compile: {cw[:150]}")
    p = doc.get("pairwise_de00", {})
    if p.get("pairs"):
        lines.append("")
        lines.append(f"pairwise dE00 ({p['population']}, n={p['n']}): "
                     f"min {p['min_mean_de00']:.3f} "
                     f"[{p['worst_pair']['a']} vs {p['worst_pair']['b']}]  "
                     f"-> {p['status'].upper()}  (REVIEW_r1 distinctiveness: "
                     f"FAIL < {QC.PAIRWISE_DE00[0]}, WARN < {QC.PAIRWISE_DE00[1]}"
                     f")")
        if p.get("status") == QC.FAIL:
            lines.append("    ^ DISTINCTIVENESS FAIL: those two looks are one "
                         "look on the photo sample.  It does not change the "
                         "exit code (REVIEW_r1: HARD list only) — it is round-2 "
                         "work for the colourist.")
        if verbose:
            for row in p["pairs"]:
                lines.append(f"    {row['a']:>10} vs {row['b']:<10} {row['mean_de00']:7.3f}")
    for err in doc.get("errors", []):
        lines.append(f"ERROR {err['source']}: {err['error']}")
    g = doc["gates"]
    lines.append("")
    lines.append(f"{doc['n_looks']} look(s), {g['fail_count']} failing gate(s) "
                 f"({g.get('hard_fail_count', 0)} of them on GATES_v3's HARD "
                 f"list), manifest {doc.get('manifest')}  -> "
                 + ("OK" if g["ok"] else "NOT SHIPPABLE"))
    lines.append("shippability = GATES_v3 HARD fails only (+ a look that does "
                 "not compile).  A FAIL on any other gate is printed above as "
                 "'other FAIL' and does not block shipping; `ap.banding` is an "
                 "info alias of d2.interior_p99_9 and is never counted at all; "
                 f"the pairwise dE00 FAIL line ({QC.PAIRWISE_DE00[0]}) is "
                 "reported, not enforced.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tools/build.py",
        description="Compile every look to out/LUTs/<name>.cube, QC it, write out/manifest.json.",
    )
    ap.add_argument("--looks-dir", type=Path, default=DEFAULT_LOOKS,
                    help="directory of look .json files (default looks/)")
    ap.add_argument("--tag", default=None,
                    help="build a variant: cubes to out/LUTs_<tag>/, reports to "
                         "work.nosync/review/qc_<tag>/, manifest to "
                         "out/manifest_<tag>.json — out/LUTs is left untouched")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--review-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--include-demo", action="store_true",
                    help="also build looks/_demo_*.json")
    ap.add_argument("--include-all", action="store_true",
                    help="also build every other _*.json (e.g. _identity)")
    ap.add_argument("--only", action="append", default=[],
                    help="build only this look (file stem or look name); repeatable")
    ap.add_argument("--size", type=int, default=33)
    ap.add_argument("--no-qc", action="store_true")
    ap.add_argument("--no-fingerprint", action="store_true")
    ap.add_argument("--no-jacobian", action="store_true")
    ap.add_argument("--no-photo", action="store_true", help="ignore photo_sample.npy")
    ap.add_argument("--no-diag-lim", action="store_true",
                    help="do not call engine.pipeline.diagnostics() looking for `lim`")
    ap.add_argument("--grid-n", type=int, default=200_000)
    ap.add_argument("--jac-size", type=int, default=65)
    ap.add_argument("--gates", action="store_true",
                    help="also print the full gate table for every look")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    paths = discover_looks(a.looks_dir, include_demo=a.include_demo,
                           include_all=a.include_all, only=a.only)
    if not paths:
        print("no look files matched")
        return 2
    where = tag_paths(a.tag)
    out_dir = a.out_dir or where["out_dir"]
    review_dir = a.review_dir or where["review_dir"]
    manifest = a.manifest or where["manifest"]
    if a.tag:
        print(f"tag {a.tag!r}: cubes -> {out_dir}\n"
              f"{' ' * (len(a.tag) + 7)}reports -> {review_dir}\n"
              f"{' ' * (len(a.tag) + 7)}manifest -> {manifest}\n"
              f"{' ' * (len(a.tag) + 7)}looks -> {a.looks_dir}")
    doc = build_all(
        paths, out_dir=out_dir, review_dir=review_dir, manifest=manifest,
        photo_sample=False if a.no_photo else None,
        size=a.size, run_qc=not a.no_qc, run_fingerprint=not a.no_fingerprint,
        grid_n=a.grid_n, jacobian=not a.no_jacobian, jac_size=a.jac_size,
        lim_from_diagnostics=not a.no_diag_lim,
    )
    doc["tag"] = a.tag
    if a.gates:
        for name, rep in doc["_reports"].items():
            if rep and rep.get("gates"):
                print(QC.format_table(rep, verbose=a.verbose))
                print()
    print(format_summary(doc, verbose=a.verbose))
    return 0 if doc["gates"]["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
