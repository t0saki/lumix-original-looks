"""P6 delivery gallery: a self-contained local HTML try-out book.

No CDN, no fetch(), no build step — every page is plain HTML with its own
inline ``<style>`` and a few lines of inline ``<script>``, and every picture is
a JPEG file sitting next to it.  Double-clicking ``index.html`` from the
unzipped folder works offline, which is the whole point (``file://`` blocks
``fetch``, so all data is inlined as a JS literal).

Pages
-----
``index.html``
    The twelve looks as cards on one hero scene, the "how to use" text, and
    links to every look page and to the contact sheet.
``look_<name>.html``
    Hero scenes with a draggable **split line** (left = the untouched base,
    right = the look), a **70 % toggle** that swaps the graded half for the
    70 %-strength render, the **C1 / C3 chart pair** (original above, through
    the LUT below), the **fingerprint essentials** as a small table, and the
    when-to-use / when-to-avoid / EV / strength text.
``contact.html``
    One strip per scene: the untouched base followed by all twelve looks.

Text comes from ``docs/使用说明.md`` when it exists — a heading per look whose
text contains the look's name, ``01Glaze``, ``Glaze`` or its Chinese name, with
labelled lines (``适合：`` / ``避免：`` / ``曝光：`` / ``强度：`` or the English
``when to use:`` …).  Anything missing becomes a visible placeholder rather
than a silent gap.

CLI::

    py tools/gallery.py --cubes out/LUTs --out out/gallery
    py tools/gallery.py --cubes out/LUTs_r1 --out out/_selftest_gallery --heroes PANA0116,PANA0005
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont

from tools import fingerprint as fp_mod, render
from tools.render import ROOT, WORK

__all__ = ["HERO_SCENES", "parse_guide", "essentials", "build", "main"]

#: the hero scenes a look page shows (portrait / golden hour / night / green).
HERO_SCENES = ("PANA0116", "P1038265", "PANA0005", "PANA0078")

HERO_PX = 1200          # <= 1600 px per spec; 1200 keeps the zip under 60 MB
THUMB_PX = 460
CHART_PX = 1200
STRIP_PX = 1500
QUALITY = 88

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}
_ICC: bytes | None = None


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                try:
                    _font_cache[size] = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
        else:  # pragma: no cover
            _font_cache[size] = ImageFont.load_default(size=size)
    return _font_cache[size]


def _icc() -> bytes:
    global _ICC
    if _ICC is None:
        _ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return _ICC


def _save_jpeg(img: np.ndarray | Image.Image, path: Path, long_edge: int | None, quality: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, Image.Image):
        pil = img
        if long_edge and max(pil.size) > long_edge:
            scale = long_edge / float(max(pil.size))
            pil = pil.resize((max(1, round(pil.width * scale)), max(1, round(pil.height * scale))),
                             Image.LANCZOS)
    else:
        arr = np.asarray(img, dtype=np.float64)
        if long_edge and max(arr.shape[:2]) > long_edge:
            arr = render.resize_long_edge(arr, long_edge)
        pil = render.to_pil(arr)
    pil.save(path, "JPEG", quality=int(quality), subsampling=0, optimize=True, icc_profile=_icc())
    return path


# --------------------------------------------------------------------------- #
# look metadata
# --------------------------------------------------------------------------- #
def look_meta(stem: str, looks_dir: Path | None = None) -> dict:
    """``{"name", "cn", "title", "order"}`` from ``looks/<stem>.json`` if present."""
    looks_dir = looks_dir if looks_dir is not None else ROOT / "looks"
    path = Path(looks_dir) / f"{stem}.json"
    meta = {"name": stem, "cn": "", "title": stem, "order": ""}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return meta
        for key in ("cn", "title", "order"):
            if data.get(key):
                meta[key] = str(data[key])
    m = re.match(r"^(\d+)(.*)$", stem)
    meta["number"] = m.group(1) if m else ""
    if meta["title"] == stem and m:
        meta["title"] = m.group(2)
    return meta


# --------------------------------------------------------------------------- #
# 使用说明.md
# --------------------------------------------------------------------------- #
_LABELS: dict[str, tuple[str, ...]] = {
    "when_to_use": ("适合", "何时使用", "什么时候用", "推荐场景", "when to use", "use when", "use for"),
    "when_to_avoid": ("避免", "不适合", "何时避免", "慎用", "when to avoid", "avoid", "avoid when"),
    "ev": ("曝光", "ev", "exposure"),
    "strength": ("强度", "strength", "opacity"),
}
_PLACEHOLDER = {
    "when_to_use": "（待填：docs/使用说明.md 里这支的「适合」一段）",
    "when_to_avoid": "（待填：docs/使用说明.md 里这支的「避免」一段）",
    "ev": "（待填：建议曝光）",
    "strength": "（待填：建议强度）",
}


def _label_of(line: str) -> tuple[str | None, str]:
    text = line.strip().lstrip("-*• \t")
    text = re.sub(r"^\*\*(.+?)\*\*", r"\1", text)
    for sep in ("：", ":"):
        if sep in text:
            head, _, rest = text.partition(sep)
            key = head.strip().strip("*_ ").lower()
            for name, synonyms in _LABELS.items():
                if any(key == s or key.startswith(s) for s in synonyms):
                    return name, rest.strip()
            break
    return None, text


def parse_guide(path: str | Path | None, looks: Sequence[dict]) -> dict:
    """Split ``使用说明.md`` into an intro plus a section per look.

    Tolerant by design: any heading level, and a section is matched to a look
    when the heading mentions its file stem, its English title or its Chinese
    name.  Returns ``{"intro": md, "looks": {stem: {...}}, "source": str|None}``.
    """
    out = {"intro": "", "looks": {}, "source": None, "found": []}
    if not path or not Path(path).exists():
        return out
    out["source"] = str(path)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"(?m)^(#{1,6})\s*(.+?)\s*$", text)
    intro = parts[0].strip()
    sections: list[tuple[int, str, str]] = []
    for i in range(1, len(parts), 3):
        sections.append((len(parts[i]), parts[i + 1], parts[i + 2]))

    def matches(heading: str, look: dict) -> bool:
        low = heading.lower()
        for token in (look["name"], look.get("title", ""), look.get("cn", "")):
            if token and str(token).lower() in low:
                return True
        return False

    consumed: set[int] = set()
    for look in looks:
        for idx, (_lvl, heading, body) in enumerate(sections):
            if idx in consumed or not matches(heading, look):
                continue
            consumed.add(idx)
            entry: dict = {"heading": heading, "summary": [], "body_md": body.strip()}
            for line in body.splitlines():
                if not line.strip():
                    continue
                key, value = _label_of(line)
                if key:
                    entry[key] = value
                elif not line.lstrip().startswith("#"):
                    entry["summary"].append(line.strip())
            entry["summary"] = " ".join(entry["summary"])[:600]
            out["looks"][look["name"]] = entry
            out["found"].append(look["name"])
            break
    # everything before the first matched look heading is the general intro
    if sections and consumed:
        first = min(consumed)
        intro_parts = [intro] + [f"{'#' * lvl} {h}\n{b}" for lvl, h, b in sections[:first]]
        intro = "\n\n".join(p for p in intro_parts if p.strip())
    out["intro"] = intro.strip()
    return out


def md_to_html(text: str) -> str:
    """A deliberately tiny Markdown subset: headings, bullets, bold, paragraphs."""
    if not text.strip():
        return ""
    lines = text.splitlines()
    out: list[str] = []
    in_list = False

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append("<p>" + inline(" ".join(buf)) + "</p>")
            buf.clear()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush()
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        m = re.match(r"^(#{1,6})\s*(.+)$", line)
        if m:
            flush()
            if in_list:
                out.append("</ul>")
                in_list = False
            level = min(6, len(m.group(1)) + 1)
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            continue
        if re.match(r"^\s*[-*•]\s+", line):
            flush()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + inline(re.sub(r"^\s*[-*•]\s+", "", line)) + "</li>")
            continue
        buf.append(line.strip())
    flush()
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# fingerprint essentials
# --------------------------------------------------------------------------- #
def essentials(fp: dict) -> list[tuple[str, str]]:
    """The handful of fingerprint numbers a colourist actually reads."""
    ne, sk, hu, gl = fp["neutral"], fp["skin"], fp["hue"], fp["global"]
    rows: list[tuple[str, str]] = []
    rows.append(("黑点 / 白点 (8-bit)", f"{ne['black'][1]:.1f} / {ne['white'][1]:.1f}"))
    rows.append(("色调斜率 趾 / 中 / 肩", f"{ne['toe']:.3f} / {ne['mid']:.3f} / {ne['shoulder']:.3f}"))
    ramp = {int(c): v for c, v in zip(ne["codes"], ne["ramp8"])}
    rows.append((
        "灰阶 in → out",
        " · ".join(f"{c}→{ramp[c]:.0f}" for c in (31, 64, 115, 166, 209) if c in ramp),
    ))
    t = list(ne["t"])
    picks = [t.index(v) for v in (0.18, 0.50, 0.80) if v in t]
    rows.append((
        "中性偏色 R−G / B−G (18/50/80 %)",
        " · ".join(f"{ne['tint_rg'][i]:+.1f}/{ne['tint_bg'][i]:+.1f}" for i in picks),
    ))
    labels = ("中间肤色", "暗部肤色", "高光肤色")
    for i, label in enumerate(labels):
        rows.append((
            f"{label} {tuple(sk['patches'][i])}",
            f"Δh {sk['dh'][i]:+.1f}°  C ×{sk['cr'][i]:.3f}  ΔL {sk['dl'][i]:+.3f}  dE00 {sk['de00'][i]:.2f}",
        ))
    dh = [v for v in hu["c10"]["dh"] if v is not None]
    cr = [v for v in hu["c10"]["cr"] if v is not None]
    if dh and cr:
        worst = max(dh, key=abs)
        idx = hu["c10"]["dh"].index(worst)
        rows.append((
            "色相环 L=0.65 C=0.10",
            f"最大 Δh {worst:+.1f}° @ h={hu['h'][idx]:.0f}°  ·  平均彩度 ×{np.mean(cr):.3f}",
        ))
    photo = gl.get("photo")
    if photo:
        rows.append(("整体强度 (照片像素 dE00)", f"平均 {photo['de00_mean']:.2f}  p95 {photo['de00_p95']:.2f}"))
    lattice = next((gl[k] for k in gl if k.startswith("lattice")), None)
    if lattice:
        rows.append(("整体强度 (均匀格点 dE00)", f"平均 {lattice['de00_mean']:.2f}  p95 {lattice['de00_p95']:.2f}"))
    rows.append(("灰阶单调", "是" if ne.get("monotone") else "否"))
    return rows


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
CSS = """
:root{color-scheme:dark;--paper:#efece4;--muted:#a7a59e;--ground:#171819;--card:#202223;--line:#3a3c3d}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--paper);
 font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB",sans-serif}
a{color:#d9d6ce;text-underline-offset:4px}
main{max-width:1180px;margin:0 auto;padding:26px 28px 70px}
header.top{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;
 padding-bottom:18px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.eyebrow{letter-spacing:.22em;font-size:11px;color:var(--muted);margin-bottom:6px}
h1{font:400 clamp(26px,4vw,40px)/1.15 Georgia,"Songti SC",serif;margin:0}
h1 .cn{font:400 18px/1.4 -apple-system,"PingFang SC",sans-serif;margin-left:14px;color:var(--muted)}
h2{font:400 21px/1.3 Georgia,"Songti SC",serif;margin:34px 0 12px;
 border-bottom:1px solid var(--line);padding-bottom:7px}
h3{font:600 15px/1.4 -apple-system,"PingFang SC",sans-serif;margin:18px 0 6px}
nav.looknav{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 0;font-size:13px}
nav.looknav a{padding:5px 10px;border:1px solid var(--line);text-decoration:none;border-radius:2px}
nav.looknav a.here{background:var(--paper);color:#171819;border-color:var(--paper)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;margin-top:18px}
.card{background:var(--card);border:1px solid var(--line);text-decoration:none;color:inherit;display:block}
.card img{display:block;width:100%;aspect-ratio:3/2;object-fit:cover}
.card b{display:block;font:400 18px Georgia,"Songti SC",serif;margin:9px 11px 0}
.card span{display:block;font-size:12.5px;color:var(--muted);margin:1px 11px 10px}
.scenebar{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 10px}
.scenebar button{font:inherit;font-size:13px;color:var(--paper);background:#262829;
 border:1px solid var(--line);padding:6px 12px;cursor:pointer;border-radius:2px}
.scenebar button[aria-pressed=true]{background:var(--paper);color:#171819;border-color:var(--paper)}
.stage{position:relative;width:100%;aspect-ratio:3/2;background:#0f1011;overflow:hidden;
 border:1px solid var(--line)}
.stage img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block}
.stage .after-wrap{position:absolute;inset:0;clip-path:inset(0 0 0 50%)}
.stage .divider{position:absolute;top:0;bottom:0;left:50%;width:1px;background:#f2efe6;pointer-events:none}
.stage .divider:after{content:"↔";position:absolute;top:50%;left:0;transform:translate(-50%,-50%);
 width:34px;height:34px;border-radius:50%;border:1px solid #c9c6bd;background:#242525;
 display:grid;place-items:center;font-size:15px}
.stage .badge{position:absolute;top:12px;background:rgba(15,16,17,.76);padding:3px 9px;font-size:12px;z-index:2}
.stage .badge.l{left:12px}.stage .badge.r{right:12px}
.stage input[type=range]{position:absolute;inset:0;width:100%;height:100%;margin:0;opacity:0;cursor:ew-resize}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:11px 0 0;font-size:13px;color:var(--muted)}
.controls label{color:var(--paper);display:flex;gap:7px;align-items:center;cursor:pointer}
figure{margin:0 0 18px}
figure img{width:100%;display:block;border:1px solid var(--line)}
figcaption{font-size:12.5px;color:var(--muted);margin-top:6px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
table.fp{border-collapse:collapse;width:100%;font-size:13.5px}
table.fp td{border-top:1px solid var(--line);padding:7px 10px;vertical-align:top}
table.fp td:first-child{color:var(--muted);width:42%}
table.fp td:last-child{font-variant-numeric:tabular-nums}
.textblock{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
.textblock .box{background:var(--card);border:1px solid var(--line);padding:14px 16px}
.textblock .box h3{margin-top:0}
.placeholder{color:#c08a7a}
.strips figure{margin-bottom:26px}
footer{border-top:1px solid var(--line);margin-top:40px;padding-top:14px;font-size:12px;color:var(--muted)}
.dl{display:inline-block;border:1px solid #6a6b63;padding:8px 15px;text-decoration:none;font-size:13px}
@media (max-width:640px){main{padding:18px 14px 44px}.stage{aspect-ratio:4/3}}
"""

_JS = """
(function(){
 var S=DATA.scenes, i=0, s70=false;
 var st=document.getElementById('stage'), before=document.getElementById('before'),
     after=document.getElementById('after'), wrap=document.getElementById('afterwrap'),
     div=document.getElementById('divider'), drag=document.getElementById('drag'),
     bl=document.getElementById('badge-l'), br=document.getElementById('badge-r'),
     cap=document.getElementById('scene-cap'), box=document.getElementById('scenebar');
 function split(){var v=drag.value; wrap.style.clipPath='inset(0 0 0 '+v+'%)'; div.style.left=v+'%';}
 function paint(){var sc=S[i];
  before.src=sc.base; before.alt=sc.label+' 基底';
  after.src=s70?sc.s70:sc.s100; after.alt=sc.label+' '+DATA.look+(s70?' 70%':' 100%');
  br.textContent=DATA.look+' · '+(s70?'70%':'100%');
  cap.textContent=sc.label+'（'+sc.id+'）';
  [].forEach.call(box.children,function(b,k){b.setAttribute('aria-pressed',k===i?'true':'false');});
  split();}
 [].forEach.call(box.children,function(b,k){b.addEventListener('click',function(){i=k;paint();});});
 drag.addEventListener('input',split);
 var t=document.getElementById('t70');
 if(t){t.addEventListener('change',function(){s70=t.checked;paint();});}
 paint();
})();
"""


def _page(title: str, body: str, script: str = "") -> str:
    return (
        "<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
        f"{body}\n" + (f"<script>\n{script}\n</script>\n" if script else "") + "</body>\n</html>\n"
    )


def _nav(looks: list[dict], here: str | None) -> str:
    items = ['<a href="index.html"%s>总览</a>' % (' class="here"' if here is None else "")]
    for lk in looks:
        cls = ' class="here"' if lk["name"] == here else ""
        items.append(f'<a href="look_{lk["name"]}.html"{cls}>{html.escape(lk["name"])}</a>')
    items.append('<a href="contact.html">全场景对照</a>')
    return '<nav class="looknav">' + "".join(items) + "</nav>"


# --------------------------------------------------------------------------- #
# image builders
# --------------------------------------------------------------------------- #
def _chart_figure(chart: np.ndarray, table, name: str, out: Path, long_edge: int, quality: int) -> Path:
    """Original above / through the LUT below, one JPEG."""
    src = np.asarray(chart, dtype=np.float64)
    out_img = render.apply_table(src, table)
    rows = [render.resize_long_edge(a, long_edge) if max(a.shape[:2]) > long_edge else a
            for a in (src, out_img)]
    h = sum(r.shape[0] for r in rows)
    w = max(r.shape[1] for r in rows)
    label_h = 26
    canvas = Image.new("RGB", (w, h + 2 * label_h + 6), (23, 24, 25))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for row, text in zip(rows, (f"{name} 原图", f"{name} 经 LUT")):
        draw.text((6, y + 4), text, font=_font(16), fill=(232, 229, 221))
        y += label_h
        canvas.paste(render.to_pil(row), ((w - row.shape[1]) // 2, y))
        y += row.shape[0] + 3
    return _save_jpeg(canvas, out, long_edge, quality)


def _contact_strip(scene: str, base: np.ndarray, looks: list[dict], tables: dict,
                   out: Path, long_edge: int, quality: int, cols: int = 5) -> Path:
    cell_w = long_edge // cols
    h, w = base.shape[:2]
    cell_h = max(1, int(round(cell_w * h / w)))
    label_h = 24
    # The LUT runs on a 2x-of-cell copy, not on the full 1600 px frame: a contact
    # thumbnail is 300 px wide and this is ~30x cheaper for 20 scenes x 12 looks.
    src = render.resize_long_edge(base, 2 * max(cell_w, cell_h))
    items = [("base", "原图 / 未套用", src)]
    for lk in looks:
        items.append((lk["name"], f"{lk['name']} {lk['cn']}".strip(),
                      render.apply_table(src, tables[lk["name"]])))
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cell_w * cols, rows * (cell_h + label_h)), (23, 24, 25))
    draw = ImageDraw.Draw(canvas)
    for idx, (_key, label, img) in enumerate(items):
        r, c = divmod(idx, cols)
        small = render.resize_long_edge(img, max(cell_w, cell_h))
        pil = render.to_pil(small).resize((cell_w, cell_h), Image.LANCZOS)
        x, y = c * cell_w, r * (cell_h + label_h)
        canvas.paste(pil, (x, y))
        draw.rectangle([x, y + cell_h, x + cell_w - 1, y + cell_h + label_h - 1], fill=(42, 44, 45))
        draw.text((x + 6, y + cell_h + 3), label, font=_font(15), fill=(232, 229, 221))
    return _save_jpeg(canvas, out, long_edge, quality)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(
    cubes_dir: str | Path,
    out_dir: str | Path,
    *,
    heroes: Sequence[str] | None = None,
    contact_scenes: Sequence[str] | None = None,
    guide: str | Path | None = None,
    hero_px: int = HERO_PX,
    chart_px: int = CHART_PX,
    strip_px: int = STRIP_PX,
    quality: int = QUALITY,
    looks: Sequence[str] | None = None,
    lut_href: str | None = None,
    progress: bool = True,
) -> dict:
    """Write the whole gallery under *out_dir* and return a small report."""
    t0 = time.perf_counter()
    cubes_dir = Path(cubes_dir)
    out_dir = Path(out_dir)
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(cubes_dir.glob("*.cube"))
    if looks:
        wanted = {s.strip() for s in looks}
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise FileNotFoundError(f"no .cube files in {cubes_dir}")
    metas = [dict(look_meta(p.stem), cube=str(p)) for p in paths]
    tables = {p.stem: render.table_of(p) for p in paths}

    hero_ids = list(heroes) if heroes else [s for s in HERO_SCENES]
    if contact_scenes is not None:
        contact_ids = list(contact_scenes)
    else:
        core = [s for s in (WORK / "bases_core.txt").read_text().split() if s.strip()]
        hold = [s for s in (WORK / "bases_holdout.txt").read_text().split() if s.strip()]
        contact_ids = core + hold
    try:
        index = json.loads((WORK / "cal" / "bases_index.json").read_text())
    except (OSError, json.JSONDecodeError):
        index = {}

    guide_path = Path(guide) if guide else (ROOT / "docs" / "使用说明.md")
    guide_data = parse_guide(guide_path if Path(guide_path).exists() else None, metas)

    # ---- images -------------------------------------------------------- #
    written: list[Path] = []
    hero_data: dict[str, list[dict]] = {m["name"]: [] for m in metas}
    for sid in hero_ids:
        base = render.load_base(sid)
        base_file = img_dir / f"base_{sid}.jpg"
        written.append(_save_jpeg(base, base_file, hero_px, quality))
        label = (index.get(sid, {}) or {}).get("scene", sid)
        for m in metas:
            r100 = render.apply_table(base, tables[m["name"]])
            r070 = render.apply_table(base, tables[m["name"]], strength=0.7)
            f100 = img_dir / f"{m['name']}_{sid}_100.jpg"
            f070 = img_dir / f"{m['name']}_{sid}_070.jpg"
            written.append(_save_jpeg(r100, f100, hero_px, quality))
            written.append(_save_jpeg(r070, f070, hero_px, quality))
            hero_data[m["name"]].append({
                "id": sid,
                "label": label.split("：")[0][:70] if label else sid,
                "base": f"img/{base_file.name}",
                "s100": f"img/{f100.name}",
                "s70": f"img/{f070.name}",
            })
            if sid == hero_ids[0]:
                written.append(_save_jpeg(r100, img_dir / f"thumb_{m['name']}.jpg", THUMB_PX, quality))
        if progress:
            print(f"  hero {sid}: {1 + 2 * len(metas)} images", flush=True)

    from tools import charts as charts_mod

    chart_imgs = {}
    for cname in ("C1", "C3"):
        try:
            chart_imgs[cname] = charts_mod.load_chart(cname)
        except Exception as exc:  # pragma: no cover - charts are built by T3
            print(f"  WARNING: chart {cname} unavailable ({exc})", flush=True)
    for m in metas:
        for cname, chart in chart_imgs.items():
            written.append(_chart_figure(chart, tables[m["name"]], cname,
                                         img_dir / f"chart_{cname}_{m['name']}.jpg", chart_px, quality))
    if progress and chart_imgs:
        print(f"  charts: {len(metas) * len(chart_imgs)} images", flush=True)

    strip_files: list[tuple[str, str, str]] = []
    for sid in contact_ids:
        try:
            base = render.load_base(sid)
        except FileNotFoundError:
            print(f"  WARNING: contact scene {sid} not found, skipped", flush=True)
            continue
        out = _contact_strip(sid, base, metas, tables, img_dir / f"strip_{sid}.jpg", strip_px, quality)
        written.append(out)
        label = (index.get(sid, {}) or {}).get("scene", sid)
        strip_files.append((sid, label, f"img/{out.name}"))
    if progress:
        print(f"  contact strips: {len(strip_files)}", flush=True)

    # ---- fingerprints --------------------------------------------------- #
    fps = {}
    for m in metas:
        sampler = fp_mod.M.sampler_from_table(tables[m["name"]], title=m["name"])
        fps[m["name"]] = fp_mod.fingerprint(sampler, name=m["name"])
    if progress:
        print(f"  fingerprints: {len(fps)}", flush=True)

    # ---- pages ---------------------------------------------------------- #
    pages: list[Path] = []
    for i, m in enumerate(metas):
        pages.append(_write_look_page(out_dir, m, metas, i, hero_data[m["name"]],
                                      chart_imgs.keys(), fps[m["name"]], guide_data, cubes_dir,
                                      lut_href))
    pages.append(_write_index(out_dir, metas, hero_data, guide_data, cubes_dir, hero_ids))
    pages.append(_write_contact(out_dir, metas, strip_files))

    total_bytes = sum(p.stat().st_size for p in written) + sum(p.stat().st_size for p in pages)
    report = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cubes": str(cubes_dir.resolve()),
        "out": str(out_dir.resolve()),
        "looks": [m["name"] for m in metas],
        "hero_scenes": hero_ids,
        "contact_scenes": [s for s, _l, _f in strip_files],
        "charts": sorted(chart_imgs),
        "guide": guide_data["source"],
        "lut_href": lut_href,
        "guide_sections_found": guide_data["found"],
        "pages": [p.name for p in pages],
        "images": len(written),
        "max_image_px": hero_px,
        "quality": quality,
        "bytes": total_bytes,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "gallery.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    if progress:
        print(
            f"gallery: {len(pages)} pages + {len(written)} images, "
            f"{total_bytes / 1e6:.1f} MB, {report['elapsed_s']} s -> {out_dir}",
            flush=True,
        )
    return report


def _cube_href(out_dir: Path, cube: Path, prefix: str | None = None) -> str:
    """Where the page points for the ``.cube`` download.

    When the cubes sit right next to the gallery folder the real relative path
    is used (so the link works while browsing the working tree).  Otherwise the
    delivery layout is assumed — ``gallery/`` and ``LUTs/`` siblings inside the
    zip — because that is where the page will actually be opened.  *prefix*
    (``--lut-href``) forces the delivery form.
    """
    if prefix:
        return f"{prefix.rstrip('/')}/{cube.name}"
    try:
        rel = os.path.relpath(cube, out_dir).replace(os.sep, "/")
    except ValueError:  # pragma: no cover - different drives, impossible on macOS
        rel = ""
    if rel and not rel.startswith("/") and rel.count("../") <= 1:
        return rel
    return f"../LUTs/{cube.name}"


def _text_box(title: str, key: str, entry: dict | None) -> str:
    value = (entry or {}).get(key)
    if value:
        return f'<div class="box"><h3>{html.escape(title)}</h3><p>{html.escape(str(value))}</p></div>'
    return (
        f'<div class="box"><h3>{html.escape(title)}</h3>'
        f'<p class="placeholder">{html.escape(_PLACEHOLDER[key])}</p></div>'
    )


def _write_look_page(out_dir: Path, m: dict, metas: list[dict], i: int, scenes: list[dict],
                     chart_names, fp: dict, guide: dict, cubes_dir: Path,
                     lut_href: str | None = None) -> Path:
    entry = guide["looks"].get(m["name"])
    prev_m, next_m = metas[i - 1], metas[(i + 1) % len(metas)]
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in essentials(fp)
    )
    charts = "".join(
        f'<figure><img src="img/chart_{c}_{m["name"]}.jpg" alt="{c} 图表">'
        f"<figcaption>{c} — 上：原图，下：经 LUT。合成图表用来看色带、偏色与压缩。</figcaption></figure>"
        for c in chart_names
    )
    summary = (entry or {}).get("summary") or ""
    body = f"""<main>
<header class="top">
 <div><div class="eyebrow">LATENT 2026 · {html.escape(m.get('number', ''))}</div>
  <h1>{html.escape(m['title'])}<span class="cn">{html.escape(m['cn'])}</span></h1>
  <p style="margin:8px 0 0;color:var(--muted);font-size:13px">{html.escape(summary)[:220]}</p></div>
 <a class="dl" href="{html.escape(_cube_href(out_dir, Path(m['cube']), lut_href))}" download>下载 {html.escape(m['name'])}.cube</a>
</header>
{_nav(metas, m['name'])}

<h2>场景对照</h2>
<div class="scenebar" id="scenebar">{''.join(
 f'<button type="button" aria-pressed="{"true" if k == 0 else "false"}">{html.escape(s["id"])}</button>'
 for k, s in enumerate(scenes))}</div>
<div class="stage" id="stage">
 <img id="before" alt="基底"><div class="after-wrap" id="afterwrap"><img id="after" alt="套用 LUT"></div>
 <div class="badge l">原图 / 未套用</div><div class="badge r" id="badge-r"></div>
 <div class="divider" id="divider"></div>
 <input id="drag" type="range" min="0" max="100" value="50" aria-label="左右拖动分界线">
</div>
<div class="controls">
 <label><input type="checkbox" id="t70"> 以 70 % 强度预览</label>
 <span>拖动画面上的分界线比较；左边始终是未套用的基底。</span>
 <span id="scene-cap"></span>
</div>

<h2>图表 C1 / C3</h2>
<div class="charts">{charts}</div>

<h2>指纹要点</h2>
<table class="fp">{rows}</table>

<h2>怎么用</h2>
<div class="textblock">
 {_text_box('适合', 'when_to_use', entry)}
 {_text_box('避免', 'when_to_avoid', entry)}
 {_text_box('曝光', 'ev', entry)}
 {_text_box('强度', 'strength', entry)}
</div>
{('<h2>说明原文</h2>' + md_to_html(entry['body_md'])) if entry and entry.get('body_md') else ''}

<footer>
 上一支 <a href="look_{html.escape(prev_m['name'])}.html">{html.escape(prev_m['name'])}</a> ·
 下一支 <a href="look_{html.escape(next_m['name'])}.html">{html.escape(next_m['name'])}</a> ·
 <a href="contact.html">全场景对照</a> · <a href="index.html">总览</a><br>
 相机：Standard + sRGB，33 点 CUBE。以上为电脑端预览，并非机内实拍；强度滑杆按
 s·LUT(x)+(1−s)·x 在 sRGB 码值上混合。
</footer>
</main>"""
    script = "const DATA=" + json.dumps(
        {"look": m["name"], "cn": m["cn"], "scenes": scenes}, ensure_ascii=False
    ) + ";\n" + _JS
    path = out_dir / f"look_{m['name']}.html"
    path.write_text(_page(f"{m['title']} {m['cn']} — Latent 2026", body, script), encoding="utf-8")
    return path


def _write_index(out_dir: Path, metas: list[dict], hero_data: dict, guide: dict,
                 cubes_dir: Path, hero_ids: list[str]) -> Path:
    cards = "".join(
        f'<a class="card" href="look_{html.escape(m["name"])}.html">'
        f'<img src="img/thumb_{html.escape(m["name"])}.jpg" alt="{html.escape(m["name"])}">'
        f'<b>{html.escape(m["title"])}</b><span>{html.escape(m["cn"])} · {html.escape(m["name"])}</span></a>'
        for m in metas
    )
    intro = md_to_html(guide["intro"]) if guide["intro"] else (
        '<p class="placeholder">（待填：docs/使用说明.md 尚未写好；这里会放整体使用说明、'
        '相机设置与强度建议。）</p>'
    )
    missing = [m["name"] for m in metas if m["name"] not in guide["looks"]]
    warn = (
        f'<p class="placeholder">尚无说明文字的 LUT：{html.escape(", ".join(missing))}</p>'
        if missing else ""
    )
    body = f"""<main>
<header class="top">
 <div><div class="eyebrow">A STUDY IN COLOUR · 2026</div>
  <h1>Latent 2026<span class="cn">{len(metas)} 支原创 LUT</span></h1>
  <p style="margin:8px 0 0;color:var(--muted);font-size:13px">
   Panasonic Lumix S9 · Real Time LUT · Standard + sRGB · 33 点 CUBE</p></div>
 <a class="dl" href="使用说明.md">使用说明</a>
</header>
{_nav(metas, None)}
<h2>十二支</h2>
<div class="cards">{cards}</div>
<h2>怎么用</h2>
{intro}
{warn}
<h2>全场景对照</h2>
<p>每个场景一条：未套用的基底在最前，随后是十二支的 100 % 效果 —
 <a href="contact.html">打开对照页</a>。</p>
<footer>
 首页示例场景：{html.escape(", ".join(hero_ids))}。全部图片与页面都在本地文件夹内，离线可用。
 LUT 文件：{html.escape(str(cubes_dir.name))}/。
</footer>
</main>"""
    path = out_dir / "index.html"
    path.write_text(_page("Latent 2026 — 试色册", body), encoding="utf-8")
    return path


def _write_contact(out_dir: Path, metas: list[dict], strips: list[tuple[str, str, str]]) -> Path:
    figs = "".join(
        f'<figure><img src="{html.escape(src)}" alt="{html.escape(sid)} 全部 LUT 对照" loading="lazy">'
        f"<figcaption><b>{html.escape(sid)}</b> — {html.escape(label)}</figcaption></figure>"
        for sid, label, src in strips
    )
    body = f"""<main>
<header class="top">
 <div><div class="eyebrow">LATENT 2026</div><h1>全场景对照<span class="cn">每个场景一条</span></h1>
 <p style="margin:8px 0 0;color:var(--muted);font-size:13px">
  每条的第一格是未套用的基底，随后是十二支的 100 % 效果。</p></div>
 <a class="dl" href="index.html">返回总览</a>
</header>
{_nav(metas, "__contact__")}
<div class="strips">{figs}</div>
<footer>{len(strips)} 个场景 × {len(metas)} 支 LUT。核心场景与 holdout 场景都在其中。</footer>
</main>"""
    path = out_dir / "contact.html"
    path.write_text(_page("Latent 2026 — 全场景对照", body), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gallery.py", description=__doc__.split("\n")[0])
    ap.add_argument("--cubes", default=str(ROOT / "out" / "LUTs"))
    ap.add_argument("--out", default=str(ROOT / "out" / "gallery"))
    ap.add_argument("--heroes", default=None, help="comma-separated hero scene ids")
    ap.add_argument("--contact-scenes", default=None)
    ap.add_argument("--looks", default=None)
    ap.add_argument("--guide", default=None, help="path to 使用说明.md")
    ap.add_argument("--lut-href", default=None,
                    help="force the per-look download link prefix (delivery layout: ../LUTs)")
    ap.add_argument("--hero-px", type=int, default=HERO_PX)
    ap.add_argument("--chart-px", type=int, default=CHART_PX)
    ap.add_argument("--strip-px", type=int, default=STRIP_PX)
    ap.add_argument("--quality", type=int, default=QUALITY)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if max(args.hero_px, args.chart_px, args.strip_px) > 1600:
        raise SystemExit("gallery images must stay at or below 1600 px on the long edge")
    build(
        args.cubes,
        args.out,
        heroes=args.heroes.split(",") if args.heroes else None,
        contact_scenes=args.contact_scenes.split(",") if args.contact_scenes else None,
        looks=args.looks.split(",") if args.looks else None,
        lut_href=args.lut_href,
        guide=args.guide,
        hero_px=args.hero_px,
        chart_px=args.chart_px,
        strip_px=args.strip_px,
        quality=args.quality,
        progress=not args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
