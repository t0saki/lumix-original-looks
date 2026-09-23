# Latent 潜影 — 12 film / "German-look" LUTs for LUMIX S9 (Standard base)

A second set, designed from scratch: sRGB→sRGB, 33-point `.cube`, header `#LUMIXPHOTOSTYLE STD`, file stems ≤ 8 characters —
copy `luts/*.cube` to the SD card's `LUT` folder. Full guide (Chinese): [使用说明.md](使用说明.md).

| file | name | idea |
|---|---|---|
| 01Glaze 琉璃 | Leica-like | chroma-gated red-gold / ultramarine, clean whites, solid blacks |
| 02Burin 铜版 | Zeiss/Contax-like | chroma arch: inky shadows, dense mids, clean highlights |
| 03Gilt 鎏金 | Kodachrome on a reversal skeleton | neutrals darker, colours glow; skin exempt from gold |
| 04Viride 苍翠 | tropical Velvia | deep, wet greens; protected skin |
| 05Clear 澄明 | Provia-like honesty | brightest mids, cleanest whites, azure skies |
| 06Almond 杏色 | Portra-like | creamy skin, soft tone, olive greens |
| 07Voile 轻纱 | over-exposed 400H | airy pastels, mint and blush, blacks kept |
| 08Arcade 骑楼 | Classic Neg street | cyan-shadow → magenta-mid crossover, teal greens |
| 09Tinsel 千禧金 | Gold / 2000s compact | warm/cool chroma asymmetry, honey highlights |
| 10Splice 放映 | 2383 print | print toe, slate → ivory axis, teal blues |
| 11Sodium 夜灯 | 800T night | indigo air, golden lamps, neon without banding |
| 12Argent 银盐 | yellow-filter Tri-X | per-hue tonal separation (100 % only) |

Quality gates on the shipped files: zero material folds (sign-normalised tetrahedra), interior second difference p99.9 4.1–5.9 codes,
exact white, clean near-whites, skin chroma/hue inside per-look windows; robustness to ±EV, WB/tint shifts and ISO 12800 noise.

`engine/` + `looks/*.json` regenerate the cubes (Python ≥ 3.12, numpy, scipy): order S (input saturation) → P (OKLCh hue-table field)
→ G (code-domain 8-norm gamut compressor) → N (per-channel tone + Gaussian-bump tints). `tools/build.py` compiles and QCs.
The photo-based review tooling needs a private base library and is included for reference only.
