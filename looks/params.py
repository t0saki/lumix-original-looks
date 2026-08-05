"""The look recipes.

Conventions: tone control points in the recipe's native domain ("lstar" = CIE
L* of the neutral axis, "code" = sRGB code); hue centers/halfwidths in OKLCh
degrees; luminance windows in CIE L*; chroma in OKLab units. Midgray targets
below are the suite ladder at reference input sRGB 0.466 (L* ~50.0).

Where a recipe states a qualitative bias ("+3 toward amber") the chroma value
follows the suite convention established by Fieldnote (+3 ~ C 0.005).
"""

from __future__ import annotations

VERSION = "1.0"

LOOKS: dict[str, dict] = {}


def _boost(look: dict, tone: float = 1.0, colr: float = 1.0, hue: float = 1.0, sat: float = 1.0) -> dict:
    """Scale a recipe's expressive magnitude while keeping every structural
    guarantee (monotone tone via re-fit, white->white, skin window intact).

    A literal translation of the design-panel recipes lands far weaker than
    intended (a reference film look measures dE00 ~9 vs identity; the literal
    recipes ~2), so each recipe carries strength factors: tone = tonal
    deviation from identity; colr = tint chromas (splits/bias/toe); hue = hue
    rotations/attractors; sat = chroma gain deviations. Skin residuals tighten
    in proportion so absolute skin drift stays within the QC gates."""
    import copy

    lk = copy.deepcopy(look)
    pts = [(x, x + tone * (y - x)) for x, y in lk["tone"]["points"]]
    # safety: keep the scaled curve inside the domain range and strictly
    # increasing (a deep calibrated toe times a large factor can otherwise go
    # negative). lstar-domain curves live in 0-100, code-domain in 0-1.
    hi = 100.0 if lk["tone"]["domain"] == "lstar" else 1.0
    eps = hi * 1e-4
    lo = 0.0
    safe = []
    for i, (x, y) in enumerate(pts):
        y = min(max(y, 0.0 if i == 0 else lo + eps), hi)
        safe.append((x, y))
        lo = y
    lk["tone"]["points"] = safe
    if "slopes" in lk["tone"]:
        lk["tone"]["slopes"] = {
            i: max(0.3, 1.0 + tone * (s - 1.0)) for i, s in lk["tone"]["slopes"].items()
        }
    if "norm_shoulder" in lk:
        ns = lk["norm_shoulder"]
        ns["end_slope"] = max(0.2, 1.0 + tone * (ns["end_slope"] - 1.0))
    # materialize implicit skin residuals BEFORE scaling so tint ops that
    # relied on the look-level defaults also get tightened
    _tone_res = lk.get("skin_tone_residual", 0.3)
    _hue_res = lk.get("skin_hue_residual", 0.15)
    if "midtone_bias" in lk:
        lk["midtone_bias"]["chroma"] *= colr
        lk["midtone_bias"].setdefault("skin_residual", _tone_res)
        lk["midtone_bias"]["skin_residual"] /= colr
    for sp in lk.get("splits", []):
        sp["c_peak"] *= colr
        sp.setdefault("skin_residual", _tone_res)
        sp["skin_residual"] /= colr
    if "toe_tint" in lk:
        lk["toe_tint"]["chroma"] *= colr
    for op in lk.get("hue_shifts", []):
        op["delta"] *= hue
        op["halfwidth"] = min(op["halfwidth"] * (1.0 + 0.18 * (hue - 1.0)), 60.0)
    for op in lk.get("attractors", []):
        op["strength"] = min(0.55, op["strength"] * hue)
        op.setdefault("skin_residual", _hue_res)
        op["skin_residual"] /= hue
    if "layered_hue" in lk:
        lk["layered_hue"]["delta_lo"] *= hue
        lk["layered_hue"]["delta_hi"] *= hue
        lk["layered_hue"]["halfwidth"] = min(
            lk["layered_hue"]["halfwidth"] * (1.0 + 0.18 * (hue - 1.0)), 55.0
        )
    for op in lk.get("sat_ops", []):
        op["gain"] = max(0.4, 1.0 + sat * (op["gain"] - 1.0))
    if "global_sat" in lk:
        lk["global_sat"] = max(0.5, 1.0 + sat * (lk["global_sat"] - 1.0))
    if "vibrance" in lk:
        lk["vibrance"]["gain"] = 1.0 + sat * (lk["vibrance"]["gain"] - 1.0)
    if "sat_compress_high" in lk:
        sc = lk["sat_compress_high"]
        sc["gain"] = 1.0 + sat * (sc["gain"] - 1.0)
    if "lum_desat_shadow" in lk:
        d = lk["lum_desat_shadow"]
        d["gain_at_0"] = max(0.4, 1.0 + sat * (d["gain_at_0"] - 1.0))
    if "lum_desat_high" in lk:
        d = lk["lum_desat_high"]
        d["gain_end"] = max(0.0, 1.0 + sat * (d["gain_end"] - 1.0))
    for op in lk.get("hue_lum", []):
        op["gain"] = max(0.5, 1.0 + sat * (op["gain"] - 1.0))
    if "micro_density" in lk:
        lk["micro_density"]["strength"] *= sat
    if "skin_lightening" in lk:
        lk["skin_lightening"]["delta_lstar"] *= tone
    if "chroma_knee" in lk and colr > 1.5:
        lk["chroma_knee"]["start"] *= 1.15
        if "cap" in lk["chroma_knee"]:
            lk["chroma_knee"]["cap"] *= 1.15
    if "gray_guard" in lk:
        lk["gray_guard"]["limit"] *= colr
    lk["skin_hue_residual"] = _hue_res / hue
    lk["skin_tone_residual"] = _tone_res / colr
    return lk

# ---------------------------------------------------------------------------
# A. Generalists
# ---------------------------------------------------------------------------

LOOKS["Fieldnote"] = {
    "name": "Fieldnote",
    "description": "Neutral C-41 style daily negative: long toe, slate/straw split, restrained color",
    "tone": {
        "domain": "lstar",
        "points": [(0.0, 3.0), (30.0, 29.72), (46.0, 47.0), (78.0, 81.56), (100.0, 100.0)],
        "slopes": {0: 0.50, 1: 1.08, 2: 1.08, 3: 1.08, 4: 0.55},
    },
    "midtone_bias": {
        "hue": 78.0, "chroma": 0.005,
        "l_center": 55.0, "l_sigma": 16.0, "l_zero_lo": 18.0, "l_zero_hi": 88.0,
    },
    "splits": [
        {"kind": "shadow", "shape": "bump", "hue": 235.0, "c_peak": 0.010,
         "l_peak": 12.0, "halfwidth": 40.0},
        {"kind": "highlight", "shape": "rise", "hue": 90.0, "c_peak": 0.006,
         "l_rise_from": 56.0, "l_peak": 80.0},
    ],
    "hue_shifts": [
        {"center": 25.0, "halfwidth": 30.0, "delta": 3.0},
        {"center": 140.0, "halfwidth": 45.0, "delta": -6.0},
        {"center": 215.0, "halfwidth": 40.0, "delta": 5.0},
        {"center": 300.0, "halfwidth": 35.0, "delta": 4.0},
    ],
    "sat_ops": [
        {"center": 140.0, "halfwidth": 55.0, "gain": 0.92},
        {"center": 230.0, "halfwidth": 50.0, "gain": 0.95},
        {"center": 330.0, "halfwidth": 45.0, "gain": 0.90},
        {"center": 100.0, "halfwidth": 30.0, "gain": 0.96},
    ],
    "global_sat": 0.94,
    "vibrance": {"c_lo": 0.05, "c_hi": 0.09, "gain": 1.06},
    "chroma_knee": {"start": 0.11, "slope": 0.6, "cap": 0.16},
    "lum_desat_shadow": {"l_end": 20.0, "gain_at_0": 0.85},
    "lum_desat_high": {"l_start": 92.0, "l_end": 99.0, "gain_end": 0.0, "c_fade": (0.10, 0.20)},
    "skin_lightening": {"delta_lstar": 1.0, "l_center": 58.0, "l_sigma": 15.0,
                        "l_zero_lo": 35.0, "l_zero_hi": 80.0},
    "skin_hue_residual": 0.15,
    "skin_tone_residual": 0.30,
    "skin_sat_clamp": (0.97, 1.02),
    "gray_guard": {"c_in": 0.01, "limit": 0.012},
    "opacity_rec": "85-100%; night markets / infant close-ups 70-80%",
    "grain_rec": "OFF (backup: finest ~10/100, flat overcast only)",
}

LOOKS["Heartland"] = {
    "name": "Heartland",
    "description": "Warm people-first daily negative: amber midtones, slate shadows, creamy highlights",
    "tone": {
        "domain": "code",
        "points": [(0.0, 0.015), (0.10, 0.108), (0.466, 0.4858), (0.80, 0.822), (1.0, 1.0)],
        "slopes": {2: 1.08, 4: 0.60},
    },
    "midtone_bias": {
        "hue": 70.0, "chroma": 0.007,
        "l_center": 55.0, "l_sigma": 16.0, "l_zero_lo": 15.0, "l_zero_hi": 90.0,
    },
    "splits": [
        {"kind": "shadow", "shape": "low", "hue": 220.0, "c_peak": 0.008,
         "l_peak": 15.0, "l_zero": 55.0},
        {"kind": "highlight", "shape": "rise", "hue": 82.0, "c_peak": 0.007,
         "l_rise_from": 55.0, "l_peak": 80.0},
    ],
    "white_guard": (90.0, 97.0),
    "hue_shifts": [
        {"center": 140.0, "halfwidth": 40.0, "delta": -10.0},
        {"center": 250.0, "halfwidth": 35.0, "delta": -6.0},
        {"center": 20.0, "halfwidth": 15.0, "delta": 4.0},
        # declared: skin nudged 2 deg toward peach for healthy color
        {"center": 50.0, "halfwidth": 20.0, "delta": -2.0, "skin_exempt": True},
    ],
    "sat_ops": [
        {"center": 140.0, "halfwidth": 45.0, "gain": 0.90},
        {"center": 250.0, "halfwidth": 40.0, "gain": 0.94},
        {"center": 320.0, "halfwidth": 30.0, "gain": 0.92},
        {"center": 95.0, "halfwidth": 20.0, "gain": 0.95},
    ],
    "global_sat": 0.96,
    "vibrance": {"c_lo": 0.06, "c_hi": 0.10, "gain": 1.04},
    "chroma_knee": {"start": 0.12, "slope": 0.8},
    "lum_desat_high": {"l_start": 85.0, "l_end": 98.0, "gain_end": 0.80},
    "skin_hue_residual": 0.15,
    "skin_tone_residual": 0.40,
    "skin_sat_clamp": (0.97, 1.03),
    "gray_guard": {"c_in": 0.01, "limit": 0.012},
    "opacity_rec": "75-85%",
    "grain_rec": "OFF (backup: ~5/100, dim indoor portraits only)",
}

LOOKS["Meridian"] = {
    "name": "Meridian",
    "description": "Honest bright daylight: measured-neutral gray axis, open shadows, airy whites",
    "tone": {
        "domain": "code",
        "points": [(0.0, 0.0), (0.02, 0.025), (0.10, 0.112), (0.466, 0.4956),
                   (0.83, 0.848), (1.0, 1.0)],
        "slopes": {0: 1.3, 3: 1.05, 5: 0.75},
    },
    "hue_c_gate": (0.03, 0.06),
    "hue_shifts": [
        {"center": 200.0, "halfwidth": 25.0, "delta": 6.0, "gamut_fade": (0.55, 0.78)},
        {"center": 255.0, "halfwidth": 30.0, "delta": 3.0, "gamut_fade": (0.55, 0.78)},
        {"center": 145.0, "halfwidth": 40.0, "delta": -6.0},
        {"center": 25.0, "halfwidth": 15.0, "delta": 3.0},
    ],
    "sat_ops": [
        {"center": 250.0, "halfwidth": 30.0, "gain": 1.05},
        {"center": 145.0, "halfwidth": 40.0, "gain": 0.95},
        {"center": 100.0, "halfwidth": 20.0, "gain": 0.95},
    ],
    "vibrance": {"c_lo": 0.06, "c_hi": 0.12, "gain": 1.06},
    "sat_compress_high": {"c_start": 0.20, "gain": 0.94},
    "lum_desat_shadow": {"l_end": 15.0, "gain_at_0": 0.90},
    "lum_desat_high": {"l_start": 88.0, "l_end": 98.0, "gain_end": 0.85},
    "skin_lightening": {"delta_lstar": 1.5, "l_center": 57.0, "l_sigma": 12.0,
                        "l_zero_lo": 40.0, "l_zero_hi": 75.0},
    "skin_hue_residual": 0.20,
    "skin_sat_clamp": (0.97, 1.03),
    "gray_guard": {"c_in": 0.015, "limit": 0.0},
    "opacity_rec": "100% daily; 70% ~ brighter cleaner Standard",
    "grain_rec": "OFF",
}

LOOKS["Matinee"] = {
    "name": "Matinee",
    "description": "All-day cinema print: deep structured blacks, zero-cast midband, slate/ivory whisper",
    # Tone: the recipe's own 10-point calibration table IS the curve (nodes exact).
    "tone": {
        "domain": "code",
        "points": [(0.0, 0.0), (0.05, 0.0248), (0.10, 0.0683), (0.20, 0.1684),
                   (0.30, 0.2750), (0.40, 0.3841), (0.466, 0.4561), (0.60, 0.6002),
                   (0.75, 0.7559), (0.90, 0.9164), (1.0, 1.0)],
        "slopes": {10: 0.61},
    },
    "splits": [
        {"kind": "shadow", "shape": "bump", "hue": 258.0, "c_peak": 0.009,
         "l_peak": 18.0, "halfwidth": 20.0, "black_guard": 6.0, "skin_residual": 0.35},
        {"kind": "highlight", "shape": "bump", "hue": 78.0, "c_peak": 0.006,
         "l_peak": 78.0, "halfwidth": 22.0, "skin_residual": 0.65},
    ],
    "white_guard": (90.0, 96.0),
    "hue_shifts": [
        {"center": 140.0, "halfwidth": 35.0, "delta": -8.0},
        {"center": 255.0, "halfwidth": 25.0, "delta": 4.0},
        {"center": 25.0, "halfwidth": 15.0, "delta": 3.0},
    ],
    "sat_ops": [
        {"center": 210.0, "halfwidth": 30.0, "gain": 0.90},
        {"center": 140.0, "halfwidth": 40.0, "gain": 0.90},
        {"center": 260.0, "halfwidth": 25.0, "gain": 0.96},
        {"center": 330.0, "halfwidth": 30.0, "gain": 0.94},
    ],
    "global_sat": 0.97,
    "lum_desat_shadow": {"l_end": 30.0, "gain_at_0": 0.90},
    "lum_desat_high": {"l_start": 85.0, "l_end": 100.0, "gain_end": 0.94},
    "chroma_knee": {"start": 0.11, "slope": 0.65},
    "skin_hue_residual": 0.0,
    "skin_sat_clamp": (0.982, 1.02),
    "gray_guard": {"c_in": 0.01, "limit": 0.010},
    "opacity_rec": "100% ok all day; daily sweet spot 80%; below 60% not recommended",
    "grain_rec": "OFF (backup: weakest level, night/overcast only)",
}

LOOKS["Postcard"] = {
    "name": "Postcard",
    "description": "Reversal-film travel look: clean whites, hue-purified confident color, micro-density",
    "tone": {
        "domain": "lstar",
        "points": [(0.0, 0.0), (12.0, 9.5), (25.0, 22.0), (50.0, 49.5),
                   (70.0, 72.5), (85.0, 86.5), (100.0, 100.0)],
    },
    "splits": [
        {"kind": "shadow", "shape": "low", "hue": 250.0, "c_peak": 0.008,
         "l_peak": 12.0, "l_zero": 42.0},
        {"kind": "highlight", "shape": "rise", "hue": 85.0, "c_peak": 0.004,
         "l_rise_from": 56.0, "l_peak": 78.0},
    ],
    "white_guard": (90.0, 96.0),
    "hue_shifts": [
        {"center": 222.0, "halfwidth": 50.0, "delta": 9.0,
         "l_gate": {"lo": 28.0, "hi": 40.0, "floor": 0.25}},
        {"center": 265.0, "halfwidth": 30.0, "delta": -4.0},
        {"center": 120.0, "halfwidth": 44.0, "delta": 7.0},
        {"center": 155.0, "halfwidth": 34.0, "delta": -6.0},
        {"center": 18.0, "halfwidth": 26.0, "delta": 4.0},
        {"center": 100.0, "halfwidth": 30.0, "delta": -6.0},
    ],
    "sat_ops": [
        {"center": 240.0, "halfwidth": 60.0, "gain": 1.10},
        {"center": 135.0, "halfwidth": 50.0, "gain": 1.06},
        {"center": 20.0, "halfwidth": 28.0, "gain": 1.08},
        {"center": 95.0, "halfwidth": 30.0, "gain": 1.05},
    ],
    "global_sat": 1.0,
    "vibrance": {"c_lo": 0.04, "c_hi": 0.13, "gain": 1.06},
    "chroma_knee": {"start": 0.135, "slope": 0.3333, "cap": 0.19},
    "lum_desat_shadow": {"l_end": 8.0, "gain_at_0": 0.88},
    "lum_desat_high": {"l_start": 88.0, "l_end": 97.0, "gain_end": 0.80},
    "micro_density": {"strength": 0.03, "c_lo": 0.06, "c_hi": 0.16,
                      "l_lo": 30.0, "l_hi": 80.0, "skin_residual": 0.4},
    "skin_hue_residual": 0.25,
    "skin_sat_clamp": (0.98, 1.016),
    "gray_guard": {"c_in": 0.01, "limit": 0.009},
    "opacity_rec": "sweet spot 75%; postcard subjects 100%",
    "grain_rec": "OFF (backup: lowest, only as dither for big-sky banding worries)",
}

# ---------------------------------------------------------------------------
# B. Scene specialists
# ---------------------------------------------------------------------------

LOOKS["NightMarket"] = {
    "name": "NightMarket",
    "description": "Standard-base mixed-light night: slate shadows, gold-amber lamps, LED-safe chroma",
    "norm_shoulder": {"start": 0.70, "end_slope": 0.40},
    "tone": {
        "domain": "code",
        "points": [(0.0, 0.018), (0.10, 0.105), (0.466, 0.4512), (1.0, 1.0)],
        "slopes": {2: 1.10, 3: 1.0},
    },
    "toe_tint": {"hue": 255.0, "chroma": 0.008, "l_end": 14.0},
    "splits": [
        {"kind": "shadow", "shape": "low", "hue": 255.0, "c_peak": 0.012,
         "l_peak": 12.0, "l_zero": 45.0},
        {"kind": "highlight", "shape": "rise", "hue": 65.0, "c_peak": 0.006,
         "l_rise_from": 45.0, "l_peak": 80.0, "skin_residual": 0.2},
    ],
    "attractors": [
        {"target": 48.0, "lo": 20.0, "hi": 75.0, "strength": 0.30, "skin_residual": 0.20},
    ],
    "hue_shifts": [
        {"center": 258.0, "halfwidth": 28.0, "delta": -8.0},
        {"center": 150.0, "halfwidth": 35.0, "delta": 8.0},
        {"center": 320.0, "halfwidth": 25.0, "delta": -5.0},
    ],
    "sat_ops": [
        # blown-orange lampshades: only bright AND rich pixels; faces stay
        {"center": 48.0, "halfwidth": 25.0, "gain": 0.85,
         "l_band": (63.0, 100.0, 8.0, 1.0), "c_gate": (0.13, 0.17)},
        {"center": 150.0, "halfwidth": 35.0, "gain": 0.85},
        {"center": 258.0, "halfwidth": 28.0, "gain": 0.95},
        {"center": 320.0, "halfwidth": 25.0, "gain": 0.90},
    ],
    "chroma_caps": [{"center": 258.0, "halfwidth": 28.0, "cap": 0.24}],
    "global_sat": 0.94,
    "lum_desat_shadow": {"l_end": 12.0, "gain_at_0": 0.85},
    "lum_desat_high": {"l_start": 80.0, "l_end": 95.0, "gain_end": 0.75},
    "chroma_knee": {"start": 0.20, "slope": 0.6, "cap": 0.32},
    "skin_hue_residual": 0.40,
    "skin_sat_clamp": (0.96, 1.02),
    "gray_guard": {"c_in": 0.01, "limit": 0.013},
    "opacity_rec": "75-85%",
    "grain_rec": "OFF (backup: ~8/100, ISO<=3200 only)",
}

LOOKS["DuskTide"] = {
    "name": "DuskTide",
    "description": "Blue hour by the water: indigo shadows, lamp-gold highlights, two-layer blues",
    "tone": {
        "domain": "code",
        "points": [(0.0, 0.012), (0.10, 0.100), (0.466, 0.4464), (0.80, 0.790), (1.0, 1.0)],
        "slopes": {2: 1.10, 4: 0.60},
    },
    "tone_skin_relief": 0.5,
    "toe_tint": {"hue": 265.0, "chroma": 0.006, "l_end": 12.0},
    "midtone_bias": {
        "hue": 275.0, "chroma": 0.006,
        "l_center": 50.0, "l_sigma": 16.0, "l_zero_lo": 12.0, "l_zero_hi": 88.0,
        "skin_residual": 0.0,
    },
    "splits": [
        {"kind": "shadow", "shape": "low", "hue": 262.0, "c_peak": 0.012,
         "l_peak": 12.0, "l_zero": 50.0, "skin_residual": 0.1},
        {"kind": "highlight", "shape": "rise", "hue": 58.0, "c_peak": 0.010,
         "l_rise_from": 50.0, "l_peak": 80.0, "skin_residual": 0.1},
    ],
    "white_guard": (90.0, 97.0),
    "hue_shifts": [
        {"center": 205.0, "halfwidth": 25.0, "delta": 10.0},
        {"center": 250.0, "halfwidth": 30.0, "delta": 6.0},
        {"center": 320.0, "halfwidth": 25.0, "delta": 4.0},
        {"center": 140.0, "halfwidth": 35.0, "delta": -6.0},
    ],
    "attractors": [
        {"target": 50.0, "lo": 25.0, "hi": 70.0, "strength": 0.25, "skin_residual": 0.2},
    ],
    "sat_ops": [
        {"center": 255.0, "halfwidth": 35.0, "gain": 1.05, "l_band": (25.0, 65.0, 10.0, 10.0)},
        {"center": 48.0, "halfwidth": 25.0, "gain": 0.95},
        {"center": 140.0, "halfwidth": 35.0, "gain": 0.88},
    ],
    "global_sat": 0.97,
    "lum_desat_shadow": {"l_end": 12.0, "gain_at_0": 0.92, "exempt_hue": (240.0, 290.0)},
    "lum_desat_high": {"l_start": 85.0, "l_end": 97.0, "gain_end": 0.85},
    "skin_hue_residual": 0.20,
    "skin_sat_clamp": (0.95, 1.02),
    "gray_guard": {"c_in": 0.01, "limit": 0.013},
    "opacity_rec": "70-85% (designed dense; 70% is a finished quiet dusk)",
    "grain_rec": "OFF (backup: ~4/100 fine)",
}

LOOKS["Canopy"] = {
    "name": "Canopy",
    "description": "Tropical greenery daily: luminance-layered greens, deep not neon, skin isolated",
    "tone": {
        "domain": "code",
        "points": [(0.0, 0.012), (0.466, 0.4752), (0.75, 0.772), (1.0, 1.0)],
        "slopes": {1: 1.08, 3: 0.50},
    },
    "toe_tint": {"hue": 165.0, "chroma": 0.005, "l_end": 12.0},
    "midtone_bias": {  # sun-through-leaves amber, highlights only, white-guarded
        "hue": 90.0, "chroma": 0.004,
        "l_center": 80.0, "l_sigma": 8.0, "l_zero_lo": 68.0, "l_zero_hi": 92.0,
    },
    "splits": [
        {"kind": "shadow", "shape": "low", "hue": 172.0, "c_peak": 0.006,
         "l_peak": 10.0, "l_zero": 40.0, "skin_residual": 0.3},
        {"kind": "highlight", "shape": "rise", "hue": 85.0, "c_peak": 0.006,
         "l_rise_from": 52.0, "l_peak": 80.0},
    ],
    "layered_hue": {"center": 128.0, "halfwidth": 40.0, "delta_lo": 12.0,
                    "delta_hi": -8.0, "l_lo": 40.0, "l_hi": 65.0, "hard_cut_below": 90.0,
                    "gamut_fade": (0.65, 0.85)},
    "hue_shifts": [
        {"center": 195.0, "halfwidth": 20.0, "delta": 4.0},
        {"center": 18.0, "halfwidth": 18.0, "delta": 4.0},
    ],
    "hue_lum": [
        {"center": 135.0, "halfwidth": 40.0, "gain": 0.94,
         "l_band": (25.0, 70.0, 10.0, 10.0), "hard_cut_below": 90.0},
    ],
    "sat_ops": [
        {"center": 135.0, "halfwidth": 40.0, "gain": 0.92, "hard_cut_below": 90.0},
        {"center": 105.0, "halfwidth": 18.0, "gain": 0.90},
        {"center": 200.0, "halfwidth": 20.0, "gain": 0.98},
    ],
    "chroma_caps": [{"center": 135.0, "halfwidth": 45.0, "cap": 0.19}],  # anti-neon knee
    "global_sat": 0.98,
    "vibrance": {"c_lo": 0.05, "c_hi": 0.10, "gain": 1.05},
    "chroma_knee": {"start": 0.15, "slope": 0.7},
    "lum_desat_high": {"l_start": 88.0, "l_end": 98.0, "gain_end": 0.88},
    "skin_hue_residual": 0.15,
    "skin_sat_clamp": (0.96, 1.03),
    "gray_guard": {"c_in": 0.01, "limit": 0.008},
    "opacity_rec": "75-90%",
    "grain_rec": "OFF (backup: ~4/100 fine)",
}

# ---------------------------------------------------------------------------
# Looks designed directly at full strength (no strength factors), from first
# principles + the direction visible in the author's own hand-edited finals
# (bold luminosity restructuring, confident clean color, no vintage cast).
# ---------------------------------------------------------------------------

LOOKS["Skylight"] = {
    "name": "Skylight",
    # The user's own editing move, made mountable: open shadows + compressed
    # highlights (ratio restructuring) + clean confident color. No tints.
    "description": "Editor's daily: opened shadows, compressed highlights, vivid clean color, zero cast",
    "tone": {
        "domain": "code",
        "points": [(0.0, 0.02), (0.10, 0.145), (0.25, 0.302), (0.466, 0.500),
                   (0.75, 0.775), (1.0, 1.0)],
        "slopes": {3: 1.05, 5: 0.60},
    },
    "hue_shifts": [
        {"center": 205.0, "halfwidth": 28.0, "delta": 6.0, "gamut_fade": (0.60, 0.82)},
        {"center": 140.0, "halfwidth": 40.0, "delta": -4.0},
    ],
    "sat_ops": [
        {"center": 240.0, "halfwidth": 35.0, "gain": 1.10},
        {"center": 135.0, "halfwidth": 40.0, "gain": 1.06},
        {"center": 20.0, "halfwidth": 25.0, "gain": 1.05},
    ],
    "global_sat": 1.02,
    "vibrance": {"c_lo": 0.06, "c_hi": 0.14, "gain": 1.15},
    "chroma_knee": {"start": 0.16, "slope": 0.6, "cap": 0.24},
    "lum_desat_shadow": {"l_end": 10.0, "gain_at_0": 0.92},
    "lum_desat_high": {"l_start": 90.0, "l_end": 98.0, "gain_end": 0.85, "c_fade": (0.10, 0.20)},
    "skin_lightening": {"delta_lstar": 1.5, "l_center": 58.0, "l_sigma": 14.0,
                        "l_zero_lo": 35.0, "l_zero_hi": 80.0},
    "skin_hue_residual": 0.15,
    "skin_sat_clamp": (0.98, 1.05),
    "gray_guard": {"c_in": 0.015, "limit": 0.0},
    "opacity_rec": "75-100%; harsh-light rescue at 100%",
    "grain_rec": "OFF",
}

LOOKS["Lowsun"] = {
    "name": "Lowsun",
    # Golden-hour companion to DuskTide: gold-amber midtones and highlights
    # against cool blue shadows (the low-sun light structure itself), early
    # shoulder so sunset highlights hold.
    "description": "Golden hour: gold midtones/highlights vs cool shadows, early shoulder, warm skin",
    "tone": {
        "domain": "code",
        "points": [(0.0, 0.012), (0.10, 0.096), (0.466, 0.4562), (0.72, 0.718), (1.0, 1.0)],
        "slopes": {2: 1.12, 4: 0.55},
    },
    "midtone_bias": {
        "hue": 60.0, "chroma": 0.014,
        "l_center": 55.0, "l_sigma": 18.0, "l_zero_lo": 12.0, "l_zero_hi": 90.0,
    },
    "splits": [
        {"kind": "shadow", "shape": "low", "hue": 250.0, "c_peak": 0.014,
         "l_peak": 12.0, "l_zero": 48.0},
        {"kind": "highlight", "shape": "rise", "hue": 68.0, "c_peak": 0.016,
         "l_rise_from": 50.0, "l_peak": 82.0},
    ],
    "white_guard": (92.0, 98.0),
    "hue_shifts": [
        {"center": 40.0, "halfwidth": 25.0, "delta": 4.0},
        {"center": 15.0, "halfwidth": 18.0, "delta": 5.0},
        {"center": 250.0, "halfwidth": 30.0, "delta": -6.0, "gamut_fade": (0.60, 0.82)},
        {"center": 130.0, "halfwidth": 40.0, "delta": -10.0},
        # declared: skin rides the golden light, 2 deg toward gold
        {"center": 50.0, "halfwidth": 20.0, "delta": 2.0, "skin_exempt": True},
    ],
    "sat_ops": [
        {"center": 250.0, "halfwidth": 30.0, "gain": 0.92},
        {"center": 130.0, "halfwidth": 40.0, "gain": 0.88},
        {"center": 48.0, "halfwidth": 30.0, "gain": 1.08, "l_band": (40.0, 85.0, 10.0, 8.0)},
        {"center": 330.0, "halfwidth": 25.0, "gain": 0.95},
    ],
    "lum_desat_high": {"l_start": 85.0, "l_end": 97.0, "gain_end": 0.85},
    "chroma_knee": {"start": 0.17, "slope": 0.6, "cap": 0.26},
    "skin_hue_residual": 0.15,
    "skin_tone_residual": 0.30,
    "skin_sat_clamp": (0.97, 1.06),
    "gray_guard": {"c_in": 0.01, "limit": 0.018},
    "opacity_rec": "70-90%",
    "grain_rec": "OFF (backup: ~5/100)",
}

# Per-look strength factors applied to the design recipes above. Tuned
# against the strength band of real film looks (commonly mounted LUTs sit at
# identity-distance dE00 ~9-11) and the boldness of the author's own
# hand-edited finals.
BOOST = {
    "Fieldnote": dict(tone=2.0, colr=2.2, hue=1.8, sat=1.9),
    "Heartland": dict(tone=2.0, colr=2.6, hue=1.8, sat=1.9),
    "Meridian": dict(tone=1.8, colr=1.5, hue=1.6, sat=1.8),
    "Matinee": dict(tone=1.55, colr=3.0, hue=1.9, sat=2.3),
    "Postcard": dict(tone=1.9, colr=2.4, hue=1.6, sat=2.0),
    "NightMarket": dict(tone=2.0, colr=2.8, hue=1.9, sat=1.9),
    "DuskTide": dict(tone=2.0, colr=2.4, hue=1.8, sat=1.9),
    "Canopy": dict(tone=1.9, colr=2.8, hue=2.2, sat=2.4),
}
for _name, _k in BOOST.items():
    LOOKS[_name] = _boost(LOOKS[_name], **_k)

# Suite midgray ladder at input sRGB 0.466, in delta-L* vs input (regression
# targets; recipe design values scaled by each look's tone factor)
_LADDER_BASE = {
    "Meridian": 3.0, "Heartland": 2.0, "Fieldnote": 1.3, "Canopy": 1.0,
    "Postcard": -0.5, "Matinee": -1.0, "NightMarket": -1.5, "DuskTide": -2.0,
}
MIDGRAY_LADDER = {n: v * BOOST[n]["tone"] for n, v in _LADDER_BASE.items()}
MIDGRAY_LADDER["Skylight"] = 3.4   # designed at full strength, no factor
MIDGRAY_LADDER["Lowsun"] = -1.0    # designed at full strength, no factor

GENERALISTS = ["Fieldnote", "Heartland", "Meridian", "Matinee", "Postcard", "Skylight"]
SPECIALISTS = ["NightMarket", "DuskTide", "Canopy", "Lowsun"]


if __name__ == "__main__":
    import numpy as np

    from . import color
    from .pipeline import apply_look

    t = np.linspace(0.0, 1.0, 257)
    gray = np.stack([t, t, t], axis=-1)
    ref = np.array([[0.466, 0.466, 0.466]])
    ref_lstar = float(color.lstar_from_oklabL(color.oklabL_from_code(np.float64(0.466))))
    print(f"{'look':<12} {'monoL':>5} {'chDip':>8} {'white':>9} {'black':>7} {'dL* mid':>8} {'target':>7}")
    ok = True
    for name, look in LOOKS.items():
        out = apply_look(gray, look)
        # Lightness monotonicity is judged in OKLab L (declared gray tints may
        # legitimately ripple individual channels by a hair).
        L_out = color.linear_srgb_to_oklab(color.srgb_decode(out))[:, 0]
        mono = bool(np.all(np.diff(L_out) > -1e-9))
        ch_dip = float(np.min(np.diff(out, axis=0)))  # worst per-channel decrease
        white = float(np.max(np.abs(out[-1] - 1.0)))
        black = float(out[0].mean())
        mid = apply_look(ref, look)[0]
        L_mid = color.linear_srgb_to_oklab(color.srgb_decode(mid[None, :]))[0, 0]
        d_mid = float(color.lstar_from_oklabL(L_mid)) - ref_lstar
        target = MIDGRAY_LADDER[name]
        flag = "" if (mono and white < 1e-9 and abs(d_mid - target) < 0.25 and ch_dip > -2e-3) else "  <-- CHECK"
        if flag:
            ok = False
        print(f"{name:<12} {str(mono):>5} {ch_dip:8.1e} {white:9.2e} {black:7.4f} {d_mid:8.2f} {target:7.1f}{flag}")
    print("params.py quick-check", "OK" if ok else "FAILED")
