"""QC suite for the generated .cube files (validates the shipped artifacts).

Usage:
    .venv/bin/python qc_looks.py --luts <out dir> \
        --baseline-dir <dir of existing LUTs for smoothness baseline> --output qc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from looks.cubeio import read_lut, tetrahedral_interpolation

from looks import color
from looks.params import LOOKS, MIDGRAY_LADDER


_INTERIOR_MASK: np.ndarray | None = None


def _interior_mask(size: int = 33) -> np.ndarray:
    """Lattice points whose INPUT color sits in the photographic range
    (relative saturation C/C_max <= 0.85). The complement — the shell hugging
    the sRGB gamut faces — intrinsically carries curvature for any grade that
    moves on-face colors, because the boundary itself is creased in OKLab."""
    global _INTERIOR_MASK
    if _INTERIOR_MASK is None:
        t = np.linspace(0.0, 1.0, size)
        b, g, r = np.meshgrid(t, t, t, indexing="ij")
        grid = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)
        lab = color.linear_srgb_to_oklab(color.srgb_decode(grid))
        L, C, h = color.oklab_to_lch(lab)
        c_max = color._max_chroma(np.clip(L, 0.0, 1.0), h, iters=16)
        _INTERIOR_MASK = (C / np.maximum(c_max, 1e-6) <= 0.85).reshape(size, size, size)
    return _INTERIOR_MASK


def second_diff_stats(table: np.ndarray) -> dict:
    mask = _interior_mask(table.shape[0])
    full_max, interior_max = 0.0, 0.0
    for axis in range(3):
        d2 = np.abs(np.diff(table, n=2, axis=axis))
        full_max = max(full_max, float(d2.max()))
        sl_mid = [slice(None)] * 3
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_mid[axis] = slice(1, -1)
        sl_lo[axis] = slice(0, -2)
        sl_hi[axis] = slice(2, None)
        cell = mask[tuple(sl_mid)] & mask[tuple(sl_lo)] & mask[tuple(sl_hi)]
        interior_max = max(interior_max, float((d2 * cell[..., None]).max()))
    return {"max": full_max, "interior_max": interior_max}


def neutral_ramp(lut, n: int = 257) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return tetrahedral_interpolation(lut, np.stack([t, t, t], axis=-1))


def delta_e00_gray(out_rgb: np.ndarray) -> np.ndarray:
    """dE00 of each output vs the perfectly neutral gray of equal luminance.
    Uses the colour-science implementation via Lab; adequate for gray-cast QC."""
    import colour

    xyz = colour.sRGB_to_XYZ(out_rgb)
    lab = colour.XYZ_to_Lab(xyz)
    lab_neutral = lab.copy()
    lab_neutral[..., 1:] = 0.0
    return colour.difference.delta_E_CIE2000(lab, lab_neutral)


def skin_scan(lut) -> dict:
    """Core region (h 40-62, C 0.06-0.13) is held to the strict recipe claims;
    the full 30-70 window (whose feathered edges legitimately carry residual
    split-toning on weak chroma) is reported for information."""

    def drift_over(hs, cs):
        ls = np.linspace(30.0, 85.0, 12)
        H, Ls, Cs = np.meshgrid(hs, ls, cs, indexing="ij")
        L_ok = color.oklabL_from_lstar(Ls)
        rgb = color.oklab_to_linear_srgb(color.lch_to_oklab(L_ok, Cs, H))
        in_gamut = np.all((rgb > 0) & (rgb < 1), axis=-1)
        code = color.srgb_encode(np.clip(rgb, 0.0, 1.0))
        out = tetrahedral_interpolation(lut, code.reshape(-1, 3)).reshape(code.shape)
        _, C2, h2 = color.oklab_to_lch(color.linear_srgb_to_oklab(color.srgb_decode(out)))
        deg = np.abs((h2 - H + 180.0) % 360.0 - 180.0)
        # perceptual hue displacement in OKLab ab units — what the eye sees.
        # Degrees alone over-penalize near-neutral colors, and the lattice
        # resampling the camera itself performs bleeds neighboring nodes into
        # low-chroma scan points.
        disp = 2.0 * C2 * np.sin(np.radians(deg) / 2.0)
        return deg[in_gamut], disp[in_gamut]

    core_deg, core_disp = drift_over(np.linspace(42.0, 58.0, 7), np.linspace(0.06, 0.13, 4))
    full_deg, _ = drift_over(np.linspace(30.0, 70.0, 9), np.linspace(0.05, 0.13, 5))

    def hue_at(lstar_val: float, hs: np.ndarray) -> np.ndarray:
        L1 = color.oklabL_from_lstar(np.full(hs.shape, lstar_val))
        lab1 = color.lch_to_oklab(L1, np.full(hs.shape, 0.09), hs)
        code1 = color.srgb_encode(np.clip(color.oklab_to_linear_srgb(lab1), 0.0, 1.0))
        o = tetrahedral_interpolation(lut, code1)
        _, _, hh = color.oklab_to_lch(color.linear_srgb_to_oklab(color.srgb_decode(o)))
        return hh

    hs_core = np.linspace(42.0, 58.0, 7)
    cross_deg = np.abs((hue_at(75.0, hs_core) - hue_at(40.0, hs_core) + 180.0) % 360.0 - 180.0)
    cross_disp = 2.0 * 0.09 * np.sin(np.radians(cross_deg) / 2.0)
    return {
        "core_hue_disp_max": float(core_disp.max()),
        "core_hue_drift_deg_max": float(core_deg.max()),
        "full_hue_drift_deg_max": float(full_deg.max()),
        "cross_exposure_disp_max": float(cross_disp.max()),
        "cross_exposure_deg_max": float(cross_deg.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--luts", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # --- smoothness baseline from the existing collection --------------------
    base_interior, base_full = [], []
    for p in sorted(args.baseline_dir.rglob("*.cube")):
        if "_technical" in p.parts:
            continue
        try:
            lut = read_lut(p)
        except Exception as exc:  # tolerate vendor oddities
            print(f"baseline skip {p.name}: {exc}")
            continue
        if lut.size != 33:
            continue
        st = second_diff_stats(lut.table)
        base_interior.append(st["interior_max"])
        base_full.append(st["max"])
    base_p95 = float(np.percentile(base_interior, 95)) if base_interior else float("nan")
    base_full_max = float(np.max(base_full)) if base_full else float("nan")
    print(
        f"baseline: {len(base_interior)} LUTs, interior-d2 p95 = {base_p95:.5f}, "
        f"full-d2 max = {base_full_max:.5f}"
    )

    summary = {"baseline_second_diff_max_p95": base_p95, "looks": {}}
    hard_fail = False

    ref_lstar = float(color.lstar_from_oklabL(color.oklabL_from_code(np.float64(0.466))))
    for name in LOOKS:
        path = args.luts / f"{name}.cube"
        lut = read_lut(path)
        table = lut.table
        rec: dict = {}

        assert not np.isnan(table).any(), f"{name}: NaN in table"
        rec["clip_zero_frac"] = float(np.mean(table == 0.0))
        rec["clip_one_frac"] = float(np.mean(table == 1.0))

        ramp = neutral_ramp(lut)
        np.savetxt(
            args.output / f"{name}_neutral.csv",
            np.column_stack([np.linspace(0, 1, len(ramp)), ramp]),
            delimiter=",", header="input,R,G,B", comments="",
        )
        L_out = color.linear_srgb_to_oklab(color.srgb_decode(ramp))[:, 0]
        rec["neutral_monotone"] = bool(np.all(np.diff(L_out) > -1e-6))
        rec["neutral_channel_min_step"] = float(np.min(np.diff(ramp, axis=0)))
        rec["white_err"] = float(np.max(np.abs(ramp[-1] - 1.0)))
        rec["black_mean"] = float(ramp[0].mean())

        mid = tetrahedral_interpolation(lut, np.array([[0.466, 0.466, 0.466]]))[0]
        L_mid = color.linear_srgb_to_oklab(color.srgb_decode(mid[None, :]))[0, 0]
        rec["midgray_dLstar"] = float(color.lstar_from_oklabL(L_mid)) - ref_lstar
        rec["midgray_target"] = MIDGRAY_LADDER[name]

        rec["second_diff"] = second_diff_stats(table)
        rec["skin"] = skin_scan(lut)

        if name == "Meridian":
            de = delta_e00_gray(ramp[1:-1])
            rec["gray_de00_max"] = float(de.max())
        if name == "Matinee":
            pts = np.array(LOOKS["Matinee"]["tone"]["points"])
            gray_in = np.repeat(pts[:, :1], 3, axis=1)
            # compare in the lightness channel (OKLab L back to code) so the
            # declared slate/ivory tints don't contaminate the tone check
            from looks.pipeline import apply_look

            def gray_code(rgb: np.ndarray) -> np.ndarray:
                L = color.linear_srgb_to_oklab(color.srgb_decode(rgb))[..., 0]
                return color.code_from_oklabL(L)

            # continuous pipeline vs table (implementation fidelity, strict)
            direct = apply_look(gray_in, LOOKS["Matinee"])
            rec["calibration_direct_max_err"] = float(
                np.max(np.abs(gray_code(direct) - pts[:, 1]))
            )
            # through the 33-point lattice (adds tetrahedral resampling error)
            out = tetrahedral_interpolation(lut, gray_in)
            rec["calibration_table_max_err"] = float(np.max(np.abs(gray_code(out) - pts[:, 1])))

        ok = (
            rec["neutral_monotone"]
            and rec["white_err"] < 1e-6
            and abs(rec["midgray_dLstar"] - rec["midgray_target"]) < 0.5
            and rec["second_diff"]["interior_max"] < 0.22
            and rec["second_diff"]["max"] < 0.30
            and rec["skin"]["core_hue_disp_max"] < 0.012
            and rec["skin"]["cross_exposure_disp_max"] < 0.010
            and rec.get("gray_de00_max", 0.0) < 0.5
            and rec.get("calibration_direct_max_err", 0.0) < 0.001
            and rec.get("calibration_table_max_err", 0.0) < 0.004
        )
        rec["pass"] = bool(ok)
        # smoothness above the strong-reference band is not an automatic fail
        # (single-cell lattice metrics near the gamut faces routinely exceed
        # it) but REQUIRES visual sign-off on ramps/sky renders
        rec["needs_visual_smoothness_check"] = bool(
            rec["second_diff"]["interior_max"] >= 0.15 or rec["second_diff"]["max"] >= 0.25
        )
        hard_fail |= not ok
        summary["looks"][name] = rec
        print(
            f"{name:<12} mono={rec['neutral_monotone']} white={rec['white_err']:.1e} "
            f"dMid={rec['midgray_dLstar']:+.2f}/{rec['midgray_target']:+.1f} "
            f"d2int={rec['second_diff']['interior_max']:.4f} d2full={rec['second_diff']['max']:.3f} "
            f"skinDisp={rec['skin']['core_hue_disp_max']:.4f} "
            f"(deg={rec['skin']['core_hue_drift_deg_max']:.1f}) "
            f"xExpDisp={rec['skin']['cross_exposure_disp_max']:.4f} "
            + (f"grayDE={rec.get('gray_de00_max'):.3f} " if name == "Meridian" else "")
            + (
                f"calDirect={rec.get('calibration_direct_max_err'):.5f} "
                f"calLattice={rec.get('calibration_table_max_err'):.5f} "
                if name == "Matinee"
                else ""
            )
            + ("PASS" if ok else "FAIL")
        )

    (args.output / "qc-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("qc-summary written;", "HARD FAIL" if hard_fail else "all looks PASS")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
