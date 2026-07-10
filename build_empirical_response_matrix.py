#!/usr/bin/env python3
"""
build_empirical_response_matrix.py
====================================
Fits a data-driven, ENERGY-DEPENDENT longitudinal response kernel per
calorimeter module against the Sim-Truth (DoseActor) vs raw ΔT-reconstructed
profile pairs from the sweep.

CHANGE FROM PREVIOUS VERSION
-----------------------------
Previously this script fit one static kernel pooled across all energies and
saved it as a fixed <module>_response_matrix.npy, which the consumer
(unfold_profile_analysis.py) loaded unconditionally -- so every energy point
was deconvolved with the same kernel width regardless of its actual timing
resolution.

Now the kernel width has an explicit energy term:

    sigma(j, E) = sigma0 + sigma_slope*(j - n_true/2) + sigma_E_slope*log(E/E_REF)

fit jointly across all energy points (each pair still gets its OWN R built at
its own energy during the fit -- only the underlying parameters are shared).
The output is the fitted PARAMETER VECTOR (saved as JSON), not a frozen
matrix. The consumer builds R on the fly per energy point via
response_kernel.build_kernel_matrix(..., energy_val=...).

A reference matrix at E_REF_GEV is still saved as .npy purely for quick
visual inspection -- it is NOT what gets used in the actual unfolding.

OUTPUT
------
response_matrices/<module>_response_params.json      -- fitted [sigma0, sigma_slope,
                                                          skew, tail_frac, tail_mult,
                                                          sigma_E_slope] + provenance
response_matrices/<module>_response_matrix_Eref.npy  -- reference matrix at E_REF_GEV (diagnostic only)
response_matrices/<module>_response_fit_quality.png  -- truth vs observed vs R@truth per energy
"""

import sys
import json
import warnings
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import response_kernel as rk

# Reuse geometry/extraction/truth-loading code so this matrix is built from
# EXACTLY the same pipeline that will eventually consume it.
import rl_deconv_profile as base

try:
    import analysis_utils as utils
except ImportError:
    utils = None
    print("[Fatal] analysis_utils not found -- truth curves are required to fit R.")
    sys.exit(1)

PAD_LAYERS = 5
MIN_COINCIDENCES = 30       # skip energy points with too few coincidences to trust
MIN_ENERGY_POINTS = 3       # need >=3 distinct energies to constrain an energy SLOPE
                             # (2 points fits *a* line trivially with zero residual --
                             # not a real constraint)


# ─────────────────────────────────────────────────────────────────────────────
# Data collection -- now also records energy_val per pair
# ─────────────────────────────────────────────────────────────────────────────
def collect_truth_observed_pairs(mod, target_sweep, is_hex):
    energy_dirs = sorted(
        [d for d in target_sweep.iterdir() if d.is_dir() and "GeV" in d.name],
        key=lambda p: base.extract_numerical_energy(p.name),
    )

    pairs = []  # (truth_norm, raw_norm, label, n_coincidences, energy_val)
    lyso_bounds_ref = None

    for edir in energy_dirs:
        energy_val = base.extract_numerical_energy(edir.name)
        res = base.extract_profile_data_unfold(edir, is_hex, mod)
        if res is None or len(res["raw_z_emits"]) < MIN_COINCIDENCES:
            print(f"    [skip] {edir.name}: insufficient coincidences "
                  f"({0 if res is None else len(res['raw_z_emits'])} < {MIN_COINCIDENCES})")
            continue

        lyso_bounds = res["lyso_bounds"]
        lyso_bounds_ref = lyso_bounds
        calor_thick = res["calor_thick"]
        edges = np.array([b[0] for b in lyso_bounds] + [lyso_bounds[-1][1]])

        raw_hist, _ = np.histogram(res["raw_z_emits"], bins=edges)
        raw_hist = raw_hist.astype(float)
        if raw_hist.sum() <= 0:
            print(f"    [skip] {edir.name}: empty raw histogram")
            continue
        raw_norm = raw_hist / raw_hist.sum()

        run_dirs = sorted(list(set(fp.parent for fp in edir.rglob("detector_hits_*.root"))))
        if not run_dirs:
            print(f"    [skip] {edir.name}: no run dirs for truth lookup")
            continue
        try:
            long_arr, _ = utils.load_calorimeter_mhd(
                run_dirs, long_glob="run_Dose_edep.mhd", trans_glob="transverse_shower_max_edep.mhd"
            )
        except Exception as e:
            print(f"    [skip] {edir.name}: truth load failed ({e})")
            continue
        if long_arr is None:
            print(f"    [skip] {edir.name}: truth array is None")
            continue

        dz_mm, avg = 0.1, long_arr / max(len(run_dirs), 1)
        layer_edeps = []
        for (z_start, z_end) in lyso_bounds:
            z0 = z_start - (-calor_thick / 2)
            z1 = z_end - (-calor_thick / 2)
            i0 = max(0, min(int(round(z0 / dz_mm)), len(avg)))
            i1 = max(0, min(int(round(z1 / dz_mm)), len(avg)))
            layer_edeps.append(float(np.sum(avg[i0:i1])))
        truth_curve = np.array(layer_edeps)
        if truth_curve.sum() <= 0:
            print(f"    [skip] {edir.name}: empty truth curve")
            continue
        truth_norm = truth_curve / truth_curve.sum()

        pairs.append((truth_norm, raw_norm, edir.name, res["n_e_coincidences"], energy_val))
        print(f"    [ok]   {edir.name}: E={energy_val:.0f}GeV N={res['n_e_coincidences']} "
              f"coincidences, truth+raw loaded")

    return pairs, lyso_bounds_ref


# ─────────────────────────────────────────────────────────────────────────────
# Fit -- each pair now gets ITS OWN R, built at its own energy, from the
# SAME shared parameter vector. This is what actually constrains sigma_E_slope.
# ─────────────────────────────────────────────────────────────────────────────
def fit_response_matrix(pairs, n_reco, pad_layers):
    energies = sorted(set(p[4] for p in pairs))
    if len(energies) < MIN_ENERGY_POINTS:
        print(f"    [warn] only {len(energies)} distinct energy points "
              f"({energies}) -- sigma_E_slope will be poorly constrained. "
              f"Consider fixing sigma_E_slope=0 instead of trusting this fit.")

    def residuals(params):
        res = []
        for truth_norm, raw_norm, label, n_coinc, energy_val in pairs:
            R = rk.build_kernel_matrix(n_reco, pad_layers, params, energy_val=energy_val)
            x_ext = np.zeros(n_reco + 2 * pad_layers)
            x_ext[pad_layers: pad_layers + n_reco] = truth_norm
            pred = R @ x_ext
            pred_norm = pred / pred.sum() if pred.sum() > 0 else pred
            # sqrt(N) weighting: higher-stat energy points count more, but
            # don't let one high-N point completely dominate the shape fit
            weight = np.sqrt(max(n_coinc, 1))
            res.append(weight * (pred_norm - raw_norm))
        return np.concatenate(res)

    # params: sigma0, sigma_slope, skew, tail_frac, tail_width_mult, sigma_E_slope
    p0        = [1.5,  0.0,   0.0, 0.10, 3.0, 0.0]
    bounds_lo = [0.2, -0.15, -8.0, 0.00, 1.5, -3.0]
    bounds_hi = [6.0,  0.15,  8.0, 0.50, 8.0,  3.0]

    return least_squares(residuals, p0, bounds=(bounds_lo, bounds_hi), verbose=0)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plot -- builds R per-pair at that pair's own energy, same as fit
# ─────────────────────────────────────────────────────────────────────────────
def plot_fit_quality(mod, pairs, params, n_reco, pad_layers, out_path):
    import matplotlib.pyplot as plt
    n = len(pairs)
    ncols = 2 if n >= 2 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), squeeze=False)
    layers = np.arange(1, n_reco + 1)

    for i, (truth_norm, raw_norm, label, n_coinc, energy_val) in enumerate(pairs):
        ax = axes[i // ncols][i % ncols]
        R = rk.build_kernel_matrix(n_reco, pad_layers, params, energy_val=energy_val)
        x_ext = np.zeros(n_reco + 2 * pad_layers)
        x_ext[pad_layers: pad_layers + n_reco] = truth_norm
        pred = R @ x_ext
        pred_norm = pred / pred.sum() if pred.sum() > 0 else pred

        ax.bar(layers, truth_norm, color="#00bcd4", alpha=0.25, edgecolor="#00838f", label="Truth (DoseActor)")
        ax.plot(layers, raw_norm, "k.:", label="Observed raw ΔT", alpha=0.7)
        ax.plot(layers, pred_norm, "r-o", markersize=3, linewidth=1.5, label="R @ Truth (fitted)")
        sigma_disp = rk.sigma_of_layer(n_reco / 2.0 + pad_layers, n_reco + 2 * pad_layers, energy_val, params)
        ax.set_title(f"{label}  (N={n_coinc}, σ_mid≈{sigma_disp:.2f} layers)", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, linestyle=":")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Empirical Response Kernel Fit Quality — {mod}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(__file__).resolve().parent / "response_matrices"
    out_dir.mkdir(exist_ok=True)

    for mod, target_sweep in base.TARGET_SWEEPS.items():
        if not target_sweep.exists():
            print(f"Skipping '{mod}' - sweep not found: {target_sweep}")
            continue

        print(f"\n=== Building empirical response kernel for '{mod}' ===")
        is_hex = "hex" in mod
        pairs, lyso_bounds = collect_truth_observed_pairs(mod, target_sweep, is_hex)

        if len(pairs) < 2:
            print(f"  [Fatal] Need >= 2 usable energy points to fit anything "
                  f"(got {len(pairs)}). Skipping '{mod}'.")
            continue

        n_reco = len(lyso_bounds)
        fit = fit_response_matrix(pairs, n_reco, PAD_LAYERS)
        sigma0, sigma_slope, skew, tail_frac, tail_mult, sigma_E_slope = fit.x

        print(f"  Fitted kernel: sigma0={sigma0:.3f}  sigma_slope={sigma_slope:+.4f}  "
              f"skew={skew:+.2f}  tail_frac={tail_frac:.3f}  tail_mult={tail_mult:.2f}  "
              f"sigma_E_slope={sigma_E_slope:+.4f}")
        print(f"  Final cost={fit.cost:.6f}  residual_norm={np.linalg.norm(fit.fun):.4f}")

        meta = {
            "module": mod,
            "n_reco": n_reco,
            "pad_layers": PAD_LAYERS,
            "E_REF_GEV": rk.E_REF_GEV,
            "params": dict(zip(rk.PARAM_NAMES, fit.x.tolist())),
            "params_vector": fit.x.tolist(),   # convenience: ordered list matching PARAM_NAMES
            "fit_cost": float(fit.cost),
            "n_energy_points_used": len(pairs),
            "energy_points": [{"label": p[2], "energy_gev": p[4], "n_coinc": p[3]} for p in pairs],
        }
        with open(out_dir / f"{mod}_response_params.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Diagnostic-only reference matrix at E_REF_GEV -- NOT used by the consumer.
        R_ref = rk.build_kernel_matrix(n_reco, PAD_LAYERS, fit.x, energy_val=rk.E_REF_GEV)
        np.save(out_dir / f"{mod}_response_matrix_Eref.npy", R_ref)

        plot_path = out_dir / f"{mod}_response_fit_quality.png"
        plot_fit_quality(mod, pairs, fit.x, n_reco, PAD_LAYERS, plot_path)

        print(f"  Saved: {mod}_response_params.json, {mod}_response_matrix_Eref.npy (diagnostic), "
              f"{plot_path.name}")

    print("\nDone. Consumer loads per-energy R via:")
    print("  import response_kernel as rk, json")
    print("  meta = json.load(open('response_matrices/<module>_response_params.json'))")
    print("  R = rk.build_kernel_matrix(meta['n_reco'], meta['pad_layers'],")
    print("                             meta['params_vector'], energy_val=<this energy in GeV>)")


if __name__ == "__main__":
    main()