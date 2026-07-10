#!/usr/bin/env python3
"""
build_empirical_response_matrix.py
====================================
Constructs a data-driven longitudinal response matrix per calorimeter module by
jointly fitting a parametric (position-dependent, skewed) kernel against the
Sim-Truth (DoseActor) vs raw ΔT-reconstructed profile pairs that already come
out of the existing sweep -- no new point-source calibration simulations
required.

APPROACH
--------
At each energy point we already have:
  - truth_norm : normalized DoseActor energy deposit per LYSO layer
  - raw_norm   : normalized histogram of raw_z_emits (BestMinus ToF z_est)

The raw histogram is a blurred version of the truth: y_obs = R @ x_true (+noise).
Rather than assume R is a single fixed symmetric Gaussian (as in
unfold_profile_analysis.py's extended_response_matrix), we fit a handful of
shape parameters of R -- width, width-vs-depth slope, skew, and a wide-tail
mixture fraction for leakage -- against ALL energy points *simultaneously*.
Different energies populate different depths/widths in the calorimeter, so
pooling them gives real constraining power on kernel shape that a single
truth/observed pair could not.

This script fits SHAPE only. It deliberately does not fit for a peak offset/
shift (that's V_EFF_MM_NS / BOUNCE_FACTOR / T0_OFFSET calibration, handled
separately) -- letting the kernel absorb a shift would just hide the real bug.

CAVEAT: the truth curves themselves are finite-statistics DoseActor
histograms. For modules/energies with sparse truth stats (rc_hex_triple is
noted as spiky at low N), the fit is only as good as that truth curve --
inspect the fit-quality plots before trusting a given module's matrix, and
consider excluding energy points where the truth histogram itself looks
noisy rather than physical.

OUTPUT
------
response_matrices/<module>_response_matrix.npy       -- the (n_reco, n_true) matrix
response_matrices/<module>_response_matrix_meta.json -- fitted params + provenance
response_matrices/<module>_response_fit_quality.png  -- truth vs observed vs R@truth per energy

USAGE
-----
Place in the same directory as unfold_profile_analysis.py (it imports geometry/
extraction code from there so the matrix stays self-consistent with whatever
consumes it) and run:

    python3 build_empirical_response_matrix.py

Then in unfold_profile_analysis.py, replace the analytic
extended_response_matrix(...) call in bootstrap_unfold with e.g.:

    R_sliced = np.load(f"response_matrices/{module_name}_response_matrix.npy")
"""

import sys
import json
import warnings
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import skewnorm

warnings.filterwarnings("ignore")

# Reuse geometry/extraction/truth-loading code so this matrix is built from
# EXACTLY the same pipeline that will eventually consume it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_deconv_profile as base

try:
    import analysis_utils as utils
except ImportError:
    utils = None
    print("[Fatal] analysis_utils not found -- truth curves are required to fit R.")
    sys.exit(1)

PAD_LAYERS = 5
MIN_COINCIDENCES = 30       # skip energy points with too few coincidences to trust
MIN_ENERGY_POINTS = 2       # minimum usable points needed to fit a kernel at all


# ─────────────────────────────────────────────────────────────────────────────
# Parametric kernel model
# ─────────────────────────────────────────────────────────────────────────────
def build_kernel_matrix(n_reco, pad_layers, params):
    """
    params = [sigma0, sigma_slope, skew, tail_frac, tail_width_mult]

    sigma(j) = sigma0 + sigma_slope * (j - n_true/2), floor-clipped
    Each column j (the response to a true emission in true-layer j) is a
    skew-normal centered at j with width sigma(j), mixed with a wider
    skew-normal "tail" component (same skew, wider scale) to capture
    leakage/scatter a single peak can't describe. Columns are normalized to
    sum to 1 (probability-conserving), then rows are sliced down to the
    physical n_reco window -- same padding convention as
    extended_response_matrix() in unfold_profile_analysis.py.
    """
    sigma0, sigma_slope, skew, tail_frac, tail_mult = params
    n_true = n_reco + 2 * pad_layers
    idx = np.arange(n_true)

    R_full = np.zeros((n_true, n_true))
    for j in idx:
        sigma_j = max(sigma0 + sigma_slope * (j - n_true / 2.0), 0.15)
        x = idx - j
        core = skewnorm.pdf(x, a=skew, loc=0.0, scale=sigma_j)
        tail = skewnorm.pdf(x, a=skew, loc=0.0, scale=sigma_j * max(tail_mult, 1.0))
        col = (1.0 - tail_frac) * core + tail_frac * tail
        s = col.sum()
        if s > 0:
            col /= s
        R_full[:, j] = col

    return R_full[pad_layers: pad_layers + n_reco, :]


# ─────────────────────────────────────────────────────────────────────────────
# Data collection -- reuses extract_profile_data_unfold + the same truth-curve
# loading block as unfold_profile_analysis.py's main(), so nothing drifts.
# ─────────────────────────────────────────────────────────────────────────────
def collect_truth_observed_pairs(mod, target_sweep, is_hex):
    energy_dirs = sorted(
        [d for d in target_sweep.iterdir() if d.is_dir() and "GeV" in d.name],
        key=lambda p: base.extract_numerical_energy(p.name),
    )

    pairs = []  # (truth_norm, raw_norm, label, n_coincidences)
    lyso_bounds_ref = None

    for edir in energy_dirs:
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

        pairs.append((truth_norm, raw_norm, edir.name, res["n_e_coincidences"]))
        print(f"    [ok]   {edir.name}: N={res['n_e_coincidences']} coincidences, truth+raw loaded")

    return pairs, lyso_bounds_ref


# ─────────────────────────────────────────────────────────────────────────────
# Fit
# ─────────────────────────────────────────────────────────────────────────────
def fit_response_matrix(pairs, n_reco, pad_layers):
    def residuals(params):
        R = build_kernel_matrix(n_reco, pad_layers, params)
        res = []
        for truth_norm, raw_norm, label, n_coinc in pairs:
            x_ext = np.zeros(n_reco + 2 * pad_layers)
            x_ext[pad_layers: pad_layers + n_reco] = truth_norm
            pred = R @ x_ext
            pred_norm = pred / pred.sum() if pred.sum() > 0 else pred
            # sqrt(N) weighting: higher-stat energy points count more, but
            # don't let one high-N point completely dominate the shape fit
            weight = np.sqrt(max(n_coinc, 1))
            res.append(weight * (pred_norm - raw_norm))
        return np.concatenate(res)

    # params: sigma0, sigma_slope, skew, tail_frac, tail_width_mult
    p0        = [1.5,  0.0,   0.0, 0.10, 3.0]
    bounds_lo = [0.2, -0.15, -8.0, 0.00, 1.5]
    bounds_hi = [6.0,  0.15,  8.0, 0.50, 8.0]

    return least_squares(residuals, p0, bounds=(bounds_lo, bounds_hi), verbose=0)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_fit_quality(mod, pairs, R, n_reco, pad_layers, out_path):
    import matplotlib.pyplot as plt
    n = len(pairs)
    ncols = 2 if n >= 2 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), squeeze=False)
    layers = np.arange(1, n_reco + 1)

    for i, (truth_norm, raw_norm, label, n_coinc) in enumerate(pairs):
        ax = axes[i // ncols][i % ncols]
        x_ext = np.zeros(n_reco + 2 * pad_layers)
        x_ext[pad_layers: pad_layers + n_reco] = truth_norm
        pred = R @ x_ext
        pred_norm = pred / pred.sum() if pred.sum() > 0 else pred

        ax.bar(layers, truth_norm, color="#00bcd4", alpha=0.25, edgecolor="#00838f", label="Truth (DoseActor)")
        ax.plot(layers, raw_norm, "k.:", label="Observed raw ΔT", alpha=0.7)
        ax.plot(layers, pred_norm, "r-o", markersize=3, linewidth=1.5, label="R @ Truth (fitted)")
        ax.set_title(f"{label}  (N={n_coinc})", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, linestyle=":")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Empirical Response Matrix Fit Quality — {mod}", fontsize=13, fontweight="bold")
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

        print(f"\n=== Building empirical response matrix for '{mod}' ===")
        is_hex = "hex" in mod
        pairs, lyso_bounds = collect_truth_observed_pairs(mod, target_sweep, is_hex)

        if len(pairs) < MIN_ENERGY_POINTS:
            print(f"  [Fatal] Need >= {MIN_ENERGY_POINTS} usable energy points to fit a kernel "
                  f"(got {len(pairs)}). Skipping '{mod}'.")
            continue

        n_reco = len(lyso_bounds)
        fit = fit_response_matrix(pairs, n_reco, PAD_LAYERS)
        sigma0, sigma_slope, skew, tail_frac, tail_mult = fit.x

        print(f"  Fitted kernel: sigma0={sigma0:.3f}  sigma_slope={sigma_slope:+.4f}  "
              f"skew={skew:+.2f}  tail_frac={tail_frac:.3f}  tail_mult={tail_mult:.2f}")
        print(f"  Final cost={fit.cost:.6f}  residual_norm={np.linalg.norm(fit.fun):.4f}")

        R = build_kernel_matrix(n_reco, PAD_LAYERS, fit.x)

        npy_path = out_dir / f"{mod}_response_matrix.npy"
        np.save(npy_path, R)

        meta = {
            "module": mod,
            "n_reco": n_reco,
            "pad_layers": PAD_LAYERS,
            "params": {
                "sigma0": sigma0, "sigma_slope": sigma_slope, "skew": skew,
                "tail_frac": tail_frac, "tail_width_mult": tail_mult,
            },
            "fit_cost": float(fit.cost),
            "n_energy_points_used": len(pairs),
            "energy_points": [p[2] for p in pairs],
        }
        with open(out_dir / f"{mod}_response_matrix_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        plot_path = out_dir / f"{mod}_response_fit_quality.png"
        plot_fit_quality(mod, pairs, R, n_reco, PAD_LAYERS, plot_path)

        print(f"  Saved: {npy_path.name}, {plot_path.name}")

    print("\nDone. Load a matrix with:")
    print("  R = np.load('response_matrices/<module>_response_matrix.npy')")


if __name__ == "__main__":
    main()