import argparse
import time
from pathlib import Path
from scipy.optimize import curve_fit
import numpy as np
import uproot
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_TYVEK_MM      = 0.008 * 25.4
_CALOR_XY_MM   = 14.0 + 2 * _TYVEK_MM
_HOLE_INSET_MM = 3.5
_HOLE_OFFSET   = _CALOR_XY_MM / 2 - _HOLE_INSET_MM

TIME = "LocalTime" 

# Realistic Hardware Jitter Parameters
SIPM_SPTR_PS        = 50.0  # Single Photon Time Resolution (fluctuations in the avalanche)
DIGITIZER_JITTER_PS = 17.5  # Constant electronics/TDC baseline jitter (RADiCAL paper floor)

# Correct alignment mapping: Indices 0, 1 -> T-type | Indices 2, 3 -> E-type
CAP_XY_MM = np.array([
    [ _HOLE_OFFSET,  _HOLE_OFFSET],   # 0 — T-type (Top-Right)
    [-_HOLE_OFFSET, -_HOLE_OFFSET],   # 1 — T-type (Bottom-Left)
    [-_HOLE_OFFSET,  _HOLE_OFFSET],   # 2 — E-type (Top-Left)
    [ _HOLE_OFFSET, -_HOLE_OFFSET],   # 3 — E-type (Bottom-Right)
])

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
ARRIVAL_QUANTILE = 0.10
MIN_PHOTONS_PER_FACE = 5

# ─────────────────────────────────────────────────────────────────────────────
#  GAUSSIAN FITTER
# ─────────────────────────────────────────────────────────────────────────────
def standard_gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian_to_peak(data, n_bins=40):
    if len(data) < 8:
        return 0.0, float(np.median(data)), float(np.std(data)), np.nan, 0.0

    q75, q25  = np.percentile(data, [75, 25])
    iqr_sigma = max((q75 - q25) / 1.349, 1.0)
    center    = np.median(data)
    lo, hi    = center - 3.0 * iqr_sigma, center + 3.0 * iqr_sigma

    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    mids = 0.5 * (edges[:-1] + edges[1:])

    smoothed  = gaussian_filter1d(counts.astype(float), sigma=2.0)
    peak_idx  = int(np.argmax(smoothed))
    mu0, A0   = float(mids[peak_idx]), float(smoothed[peak_idx])

    fit_mask = np.abs(mids - mu0) < 3.0 * iqr_sigma
    if fit_mask.sum() < 5:
        return A0, mu0, iqr_sigma, np.nan, 0.0

    try:
        popt, pcov = curve_fit(
            standard_gaussian, mids[fit_mask], counts[fit_mask],
            p0=[A0, mu0, iqr_sigma * 0.8],
            bounds=([0.5, mu0 - iqr_sigma, iqr_sigma * 0.05], [A0 * 3.0, mu0 + iqr_sigma, iqr_sigma * 2.0]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        return float(popt[0]), float(popt[1]), float(popt[2]), float(perr[2]), 0.0
    except Exception:
        return A0, mu0, iqr_sigma, np.nan, 0.0

def clean_around_mode(arr, window_ps=60.0):
    if len(arr) == 0: return arr
    counts, edges = np.histogram(arr, bins=40)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=2.0)
    mode_center = 0.5 * (edges[np.argmax(smoothed)] + edges[np.argmax(smoothed) + 1])
    return arr[np.abs(arr - mode_center) < window_ps]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def run(batch_dir: Path):
    t_start = time.perf_counter()
    
    # Traverse directories robustly
    raw_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    run_dirs = []
    for d in raw_dirs:
        nested = d / d.name
        run_dirs.append(nested if nested.is_dir() else d)
    
    hit_files = [p for d in run_dirs for p in sorted(d.glob("**/detector_hits_*.root"))]

    if not hit_files:
        print("  WARNING: No detector_hits_*.root files found.")
        return None

    # Dictionaries to accumulate UP and DW times avoiding EventID overlaps
    up_times_by_ev = {}
    dw_times_by_ev = {}

    print(f"  [Checkpoint 1/4] Parsing {len(hit_files)} ROOT files via dict mapping...")
    
    for fpath in hit_files:
        run_tag = fpath.parent.name
        try:
            with uproot.open(fpath) as f:
                tree_key = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tree_key: continue
                tree = f[tree_key]
                if tree.num_entries == 0: continue

                x  = tree["Position_X"].array(library="np")
                y  = tree["Position_Y"].array(library="np")
                z  = tree["Position_Z"].array(library="np")
                lt = tree[TIME].array(library="np")
                ev = tree["EventID"].array(library="np")
                pn = tree["ParticleName"].array(library="np")
        except Exception:
            continue

        # Filter for optical photons
        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")
        x, y, z, lt, ev = x[is_optical], y[is_optical], z[is_optical], lt[is_optical], ev[is_optical]
        
        if len(ev) == 0: continue

        # Convert to ps and apply Single Photon SiPM avalanche jitter
        lt_ps = lt * 1000.0
        if SIPM_SPTR_PS > 0:
            lt_ps += np.random.normal(0.0, SIPM_SPTR_PS, size=len(lt_ps))

        # Map to capillaries
        dx = x[:, None] - CAP_XY_MM[:, 0]
        dy = y[:, None] - CAP_XY_MM[:, 1]
        cap_idx = np.argmin(np.hypot(dx, dy), axis=1)

        # T-Type Only (0 and 1)
        is_t_type = (cap_idx == 0) | (cap_idx == 1)
        mask_up = is_t_type & (z < 0)
        mask_dw = is_t_type & (z > 0)

        for e, t in zip(ev[mask_up], lt_ps[mask_up]):
            up_times_by_ev.setdefault((run_tag, int(e)), []).append(t)

        for e, t in zip(ev[mask_dw], lt_ps[mask_dw]):
            dw_times_by_ev.setdefault((run_tag, int(e)), []).append(t)

    print("  [Checkpoint 2/4] Extracting Quantiles & applying Digitizer jitter...")
    all_keys = set(up_times_by_ev.keys()) & set(dw_times_by_ev.keys())
    
    best_minus_ps, dw_only_ps, up_only_ps = [], [], []
    diag_dw_n, diag_up_n = [], []

    for key in all_keys:
        up_times = up_times_by_ev[key]
        dw_times = dw_times_by_ev[key]
        
        dw_num = len(dw_times)
        up_num = len(up_times)
        diag_dw_n.append(dw_num)
        diag_up_n.append(up_num)

        if dw_num < MIN_PHOTONS_PER_FACE or up_num < MIN_PHOTONS_PER_FACE:
            continue

        # Extract timing markers and apply constant electronics floor jitter
        t_dw = np.quantile(dw_times, ARRIVAL_QUANTILE)
        t_up = np.quantile(up_times, ARRIVAL_QUANTILE)

        if DIGITIZER_JITTER_PS > 0:
            t_dw += np.random.normal(0.0, DIGITIZER_JITTER_PS)
            t_up += np.random.normal(0.0, DIGITIZER_JITTER_PS)

        best_minus_ps.append((t_dw - t_up) / 2.0)
        dw_only_ps.append(t_dw)
        up_only_ps.append(t_up)

    n_ev = len(all_keys)
    if n_ev == 0 or len(best_minus_ps) == 0:
        print("  WARNING: 0 Events met the minimum photon threshold. Exiting early.")
        return None

    diag_dw_n, diag_up_n = np.array(diag_dw_n), np.array(diag_up_n)

    print(f"\n  ── Asymmetry Diagnostics (Pure T-Type Filament Isolation) ──")
    print(f"  Coincident events      : {len(best_minus_ps)} / {n_ev}")
    print(f"  DW median photons/ev   : {np.median(diag_dw_n):.1f}")
    print(f"  UP median photons/ev   : {np.median(diag_up_n):.1f}")

    print("  [Checkpoint 3/4] Filtering and Fitting Gaussian Profiles...")
    clean_dw = clean_around_mode(np.array(dw_only_ps), window_ps=150.0)
    clean_up = clean_around_mode(np.array(up_only_ps), window_ps=150.0)
    clean_bm = clean_around_mode(np.array(best_minus_ps), window_ps=300.0)

    bm_amp, bm_mu, bm_sigma, bm_sigma_err, _ = fit_gaussian_to_peak(clean_bm)
    dw_amp, dw_mu, dw_sigma, dw_sigma_err, _ = fit_gaussian_to_peak(clean_dw)
    up_amp, up_mu, up_sigma, up_sigma_err, _ = fit_gaussian_to_peak(clean_up)

    print("  [Checkpoint 4/4] Rendering Visualizations...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Timing Distributions for {batch_dir.name}\nDirect Q={ARRIVAL_QUANTILE:.2f}  |  min {MIN_PHOTONS_PER_FACE} photons/face", fontsize=13, fontweight="bold")

    distributions = [
        {"data": clean_dw, "amp": dw_amp, "mu": dw_mu, "sigma": dw_sigma, "sigma_err": dw_sigma_err, "title": "Downstream T-Type ($t_{DW}$)", "color": "royalblue"},
        {"data": clean_up, "amp": up_amp, "mu": up_mu, "sigma": up_sigma, "sigma_err": up_sigma_err, "title": "Upstream T-Type ($t_{UP}$)", "color": "crimson"},
        {"data": clean_bm, "amp": bm_amp, "mu": bm_mu, "sigma": bm_sigma, "sigma_err": bm_sigma_err, "title": "BestMinus $(t_{DW} - t_{UP})/2$", "color": "darkorchid"},
    ]

    for ax, dist in zip(axs, distributions):
        if len(dist["data"]) == 0: continue
        lo, hi = dist["mu"] - 3.0 * dist["sigma"], dist["mu"] + 3.0 * dist["sigma"]
        counts, edges, _ = ax.hist(dist["data"], bins=100, range=(lo, hi), color=dist["color"], alpha=0.6, edgecolor="black", label="Data")
        
        q75, q25 = np.percentile(dist["data"], [75, 25])
        iqr_sigma = max((q75 - q25) / 1.349, 1.0)
        scale_factor = ((hi - lo) / 100.0) / ((6.0 * iqr_sigma) / 40.0)
        
        x_fit = np.linspace(lo, hi, 5000)
        amplitude = dist["amp"] * scale_factor if dist["amp"] > 0 else counts.max()
        ax.plot(x_fit, standard_gaussian(x_fit, amplitude, dist["mu"], dist["sigma"]), color="black", linestyle="--", linewidth=2.5,
                label=f"Fit:\n$\\mu$={dist['mu']:.1f}\n$\\sigma$={dist['sigma']:.1f}")

        ax.set_title(dist["title"], fontsize=12)
        ax.set_xlabel("Time (ps)", fontsize=10)
        ax.legend(loc="upper right")

    plt.tight_layout()
    plot_path = batch_dir / "direct_timing_histograms.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"  [Execution Completed in {time.perf_counter() - t_start:.2f} seconds]")

    return {
        "sigma_t_ps": bm_sigma, "dw_sigma": dw_sigma, "up_sigma": up_sigma
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=str, required=True)
    args = parser.parse_args()

    batch_path   = Path(args.batch_dir)
    print(f"\n{'─'*60}\n  Batch : {batch_path.name}\n{'─'*60}")
    
    result = run(batch_path)

    if result is not None:
        print(f"\n  BestMinus σ_t  =  {result['sigma_t_ps']:.2f} ps")
        print(f"  Downstream σ   =  {result['dw_sigma']:.2f} ps")
        print(f"  Upstream σ     =  {result['up_sigma']:.2f} ps")