import argparse
import warnings
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.special import erf
import numpy as np
import uproot
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_TYVEK_MM      = 0.008 * 25.4
_CAP_LENGTH_MM = 183.0
_SIPM_THICK_MM = 0.3
_SIPM_Z_MM     = _CAP_LENGTH_MM / 2 + _SIPM_THICK_MM / 2

_CALOR_XY_MM   = 14.0 + 2 * _TYVEK_MM
_HOLE_INSET_MM = 3.5
_HOLE_OFFSET   = _CALOR_XY_MM / 2 - _HOLE_INSET_MM

# Correct alignment mapping matching worlds/radi_cal_energy.py:
# Indices 0, 1 -> T-type
# Indices 2, 3 -> E-type
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
# CHANNEL ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
def assign_channel(x_mm, y_mm, z_mm):
    if abs(abs(z_mm) - _SIPM_Z_MM) > 1.0:
        return -1
    dists = np.hypot(CAP_XY_MM[:, 0] - x_mm, CAP_XY_MM[:, 1] - y_mm)
    cap_idx = int(np.argmin(dists))
    return cap_idx if z_mm < 0 else cap_idx + 4


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
    lo = center - 3.0 * iqr_sigma
    hi = center + 3.0 * iqr_sigma

    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    mids = 0.5 * (edges[:-1] + edges[1:])

    smoothed  = gaussian_filter1d(counts.astype(float), sigma=2.0)
    peak_idx  = int(np.argmax(smoothed))
    mu0       = float(mids[peak_idx])
    A0        = float(smoothed[peak_idx])

    # Narrow the window slightly to avoid the broad flat skirts pulling the peak amplitude up
    fit_mask = np.abs(mids - mu0) < 1.2 * iqr_sigma
    if fit_mask.sum() < 5:
        return A0, mu0, iqr_sigma, np.nan, 0.0

    try:
        popt, pcov = curve_fit(
            standard_gaussian,  # Use the new function
            mids[fit_mask], counts[fit_mask],
            p0=[A0, mu0, iqr_sigma * 0.8],  # Removed alpha initial guess
            bounds=(
                [0.5, mu0 - iqr_sigma, 2.0],        # Lower bounds (removed alpha)
                [A0 * 3.0, mu0 + iqr_sigma, iqr_sigma * 2.0]  # Upper bounds (removed alpha)
            ),
            maxfev=10000,
        )
        A_fit, mu_fit, sig_fit = popt
        perr = np.sqrt(np.diag(pcov))
        
        # Return 0.0 for alpha and its error to keep compatibility with the rest of your script
        return float(A_fit), float(mu_fit), float(sig_fit), float(perr[2]), 0.0
    except Exception:
        return A0, mu0, iqr_sigma, np.nan, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def run(batch_dir: Path):
    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"))
    if not hit_files:
        print("  No detector_hits_*.root files found.")
        return None

    all_event_id  = []
    all_x, all_y, all_z = [], [], []
    all_time_ns   = []
    all_particle  = []

    grouped_files = defaultdict(list)
    for fpath in hit_files:
        grouped_files[fpath.parent].append(fpath)

    global_offset = 0
    for parent_dir, files in grouped_files.items():
        max_ev_in_dir = 0
        for fpath in files:
            try:
                with uproot.open(fpath) as f:
                    if not f.keys():
                        continue                    # skip empty ROOT files
                    tree = f[f.keys()[0]]
                    ev   = tree["EventID"].array(library="np").astype(int)
                    x    = tree["Position_X"].array(library="np")
                    y    = tree["Position_Y"].array(library="np")
                    z    = tree["Position_Z"].array(library="np")
                    t    = tree["LocalTime"].array(library="np")
                    pn   = tree["ParticleName"].array(library="np")

                    all_event_id.append(ev + global_offset)
                    all_x.append(x);  all_y.append(y);  all_z.append(z)
                    all_time_ns.append(t)
                    all_particle.append(pn)

                    if len(ev) > 0:
                        max_ev_in_dir = max(max_ev_in_dir, int(ev.max()))
            except Exception as e:
                print(f"  Warning: could not read {fpath.name}: {e}")
        global_offset += max_ev_in_dir + 1

    event_id  = np.concatenate(all_event_id)
    x_mm      = np.concatenate(all_x)
    y_mm      = np.concatenate(all_y)
    z_mm      = np.concatenate(all_z)
    time_ns   = np.concatenate(all_time_ns)
    particle  = np.concatenate(all_particle)

    is_optical = (particle == b"opticalphoton") | (particle == "opticalphoton")
    event_id   = event_id[is_optical]
    x_mm       = x_mm[is_optical]
    y_mm       = y_mm[is_optical]
    z_mm       = z_mm[is_optical]
    time_ns    = time_ns[is_optical]

    channels = np.array([assign_channel(x, y, z) for x, y, z in zip(x_mm, y_mm, z_mm)])
    on_sipm   = channels >= 0
    event_id  = event_id[on_sipm]
    time_ns   = time_ns[on_sipm]
    channels  = channels[on_sipm]

    unique_events = np.unique(event_id)
    best_minus_ps, dw_only_ps, up_only_ps = [], [], []
    dw_counts, up_counts = [], []

    diag_dw_n, diag_up_n       = [], []
    diag_dw_valid, diag_up_valid = [], []

    for ev_id in unique_events:
        mask = event_id == ev_id
        ev_times_ps = time_ns[mask] * 1000.0
        ev_channels = channels[mask]

        # ── EXPLICIT T-TYPE ISOLATION ──
        up_times = ev_times_ps[(ev_channels == 0) | (ev_channels == 1)]
        dw_times = ev_times_ps[(ev_channels == 4) | (ev_channels == 5)]

        dw_num = len(dw_times)
        up_num = len(up_times)
        diag_dw_n.append(dw_num)
        diag_up_n.append(up_num)

        dw_valid = dw_num >= MIN_PHOTONS_PER_FACE
        up_valid = up_num >= MIN_PHOTONS_PER_FACE
        diag_dw_valid.append(dw_valid)
        diag_up_valid.append(up_valid)

        if not dw_valid or not up_valid:
            continue

        t_dw = np.quantile(dw_times, ARRIVAL_QUANTILE)
        t_up = np.quantile(up_times, ARRIVAL_QUANTILE)

        best_minus_ps.append((t_dw - t_up) / 2.0)
        dw_only_ps.append(t_dw)
        up_only_ps.append(t_up)
        dw_counts.append(dw_num)
        up_counts.append(up_num)

    best_minus_ps = np.array(best_minus_ps)
    dw_only_ps    = np.array(dw_only_ps)
    up_only_ps    = np.array(up_only_ps)
    delta_t_ps    = dw_only_ps - up_only_ps
    dw_counts     = np.array(dw_counts)
    up_counts     = np.array(up_counts)

    # ── Asymmetry diagnostics ───────────────────────────────────────────────
    diag_dw_n   = np.array(diag_dw_n)
    diag_up_n   = np.array(diag_up_n)
    diag_dw_v   = np.array(diag_dw_valid)
    diag_up_v   = np.array(diag_up_valid)
    n_ev        = len(unique_events)
    print(f"\n  ── Asymmetry Diagnostics (Pure T-Type Filament Isolation) ──")
    print(f"  Total events           : {n_ev}")
    print(f"  DW median photons/ev   : {np.median(diag_dw_n):.1f}   (mean {np.mean(diag_dw_n):.1f})")
    print(f"  UP median photons/ev   : {np.median(diag_up_n):.1f}   (mean {np.mean(diag_up_n):.1f})")
    print(f"  DW yield valid timing  : {diag_dw_v.sum()}/{n_ev} ({100*diag_dw_v.mean():.0f}%)")
    print(f"  UP yield valid timing  : {diag_up_v.sum()}/{n_ev} ({100*diag_up_v.mean():.0f}%)")
    ratio = np.mean(diag_dw_n) / max(np.mean(diag_up_n), 0.001)
    print(f"  DW/UP photon ratio     : {ratio:.2f}")

    selected_bm = best_minus_ps
    selected_dw = dw_only_ps
    selected_up = up_only_ps

    # ── Outlier removal ───────────────────────────────────────────────────────
    def clean_around_mode(arr, window_ps=60.0):
        if len(arr) == 0:
            return arr
        counts, edges = np.histogram(arr, bins=40)
        smoothed = gaussian_filter1d(counts.astype(float), sigma=2.0)
        peak_bin = np.argmax(smoothed)
        mode_center = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
        return arr[np.abs(arr - mode_center) < window_ps]

    clean_dw = clean_around_mode(selected_dw, window_ps=150.0)
    clean_up = clean_around_mode(selected_up, window_ps=150.0)
    clean_bm = clean_around_mode(selected_bm, window_ps=100.0)

    # ── Skewed Gaussian peak fits ───────────────────────────────────────────
    bm_amp, bm_mu, bm_sigma, bm_sigma_err, bm_alpha = fit_gaussian_to_peak(clean_bm)
    dw_amp, dw_mu, dw_sigma, dw_sigma_err, dw_alpha = fit_gaussian_to_peak(clean_dw)
    up_amp, up_mu, up_sigma, up_sigma_err, up_alpha = fit_gaussian_to_peak(clean_up)

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    energy_label = batch_dir.name
    fig.suptitle(
        f"Direct LocalTime Distributions for {energy_label} (Pure T-Type Filaments)\n"
        f"Skewed Gaussian peak fit  |  Direct Q={ARRIVAL_QUANTILE:.2f} time marker  |  min {MIN_PHOTONS_PER_FACE} photons/face",
        fontsize=13, fontweight="bold"
    )

    # Updated third entry to track and plot the BestMinus Resolution matching the paper
    distributions = [
        {"data": clean_dw, "amp": dw_amp, "mu": dw_mu, "sigma": dw_sigma, "sigma_err": dw_sigma_err, "alpha": dw_alpha,
         "title": "Downstream T-Type Direct Time ($t_{DW}$)", "color": "royalblue"},
        {"data": clean_up, "amp": up_amp, "mu": up_mu, "sigma": up_sigma, "sigma_err": up_sigma_err, "alpha": up_alpha,
         "title": "Upstream T-Type Direct Time ($t_{UP}$)",   "color": "crimson"},
        {"data": clean_bm, "amp": bm_amp, "mu": bm_mu, "sigma": bm_sigma, "sigma_err": bm_sigma_err, "alpha": bm_alpha,
         "title": "BestMinus Timing Resolution $(t_{DW} - t_{UP})/2$", "color": "darkorchid"},
    ]

    for ax, dist in zip(axs, distributions):
        data = dist["data"]
        if len(data) == 0:
            continue

        # 1. Plot ranges (based on fit results)
        plot_center = dist["mu"]
        plot_sigma  = dist["sigma"]
        lo = plot_center - 3.0 * plot_sigma
        hi = plot_center + 3.0 * plot_sigma

        counts, edges, _ = ax.hist(
            data, bins=100, range=(lo, hi),
            color=dist["color"], alpha=0.6, edgecolor="black", label="Data"
        )

        # 2. Recalculate the original bin width used during the fit
        q75, q25      = np.percentile(data, [75, 25])
        iqr_sigma     = max((q75 - q25) / 1.349, 1.0)
        fit_bin_width = (6.0 * iqr_sigma) / 40.0

        # 3. Calculate the new bin width used for the plot
        plot_bin_width = (hi - lo) / 100.0
        
        # 4. The true scaling factor
        scale_factor = plot_bin_width / fit_bin_width

        x_fit     = np.linspace(lo, hi, 5000)
        amplitude = dist["amp"] * scale_factor if dist["amp"] > 0 else counts.max()
        y_fit     = standard_gaussian(x_fit, amplitude, dist["mu"], dist["sigma"], dist["alpha"])

        err_str = f" ± {dist['sigma_err']:.1f}" if not np.isnan(dist["sigma_err"]) else " (IQR fallback)"
        ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2.5,
                label=f"Skewed Gaussian fit:\n$\\mu$ = {dist['mu']:.1f} ps\n$\\sigma$ = {dist['sigma']:.1f}{err_str} ps\n$\\alpha$ = {dist['alpha']:.2f}")

        ax.set_title(dist["title"], fontsize=12)
        ax.set_xlabel("LocalTime (ps)", fontsize=10)
        ax.set_ylabel("Events / Bin", fontsize=10)
        ax.set_xlim(lo, hi)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plot_path = batch_dir / "direct_timing_histograms.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n  Saved plot → {plot_path}")

    return {
        "sigma_t_ps":        bm_sigma,
        "sigma_t_err_ps":    bm_sigma_err,
        "mu_ps":             bm_mu,
        "dw_sigma":          dw_sigma,
        "dw_mu":             dw_mu,
        "up_sigma":          up_sigma,
        "up_mu":             up_mu,
        "n_events_total":    len(best_minus_ps),
        "n_events_selected": len(clean_bm),
        "dw_median_photons": float(np.median(diag_dw_n)),
        "up_median_photons": float(np.median(diag_up_n)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=str, required=True)
    parser.add_argument("--wls-filter-ns", type=float, default=0.0, help="Ignored in direct mode")
    args = parser.parse_args()

    batch_path   = Path(args.batch_dir)
    energy_label = batch_path.name
    print(f"\n{'─'*60}")
    print(f"  Timing Resolution — Pure Direct LocalTime Mode")
    print(f"  No electronics modeling | Arrival Marker Quantile: {ARRIVAL_QUANTILE}")
    print(f"  Batch : {energy_label}")
    print(f"{'─'*60}")

    result = run(batch_path)

    if result is not None:
        s  = result["sigma_t_ps"]
        se = result["sigma_t_err_ps"]
        err_str = f" ± {se:.2f}" if not np.isnan(se) else " (IQR fallback)"

        print(f"\n  BestMinus σ_t  =  {s:.2f}{err_str} ps")
        print(f"  Downstream σ   =  {result['dw_sigma']:.2f} ps")
        print(f"  Upstream σ     =  {result['up_sigma']:.2f} ps")
        print(f"  DW median hits =  {result['dw_median_photons']:.0f} photons/event")
        print(f"  UP median hits =  {result['up_median_photons']:.0f} photons/event")

        out_txt = batch_path / "direct_timing_resolution.txt"
        with open(out_txt, "w") as f:
            f.write(f"Method          : Direct LocalTime (No Electronics Baseline)\n")
            f.write(f"Arrival Quantile: {ARRIVAL_QUANTILE}\n")
            f.write(f"sigma_t_ps      : {s:.4f}\n")
            f.write(f"sigma_t_err_ps  : {se:.4f}\n")
            f.write(f"mu_ps           : {result['mu_ps']:.4f}\n")
            f.write(f"dw_sigma_ps     : {result['dw_sigma']:.4f}\n")
            f.write(f"up_sigma_ps     : {result['up_sigma']:.4f}\n")
            f.write(f"n_events_total  : {result['n_events_total']}\n")
            f.write(f"n_events_sel    : {result['n_events_selected']}\n")
            f.write(f"dw_median_phot  : {result['dw_median_photons']:.1f}\n")
            f.write(f"up_median_phot  : {result['up_median_photons']:.1f}\n")
        print(f"  Saved metrics → {out_txt}")