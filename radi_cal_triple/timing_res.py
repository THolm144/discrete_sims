import argparse
import warnings
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
SIPM_JITTER_PS = 20.0

# Correct alignment mapping matching worlds/radi_cal_energy.py:
# Indices 0, 1 -> T-type | Indices 2, 3 -> E-type
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
    """Standard symmetric Gaussian distribution."""
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
        A_fit, mu_fit, sig_fit = popt
        perr = np.sqrt(np.diag(pcov))
        return float(A_fit), float(mu_fit), float(sig_fit), float(perr[2]), 0.0
    except Exception:
        return A0, mu0, iqr_sigma, np.nan, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def run(batch_dir: Path):
    t_start = time.perf_counter()
    
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
    print(f"  [Checkpoint 1/6] Loading branches from {len(hit_files)} ROOT files...")
    for parent_dir, files in grouped_files.items():
        max_ev_in_dir = 0
        for fpath in files:
            try:
                with uproot.open(fpath) as f:
                    if not f.keys(): continue
                    tree = f[f.keys()[0]]
                    ev   = tree["EventID"].array(library="np").astype(int)
                    x    = tree["Position_X"].array(library="np")
                    y    = tree["Position_Y"].array(library="np")
                    z    = tree["Position_Z"].array(library="np")
                    t    = tree[TIME].array(library="np")
                    pn   = tree["ParticleName"].array(library="np")

                    all_event_id.append(ev + global_offset)
                    all_x.append(x);  all_y.append(y);  all_z.append(z)
                    all_time_ns.append(t)
                    all_particle.append(pn)

                    if len(ev) > 0:
                        max_ev_in_dir = max(max_ev_in_dir, int(ev.max()))
            except Exception as e:
                pass
        global_offset += max_ev_in_dir + 1

    event_id  = np.concatenate(all_event_id)
    x_mm      = np.concatenate(all_x)
    y_mm      = np.concatenate(all_y)
    z_mm      = np.concatenate(all_z)
    time_ns   = np.concatenate(all_time_ns)
    particle  = np.concatenate(all_particle)
    print(f"  [Checkpoint 2/6] Total Raw Hits: {len(event_id):,}")

    is_optical = (particle == b"opticalphoton") | (particle == "opticalphoton")
    event_id   = event_id[is_optical]
    x_mm       = x_mm[is_optical]
    y_mm       = y_mm[is_optical]
    z_mm       = z_mm[is_optical]
    time_ns    = time_ns[is_optical]
    print(f"  [Checkpoint 3/6] Total Optical Photons: {len(event_id):,}")

    if len(event_id) == 0:
        print("  ERROR: No optical photons found!")
        return None

    # ── Fast, geometry-agnostic vector tracking (No strict Z-cut) ──
    print("  [Checkpoint 4/6] Assigning channels via fast matrix tracking...")
    dx = x_mm[:, None] - CAP_XY_MM[:, 0]
    dy = y_mm[:, None] - CAP_XY_MM[:, 1]
    dists = np.hypot(dx, dy)
    cap_idx = np.argmin(dists, axis=1)
    
    # Simple threshold: z < 0 is Upstream, z > 0 is Downstream
    channels = np.where(z_mm < 0, cap_idx, cap_idx + 4)
    print(f"  --> Successfully mapped {len(channels):,} photons to SiPMs.")

    print("  [Checkpoint 5/6] Sorting and chunking unique event structures...")
    time_ps = time_ns * 1000.0
    is_up_channel = (channels == 0) | (channels == 1)
    is_dw_channel = (channels == 4) | (channels == 5)

    sort_idx = np.argsort(event_id)
    ev_sorted = event_id[sort_idx]
    time_sorted = time_ps[sort_idx]
    up_mask_sorted = is_up_channel[sort_idx]
    dw_mask_sorted = is_dw_channel[sort_idx]

    split_indices = np.where(np.diff(ev_sorted) != 0)[0] + 1
    ev_ids_split = np.split(ev_sorted, split_indices)
    times_split = np.split(time_sorted, split_indices)
    up_mask_split = np.split(up_mask_sorted, split_indices)
    dw_mask_split = np.split(dw_mask_sorted, split_indices)

    print("  [Checkpoint 6/6] Calculating quantile arrival times with SiPM Jitter...")
    best_minus_ps, dw_only_ps, up_only_ps = [], [], []
    dw_counts, up_counts = [], []

    diag_dw_n, diag_up_n       = [], []
    diag_dw_valid, diag_up_valid = [], []
    final_unique_events = []

    for ev_id_arr, t_arr, up_m, dw_m in zip(ev_ids_split, times_split, up_mask_split, dw_mask_split):
        if len(ev_id_arr) == 0:
            continue
            
        ev_id = ev_id_arr[0]
        up_times = t_arr[up_m]
        dw_times = t_arr[dw_m]

        if SIPM_JITTER_PS > 0:
            if len(up_times) > 0:
                up_times = up_times + np.random.normal(0.0, SIPM_JITTER_PS, size=len(up_times))
            if len(dw_times) > 0:
                dw_times = dw_times + np.random.normal(0.0, SIPM_JITTER_PS, size=len(dw_times))

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
        final_unique_events.append(ev_id)

    best_minus_ps = np.array(best_minus_ps)
    dw_only_ps    = np.array(dw_only_ps)
    up_only_ps    = np.array(up_only_ps)
    dw_counts     = np.array(dw_counts)
    up_counts     = np.array(up_counts)
    unique_events = np.array(final_unique_events)
    n_ev          = len(unique_events)

    print(f"\n  ── Asymmetry Diagnostics (Pure T-Type Filament Isolation) ──")
    print(f"  Total events           : {n_ev}")
    
    if n_ev == 0:
        print("  WARNING: 0 Events met the minimum photon threshold. Exiting early.")
        return None

    print(f"  DW median photons/ev   : {np.median(diag_dw_n):.1f}   (mean {np.mean(diag_dw_n):.1f})")
    print(f"  UP median photons/ev   : {np.median(diag_up_n):.1f}   (mean {np.mean(diag_up_n):.1f})")
    
    diag_dw_v = np.array(diag_dw_valid)
    diag_up_v = np.array(diag_up_valid)
    print(f"  DW yield valid timing  : {diag_dw_v.sum()}/{len(diag_dw_n)} ({100*diag_dw_v.mean():.0f}%)")
    print(f"  UP yield valid timing  : {diag_up_v.sum()}/{len(diag_up_n)} ({100*diag_up_v.mean():.0f}%)")
    ratio = np.mean(diag_dw_n) / max(np.mean(diag_up_n), 0.001)
    print(f"  DW/UP photon ratio     : {ratio:.2f}")

    def clean_around_mode(arr, window_ps=60.0):
        if len(arr) == 0: return arr
        counts, edges = np.histogram(arr, bins=40)
        smoothed = gaussian_filter1d(counts.astype(float), sigma=2.0)
        mode_center = 0.5 * (edges[np.argmax(smoothed)] + edges[np.argmax(smoothed) + 1])
        return arr[np.abs(arr - mode_center) < window_ps]

    clean_dw = clean_around_mode(dw_only_ps, window_ps=150.0)
    clean_up = clean_around_mode(up_only_ps, window_ps=150.0)
    clean_bm = clean_around_mode(best_minus_ps, window_ps=300.0)

    bm_amp, bm_mu, bm_sigma, bm_sigma_err, bm_alpha = fit_gaussian_to_peak(clean_bm)
    dw_amp, dw_mu, dw_sigma, dw_sigma_err, dw_alpha = fit_gaussian_to_peak(clean_dw)
    up_amp, up_mu, up_sigma, up_sigma_err, up_alpha = fit_gaussian_to_peak(clean_up)

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
    
    return {
        "sigma_t_ps": bm_sigma, "sigma_t_err_ps": bm_sigma_err, "mu_ps": bm_mu,
        "dw_sigma": dw_sigma, "up_sigma": up_sigma, "n_events_total": len(best_minus_ps),
        "n_events_selected": len(clean_bm), "dw_median_photons": float(np.median(diag_dw_n)),
        "up_median_photons": float(np.median(diag_up_n)),
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