import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d
import numpy as np
import uproot
import matplotlib.pyplot as plt

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
ARRIVAL_QUANTILE     = 0.10
MIN_PHOTONS_PER_FACE = 1

# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
def assign_channel(x_mm, y_mm, z_mm):
    # Find closest SiPM coordinate in the XY plane
    dists = np.hypot(CAP_XY_MM[:, 0] - x_mm, CAP_XY_MM[:, 1] - y_mm)
    cap_idx = int(np.argmin(dists))
    
    # z_mm < 0 -> Upstream (Channels 0-3)
    # z_mm > 0 -> Downstream (Channels 4-7)
    return cap_idx if z_mm < 0 else cap_idx + 4

# ─────────────────────────────────────────────────────────────────────────────
# FITTING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def skewed_gaussian(x, A, mu, sigma, alpha):
    gauss = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    skew  = 1.0 + erf(alpha * (x - mu) / (sigma * np.sqrt(2)))
    return A * gauss * skew

def gaussian(x, A, mu, sigma):
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

    fit_mask = np.abs(mids - mu0) < 1.0 * iqr_sigma
    if fit_mask.sum() < 4:
        return A0, mu0, iqr_sigma, np.nan, 0.0

    try:
        popt, pcov = curve_fit(
            gaussian,
            mids[fit_mask], counts[fit_mask],
            p0=[A0, mu0, iqr_sigma * 0.5],
            bounds=([0.5, mu0 - iqr_sigma, 1.0],
                    [A0 * 4.0, mu0 + iqr_sigma, iqr_sigma * 1.5]),
            maxfev=10000,
        )
        A_fit, mu_fit, sig_fit = popt
        perr = np.sqrt(np.diag(pcov))
        return float(A_fit), float(mu_fit), float(sig_fit), float(perr[2]), 0.0
    except Exception:
        return A0, mu0, iqr_sigma, np.nan, 0.0

def clean_around_mode(arr, window_ps=60.0):
    if len(arr) == 0:
        return arr
    counts, edges = np.histogram(arr, bins=40)
    smoothed      = gaussian_filter1d(counts.astype(float), sigma=2.0)
    peak_bin      = np.argmax(smoothed)
    mode_center   = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
    return arr[np.abs(arr - mode_center) < window_ps]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(batch_dir: Path):
    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"))
    if not hit_files:
        print("  No detector_hits_*.root files found.")
        return None

    all_event_id             = []
    all_x, all_y, all_z     = [], [], []
    all_time_ns              = []
    all_particle             = []

    grouped_files = defaultdict(list)
    for fpath in hit_files:
        grouped_files[fpath.parent].append(fpath)

    global_offset = 0
    for parent_dir, files in grouped_files.items():
        max_ev_in_dir = 0
        for fpath in files:
            try:
                with uproot.open(fpath) as f:
                    tree = f[f.keys()[0]]
                    ev   = tree["EventID"].array(library="np").astype(int)
                    x    = tree["Position_X"].array(library="np")
                    y    = tree["Position_Y"].array(library="np")
                    z    = tree["Position_Z"].array(library="np")
                    t    = tree["LocalTime"].array(library="np")
                    pn   = tree["ParticleName"].array(library="np")

                    all_event_id.append(ev + global_offset)
                    all_x.append(x); all_y.append(y); all_z.append(z)
                    all_time_ns.append(t)
                    all_particle.append(pn)

                    if len(ev) > 0:
                        max_ev_in_dir = max(max_ev_in_dir, int(ev.max()))
            except Exception as e:
                print(f"  Warning: could not read {fpath.name}: {e}")
        global_offset += max_ev_in_dir + 1

    event_id = np.concatenate(all_event_id)
    x_mm     = np.concatenate(all_x)
    y_mm     = np.concatenate(all_y)
    z_mm     = np.concatenate(all_z)
    time_ns  = np.concatenate(all_time_ns)
    particle = np.concatenate(all_particle)
    # ─────────────────────────────────────────────────────────────────────────────
    # DEBUGGING SUITE: WHERE IS THE DATA VANISHING?
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[DEBUG SET 1: RAW CONCATENATED SHAPES]")
    print(f"  Raw event_id shape  : {event_id.shape}")
    print(f"  Raw particle shape  : {particle.shape}")
    if len(particle) > 0:
        print(f"  Sample raw particle values: {particle[:5]}")
        print(f"  Raw particle type          : {particle.dtype}")

    # 1. Optical Photon Masking
    is_optical = (particle == b"opticalphoton") | (particle == "opticalphoton")
    print(f"\n[DEBUG SET 2: OPTICAL FILTERING]")
    print(f"  True elements in is_optical mask: {np.sum(is_optical)}")
    
    event_id   = event_id[is_optical]
    x_mm       = x_mm[is_optical]
    y_mm       = y_mm[is_optical]
    z_mm       = z_mm[is_optical]
    time_ns    = time_ns[is_optical]

    # 2. Channel Assignment
    channels = np.array([assign_channel(x, y, z) for x, y, z in zip(x_mm, y_mm, z_mm)])
    print(f"\n[DEBUG SET 3: CHANNEL ASSIGNMENT]")
    print(f"  Unique channels assigned before filter: {np.unique(channels)}")
    print(f"  Count of channel values == -1         : {np.sum(channels == -1)}")
    
    on_sipm   = channels >= 0
    print(f"  True elements in on_sipm mask         : {np.sum(on_sipm)}")
    
    event_id  = event_id[on_sipm]
    time_ns   = time_ns[on_sipm]
    channels  = channels[on_sipm]

    unique_events = np.unique(event_id)
    print(f"\n[DEBUG SET 4: FINAL CLEANED EVENTS]")
    print(f"  Unique event count remaining: {len(unique_events)}")
    if len(unique_events) > 0:
        print(f"  Sample remaining unique event IDs: {unique_events[:5]}")
    print("────────────────────────────────────────────────────────────\n")

    # Clean byte/string comparison that bypasses the type-casting bug
    is_optical = (particle == b"opticalphoton") | (particle == "opticalphoton")
    event_id = np.concatenate(all_event_id)
    x_mm     = np.concatenate(all_x)
    y_mm     = np.concatenate(all_y)
    z_mm     = np.concatenate(all_z)
    time_ns  = np.concatenate(all_time_ns)
    particle = np.concatenate(all_particle)

    # 1. Optical photon selection
    is_optical = (particle == b"opticalphoton") | (particle == "opticalphoton")
    
    event_id_opt = event_id[is_optical]
    x_mm_opt     = x_mm[is_optical]
    y_mm_opt     = y_mm[is_optical]
    z_mm_opt     = z_mm[is_optical]
    time_ns_opt  = time_ns[is_optical]

    # 2. Channel Assignment (Using the updated, robust function)
    channels = np.array([assign_channel(x, y, z) for x, y, z in zip(x_mm_opt, y_mm_opt, z_mm_opt)])
    
    on_sipm   = channels >= 0
    event_id  = event_id_opt[on_sipm]
    time_ns   = time_ns_opt[on_sipm]
    channels  = channels[on_sipm]

    unique_events = np.unique(event_id)
    best_minus_ps, dw_only_ps, up_only_ps = [], [], []

    diag_dw_n, diag_up_n         = [], []
    diag_dw_valid, diag_up_valid = [], []

    for ev_id in unique_events:
        mask        = event_id == ev_id
        ev_times_ps = time_ns[mask] * 1000.0
        ev_channels = channels[mask]

        # Explicit T-Type Geometry Selection
        dw_times = ev_times_ps[(ev_channels == 0) | (ev_channels == 1)]
        up_times = ev_times_ps[(ev_channels == 4) | (ev_channels == 5)]

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

    best_minus_ps = np.array(best_minus_ps)
    dw_only_ps    = np.array(dw_only_ps)
    up_only_ps    = np.array(up_only_ps)
    delta_t_ps    = dw_only_ps - up_only_ps

    diag_dw_n = np.array(diag_dw_n)
    diag_up_n = np.array(diag_up_n)
    diag_dw_v = np.array(diag_dw_valid)
    diag_up_v = np.array(diag_up_valid)
    n_ev      = len(unique_events)

    print(f"\n  ── Asymmetry Diagnostics (Pure T-Type Filament Isolation) ──")
    print(f"  Total events           : {n_ev}")
    print(f"  DW median photons/ev   : {np.median(diag_dw_n):.1f}   (mean {np.mean(diag_dw_n):.1f})")
    print(f"  UP median photons/ev   : {np.median(diag_up_n):.1f}   (mean {np.mean(diag_up_n):.1f})")
    print(f"  DW yield valid timing  : {diag_dw_v.sum()}/{n_ev} ({100*diag_dw_v.mean():.0f}%)")
    print(f"  UP yield valid timing  : {diag_up_v.sum()}/{n_ev} ({100*diag_up_v.mean():.0f}%)")
    ratio = np.mean(diag_dw_n) / max(np.mean(diag_up_n), 0.001)
    print(f"  DW/UP photon ratio     : {ratio:.2f}")

    clean_dw = clean_around_mode(dw_only_ps,    window_ps=150.0)
    clean_up = clean_around_mode(up_only_ps,    window_ps=150.0)
    clean_dt = clean_around_mode(delta_t_ps,    window_ps=200.0)
    clean_bm = clean_around_mode(best_minus_ps, window_ps=100.0)

    bm_amp, bm_mu, bm_sigma, bm_sigma_err, bm_alpha = fit_gaussian_to_peak(clean_bm)
    dw_amp, dw_mu, dw_sigma, dw_sigma_err, dw_alpha = fit_gaussian_to_peak(clean_dw)
    up_amp, up_mu, up_sigma, up_sigma_err, up_alpha = fit_gaussian_to_peak(clean_up)
    dt_amp, dt_mu, dt_sigma, dt_sigma_err, dt_alpha = fit_gaussian_to_peak(clean_dt)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    energy_label = batch_dir.name
    fig.suptitle(
        f"Direct LocalTime Distributions for {energy_label} (Pure T-Type Filaments)\n"
        f"Gaussian peak fit  |  Q={ARRIVAL_QUANTILE:.2f} arrival marker  |  min {MIN_PHOTONS_PER_FACE} photons/face",
        fontsize=13, fontweight="bold"
    )

    distributions = [
        {"data": clean_dw, "amp": dw_amp, "mu": dw_mu, "sigma": dw_sigma, "sigma_err": dw_sigma_err, "alpha": dw_alpha,
         "title": "Downstream T-Type Direct Time ($t_{DW}$)", "color": "royalblue"},
        {"data": clean_up, "amp": up_amp, "mu": up_mu, "sigma": up_sigma, "sigma_err": up_sigma_err, "alpha": up_alpha,
         "title": "Upstream T-Type Direct Time ($t_{UP}$)",   "color": "crimson"},
        {"data": clean_dt, "amp": dt_amp, "mu": dt_mu, "sigma": dt_sigma, "sigma_err": dt_sigma_err, "alpha": dt_alpha,
         "title": "Delta t ($t_{DW} - t_{UP}$)",              "color": "darkorchid"},
    ]

    for ax, dist in zip(axs, distributions):
        data = dist["data"]
        if len(data) == 0:
            continue

        plot_center = dist["mu"]
        plot_sigma  = dist["sigma"]
        lo = plot_center - 3.0 * plot_sigma
        hi = plot_center + 3.0 * plot_sigma

        counts, edges, _ = ax.hist(
            data, bins=100, range=(lo, hi),
            color=dist["color"], alpha=0.6, edgecolor="black", label="Data"
        )

        x_fit     = np.linspace(lo, hi, 5000)
        amplitude = dist["amp"] if dist["amp"] > 0 else counts.max()
        y_fit     = skewed_gaussian(x_fit, amplitude, dist["mu"], dist["sigma"], dist["alpha"])

        err_str = f" ± {dist['sigma_err']:.1f}" if not np.isnan(dist["sigma_err"]) else " (IQR fallback)"
        ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2.5,
                label=f"Gaussian fit:\n$\\mu$ = {dist['mu']:.1f} ps\n$\\sigma$ = {dist['sigma']:.1f}{err_str} ps")

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
        "dt_sigma":          dt_sigma,
        "dt_mu":             dt_mu,
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
        print(f"  Delta t σ      =  {result['dt_sigma']:.2f} ps")
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
            f.write(f"dt_sigma_ps     : {result['dt_sigma']:.4f}\n")
            f.write(f"n_events_total  : {result['n_events_total']}\n")
            f.write(f"n_events_sel    : {result['n_events_selected']}\n")
            f.write(f"dw_median_phot  : {result['dw_median_photons']:.1f}\n")
            f.write(f"up_median_phot  : {result['up_median_photons']:.1f}\n")
        print(f"  Saved metrics → {out_txt}")