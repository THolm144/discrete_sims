import argparse
import warnings
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import numpy as np
import uproot
from scipy.optimize import curve_fit
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

CAP_XY_MM = np.array([
    [ _HOLE_OFFSET,  _HOLE_OFFSET],   
    [ _HOLE_OFFSET, -_HOLE_OFFSET],   
    [-_HOLE_OFFSET,  _HOLE_OFFSET],   
    [-_HOLE_OFFSET, -_HOLE_OFFSET],   
])

# ─────────────────────────────────────────────────────────────────────────────
# DIGITIZER & FRONTLINE ELECTRONICS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_PS       = 2        # High-granularity digitizer sampling rate
WINDOW_PS       = 50_000    # 50 ns acquisition window
SPTR_SIGMA_PS   = 150       # SiPM single-photon time resolution
MCP_SIGMA_PS    = 15        # MCP reference jitter
MCP_TRUE_PS     = 0.0
FACE_JITTER_PS  = 18.5      # Uncorrelated digitizer/TDC jitter per channel 
SIGMA           = 10         # How hard to smooth to gaussian

# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
def assign_channel(x_mm, y_mm, z_mm):
    if abs(abs(z_mm) - _SIPM_Z_MM) > 1.0:
        return -1
    dists = np.hypot(CAP_XY_MM[:, 0] - x_mm, CAP_XY_MM[:, 1] - y_mm)
    cap_idx = int(np.argmin(dists))
    return cap_idx if z_mm > 0 else cap_idx + 4


# ─────────────────────────────────────────────────────────────────────────────
# ANALOG PULSE WAVEFORM BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_waveform(hit_times_ns, rng):
    # Process the heavy math on a fast, coarse 25 ps grid
    COARSE_PS = 25
    coarse_bins = np.arange(0, WINDOW_PS + COARSE_PS, COARSE_PS)
    
    # The microscopic grid for the CFD to slide along
    fine_bins = np.arange(0, WINDOW_PS + SAMPLE_PS, SAMPLE_PS)
    
    if len(hit_times_ns) == 0:
        return np.zeros(len(fine_bins) - 1, dtype=float), fine_bins

    # Convert to ps and apply SiPM sensor jitter
    times_ps = np.asarray(hit_times_ns, dtype=float) * 1000.0
    times_ps += rng.normal(0.0, SPTR_SIGMA_PS, size=len(times_ps))

    # 1. Lightning-fast histogram on the coarse grid
    counts, _ = np.histogram(times_ps, bins=coarse_bins)
    
    # 2. Lightning-fast smoothing (20 bins * 25 ps = 500 ps physical smoothing)
    coarse_wf = gaussian_filter1d(counts.astype(float), sigma=SIGMA)
    
    # 3. Smooth Upsampling: Stretch the smooth analog curve over the 2 ps grid
    wf = np.interp(fine_bins[:-1], coarse_bins[:-1], coarse_wf)
    
    return wf, fine_bins

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANT FRACTION DISCRIMINATOR (CFD INTERPOLATION)
# ─────────────────────────────────────────────────────────────────────────────
def threshold_crossing_ps(waveform, threshold_fraction=0.20):
    peak_amplitude = np.max(waveform)
    if peak_amplitude < 0.1:
        return None
        
    dynamic_threshold = peak_amplitude * threshold_fraction
    crossings = np.where(waveform >= dynamic_threshold)[0]
    if len(crossings) == 0:
        return None

    idx = crossings[0]
    if idx == 0:
        return 0.0

    # Smooth sub-sample interpolation breaker to bypass the digital floor grid
    y0, y1 = waveform[idx - 1], waveform[idx]
    frac = 0.0 if y1 == y0 else (dynamic_threshold - y0) / (y1 - y0)
    return (idx - 1 + frac) * SAMPLE_PS

# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN FITTER
# ─────────────────────────────────────────────────────────────────────────────
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian(clean_data, n_bins=120):
    if len(clean_data) < 5:
        return 0.0, 0.0, 1.0, 0.0

    actual_mean = np.mean(clean_data)
    actual_median = np.median(clean_data)

    # 1. Isolate the central core of the distribution (removes outliers/tails)
    q75, q25 = np.percentile(clean_data, [75, 25])
    iqr = q75 - q25
    
    # 2. Compute effective sigma directly from the core data width (IQR / 1.349)
    # This is mathematically identical to a Gaussian sigma if the distribution is normal,
    # but completely immune to fitting failures!
    sigma_effective = max(iqr / 1.349, 1.0) # Floor at 1 ps to avoid 0 division

    # Find the peak height for the plot label
    counts, _ = np.histogram(clean_data, bins=n_bins)
    max_counts = counts.max() if len(counts) > 0 else 1.0

    # Return: Amplitude, Mean (Median is more robust), Sigma, Error
    return float(max_counts), float(actual_median), float(sigma_effective), 0.0


def select_high_energy_events(photon_counts_per_event, fraction_lo=5/9, fraction_hi=8/9):
    lo = np.quantile(photon_counts_per_event, fraction_lo)
    hi = np.quantile(photon_counts_per_event, fraction_hi)
    return (photon_counts_per_event >= lo) & (photon_counts_per_event <= hi)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def run(batch_dir: Path):
    rng = np.random.default_rng(seed=42)

    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"))
    if not hit_files:
        print("  No detector_hits_*.root files found.")
        return None

    all_event_id  = []
    all_x, all_y, all_z = [], [], []
    all_time_ns   = []
    all_particle  = []

    from collections import defaultdict
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
                    t    = tree["GlobalTime"].array(library="np")
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
    dw_counts, up_counts = [], [] # Track light balance

    mcp_times = rng.normal(MCP_TRUE_PS, MCP_SIGMA_PS, size=len(unique_events))

    for ev_id, mcp_t in zip(unique_events, mcp_times):
        mask = event_id == ev_id
        ev_times    = time_ns[mask]
        ev_channels = channels[mask]
        
        # Track counts per face
        dw_num = int((ev_channels < 4).sum())
        up_num = int((ev_channels >= 4).sum())

        t_channel = {}   
        for ch in range(8):
            ch_times = ev_times[ev_channels == ch]
            waveform, _ = build_waveform(ch_times, rng)
            t_channel[ch] = threshold_crossing_ps(waveform, threshold_fraction=0.20)

        dw_crossings = [t_channel[ch] for ch in range(4) if t_channel[ch] is not None]
        up_crossings = [t_channel[ch] for ch in range(4, 8) if t_channel[ch] is not None]

        if not dw_crossings or not up_crossings:
            continue

        delta_dw = np.mean(dw_crossings) - mcp_t   
        delta_up = np.mean(up_crossings) - mcp_t   

        delta_dw += rng.normal(0.0, FACE_JITTER_PS)
        delta_up += rng.normal(0.0, FACE_JITTER_PS)

        best_minus_ps.append((delta_dw - delta_up) / 2.0)
        dw_only_ps.append(delta_dw)
        up_only_ps.append(delta_up)
        dw_counts.append(dw_num)
        up_counts.append(up_num)

    best_minus_ps     = np.array(best_minus_ps)
    dw_only_ps        = np.array(dw_only_ps) 
    up_only_ps        = np.array(up_only_ps)
    delta_t_ps        = dw_only_ps - up_only_ps
    
    dw_counts = np.array(dw_counts)
    up_counts = np.array(up_counts)

    # REPLACED SELECTION LOGIC: Filter for events well-centered with high statistics on BOTH sides
    dw_lo = np.quantile(dw_counts, 0.4)
    up_lo = np.quantile(up_counts, 0.4)
    energy_mask = (dw_counts >= dw_lo) & (up_counts >= up_lo)

    selected_bm = best_minus_ps[energy_mask]
    selected_dw = dw_only_ps[energy_mask] 
    selected_up = up_only_ps[energy_mask]
    selected_dt = delta_t_ps[energy_mask]

    if len(selected_bm) < 10:
        selected_bm, selected_dw, selected_up, selected_dt = best_minus_ps, dw_only_ps, up_only_ps, delta_t_ps

    # =================─────────────────────────────────────────────────────────
    # GLOBAL TIMING FILTER (Removes outlier garbage to un-squash histograms)
    # =================─────────────────────────────────────────────────────────
    clean_dw = selected_dw[abs(selected_dw - np.median(selected_dw)) < 400.0]
    clean_up = selected_up[abs(selected_up - np.median(selected_up)) < 400.0]
    clean_dt = selected_dt[abs(selected_dt - np.median(selected_dt)) < 400.0]
    clean_bm = selected_bm[abs(selected_bm - np.median(selected_bm)) < 400.0]

    # Fits executed on the zoomed, filtered curves
    bm_amp, bm_mu, bm_sigma, bm_sigma_err = fit_gaussian(clean_bm)
    dw_amp, dw_mu, dw_sigma, dw_sigma_err = fit_gaussian(clean_dw)
    up_amp, up_mu, up_sigma, up_sigma_err = fit_gaussian(clean_up)
    dt_amp, dt_mu, dt_sigma, dt_sigma_err = fit_gaussian(clean_dt)

    # Plot Layouts
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    energy_label = batch_dir.name
    fig.suptitle(f"Timing Distributions for {energy_label} (Energy Bins 6-8)", fontsize=14, fontweight="bold")

    distributions = [
        {"data": clean_dw, "amp": dw_amp, "mu": dw_mu, "sigma": dw_sigma, "title": "Downstream Time ($\\Delta t_{DW}$)", "color": "royalblue"},
        {"data": clean_up, "amp": up_amp, "mu": up_mu, "sigma": up_sigma, "title": "Upstream Time ($\\Delta t_{UP}$)", "color": "crimson"},
        {"data": clean_dt, "amp": dt_amp, "mu": dt_mu, "sigma": dt_sigma, "title": "Delta t ($\\Delta t_{DW} - \\Delta t_{UP}$)", "color": "darkorchid"}
    ]

    for ax, dist in zip(axs, distributions):
        data = dist["data"]
        if len(data) == 0:
            continue
            
        # ====================================================================
        # ROBUST PLOT ZOOM LOGIC
        # ====================================================================
        q75, q25 = np.percentile(data, [75, 25])
        core_sigma = (q75 - q25) / 1.35
        
        # Lowered the zoom limit from 5.0 to 1.0 to reveal sub-picosecond bins
        plot_sigma = np.clip(core_sigma, 1.0, 150.0) 
        plot_center = np.median(data)
        
        lo = plot_center - 4.0 * plot_sigma
        hi = plot_center + 4.0 * plot_sigma
        # ====================================================================
        
        counts, edges, _ = ax.hist(data, bins=120, range=(lo, hi), color=dist["color"], alpha=0.6, edgecolor='black', label="Data")
        
        x_fit = np.linspace(lo, hi, 5000)
        amplitude = dist["amp"] if dist["amp"] > 0 else counts.max()
        
        # Draw the fit line (even if it's currently a bad fit, so you can see what it's doing)
        y_fit = gaussian(x_fit, amplitude, dist["mu"], dist["sigma"])
        ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2.5, 
                label=f"Fit:\n$\\mu$ = {dist['mu']:.1f} ps\n$\\sigma$ = {dist['sigma']:.1f} ps")
        
        ax.set_title(dist["title"], fontsize=12)
        ax.set_xlabel("Time (ps)", fontsize=10)
        ax.set_ylabel("Events / Bin", fontsize=10)
        ax.set_xlim(lo, hi)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plot_path = batch_dir / "timing_histograms.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return {
        "sigma_t_ps":      bm_sigma,
        "sigma_t_err_ps":  bm_sigma_err,
        "mu_ps":           bm_mu,
        "dw_sigma":        dw_sigma,
        "dw_mu":           dw_mu,
        "up_sigma":        up_sigma,
        "up_mu":           up_mu,
        "dt_sigma":        dt_sigma,
        "dt_mu":           dt_mu,
        "n_events_total":  len(best_minus_ps),
        "n_events_selected": len(clean_bm),
    }

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=str, required=True)
    args   = parser.parse_args()

    batch_path = Path(args.batch_dir)
    energy_label = batch_path.name   
    print(f"\n{'─'*60}")
    print(f"  Timing Resolution — BestMinus (Method 4)")
    print(f"  Batch : {energy_label}")
    print(f"{'─'*60}")

    result = run(batch_path)

    if result is None:
        print("  Could not compute timing resolution.")
    else:
        s  = result["sigma_t_ps"]
        se = result["sigma_t_err_ps"]

        err_str = f" ± {se:.2f}" if not np.isnan(se) else ""
        print(f"\n  BestMinus σ_t  =  {s:.2f}{err_str} ps")
        print(f"  Downstream σ   =  {result['dw_sigma']:.2f} ps")
        print(f"  Upstream σ     =  {result['up_sigma']:.2f} ps")
        print(f"  Delta t σ      =  {result['dt_sigma']:.2f} ps")

        out_txt = batch_path / "timing_resolution.txt"
        with open(out_txt, "w") as f:
            f.write(f"Method          : BestMinus (paper Method 4)\n")
            f.write(f"sigma_t_ps      : {s:.4f}\n")
            f.write(f"sigma_t_err_ps  : {se:.4f}\n")
            f.write(f"mu_ps           : {result['mu_ps']:.4f}\n")
            f.write(f"dw_sigma_ps     : {result['dw_sigma']:.4f}\n")
            f.write(f"up_sigma_ps     : {result['up_sigma']:.4f}\n")
            f.write(f"dt_sigma_ps     : {result['dt_sigma']:.4f}\n")
            f.write(f"n_events_total  : {result['n_events_total']}\n")
            f.write(f"n_events_sel    : {result['n_events_selected']}\n")
        print(f"  Saved metrics → {out_txt}")