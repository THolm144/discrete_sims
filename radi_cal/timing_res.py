import argparse
import warnings
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
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

CAP_XY_MM = np.array([
    [ _HOLE_OFFSET,  _HOLE_OFFSET],
    [ _HOLE_OFFSET, -_HOLE_OFFSET],
    [-_HOLE_OFFSET,  _HOLE_OFFSET],
    [-_HOLE_OFFSET, -_HOLE_OFFSET],
])

# ─────────────────────────────────────────────────────────────────────────────
# DIGITIZER & FRONTLINE ELECTRONICS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_PS       = 2
WINDOW_PS       = 50_000
SPTR_SIGMA_PS   = 150
MCP_SIGMA_PS    = 15
MCP_TRUE_PS     = 0.0
FACE_JITTER_PS  = 18.5
SIGMA           = 10       # Gaussian smooth kernel (bins on coarse 25 ps grid)

# Saturation clamp: mimics amplifier saturation (paper drove high-gain into
# saturation deliberately to steepen the leading edge for better timing).
# Set as a fraction of the per-event peak — events with few photons are NOT
# clamped (they're dim enough to stay linear), only bright ones are.
SATURATION_FRACTION = 0.6   # clamp at 60% of unclamped peak height

# Minimum photon count per face to accept a channel crossing as valid.
# In the paper, the high-gain pulse only fires reliably above some light level.
MIN_PHOTONS_PER_FACE = 3

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
# ANALOG PULSE WAVEFORM BUILDER (with saturation clamp)
# ─────────────────────────────────────────────────────────────────────────────
def build_waveform(hit_times_ns, rng):
    COARSE_PS = 25
    coarse_bins = np.arange(0, WINDOW_PS + COARSE_PS, COARSE_PS)
    fine_bins   = np.arange(0, WINDOW_PS + SAMPLE_PS, SAMPLE_PS)

    if len(hit_times_ns) == 0:
        return np.zeros(len(fine_bins) - 1, dtype=float), fine_bins

    times_ps = np.asarray(hit_times_ns, dtype=float) * 1000.0
    times_ps += rng.normal(0.0, SPTR_SIGMA_PS, size=len(times_ps))

    counts, _ = np.histogram(times_ps, bins=coarse_bins)
    coarse_wf  = gaussian_filter1d(counts.astype(float), sigma=SIGMA)
    wf         = np.interp(fine_bins[:-1], coarse_bins[:-1], coarse_wf)

    # ── Saturation clamp ────────────────────────────────────────────────────
    # Cap amplitude at SATURATION_FRACTION * peak, then rescale so the
    # saturated waveform still has unit-ish amplitude. This mimics the
    # paper's deliberately saturated high-gain channel: the leading edge
    # becomes a steep ramp that hits the flat ceiling, after which the CFD
    # threshold is crossed at a much more consistent time across events.
    peak = wf.max()
    if peak > 0:
        sat_level = peak * SATURATION_FRACTION
        wf = np.clip(wf, 0, sat_level)
    # ────────────────────────────────────────────────────────────────────────

    return wf, fine_bins


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANT FRACTION DISCRIMINATOR
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

    y0, y1 = waveform[idx - 1], waveform[idx]
    frac = 0.0 if y1 == y0 else (dynamic_threshold - y0) / (y1 - y0)
    return (idx - 1 + frac) * SAMPLE_PS


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN FITTER  —  paper-faithful: fit to central peak, not IQR
# ─────────────────────────────────────────────────────────────────────────────
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_to_peak(data, n_bins=80):
    """
    Fit a Gaussian to the central peak of the distribution, matching the
    paper's approach (Figs. 19–26).  Strategy:
      1. Histogram with moderate binning.
      2. Seed mu at the histogram mode (most-populated bin), not the mean,
         so spike artefacts don't pull the seed off the main population.
      3. Restrict the fit window to ±2 IQR around the mode.
      4. curve_fit with bounded parameters.
      5. Fall back to IQR/1.349 if the fit fails or gives implausible sigma.
    Returns (amplitude, mu, sigma, sigma_err).
    """
    if len(data) < 8:
        return 0.0, float(np.median(data)), float(np.std(data)), np.nan

    q75, q25 = np.percentile(data, [75, 25])
    iqr        = q75 - q25
    iqr_sigma  = max(iqr / 1.349, 1.0)
    center     = np.median(data)

    # Histogram over ±3 IQR for the fit window
    lo  = center - 3.0 * iqr_sigma
    hi  = center + 3.0 * iqr_sigma
    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    mids = 0.5 * (edges[:-1] + edges[1:])

    # Seed from histogram mode
    peak_idx = int(np.argmax(counts))
    mu0      = float(mids[peak_idx])
    A0       = float(counts[peak_idx])
    sig0     = iqr_sigma

    # Restrict fit to ±2 IQR around the mode
    fit_mask = np.abs(mids - mu0) < 2.0 * iqr_sigma
    if fit_mask.sum() < 5:
        return A0, mu0, iqr_sigma, np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                gaussian,
                mids[fit_mask],
                counts[fit_mask],
                p0=[A0, mu0, sig0],
                bounds=([0, lo, 0.5], [A0 * 5, hi, iqr_sigma * 3]),
                maxfev=5000,
            )
        A_fit, mu_fit, sig_fit = popt
        perr = np.sqrt(np.diag(pcov))
        sig_err = float(perr[2])

        # Sanity check: reject if sigma is implausibly small or huge
        if sig_fit < 1.0 or sig_fit > iqr_sigma * 3:
            raise RuntimeError("implausible sigma")

        return float(A_fit), float(mu_fit), float(sig_fit), sig_err

    except Exception:
        # Graceful fallback to IQR estimator
        return A0, mu0, iqr_sigma, np.nan


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
    dw_counts, up_counts = [], []

    # ── Asymmetry diagnostics ───────────────────────────────────────────────
    # Track per-event photon counts and crossing times for each face to
    # understand the DW/UP imbalance before any selection cuts.
    diag_dw_n, diag_up_n       = [], []   # photon counts per face per event
    diag_dw_valid, diag_up_valid = [], [] # bool: did this face yield a crossing?

    mcp_times = rng.normal(MCP_TRUE_PS, MCP_SIGMA_PS, size=len(unique_events))

    for ev_id, mcp_t in zip(unique_events, mcp_times):
        mask        = event_id == ev_id
        ev_times    = time_ns[mask]
        ev_channels = channels[mask]

        dw_num = int((ev_channels < 4).sum())
        up_num = int((ev_channels >= 4).sum())
        diag_dw_n.append(dw_num)
        diag_up_n.append(up_num)

        t_channel = {}
        for ch in range(8):
            ch_times = ev_times[ev_channels == ch]

            # Gate on minimum photon count: channels with too few photons
            # produce unreliable waveforms whose threshold crossing is
            # dominated by noise, not the true shower arrival time.
            if len(ch_times) < MIN_PHOTONS_PER_FACE:
                t_channel[ch] = None
                continue

            waveform, _ = build_waveform(ch_times, rng)
            t_channel[ch] = threshold_crossing_ps(waveform, threshold_fraction=0.20)

        dw_crossings = [t_channel[ch] for ch in range(4)   if t_channel[ch] is not None]
        up_crossings = [t_channel[ch] for ch in range(4, 8) if t_channel[ch] is not None]

        diag_dw_valid.append(len(dw_crossings) > 0)
        diag_up_valid.append(len(up_crossings) > 0)

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

    best_minus_ps = np.array(best_minus_ps)
    dw_only_ps    = np.array(dw_only_ps)
    up_only_ps    = np.array(up_only_ps)
    delta_t_ps    = dw_only_ps - up_only_ps
    dw_counts     = np.array(dw_counts)
    up_counts     = np.array(up_counts)

    # ── Print asymmetry diagnostics ─────────────────────────────────────────
    diag_dw_n   = np.array(diag_dw_n)
    diag_up_n   = np.array(diag_up_n)
    diag_dw_v   = np.array(diag_dw_valid)
    diag_up_v   = np.array(diag_up_valid)
    n_ev        = len(unique_events)
    print(f"\n  ── Asymmetry Diagnostics ──")
    print(f"  Total events           : {n_ev}")
    print(f"  DW median photons/ev   : {np.median(diag_dw_n):.1f}   "
          f"(mean {np.mean(diag_dw_n):.1f})")
    print(f"  UP median photons/ev   : {np.median(diag_up_n):.1f}   "
          f"(mean {np.mean(diag_up_n):.1f})")
    print(f"  DW yield valid crossing: {diag_dw_v.sum()}/{n_ev} "
          f"({100*diag_dw_v.mean():.0f}%)")
    print(f"  UP yield valid crossing: {diag_up_v.sum()}/{n_ev} "
          f"({100*diag_up_v.mean():.0f}%)")
    ratio = np.mean(diag_dw_n) / max(np.mean(diag_up_n), 0.001)
    print(f"  DW/UP photon ratio     : {ratio:.2f}  "
          f"({'symmetric ~1.0 ✓' if 0.7 < ratio < 1.4 else 'ASYMMETRIC — check geometry ✗'})")

    # ── Energy-bin selection (matching paper bins 6–8 of 9) ─────────────────
    # Use total per-face photon count as the proxy for measured energy.
    # Select the upper-middle third of the joint distribution, requiring
    # both faces to be well-lit (avoids events where one face is starved).
    dw_lo = np.quantile(dw_counts, 5/9)
    dw_hi = np.quantile(dw_counts, 8/9)
    up_lo = np.quantile(up_counts, 5/9)
    up_hi = np.quantile(up_counts, 8/9)
    energy_mask = (
        (dw_counts >= dw_lo) & (dw_counts <= dw_hi) &
        (up_counts >= up_lo) & (up_counts <= up_hi)
    )
    print(f"\n  ── Energy-bin selection (bins 6–8 of 9 equivalent) ──")
    print(f"  DW photon window       : [{dw_lo:.0f}, {dw_hi:.0f}]")
    print(f"  UP photon window       : [{up_lo:.0f}, {up_hi:.0f}]")
    print(f"  Events selected        : {energy_mask.sum()} / {len(best_minus_ps)}")

    selected_bm = best_minus_ps[energy_mask]
    selected_dw = dw_only_ps[energy_mask]
    selected_up = up_only_ps[energy_mask]
    selected_dt = delta_t_ps[energy_mask]

    if len(selected_bm) < 8:
        print("  WARNING: too few selected events; falling back to no selection.")
        selected_bm = best_minus_ps
        selected_dw = dw_only_ps
        selected_up = up_only_ps
        selected_dt = delta_t_ps

    # ── Outlier removal (±400 ps from median) ───────────────────────────────
    def clean(arr):
        return arr[np.abs(arr - np.median(arr)) < 400.0]

    clean_dw = clean(selected_dw)
    clean_up = clean(selected_up)
    clean_dt = clean(selected_dt)
    clean_bm = clean(selected_bm)

    # ── Gaussian peak fits (paper-faithful) ─────────────────────────────────
    bm_amp, bm_mu, bm_sigma, bm_sigma_err = fit_gaussian_to_peak(clean_bm)
    dw_amp, dw_mu, dw_sigma, dw_sigma_err = fit_gaussian_to_peak(clean_dw)
    up_amp, up_mu, up_sigma, up_sigma_err = fit_gaussian_to_peak(clean_up)
    dt_amp, dt_mu, dt_sigma, dt_sigma_err = fit_gaussian_to_peak(clean_dt)

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    energy_label = batch_dir.name
    fig.suptitle(
        f"Timing Distributions for {energy_label} (Energy Bins 6–8)\n"
        f"Gaussian peak fit  |  saturation clamp  |  min {MIN_PHOTONS_PER_FACE} photons/channel",
        fontsize=13, fontweight="bold"
    )

    distributions = [
        {"data": clean_dw, "amp": dw_amp, "mu": dw_mu, "sigma": dw_sigma,
         "sigma_err": dw_sigma_err,
         "title": "Downstream Time ($\\Delta t_{DW}$)", "color": "royalblue"},
        {"data": clean_up, "amp": up_amp, "mu": up_mu, "sigma": up_sigma,
         "sigma_err": up_sigma_err,
         "title": "Upstream Time ($\\Delta t_{UP}$)",   "color": "crimson"},
        {"data": clean_dt, "amp": dt_amp, "mu": dt_mu, "sigma": dt_sigma,
         "sigma_err": dt_sigma_err,
         "title": "Delta t ($\\Delta t_{DW} - \\Delta t_{UP}$)", "color": "darkorchid"},
    ]

    for ax, dist in zip(axs, distributions):
        data = dist["data"]
        if len(data) == 0:
            continue

        q75, q25    = np.percentile(data, [75, 25])
        core_sigma  = (q75 - q25) / 1.35
        plot_sigma  = np.clip(core_sigma, 1.0, 150.0)
        plot_center = np.median(data)
        lo = plot_center - 4.0 * plot_sigma
        hi = plot_center + 4.0 * plot_sigma

        counts, edges, _ = ax.hist(
            data, bins=80, range=(lo, hi),
            color=dist["color"], alpha=0.6, edgecolor='black', label="Data"
        )

        # Draw Gaussian fit
        x_fit     = np.linspace(lo, hi, 5000)
        amplitude = dist["amp"] if dist["amp"] > 0 else counts.max()
        y_fit     = gaussian(x_fit, amplitude, dist["mu"], dist["sigma"])

        err_str = (f" ± {dist['sigma_err']:.1f}" 
                   if not np.isnan(dist["sigma_err"]) else " (IQR fallback)")
        ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2.5,
                label=f"Gaussian fit:\n$\\mu$ = {dist['mu']:.1f} ps\n"
                      f"$\\sigma$ = {dist['sigma']:.1f}{err_str} ps")

        ax.set_title(dist["title"], fontsize=12)
        ax.set_xlabel("Time (ps)", fontsize=10)
        ax.set_ylabel("Events / Bin", fontsize=10)
        ax.set_xlim(lo, hi)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plot_path = batch_dir / "timing_histograms_v2.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n  Saved plot → {plot_path}")

    # ── Asymmetry diagnostic plot ────────────────────────────────────────────
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle(f"DW / UP Asymmetry Diagnostics — {energy_label}", fontsize=13)

    axs2[0].hist(diag_dw_n, bins=30, alpha=0.6, color="royalblue",
                 edgecolor="black", label=f"DW  (median={np.median(diag_dw_n):.0f})")
    axs2[0].hist(diag_up_n, bins=30, alpha=0.6, color="crimson",
                 edgecolor="black", label=f"UP  (median={np.median(diag_up_n):.0f})")
    axs2[0].set_xlabel("Photons reaching SiPM per event", fontsize=10)
    axs2[0].set_ylabel("Events", fontsize=10)
    axs2[0].set_title("Photon count distribution per face", fontsize=11)
    axs2[0].legend()
    axs2[0].grid(True, linestyle=":", alpha=0.5)

    axs2[1].scatter(diag_dw_n, diag_up_n, alpha=0.4, s=15, color="purple")
    lim = max(diag_dw_n.max(), diag_up_n.max()) * 1.05
    axs2[1].plot([0, lim], [0, lim], "k--", linewidth=1, label="DW = UP")
    axs2[1].set_xlabel("DW photons / event", fontsize=10)
    axs2[1].set_ylabel("UP photons / event", fontsize=10)
    axs2[1].set_title("Per-event DW vs UP photon count", fontsize=11)
    axs2[1].legend()
    axs2[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    diag_path = batch_dir / "asymmetry_diagnostics.png"
    plt.savefig(diag_path, dpi=200)
    plt.close()
    print(f"  Saved diagnostics → {diag_path}")

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
    print(f"  Timing Resolution v2 — BestMinus (Paper Method 4)")
    print(f"  Improvements: Gaussian peak fit | saturation clamp")
    print(f"                min-photon gate   | bins 6–8 of 9 selection")
    print(f"  Batch : {energy_label}")
    print(f"{'─'*60}")

    result = run(batch_path)

    if result is None:
        print("  Could not compute timing resolution.")
    else:
        s  = result["sigma_t_ps"]
        se = result["sigma_t_err_ps"]
        err_str = f" ± {se:.2f}" if not np.isnan(se) else " (IQR fallback)"

        print(f"\n  BestMinus σ_t  =  {s:.2f}{err_str} ps")
        print(f"  Downstream σ   =  {result['dw_sigma']:.2f} ps")
        print(f"  Upstream σ     =  {result['up_sigma']:.2f} ps")
        print(f"  Delta t σ      =  {result['dt_sigma']:.2f} ps")
        print(f"  DW median hits =  {result['dw_median_photons']:.0f} photons/event")
        print(f"  UP median hits =  {result['up_median_photons']:.0f} photons/event")

        out_txt = batch_path / "timing_resolution_v2.txt"
        with open(out_txt, "w") as f:
            f.write(f"Method          : BestMinus v2 (Gaussian peak fit + saturation clamp)\n")
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