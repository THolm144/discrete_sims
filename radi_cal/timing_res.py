"""
timing_res.py
=============
Timing resolution analysis for the RADiCAL Shashlik calorimeter (radi_cal world).
Implements the BestMinus method (Method 4) from Perez-Lara et al. NIM A 1068 (2024).

Channel mapping (matches radi_cal.py DETECTOR_VOLUME_NAMES):
    0: sipm_front_0  (upstream,   cap 0  [+x, +y])
    1: sipm_front_1  (upstream,   cap 1  [+x, -y])
    2: sipm_front_2  (upstream,   cap 2  [-x, +y])
    3: sipm_front_3  (upstream,   cap 3  [-x, -y])
    4: sipm_back_0   (downstream, cap 0  [+x, +y])
    5: sipm_back_1   (downstream, cap 1  [+x, -y])
    6: sipm_back_2   (downstream, cap 2  [-x, +y])
    7: sipm_back_3   (downstream, cap 3  [-x, -y])

All 4 caps are treated symmetrically (T-type geometry throughout).
"""

import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import numpy as np
import uproot
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY CONSTANTS  —  matched to radi_cal.py
# ─────────────────────────────────────────────────────────────────────────────

_LYSO_XY_MM     = 14.0
_LYSO_THICK_MM  = 1.5
_TYVEK_THICK_MM = 0.008 * 25.4          # 0.2032 mm
_W_THICK_MM     = 2.5
_N_LYSO         = 29
_N_W            = 28
_GAP_THICK_MM   = _LYSO_THICK_MM + 2 * _TYVEK_THICK_MM   # 1.9064 mm
_CALOR_THICK_MM = _N_LYSO * _GAP_THICK_MM + _N_W * _W_THICK_MM  # 125.2856 mm
_CAP_LENGTH_MM  = 183.0
_SIPM_THICK_MM  = 0.3
_SIPM_Z_MM      = _CAP_LENGTH_MM / 2 + _SIPM_THICK_MM / 2   # 91.65 mm

_CALOR_XY_MM    = _LYSO_XY_MM + 2 * _TYVEK_THICK_MM         # 14.4064 mm
_HOLE_INSET_MM  = 3.5
_HOLE_OFFSET    = _CALOR_XY_MM / 2 - _HOLE_INSET_MM          # 3.7032 mm

# Cap positions matching radi_cal.py _CAP_POSITIONS_MM order exactly
CAP_XY_MM = np.array([
    [ _HOLE_OFFSET,  _HOLE_OFFSET],   # cap 0  [+x, +y]
    [ _HOLE_OFFSET, -_HOLE_OFFSET],   # cap 1  [+x, -y]
    [-_HOLE_OFFSET,  _HOLE_OFFSET],   # cap 2  [-x, +y]
    [-_HOLE_OFFSET, -_HOLE_OFFSET],   # cap 3  [-x, -y]
])

# z-tolerance for SiPM face identification (mm)
_SIPM_Z_TOL_MM  = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# DIGITIZER & ELECTRONICS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_PS           = 2        # digitizer sample period (ps)
WINDOW_PS           = 50_000   # readout window (ps)
SPTR_SIGMA_PS       = 150      # SiPM single-photon timing resolution (ps)
MCP_SIGMA_PS        = 15       # MCP reference jitter (ps)
MCP_TRUE_PS         = 0.0      # true MCP time (ps)
FACE_JITTER_PS      = 18.5     # per-face electronics jitter (ps)
SIGMA               = 10       # Gaussian smooth kernel (bins on 25 ps coarse grid)
SATURATION_FRACTION = 0.6      # clamp at 60% of unclamped peak (mimics paper saturation)
MIN_PHOTONS_PER_FACE = 3       # minimum photons for a valid waveform crossing


# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_channel(x_mm, y_mm, z_mm):
    """
    Map hit position → detector index (0–7).
    Front (upstream)   = z < 0 → channels 0–3
    Back  (downstream) = z > 0 → channels 4–7
    Returns -1 if not on a SiPM face.
    """
    if abs(abs(z_mm) - _SIPM_Z_MM) > _SIPM_Z_TOL_MM:
        return -1
    dists   = np.hypot(CAP_XY_MM[:, 0] - x_mm, CAP_XY_MM[:, 1] - y_mm)
    cap_idx = int(np.argmin(dists))
    return cap_idx if z_mm < 0 else cap_idx + 4


# ─────────────────────────────────────────────────────────────────────────────
# WAVEFORM BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_waveform(hit_times_ns, rng):
    """
    Build a digitised analog waveform from photon arrival times.
    Applies SPTR jitter, coarse histogram, Gaussian smoothing,
    fine interpolation, and saturation clamping.
    """
    COARSE_PS = 25
    coarse_bins = np.arange(0, WINDOW_PS + COARSE_PS, COARSE_PS)
    fine_bins   = np.arange(0, WINDOW_PS + SAMPLE_PS, SAMPLE_PS)

    if len(hit_times_ns) == 0:
        return np.zeros(len(fine_bins) - 1, dtype=float), fine_bins

    times_ps  = np.asarray(hit_times_ns, dtype=float) * 1000.0
    times_ps += rng.normal(0.0, SPTR_SIGMA_PS, size=len(times_ps))

    counts, _  = np.histogram(times_ps, bins=coarse_bins)
    coarse_wf  = gaussian_filter1d(counts.astype(float), sigma=SIGMA)
    wf         = np.interp(fine_bins[:-1], coarse_bins[:-1], coarse_wf)

    peak = wf.max()
    if peak > 0:
        wf = np.clip(wf, 0, peak * SATURATION_FRACTION)

    return wf, fine_bins


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANT FRACTION DISCRIMINATOR
# ─────────────────────────────────────────────────────────────────────────────

def threshold_crossing_ps(waveform, threshold_fraction=0.20):
    peak = np.max(waveform)
    if peak < 0.1:
        return None

    thr      = peak * threshold_fraction
    crossings = np.where(waveform >= thr)[0]
    if len(crossings) == 0:
        return None

    idx = crossings[0]
    if idx == 0:
        return 0.0

    y0, y1 = waveform[idx - 1], waveform[idx]
    frac   = 0.0 if y1 == y0 else (thr - y0) / (y1 - y0)
    return (idx - 1 + frac) * SAMPLE_PS


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN FITTER
# ─────────────────────────────────────────────────────────────────────────────

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_to_peak(data, n_bins=80):
    if len(data) < 8:
        return 0.0, float(np.median(data)), float(np.std(data)), np.nan

    q75, q25  = np.percentile(data, [75, 25])
    iqr        = q75 - q25
    iqr_sigma  = max(iqr / 1.349, 1.0)
    center     = np.median(data)

    lo = center - 3.0 * iqr_sigma
    hi = center + 3.0 * iqr_sigma
    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    mids = 0.5 * (edges[:-1] + edges[1:])

    peak_idx = int(np.argmax(counts))
    mu0      = float(mids[peak_idx])
    A0       = float(counts[peak_idx])
    sig0     = iqr_sigma

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
        perr    = np.sqrt(np.diag(pcov))
        sig_err = float(perr[2])

        if sig_fit < 1.0 or sig_fit > iqr_sigma * 3:
            raise RuntimeError("implausible sigma")

        return float(A_fit), float(mu_fit), float(sig_fit), sig_err

    except Exception:
        return A0, mu0, iqr_sigma, np.nan


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_hits(batch_dir: Path):
    """
    Load all detector_hits_*.root files under batch_dir.
    Tries LocalTime first, falls back to GlobalTime if absent.
    Returns arrays: event_id, x_mm, y_mm, z_mm, time_ns, particle.
    """
    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"))
    if not hit_files:
        return None

    all_event_id = []
    all_x, all_y, all_z = [], [], []
    all_time_ns  = []
    all_particle = []

    grouped = defaultdict(list)
    for fpath in hit_files:
        grouped[fpath.parent].append(fpath)

    global_offset = 0
    for parent_dir, files in grouped.items():
        max_ev = 0
        for fpath in files:
            try:
                with uproot.open(fpath) as f:
                    file_keys = f.keys()
                    # FIX: Handle empty ROOT files gracefully
                    if not file_keys:
                        print(f"  Warning: {fpath.name} is completely empty (no keys/trees found).")
                        continue

                    key  = file_keys[0].split(";")[0]
                    tree = f[key]
                    branches = tree.keys()

                    ev = tree["EventID"].array(library="np").astype(int)
                    
                    # Check if tree actually has entries
                    if len(ev) == 0:
                        print(f"  Warning: TTree in {fpath.name} contains zero entries.")
                        continue

                    x  = tree["Position_X"].array(library="np")
                    y  = tree["Position_Y"].array(library="np")
                    z  = tree["Position_Z"].array(library="np")
                    pn = tree["ParticleName"].array(library="np")

                    # LocalTime preferred; GlobalTime fallback
                    if "LocalTime" in branches:
                        t = tree["LocalTime"].array(library="np")
                    elif "GlobalTime" in branches:
                        print(f"  [fallback] LocalTime absent in {fpath.name} — using GlobalTime")
                        t = tree["GlobalTime"].array(library="np")
                    else:
                        print(f"  Warning: no time branch in {fpath.name} — skipping")
                        continue

                    all_event_id.append(ev + global_offset)
                    all_x.append(x);  all_y.append(y);  all_z.append(z)
                    all_time_ns.append(t)
                    all_particle.append(pn)

                    max_ev = max(max_ev, int(ev.max()))

            except Exception as e:
                print(f"  Warning: could not read {fpath.name}: {e}")

        global_offset += max_ev + 1

    if not all_event_id:
        return None

    return (
        np.concatenate(all_event_id),
        np.concatenate(all_x),
        np.concatenate(all_y),
        np.concatenate(all_z),
        np.concatenate(all_time_ns),
        np.concatenate(all_particle),
    )

    return (
        np.concatenate(all_event_id),
        np.concatenate(all_x),
        np.concatenate(all_y),
        np.concatenate(all_z),
        np.concatenate(all_time_ns),
        np.concatenate(all_particle),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run(batch_dir: Path):
    rng = np.random.default_rng(seed=42)

    loaded = load_hits(batch_dir)
    if loaded is None:
        print("  No detector_hits_*.root files found.")
        return None

    event_id, x_mm, y_mm, z_mm, time_ns, particle = loaded

    # ── Filter to optical photons only ───────────────────────────────────────
    is_optical = (particle == b"opticalphoton") | (particle == "opticalphoton")
    event_id   = event_id[is_optical]
    x_mm       = x_mm[is_optical]
    y_mm       = y_mm[is_optical]
    z_mm       = z_mm[is_optical]
    time_ns    = time_ns[is_optical]

    # ── Assign channels ───────────────────────────────────────────────────────
    channels  = np.array([assign_channel(x, y, z) for x, y, z in zip(x_mm, y_mm, z_mm)])
    on_sipm   = channels >= 0
    event_id  = event_id[on_sipm]
    time_ns   = time_ns[on_sipm]
    channels  = channels[on_sipm]

    unique_events = np.unique(event_id)
    mcp_times     = rng.normal(MCP_TRUE_PS, MCP_SIGMA_PS, size=len(unique_events))

    best_minus_ps, dw_only_ps, up_only_ps = [], [], []
    dw_counts, up_counts = [], []

    diag_dw_n, diag_up_n         = [], []
    diag_dw_valid, diag_up_valid  = [], []

    for ev_id, mcp_t in zip(unique_events, mcp_times):
        mask        = event_id == ev_id
        ev_times    = time_ns[mask]
        ev_channels = channels[mask]

        # front = channels 0–3 (upstream), back = channels 4–7 (downstream)
        up_num = int((ev_channels < 4).sum())
        dw_num = int((ev_channels >= 4).sum())
        diag_up_n.append(up_num)
        diag_dw_n.append(dw_num)

        t_channel = {}
        for ch in range(8):
            ch_times = ev_times[ev_channels == ch]
            if len(ch_times) < MIN_PHOTONS_PER_FACE:
                t_channel[ch] = None
                continue
            waveform, _ = build_waveform(ch_times, rng)
            t_channel[ch] = threshold_crossing_ps(waveform)

        up_crossings = [t_channel[ch] for ch in range(0, 4) if t_channel[ch] is not None]
        dw_crossings = [t_channel[ch] for ch in range(4, 8) if t_channel[ch] is not None]

        diag_up_valid.append(len(up_crossings) > 0)
        diag_dw_valid.append(len(dw_crossings) > 0)

        if not dw_crossings or not up_crossings:
            continue

        delta_up = np.mean(up_crossings) - mcp_t
        delta_dw = np.mean(dw_crossings) - mcp_t

        delta_up += rng.normal(0.0, FACE_JITTER_PS)
        delta_dw += rng.normal(0.0, FACE_JITTER_PS)

        best_minus_ps.append((delta_dw - delta_up) / 2.0)
        up_only_ps.append(delta_up)
        dw_only_ps.append(delta_dw)
        up_counts.append(up_num)
        dw_counts.append(dw_num)

    best_minus_ps = np.array(best_minus_ps)
    dw_only_ps    = np.array(dw_only_ps)
    up_only_ps    = np.array(up_only_ps)
    delta_t_ps    = dw_only_ps - up_only_ps
    dw_counts     = np.array(dw_counts)
    up_counts     = np.array(up_counts)
    diag_dw_n     = np.array(diag_dw_n)
    diag_up_n     = np.array(diag_up_n)
    diag_dw_valid = np.array(diag_dw_valid)
    diag_up_valid = np.array(diag_up_valid)
    n_ev          = len(unique_events)

    # ── Asymmetry diagnostics ─────────────────────────────────────────────────
    print(f"\n  ── Asymmetry Diagnostics ──")
    print(f"  Total events           : {n_ev}")
    print(f"  UP median photons/ev   : {np.median(diag_up_n):.1f}  (mean {np.mean(diag_up_n):.1f})")
    print(f"  DW median photons/ev   : {np.median(diag_dw_n):.1f}  (mean {np.mean(diag_dw_n):.1f})")
    print(f"  UP yield valid crossing: {diag_up_valid.sum()}/{n_ev} ({100*diag_up_valid.mean():.0f}%)")
    print(f"  DW yield valid crossing: {diag_dw_valid.sum()}/{n_ev} ({100*diag_dw_valid.mean():.0f}%)")
    ratio = np.mean(diag_dw_n) / max(np.mean(diag_up_n), 0.001)
    sym   = "symmetric ~1.0 ✓" if 0.7 < ratio < 1.4 else "ASYMMETRIC — check geometry ✗"
    print(f"  DW/UP photon ratio     : {ratio:.2f}  ({sym})")

    # ── Energy-bin selection (bins 6–8 of 9) ─────────────────────────────────
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
        print("  WARNING: too few selected events — falling back to no selection.")
        selected_bm = best_minus_ps
        selected_dw = dw_only_ps
        selected_up = up_only_ps
        selected_dt = delta_t_ps

    # ── Outlier removal ───────────────────────────────────────────────────────
    def clean(arr):
        return arr[np.abs(arr - np.median(arr)) < 400.0]

    clean_bm = clean(selected_bm)
    clean_dw = clean(selected_dw)
    clean_up = clean(selected_up)
    clean_dt = clean(selected_dt)

    # ── Gaussian fits ─────────────────────────────────────────────────────────
    bm_amp, bm_mu, bm_sigma, bm_sigma_err = fit_gaussian_to_peak(clean_bm)
    dw_amp, dw_mu, dw_sigma, dw_sigma_err = fit_gaussian_to_peak(clean_dw)
    up_amp, up_mu, up_sigma, up_sigma_err = fit_gaussian_to_peak(clean_up)
    dt_amp, dt_mu, dt_sigma, dt_sigma_err = fit_gaussian_to_peak(clean_dt)

    # ── Timing plots ──────────────────────────────────────────────────────────
    energy_label = batch_dir.name
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Timing Distributions — {energy_label}  (Energy Bins 6–8)\n"
        f"BestMinus method  |  saturation clamp  |  min {MIN_PHOTONS_PER_FACE} photons/channel",
        fontsize=13, fontweight="bold",
    )

    distributions = [
        {"data": clean_dw, "amp": dw_amp, "mu": dw_mu, "sigma": dw_sigma,
         "sigma_err": dw_sigma_err,
         "title": "Downstream ($\\Delta t_{DW}$)", "color": "royalblue"},
        {"data": clean_up, "amp": up_amp, "mu": up_mu, "sigma": up_sigma,
         "sigma_err": up_sigma_err,
         "title": "Upstream ($\\Delta t_{UP}$)", "color": "crimson"},
        {"data": clean_dt, "amp": dt_amp, "mu": dt_mu, "sigma": dt_sigma,
         "sigma_err": dt_sigma_err,
         "title": "$\\Delta t_{DW} - \\Delta t_{UP}$", "color": "darkorchid"},
    ]

    for ax, dist in zip(axs, distributions):
        data = dist["data"]
        if len(data) == 0:
            ax.set_title(dist["title"])
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        q75, q25   = np.percentile(data, [75, 25])
        core_sigma = max((q75 - q25) / 1.35, 1.0)
        center     = np.median(data)
        lo = center - 4.0 * core_sigma
        hi = center + 4.0 * core_sigma

        counts, edges, _ = ax.hist(
            data, bins=80, range=(lo, hi),
            color=dist["color"], alpha=0.6, edgecolor="black", label="Data",
        )
        x_fit = np.linspace(lo, hi, 5000)
        amp   = dist["amp"] if dist["amp"] > 0 else counts.max()
        y_fit = gaussian(x_fit, amp, dist["mu"], dist["sigma"])
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
    plot_path = batch_dir / "timing_histograms.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n  Saved plot → {plot_path}")

    # ── Asymmetry diagnostic plot ─────────────────────────────────────────────
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle(f"DW / UP Asymmetry Diagnostics — {energy_label}", fontsize=13)

    axs2[0].hist(diag_dw_n, bins=30, alpha=0.6, color="royalblue", edgecolor="black",
                 label=f"DW  (median={np.median(diag_dw_n):.0f})")
    axs2[0].hist(diag_up_n, bins=30, alpha=0.6, color="crimson",   edgecolor="black",
                 label=f"UP  (median={np.median(diag_up_n):.0f})")
    axs2[0].set_xlabel("Photons reaching SiPM per event", fontsize=10)
    axs2[0].set_ylabel("Events", fontsize=10)
    axs2[0].set_title("Photon count per face", fontsize=11)
    axs2[0].legend()
    axs2[0].grid(True, linestyle=":", alpha=0.5)

    lim = max(diag_dw_n.max(), diag_up_n.max()) * 1.05
    axs2[1].scatter(diag_dw_n, diag_up_n, alpha=0.4, s=15, color="purple")
    axs2[1].plot([0, lim], [0, lim], "k--", linewidth=1, label="DW = UP")
    axs2[1].set_xlabel("DW photons / event", fontsize=10)
    axs2[1].set_ylabel("UP photons / event", fontsize=10)
    axs2[1].set_title("Per-event DW vs UP", fontsize=11)
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
    parser = argparse.ArgumentParser(
        description="BestMinus timing resolution for radi_cal world"
    )
    parser.add_argument("--batch-dir", type=str, required=True)
    args = parser.parse_args()

    batch_path   = Path(args.batch_dir)
    energy_label = batch_path.name

    print(f"\n{'─'*60}")
    print(f"  Timing Resolution — BestMinus (Perez-Lara Method 4)")
    print(f"  Batch : {energy_label}")
    print(f"{'─'*60}")

    result = run(batch_path)

    if result is None:
        print("  Could not compute timing resolution.")
    else:
        s   = result["sigma_t_ps"]
        se  = result["sigma_t_err_ps"]
        err = f" ± {se:.2f}" if not np.isnan(se) else " (IQR fallback)"

        print(f"\n  BestMinus σ_t  =  {s:.2f}{err} ps")
        print(f"  Downstream σ   =  {result['dw_sigma']:.2f} ps")
        print(f"  Upstream σ     =  {result['up_sigma']:.2f} ps")
        print(f"  Delta t σ      =  {result['dt_sigma']:.2f} ps")
        print(f"  DW median hits =  {result['dw_median_photons']:.0f} photons/event")
        print(f"  UP median hits =  {result['up_median_photons']:.0f} photons/event")

        out_txt = batch_path / "timing_resolution.txt"
        with open(out_txt, "w") as f:
            f.write(f"Method          : BestMinus (Perez-Lara NIM A 1068 2024)\n")
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