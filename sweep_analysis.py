#!/usr/bin/env python3
import os
import sys
import argparse
"""
unified_sweep_analysis.py
==========================
Aggregates timing-resolution and ToF-reconstruction results across all four
RADiCAL geometry variants (radi_cal_energy, radi_cal_triple, rc_hex,
rc_hex_triple) and all energy points in each module's latest sweep.

Outputs (in analysis/sweep_summary_<timestamp>/):
    {module}_timing_panels.png   -- 4-panel BestMinus timing histograms
    {module}_tof_panels.png      -- 4-panel E-type ToF longitudinal reconstruction
    {module}_dw_e_hits_time.png  -- Downstream E-type SiPM hits vs Time
    {module}_dw_prompt_distance_cropped.png -- Downstream pseudo-distance kinematics
    {module}_intensity_ratio.png -- T-Type vs E-Type intensity shower tracking
    timing_resolution_vs_energy.png -- sigma_t vs energy, all 4 modules, error bars
    timing_vs_energy_report.txt  -- text summary of all sigma_t / sigma_z values
"""
import datetime
import warnings
from pathlib import Path
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

# Suppress runtime optimization warnings from noisy fits
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL KINEMATICS CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS = 299.792
REFRACTIVE_INDEX = 1.60
V_LIGHT_MM_NS = C_LIGHT_MM_NS / REFRACTIVE_INDEX
BOUNCE_FACTOR = 0.92
V_EFF_MM_NS = V_LIGHT_MM_NS * BOUNCE_FACTOR

_GT_LO_NS = 0.0
_GT_HI_NS = 50.0
_TYVEK_THICK_MM = 0.2032
_W_THICK_MM = 2.5
_N_LYSO = 29
_N_W = 28

ARRIVAL_QUANTILE = 0.10
MIN_PHOTONS_PER_FACE = 1

# ── Geometry mappings ──────────────────────────────────────────────────────
_SQUARE_HOLE_OFFSET = 3.7032
SQUARE_CAP_XY = np.array([
    [ _SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],  # 0 (T)
    [-_SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],  # 1 (T)
    [-_SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],  # 2 (E)
    [ _SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],  # 3 (E)
])

HEX_CAP_R_MM = 3.5
HEX_CAP_XY = np.array([
    [HEX_CAP_R_MM * np.cos(np.pi/2 + i*(np.pi/3)), HEX_CAP_R_MM * np.sin(np.pi/2 + i*(np.pi/3))]
    for i in range(6)
])

_KNOWN_Z_PLANES = {91.65: 1.5, 135.15: 4.5}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_lyso_layer_bounds(lyso_thick, calor_thick):
    gap_thick = lyso_thick + 2 * _TYVEK_THICK_MM
    bounds = []
    current_z = -calor_thick / 2
    for idx in range(_N_LYSO):
        z_start = current_z + _TYVEK_THICK_MM
        z_end = z_start + lyso_thick
        bounds.append((z_start, z_end))
        current_z += gap_thick + (_W_THICK_MM if idx < _N_W else 0)
    return bounds

def standard_gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian_to_peak(data, n_bins=40):
    if len(data) < 8:
        return 0.0, float(np.median(data)) if len(data) else 0.0, float(np.std(data)) if len(data) else 0.0
    center = np.median(data)
    spread = max(np.std(data), 1.0)
    lo, hi = float(np.min(data)), float(np.max(data))
    if hi <= lo:
        hi = lo + 1.0

    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    mids = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(counts.astype(float), sigma=2.0)
    peak_idx = int(np.argmax(smoothed))
    mu0, A0 = float(mids[peak_idx]), float(smoothed[peak_idx])

    try:
        popt, _ = curve_fit(
            standard_gaussian, mids, counts,
            p0=[A0, mu0, spread],
            bounds=([0.0, lo, 1e-6], [A0 * 10.0 + 1.0, hi, (hi - lo)]),
            maxfev=10000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return A0, mu0, spread

def clean_around_mode(arr, window_ps=500.0):
    if len(arr) == 0:
        return arr
    counts, edges = np.histogram(arr, bins=40)
    peak_bin = np.argmax(gaussian_filter1d(counts.astype(float), sigma=2.0))
    mode_center = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
    return arr[np.abs(arr - mode_center) < window_ps]

def extract_numerical_energy(label: str) -> float:
    try:
        return float(''.join(c for c in label if c.isdigit() or c == '.'))
    except ValueError:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE: DATA PARSING & COINCIDENCE FOLDING
# ─────────────────────────────────────────────────────────────────────────────
def analyze_energy_batch(batch_dir: Path, is_hex: bool, verbose_label: str = ""):
    hit_files = sorted(list(batch_dir.rglob("detector_hits_*.root")))
    if not hit_files:
        return None

    detected_z_sensor = None
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk:
                    continue
                z_arr = f[tk]["Position_Z"].array(library="np")
                if len(z_arr) > 0:
                    abs_z = np.abs(z_arr)
                    detected_z_sensor = float(np.median(abs_z[abs_z > (np.max(abs_z) - 5.0)]))
                    break
        except Exception:
            continue

    if detected_z_sensor is None:
        return None

    lyso_thick = min(_KNOWN_Z_PLANES.items(), key=lambda kv: abs(kv[0] - detected_z_sensor))[1]

    gap_thick_mm = lyso_thick + 2 * _TYVEK_THICK_MM
    calor_thick_mm = (_N_LYSO * gap_thick_mm) + (_N_W * _W_THICK_MM)
    lyso_bounds = get_lyso_layer_bounds(lyso_thick, calor_thick_mm)

    cap_xy_map = HEX_CAP_XY if is_hex else SQUARE_CAP_XY
    t_indices = {1, 3, 5} if is_hex else {0, 1}
    e_indices = {0, 2, 4} if is_hex else {2, 3}

    up_first, down_first = {}, {}
    up_times_by_ev, dw_times_by_ev = {}, {}
    all_bm_raw_ps = []
    all_dw_e_times = []  
    
    # Track hit totals for intensity ratio
    dw_e_hits_per_ev = {}
    dw_t_hits_per_ev = {}

    for fpath in hit_files:
        run_tag = fpath.parent.name
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk:
                    continue
                tree = f[tk]
                if tree.num_entries == 0:
                    continue
                x = tree["Position_X"].array(library="np")
                y = tree["Position_Y"].array(library="np")
                z = tree["Position_Z"].array(library="np")
                gt = tree["GlobalTime"].array(library="np")
                lt = tree["LocalTime"].array(library="np")
                ev = tree["EventID"].array(library="np")
                pn = tree["ParticleName"].array(library="np")
        except Exception:
            continue

        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(np.hypot(dx, dy), axis=1)

        near_up = np.abs(z + detected_z_sensor) < 2.5
        near_dw = np.abs(z - detected_z_sensor) < 2.5
        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")

        is_e = np.isin(channels, list(e_indices))
        is_prompt = (gt >= _GT_LO_NS) & (gt <= _GT_HI_NS)
        m_e_up, m_e_dw = is_e & is_prompt & near_up, is_e & is_prompt & near_dw

        all_dw_e_times.extend(gt[m_e_dw].astype(float).tolist())
        
        # Fast event hit counting for E-type
        e_u, e_c = np.unique(ev[m_e_dw], return_counts=True)
        for eid, count in zip(e_u, e_c):
            dw_e_hits_per_ev[(run_tag, int(eid))] = dw_e_hits_per_ev.get((run_tag, int(eid)), 0) + count

        for eid, ti in zip(ev[m_e_up], gt[m_e_up]):
            key = (run_tag, int(eid))
            if key not in up_first or ti < up_first[key]:
                up_first[key] = float(ti)
        for eid, ti in zip(ev[m_e_dw], gt[m_e_dw]):
            key = (run_tag, int(eid))
            if key not in down_first or ti < down_first[key]:
                down_first[key] = float(ti)

        is_t = np.isin(channels, list(t_indices))
        m_t_up, m_t_dw = is_t & is_optical & near_up, is_t & is_optical & near_dw
        
        # Fast event hit counting for T-type
        t_u, t_c = np.unique(ev[m_t_dw], return_counts=True)
        for eid, count in zip(t_u, t_c):
            dw_t_hits_per_ev[(run_tag, int(eid))] = dw_t_hits_per_ev.get((run_tag, int(eid)), 0) + count

        for e, t in zip(ev[m_t_up], lt[m_t_up] * 1000.0):
            up_times_by_ev.setdefault((run_tag, int(e)), []).append(t)
        for e, t in zip(ev[m_t_dw], lt[m_t_dw] * 1000.0):
            dw_times_by_ev.setdefault((run_tag, int(e)), []).append(t)

    common_t_evs = set(up_times_by_ev.keys()) & set(dw_times_by_ev.keys())
    all_bm_raw_ps = []
    for e in common_t_evs:
        if len(up_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE and len(dw_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE:
            t_up_q = np.quantile(up_times_by_ev[e], ARRIVAL_QUANTILE)
            t_dw_q = np.quantile(dw_times_by_ev[e], ARRIVAL_QUANTILE)
            all_bm_raw_ps.append((t_dw_q - t_up_q) / 2.0)

    clean_bm = clean_around_mode(np.array(all_bm_raw_ps), window_ps=500.0)
    _, _, sigma_t_ps = fit_gaussian_to_peak(clean_bm)

    common_e_keys = set(up_first) & set(down_first)
    valid_z_emits = []
    for k in common_e_keys:
        z_est = V_EFF_MM_NS * (down_first[k] - up_first[k]) / 2.0
        if -calor_thick_mm / 2 - 15.0 <= z_est <= calor_thick_mm / 2 + 15.0:
            valid_z_emits.append(z_est)

    profile_counts = np.zeros(_N_LYSO)
    valid_z_emits = np.array(valid_z_emits)
    if len(valid_z_emits) >= 5:
        kde = gaussian_kde(valid_z_emits, bw_method=0.15)
        for i, (zm, zx) in enumerate(lyso_bounds):
            profile_counts[i] = kde.evaluate((zm + zx) / 2.0)[0]
    else:
        for i, (zm, zx) in enumerate(lyso_bounds):
            profile_counts[i] = np.sum((valid_z_emits >= zm) & (valid_z_emits <= zx))
    
    profile_counts = profile_counts[::-1]

    if verbose_label:
        n_runs = len(set(fp.parent.name for fp in hit_files))
        print(f"    [{verbose_label}] {n_runs} run dirs, "
              f"{len(common_t_evs)} T-coincidences, {len(common_e_keys)} E-coincidences "
              f"(sigma_t={sigma_t_ps:.1f}ps)")

    return {
        "sigma_t_ps": sigma_t_ps,
        "raw_bm_data": np.array(all_bm_raw_ps),
        "tof_profile": profile_counts,
        "lyso_thick": lyso_thick,
        "pitch_mm": gap_thick_mm + _W_THICK_MM,
        "n_t_coincidences": len(common_t_evs),
        "n_e_coincidences": len(common_e_keys),
        "dw_e_times": np.array(all_dw_e_times),
        "dw_first_times": np.array(list(down_first.values())),
        "dw_e_total": np.array([dw_e_hits_per_ev.get(k, 0) for k in down_first.keys()]),
        "dw_t_total": np.array([dw_t_hits_per_ev.get(k, 0) for k in down_first.keys()])
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).resolve().parent
    modules = ["radi_cal_energy", "radi_cal_triple", "rc_hex", "rc_hex_triple"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_out = base_dir / "analysis" / f"sweep_summary_{timestamp}"
    analysis_out.mkdir(parents=True, exist_ok=True)

    print("Master processing engine spawned. Targeting tracking metrics...")
    print(f"Saving output to: {analysis_out.relative_to(base_dir)}\n")

    master_summary = {mod: {} for mod in modules}

    for mod in modules:
        mod_path = base_dir / mod / "runs" / mod
        if not mod_path.exists():
            print(f"  Skipping module '{mod}' (path not found)")
            continue

        sweeps = sorted(list(mod_path.glob("sweep_*")), key=lambda p: p.name)
        if not sweeps:
            continue
        target_sweep = sweeps[-1]
        print(f"Processing '{mod}' -> {target_sweep.name}")

        energy_dirs = sorted(list(target_sweep.glob("*GeV")), key=lambda p: extract_numerical_energy(p.name))
        is_hex = "hex" in mod

        for edir in energy_dirs:
            energy_label = edir.name
            print(f"    Parsing radiation slice: {energy_label}")
            res = analyze_energy_batch(edir, is_hex, verbose_label=edir.name)
            if res is not None:
                master_summary[mod][edir.name] = res

    mod_colors = {"radi_cal_energy": "#1976d2", "radi_cal_triple": "#388e3c", "rc_hex": "#d32f2f", "rc_hex_triple": "#7b1fa2"}
    mod_markers = {"radi_cal_energy": "s", "radi_cal_triple": "^", "rc_hex": "o", "rc_hex_triple": "D"}
    layers = np.arange(1, _N_LYSO + 1)

    try:
        import analysis_utils as utils
    except ImportError:
        utils = None

    for mod in modules:
        energy_keys = sorted(master_summary[mod].keys(), key=extract_numerical_energy)
        
        if not energy_keys:
            continue

        n_energies = len(energy_keys)
        ncols = 2 if n_energies >= 2 else 1
        nrows = int(np.ceil(n_energies / ncols))

        # ─────────────────────────────────────────────────────────────────────
        # 1. TIMING HIERARCHY — DYNAMIC PER-PANEL BINNING (FREEDMAN-DIACONIS)
        # ─────────────────────────────────────────────────────────────────────
        fig_time, axs_time = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_time = axs_time.flatten()

        for idx, ekey in enumerate(energy_keys):
            ax = axs_time[idx]
            data = master_summary[mod][ekey]["raw_bm_data"]

            if len(data) > 0:
                clean = clean_around_mode(data, window_ps=500.0)
                lo, hi = float(np.min(clean)), float(np.max(clean))
                total_range = hi - lo
                
                data_std = np.std(clean) if len(clean) > 1 else 5.0
                if data_std <= 0: data_std = 1.0
                if hi <= lo: hi = lo + 1.0

                q75, q25 = np.percentile(clean, [75, 25])
                iqr = q75 - q25
                if iqr > 0 and len(clean) > 1:
                    fd_width = 2.0 * iqr / (len(clean) ** (1.0 / 3.0))
                else:
                    fd_width = 3.5 * np.std(clean) / (len(clean) ** (1.0 / 3.0)) if len(clean) > 1 else 5.0
                min_width = max(1.0, total_range / 50.0)   # never more than 50 bins from range alone
                optimal_width = max(min_width, min(fd_width, 10.0))
                plot_bins = max(3, int(np.ceil((hi - lo) / optimal_width)))
                actual_plot_width = (hi - lo) / plot_bins
                
                counts, edges, _ = ax.hist(clean, bins=plot_bins, range=(lo, hi),
                                           color=mod_colors[mod], alpha=0.6, edgecolor="black", label="Data")

                # ─────────────────────────────────────────────────────────────────────
                # BIFURCATED GAUSSIAN FIT (Handles Low-Side Tails Safely)
                # ─────────────────────────────────────────────────────────────────────
                from scipy.optimize import curve_fit

                # Define the asymmetric twin-width Gaussian
                def bifurcated_gaussian(x, amp, mu, sigma_left, sigma_right):
                    return np.where(x < mu,
                                    amp * np.exp(-0.5 * ((x - mu) / sigma_left)**2),
                                    amp * np.exp(-0.5 * ((x - mu) / sigma_right)**2))

                # Extract the bin centers from your actual plot histogram
                bin_centers = (edges[:-1] + edges[1:]) / 2.0
                x_fit = np.linspace(lo, hi, 5000)

                # Bulletproof initial guesses: [amplitude, peak position, left width, right width]
                p0 = [float(counts.max()), float(np.median(clean)), float(np.std(clean)), float(np.std(clean)) * 0.8]
                
                # Strict bounds to ensure the fit cannot run away or flatten out
                bounds = ([0.0, lo, 0.1, 0.1], [counts.max() * 5.0, hi, (hi - lo), (hi - lo)])

                try:
                    # Run chi-squared minimization on the histogram bins
                    popt, _ = curve_fit(bifurcated_gaussian, bin_centers, counts, p0=p0, bounds=bounds, maxfev=10000)
                    amp_f, mu_f, sig_l_f, sig_r_f = popt
                    
                    y_fit = bifurcated_gaussian(x_fit, amp_f, mu_f, sig_l_f, sig_r_f)
                    label_text = (f"Bifurcated Gauss\n"
                                  f"$\\mu$ = {mu_f:.1f} ps\n"
                                  f"$\\sigma_L$ = {sig_l_f:.1f} ps\n"
                                  f"$\\sigma_R$ = {sig_r_f:.1f} ps")
                except Exception:
                    # Revert back to your trusted fallback functions if optimization runs into trouble
                    _, mu, sigma = fit_gaussian_to_peak(clean, n_bins=40)
                    amplitude = (len(clean) * actual_plot_width) / (sigma * np.sqrt(2 * np.pi)) if sigma > 0 else counts.max()
                    y_fit = standard_gaussian(x_fit, amplitude, mu, sigma)
                    label_text = f"Gaussian Fallback\n$\\mu$ = {mu:.1f} ps\n$\\sigma_t$ = {sigma:.1f} ps"

                ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2.5, label=label_text)

        for idx in range(n_energies, len(axs_time)):
            fig_time.delaxes(axs_time[idx])

        fig_time.suptitle(f"Timing Resolution Distributions — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_time.tight_layout()
        fig_time.savefig(analysis_out / f"{mod}_timing_panels.png", dpi=200)
        plt.close(fig_time)

        # ─────────────────────────────────────────────────────────────────────
        # 2. TOF Profile Figures (Restored)
        # ─────────────────────────────────────────────────────────────────────
        fig_tof, axs_tof = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_tof = axs_tof.flatten()

        for idx, ekey in enumerate(energy_keys):
            ax = axs_tof[idx]
            prof = master_summary[mod][ekey]["tof_profile"]
            s_t = master_summary[mod][ekey]["sigma_t_ps"]
            pitch = master_summary[mod][ekey]["pitch_mm"]
            lyso_thick = master_summary[mod][ekey]["lyso_thick"]

            sigma_z = V_EFF_MM_NS * (s_t / 1000.0)
            sigma_layer = sigma_z / pitch

            total_hits = np.sum(prof)
            norm_prof = prof / total_hits if total_hits > 0 else prof

            sweep_dirs = sorted(list((base_dir / mod / "runs" / mod).glob("sweep_*")))
            edir_path = base_dir / mod / "runs" / mod / sweep_dirs[-1].name / ekey
            run_dirs = sorted(list(set(fp.parent for fp in edir_path.rglob("detector_hits_*.root"))))

            truth_curve = None
            if utils and run_dirs:
                try:
                    gap_thick = lyso_thick + 2 * _TYVEK_THICK_MM
                    calor_thick = (_N_LYSO * gap_thick) + (_N_W * _W_THICK_MM)
                    bounds = get_lyso_layer_bounds(lyso_thick, calor_thick)
                    long_arr, _ = utils.load_calorimeter_mhd(run_dirs, long_glob="run_Dose_edep.mhd", trans_glob="transverse_shower_max_edep.mhd")
                    if long_arr is not None:
                        dz_mm, avg = 0.1, long_arr / max(len(run_dirs), 1)
                        layer_edeps = []
                        for (z_start, z_end) in bounds:
                            z_offset_start = z_start - (-calor_thick / 2)
                            z_offset_end = z_end - (-calor_thick / 2)
                            i0 = max(0, min(int(round(z_offset_start / dz_mm)), len(avg)))
                            i1 = max(0, min(int(round(z_offset_end / dz_mm)), len(avg)))
                            layer_edeps.append(float(np.sum(avg[i0:i1])))
                        truth_curve = np.array(layer_edeps)
                except Exception:
                    truth_curve = None

            if truth_curve is not None and np.sum(truth_curve) > 0:
                norm_truth = truth_curve / np.sum(truth_curve)
                ax.bar(layers, norm_truth, color="#00bcd4", alpha=0.35, edgecolor="#00838f",
                       linewidth=0.8, width=0.8, label="Sim Truth (DoseActor)")

            n_ev = master_summary[mod][ekey]["n_e_coincidences"]
            ax.errorbar(layers, norm_prof, xerr=sigma_layer, color=mod_colors[mod],
                        linewidth=2, marker="o", markersize=4, capsize=3, capthick=1.0,
                        label=f"ΔT Coincidence (N={n_ev})")
            
            ax.set_title(f"{ekey}", fontsize=11, fontweight="bold")
            ax.set_xlabel("LYSO Layer Number", fontsize=9)
            ax.set_ylabel("Normalized Density Fraction", fontsize=9)
            ax.set_xlim(0, _N_LYSO + 1)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8)

        for idx in range(n_energies, len(axs_tof)):
            fig_tof.delaxes(axs_tof[idx])

        fig_tof.suptitle(f"Continuous E-Type ToF Reconstructions — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_tof.tight_layout()
        fig_tof.savefig(analysis_out / f"{mod}_tof_panels.png", dpi=200)
        plt.close(fig_tof)

        # ─────────────────────────────────────────────────────────────────────
        # 3. DOWNSTREAM E-TYPE SIPM HITS VS TIME 
        # ─────────────────────────────────────────────────────────────────────
        fig_dw_hits, axs_dw_hits = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_dw_hits = axs_dw_hits.flatten()

        for idx, ekey in enumerate(energy_keys):
            ax = axs_dw_hits[idx]
            dw_times = master_summary[mod][ekey]["dw_e_times"]
            
            if len(dw_times) > 0:
                ax.hist(dw_times, bins=70, range=(_GT_LO_NS, _GT_HI_NS), color=mod_colors[mod], 
                        alpha=0.7, edgecolor="black", linewidth=0.5, label=f"Downstream E-Hits\nTotal={len(dw_times)}")
                
                ax.set_title(f"Downstream Intensity: {ekey}", fontsize=11, fontweight="bold")
                ax.set_xlabel("Photon Global Time (ns)", fontsize=9)
                ax.set_ylabel("Hit Count / Bin", fontsize=9)
                ax.set_xlim(_GT_LO_NS, _GT_HI_NS)
                ax.legend(loc="upper right", fontsize=8, frameon=True)
            else:
                ax.text(0.5, 0.5, "No Downstream Hits", ha='center', va='center')
            ax.grid(True, linestyle=":", alpha=0.5)

        for idx in range(n_energies, len(axs_dw_hits)):
            fig_dw_hits.delaxes(axs_dw_hits[idx])

        fig_dw_hits.suptitle(f"Downstream E-Type SiPM Intensity Profiles — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_dw_hits.tight_layout()
        fig_dw_hits.savefig(analysis_out / f"{mod}_dw_e_hits_time.png", dpi=200)
        plt.close(fig_dw_hits)

        # ─────────────────────────────────────────────────────────────────────
        # 3.5. DOWNSTREAM PROMPT STRIKES VS DISTANCE (SINGLE-SIDED ZOOM)
        # ─────────────────────────────────────────────────────────────────────
        WLS_LAYERS = {8, 9, 10, 11} 

        fig_dist, axs_dist = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_dist = axs_dist.flatten()

        for idx, ekey in enumerate(energy_keys):
            ax = axs_dist[idx]
            first_times = master_summary[mod][ekey]["dw_first_times"]
            lyso_thick = master_summary[mod][ekey]["lyso_thick"]
            
            if len(first_times) > 0:
                distances_mm = first_times * V_EFF_MM_NS
                view_min, view_max = 120.0, 350.0 
                
                counts, edges, _ = ax.hist(distances_mm, bins=80, range=(view_min, view_max), color="#ff9800", 
                                           alpha=0.6, edgecolor="black", linewidth=0.5, 
                                           label=f"Prompt Strikes (N={len(first_times)})")
                
                if len(counts) > 0 and np.max(counts) > 0:
                    peak_idx = np.argmax(counts)
                    peak_dist = edges[peak_idx] + (edges[1] - edges[0]) / 2.0
                    ax.axvline(peak_dist, color="red", linestyle="--", linewidth=1.5, 
                               label=f"Peak: {peak_dist:.1f} mm")

                sweep_dirs = sorted(list((base_dir / mod / "runs" / mod).glob("sweep_*")))
                edir_path = base_dir / mod / "runs" / mod / sweep_dirs[-1].name / ekey
                run_dirs = sorted(list(set(fp.parent for fp in edir_path.rglob("detector_hits_*.root"))))

                if utils and run_dirs:
                    try:
                        gap_thick = lyso_thick + 2 * _TYVEK_THICK_MM
                        calor_thick = (_N_LYSO * gap_thick) + (_N_W * _W_THICK_MM)
                        bounds = get_lyso_layer_bounds(lyso_thick, calor_thick)
                        long_arr, _ = utils.load_calorimeter_mhd(run_dirs, long_glob="run_Dose_edep.mhd", trans_glob="transverse_shower_max_edep.mhd")
                        
                        if long_arr is not None:
                            dz_mm = 0.1
                            avg = long_arr / max(len(run_dirs), 1)
                            z_sensor = 110.0 
                            z_source_mm = -130.15 
                            max_hist_height = np.max(counts) if len(counts) > 0 else 1.0
                            
                            layer_dists, layer_edeps = [], []
                            
                            for i, (z_start, z_end) in enumerate(bounds):
                                z_offset_start = z_start - (-calor_thick / 2)
                                z_offset_end = z_end - (-calor_thick / 2)
                                i0 = max(0, min(int(round(z_offset_start / dz_mm)), len(avg)))
                                i1 = max(0, min(int(round(z_offset_end / dz_mm)), len(avg)))
                                edep = float(np.sum(avg[i0:i1]))
                                layer_edeps.append(edep)
                                
                                z_center = (z_start + z_end) / 2.0
                                t_expected = ((z_center - z_source_mm) / C_LIGHT_MM_NS) + (abs(z_sensor - z_center) / V_EFF_MM_NS)
                                layer_dists.append(t_expected * V_EFF_MM_NS)
                            
                            layer_edeps = np.array(layer_edeps)
                            if np.sum(layer_edeps) > 0:
                                norm_edeps = (layer_edeps / np.max(layer_edeps)) * (max_hist_height * 0.9)
                                bar_width = 2.0 
                                
                                for i, (d_val, edep_val) in enumerate(zip(layer_dists, norm_edeps)):
                                    if view_min <= d_val <= view_max: 
                                        bar_color = "#e91e63" if i in WLS_LAYERS else "#00bcd4"
                                        bar_label = "WLS Region (Sim Truth)" if (i in WLS_LAYERS and i == min(WLS_LAYERS)) else \
                                                    "Standard LYSO (Sim Truth)" if (i not in WLS_LAYERS and i == 0) else None
                                        
                                        ax.bar(d_val, edep_val, width=bar_width, color=bar_color, alpha=0.5, 
                                               edgecolor="black", linewidth=0.5, label=bar_label)
                    except Exception as e:
                        print(f"    [Warning] Could not overlay DoseActor: {e}")

                ax.set_title(f"Prompt Strikes vs Distance: {ekey}", fontsize=11, fontweight="bold")
                ax.set_xlabel(f"Kinematic Pseudo-Distance (mm)", fontsize=9)
                ax.set_ylabel("Events", fontsize=9)
                ax.set_xlim(view_min, view_max)
                
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, frameon=True)
                
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.grid(True, linestyle=":", alpha=0.5)

        for idx in range(n_energies, len(axs_dist)):
            fig_dist.delaxes(axs_dist[idx])

        fig_dist.suptitle(f"Downstream Prompt Strikes vs Distance (Cropped) — {mod}", 
                           fontsize=14, fontweight="bold", y=0.98)
        fig_dist.tight_layout()
        fig_dist.savefig(analysis_out / f"{mod}_dw_prompt_distance_cropped.png", dpi=200)
        plt.close(fig_dist)

        # ─────────────────────────────────────────────────────────────────────
        # 3.6. SHOWER-MAX TRACKING: T-TYPE vs E-TYPE INTENSITY RATIO
        # ─────────────────────────────────────────────────────────────────────
        fig_ratio, axs_ratio = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_ratio = axs_ratio.flatten()

        for idx, ekey in enumerate(energy_keys):
            ax = axs_ratio[idx]
            
            t_hits = master_summary[mod][ekey].get("dw_t_total", np.array([]))
            e_hits = master_summary[mod][ekey].get("dw_e_total", np.array([]))
            
            if len(t_hits) > 0 and len(e_hits) > 0:
                valid = e_hits > 0
                valid_t = t_hits[valid]
                valid_e = e_hits[valid]
                
                ratio = valid_t / valid_e
                
                counts, edges, _ = ax.hist(ratio, bins=80, range=(0.0, 1.0), color="#9c27b0", 
                                           alpha=0.7, edgecolor="black", linewidth=0.5, 
                                           label=f"T/E Ratio (N={len(ratio)})")
                
                if len(counts) > 0 and np.max(counts) > 0:
                    peak_idx = np.argmax(counts)
                    peak_ratio = edges[peak_idx] + (edges[1] - edges[0]) / 2.0
                    ax.axvline(peak_ratio, color="red", linestyle="--", linewidth=1.5, 
                               label=f"Mode Ratio: {peak_ratio:.3f}")
                
                ax.set_title(f"WLS Intensity Ratio: {ekey}", fontsize=11, fontweight="bold")
                ax.set_xlabel("Ratio (T-Type Hits / E-Type Hits)", fontsize=9)
                ax.set_ylabel("Events", fontsize=9)
                ax.set_xlim(0.0, 1.0)
                ax.legend(loc="upper right", fontsize=8, frameon=True)
            else:
                ax.text(0.5, 0.5, "No T/E Intensity Data", ha='center', va='center')
            
            ax.grid(True, linestyle=":", alpha=0.5)

        for idx in range(n_energies, len(axs_ratio)):
            fig_ratio.delaxes(axs_ratio[idx])

        fig_ratio.suptitle(f"Shower-Max Tracking via WLS Intensity Ratio — {mod}", 
                           fontsize=14, fontweight="bold", y=0.98)
        fig_ratio.tight_layout()
        fig_ratio.savefig(analysis_out / f"{mod}_intensity_ratio.png", dpi=200)
        plt.close(fig_ratio)

    # ─────────────────────────────────────────────────────────────────────
    # 4. UNIFIED OVERALL PERFORMANCE HORIZON COMPARISON GRAPH
    # ─────────────────────────────────────────────────────────────────────
    fig_perf, ax_perf = plt.subplots(figsize=(9, 6))
    any_points = False

    for mod in modules:
        energy_keys = sorted(master_summary[mod].keys(), key=extract_numerical_energy)
        
        if not energy_keys:
            continue

        x_energy, y_res, y_err = [], [], []
        for ekey in energy_keys:
            s_t = master_summary[mod][ekey]["sigma_t_ps"]
            n_ev = master_summary[mod][ekey]["n_t_coincidences"]
            if n_ev < 8 or s_t <= 0:
                continue
            x_energy.append(extract_numerical_energy(ekey))
            y_res.append(s_t)
            y_err.append(s_t / np.sqrt(2 * n_ev))

        if x_energy:
            any_points = True
            ax_perf.errorbar(x_energy, y_res, yerr=y_err, marker=mod_markers[mod], color=mod_colors[mod],
                             linewidth=2, markersize=7, capsize=4, capthick=1.5, linestyle="--", label=mod)

    ax_perf.set_xlabel("Incident Particle Beam Energy (GeV)", fontweight="bold")
    ax_perf.set_ylabel(r"BestMinus Timing Resolution $\sigma_t$ (ps)", fontweight="bold")
    ax_perf.set_title("Unified Performance Horizon — Timing Resolution vs Energy", fontsize=12, fontweight="bold")
    ax_perf.grid(True, linestyle=":", alpha=0.6)
    ax_perf.set_xscale("log")
    ax_perf.set_xticks([25, 50, 100, 200])
    ax_perf.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if any_points:
        ax_perf.legend(loc="upper right", frameon=True)
    else:
        ax_perf.text(0.5, 0.5, "No modules had sufficient statistics", ha='center', va='center', transform=ax_perf.transAxes)

    fig_perf.tight_layout()
    fig_perf.savefig(analysis_out / "timing_resolution_vs_energy.png", dpi=220)
    plt.close(fig_perf)

    # ─────────────────────────────────────────────────────────────────────
    # 5. EXPORT MASTER MATRIX TEXT REPORT
    # ─────────────────────────────────────────────────────────────────────
    sheet_path = analysis_out / "timing_vs_energy_report.txt"
    with open(sheet_path, "w") as f:
        f.write(f"{'='*80}\n")
        f.write(" RADiCAL SIMULATION UNIFIED RUN SUMMARY SHEET\n")
        f.write(f" Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")

        for mod in modules:
            f.write(f"MODULE: {mod}\n")
            f.write(f"{'-'*65}\n")
            f.write(f"  {'Energy':<12} | {'sigma_t (ps)':<16} | {'sigma_z (mm)':<14} | {'sigma_layer':<12} | {'N events (T/E)':<15}\n")
            f.write(f"{'-'*65}\n")

            energy_keys = sorted(master_summary[mod].keys(), key=extract_numerical_energy)
            for ekey in energy_keys:
                s_t = master_summary[mod][ekey]["sigma_t_ps"]
                s_z = V_EFF_MM_NS * (s_t / 1000.0)
                pitch = master_summary[mod][ekey]["pitch_mm"]
                s_layer = s_z / pitch if pitch > 0 else 0
                n_t = master_summary[mod][ekey]["n_t_coincidences"]
                n_e = master_summary[mod][ekey]["n_e_coincidences"]
                f.write(f"  {ekey:<12} | {s_t:<16.1f} | {s_z:<14.2f} | {s_layer:<12.2f} | {n_t}/{n_e}\n")
            f.write("\n")

if __name__ == "__main__":
    main()