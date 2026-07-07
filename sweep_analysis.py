#!/usr/bin/env python3
import os
import sys
import argparse
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
C_LIGHT_MM_NS    = 299.792
REFRACTIVE_INDEX = 1.60                          
V_LIGHT_MM_NS    = C_LIGHT_MM_NS / REFRACTIVE_INDEX
BOUNCE_FACTOR    = 0.92                          
V_EFF_MM_NS      = V_LIGHT_MM_NS * BOUNCE_FACTOR

_GT_LO_NS = 0.0  
_GT_HI_NS = 50.0
_TYVEK_THICK_MM  = 0.2032
_W_THICK_MM      = 2.5
_N_LYSO          = 29
_N_W             = 28

ARRIVAL_QUANTILE     = 0.10
MIN_PHOTONS_PER_FACE = 1

# Geometry mappings
_SQUARE_HOLE_OFFSET = 3.7032
SQUARE_CAP_XY = np.array([
    [ _SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET], # 0 (T)
    [-_SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET], # 1 (T)
    [-_SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET], # 2 (E)
    [ _SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET], # 3 (E)
])

HEX_CAP_R_MM = 3.5
HEX_CAP_XY   = np.array([
    [HEX_CAP_R_MM * np.cos(np.pi/2 + i*(np.pi/3)), HEX_CAP_R_MM * np.sin(np.pi/2 + i*(np.pi/3))]
    for i in range(6)
])

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY & LOGIC RESOLVERS
# ─────────────────────────────────────────────────────────────────────────────
def get_lyso_layer_bounds(lyso_thick, calor_thick):
    gap_thick = lyso_thick + 2 * _TYVEK_THICK_MM
    bounds = []
    current_z = -calor_thick / 2
    for idx in range(_N_LYSO):
        z_start = current_z + _TYVEK_THICK_MM
        z_end   = z_start + lyso_thick
        bounds.append((z_start, z_end))
        current_z += gap_thick + (_W_THICK_MM if idx < _N_W else 0)
    return bounds

def standard_gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian_to_peak(data, n_bins=40):
    if len(data) < 8: 
        return 0.0, float(np.median(data)), float(np.std(data))
    center = np.median(data)
    spread = max(np.std(data), 1.0)
    lo, hi = float(np.min(data)), float(np.max(data))
    if hi <= lo: hi = lo + 1.0

    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    mids = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(counts.astype(float), sigma=2.0)
    peak_idx = int(np.argmax(smoothed))
    mu0, A0  = float(mids[peak_idx]), float(smoothed[peak_idx])

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

def clean_around_mode(arr, window_ps=80.0):
    if len(arr) == 0: return arr
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
def analyze_energy_batch(batch_dir: Path, is_hex: bool):
    hit_files = sorted(list(batch_dir.rglob("detector_hits_*.root")))
    if not hit_files:
        return None

    # Determine Z plane geometry metrics dynamically
    detected_z_sensor = None
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                z_arr = f[tk]["Position_Z"].array(library="np")
                if len(z_arr) > 0:
                    abs_z = np.abs(z_arr)
                    detected_z_sensor = float(np.median(abs_z[abs_z > (np.max(abs_z) - 5.0)]))
                    break
        except: continue

    if detected_z_sensor is None:
        return None

    lyso_thick = 1.5 if abs(detected_z_sensor - 91.65) < 3.0 else 4.5
    gap_thick_mm = lyso_thick + 2 * _TYVEK_THICK_MM
    calor_thick_mm = (_N_LYSO * gap_thick_mm) + (_N_W * _W_THICK_MM)
    lyso_bounds = get_lyso_layer_bounds(lyso_thick, calor_thick_mm)

    cap_xy_map = HEX_CAP_XY if is_hex else SQUARE_CAP_XY
    num_caps   = 6 if is_hex else 4
    t_indices  = {1, 3, 5} if is_hex else {0, 1}
    e_indices  = {0, 2, 4} if is_hex else {2, 3}

    up_first, down_first = {}, {}
    up_times_by_ev, dw_times_by_ev = {}, {}
    t_type_best_minus_ps = []
    all_bm_raw_ps = []

    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                tree = f[tk]
                if tree.num_entries == 0: continue
                x  = tree["Position_X"].array(library="np")
                y  = tree["Position_Y"].array(library="np")
                z  = tree["Position_Z"].array(library="np")
                gt = tree["GlobalTime"].array(library="np")
                lt = tree["LocalTime"].array(library="np")
                ev = tree["EventID"].array(library="np")
                pn = tree["ParticleName"].array(library="np")
        except: continue

        # Compute dynamic matching channels
        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(np.hypot(dx, dy), axis=1)

        near_up = np.abs(z + detected_z_sensor) < 2.5
        near_dw = np.abs(z - detected_z_sensor) < 2.5
        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")

        # 1. E-Type Parsing
        is_e = np.isin(channels, list(e_indices))
        is_prompt = (gt >= _GT_LO_NS) & (gt <= _GT_HI_NS)
        m_e_up, m_e_dw = is_e & is_prompt & near_up, is_e & is_prompt & near_dw

        for eid, ti in zip(ev[m_e_up], gt[m_e_up]):
            key = int(eid)
            if key not in up_first or ti < up_first[key]: up_first[key] = float(ti)
        for eid, ti in zip(ev[m_e_dw], gt[m_e_dw]):
            key = int(eid)
            if key not in down_first or ti < down_first[key]: down_first[key] = float(ti)

        # 2. T-Type Timing Parsing
        is_t = np.isin(channels, list(t_indices))
        m_t_up, m_t_dw = is_t & is_optical & near_up, is_t & is_optical & near_dw

        for e, t in zip(ev[m_t_up], lt[m_t_up] * 1000.0): up_times_by_ev.setdefault(int(e), []).append(t)
        for e, t in zip(ev[m_t_dw], lt[m_t_dw] * 1000.0): dw_times_by_ev.setdefault(int(e), []).append(t)

    # Fold coincidences
    common_t_evs = set(up_times_by_ev.keys()) & set(dw_times_by_ev.keys())
    for e in common_t_evs:
        if len(up_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE and len(dw_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE:
            t_up_q = np.quantile(up_times_by_ev[e], ARRIVAL_QUANTILE)
            t_dw_q = np.quantile(dw_times_by_ev[e], ARRIVAL_QUANTILE)
            all_bm_raw_ps.append((t_dw_q - t_up_q) / 2.0)

    clean_bm = clean_around_mode(np.array(all_bm_raw_ps), window_ps=120.0)
    _, _, sigma_t_ps = fit_gaussian_to_peak(clean_bm)

    # Map out ToF Z profiles
    common_e_keys = set(up_first) & set(down_first)
    valid_z_emits = []
    for k in common_e_keys:
        z_est = V_EFF_MM_NS * (down_first[k] - up_first[k]) / 2.0
        if -calor_thick_mm/2 - 15.0 <= z_est <= calor_thick_mm/2 + 15.0:
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
    
    # Invert to realign with physics injection vector definition
    profile_counts = profile_counts[::-1]

    return {
        "sigma_t_ps": sigma_t_ps,
        "raw_bm_data": np.array(all_bm_raw_ps),
        "tof_profile": profile_counts,
        "lyso_thick": lyso_thick,
        "pitch_mm": gap_thick_mm + _W_THICK_MM
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCHEDULER & AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).resolve().parent
    modules  = ["radi_cal_energy", "radi_cal_triple", "rc_hex", "rc_hex_triple"]

    # Initialize execution environment paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_out = base_dir / "analysis" / f"sweep_summary_{timestamp}"
    analysis_out.mkdir(parents=True, exist_ok=True)

    print(f"Master processing engine spawned. Targeting tracking metrics...")
    print(f"Saving output arrays to: {analysis_out.relative_to(base_dir)}\n")

    master_summary = {mod: {} for mod in modules}

    for mod in modules:
        mod_path = base_dir / mod / "runs" / mod
        if not mod_path.exists():
            print(f"  Skipping module target: '{mod}' (Directory path not found)")
            continue

        # Crawl most recent timestamp directory layout
        sweeps = sorted(list(mod_path.glob("sweep_*")), key=lambda p: p.name)
        if not sweeps:
            continue
        target_sweep = sweeps[-1]
        print(f"Processing '{mod}' -> target tracking node: {target_sweep.name}")

        # Extract dynamic energy nodes
        energy_dirs = sorted(list(target_sweep.glob("*GeV")), key=lambda p: extract_numerical_energy(p.name))
        is_hex = "hex" in mod

        for edir in energy_dirs:
            energy_label = edir.name
            print(f"    Parsing radiation slice: {energy_label}")
            res = analyze_energy_batch(edir, is_hex)
            if res is not None:
                master_summary[mod][energy_label] = res

    # ─────────────────────────────────────────────────────────────────────────
    # MATRIX GRAPH VISUALIZATION ENGINE (WITH FIT LINES & TRUTH OVERLAYS)
    # ─────────────────────────────────────────────────────────────────────────
    mod_colors = {"radi_cal_energy": "#1976d2", "radi_cal_triple": "#388e3c", "rc_hex": "#d32f2f", "rc_hex_triple": "#7b1fa2"}
    mod_markers = {"radi_cal_energy": "s", "radi_cal_triple": "^", "rc_hex": "o", "rc_hex_triple": "D"}
    layers = np.arange(1, _N_LYSO + 1)

    # Attempt to pull in the OpenGATE simulation utilities for DoseActor extraction
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

        # 1. Dedicated Multi-Panel TIMING Histogram Figure (With Gaussian Fits)
        fig_time, axs_time = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_time = axs_time.flatten()

        for idx, ekey in enumerate(energy_keys):
            ax = axs_time[idx]
            data = master_summary[mod][ekey]["raw_bm_data"]
            
            if len(data) > 0:
                clean = clean_around_mode(data, window_ps=150.0)
                lo, hi = float(np.min(clean)), float(np.max(clean))
                if hi <= lo: hi = lo + 1.0
                
                # Plot Data Histogram
                counts, edges, _ = ax.hist(clean, bins="auto", color=mod_colors[mod], alpha=0.6, edgecolor="black", label="Data")
                
                # Refit and overlay the continuous Gaussian line shape
                amp, mu, sigma = fit_gaussian_to_peak(clean, n_bins=50)
                x_fit = np.linspace(lo, hi, 1000)
                y_fit = standard_gaussian(x_fit, amp, mu, sigma)
                
                ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2.0, 
                        label=f"Gaussian Fit\n$\\mu$ = {mu:.1f} ps\n$\\sigma_t$ = {sigma:.1f} ps")
                
                ax.set_title(f"Energy Sweep Slice: {ekey}", fontsize=11, fontweight="bold")
                ax.set_xlabel("BestMinus LocalTime (ps)", fontsize=9)
                ax.set_ylabel("Events / Bin", fontsize=9)
                ax.legend(loc="upper right", fontsize=8, frameon=True)
            else:
                ax.text(0.5, 0.5, "Empty Dataset", ha='center', va='center')
            ax.grid(True, linestyle=":", alpha=0.5)

        for idx in range(n_energies, len(axs_time)):
            fig_time.delaxes(axs_time[idx])

        fig_time.suptitle(f"Timing Resolution Distributions — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_time.tight_layout()
        fig_time.savefig(analysis_out / f"{mod}_timing_panels.png", dpi=200)
        plt.close(fig_time)

        # 2. Dedicated Multi-Panel TOF Reconstruction Figure (With DoseActor Overlay)
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

            # Cross-crawl directories to grab simulation truth files (.mhd)
            edir_path = base_dir / mod / "runs" / mod / sorted(list((base_dir / mod / "runs" / mod).glob("sweep_*")))[-1].name / ekey
            run_dirs = sorted(list(set(fpath.parent for fpath in edir_path.rglob("detector_hits_*.root"))))
            
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
                            z_offset_end   = z_end   - (-calor_thick / 2)
                            i0 = max(0, min(int(round(z_offset_start / dz_mm)), len(avg)))
                            i1 = max(0, min(int(round(z_offset_end   / dz_mm)), len(avg)))
                            layer_edeps.append(float(np.sum(avg[i0:i1])))
                        truth_curve = np.array(layer_edeps)
                except:
                    truth_curve = None

            # Plot simulation truth bar background if available
            if truth_curve is not None and np.sum(truth_curve) > 0:
                norm_truth = truth_curve / np.sum(truth_curve)
                ax.bar(layers, norm_truth, color="#00bcd4", alpha=0.35, edgecolor="#00838f", linewidth=0.8, width=0.8, label="Sim Truth (DoseActor)")

            # Plot reconstruction overlay line
            ax.errorbar(layers, norm_prof, xerr=sigma_layer, color=mod_colors[mod], 
                        linewidth=2, marker="o", markersize=4, capsize=3, capthick=1.0, label="ΔT Coincidence")
            
            ax.set_title(f"Energy Sweep Slice: {ekey}", fontsize=11, fontweight="bold")
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


    # ─────────────────────────────────────────────────────────────────────────
    # 3. UNIFIED OVERALL PERFORMANCE HORIZON COMPARISON GRAPH
    # ─────────────────────────────────────────────────────────────────────────
    fig_perf, ax_perf = plt.subplots(figsize=(9, 6))

    for mod in modules:
        energy_keys = sorted(master_summary[mod].keys(), key=extract_numerical_energy)
        if not energy_keys: continue
        
        x_energy, y_res, y_err = [], [], []
        for ekey in energy_keys:
            x_energy.append(extract_numerical_energy(ekey))
            s_t = master_summary[mod][ekey]["sigma_t_ps"]
            y_res.append(s_t)
            y_err.append(s_t * 0.04 + 1.0)

        ax_perf.errorbar(x_energy, y_res, yerr=y_err, marker=mod_markers[mod], color=mod_colors[mod],
                         linewidth=2, markersize=7, capsize=4, capthick=1.5, linestyle="--", label=mod)

    ax_perf.set_xlabel("Incident Particle Beam Energy (GeV)", fontweight="bold")
    ax_perf.set_ylabel("BestMinus Timing Resolution $\sigma_t$ (ps)", fontweight="bold")
    ax_perf.set_title("Unified Performance Horizon — Timing Resolution vs Energy", fontsize=12, fontweight="bold")
    ax_perf.grid(True, linestyle=":", alpha=0.6)
    ax_perf.set_xscale("log")
    ax_perf.set_xticks([25, 50, 100, 200])
    ax_perf.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_perf.legend(loc="upper right", frameon=True)

    fig_perf.tight_layout()
    fig_perf.savefig(analysis_out / "timing_resolution_vs_energy.png", dpi=220)
    plt.close(fig_perf)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. EXPORT MASTER MATRIX TEXT REPORT
    # ─────────────────────────────────────────────────────────────────────────
    sheet_path = analysis_out / "timing_vs_energy_report.txt"
    with open(sheet_path, "w") as f:
        f.write(f"{'═'*80}\n")
        f.write(f" RADiCAL SIMULATION UNIFIED HOVERSON RUN SUMMARY SHEET\n")
        f.write(f" Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'═'*80}\n\n")

        for mod in modules:
            f.write(f"■ TARGET DETECTOR CONFIGURATION MODULE: {mod}\n")
            f.write(f"{'─'*65}\n")
            f.write(f"  {'Energy Layer':<15} | {'Timing Resolution (σ_t)':<25} | {'Spatial Error (σ_z)':<15}\n")
            f.write(f"{'─'*65}\n")
            
            energy_keys = sorted(master_summary[mod].keys(), key=extract_numerical_energy)
            if not energy_keys:
                f.write(f"  [No valid data records isolated for target matrix configuration]\n")
            else:
                for ekey in energy_keys:
                    s_t = master_summary[mod][ekey]["sigma_t_ps"]
                    pitch = master_summary[mod][ekey]["pitch_mm"]
                    sigma_z = V_EFF_MM_NS * (s_t / 1000.0)
                    sigma_layer = sigma_z / pitch
                    
                    res_str = f"{s_t:.2f} ps"
                    err_str = f"± {sigma_layer:.2f} layers ({sigma_z:.2f} mm)"
                    f.write(f"  {ekey:<15} | {res_str:<25} | {err_str:<15}\n")
            f.write(f"{'═'*80}\n\n")

    print(f"\nProcessing execution window sealed successfully.")
    print(f"Saved master data matrix report → {sheet_path.name}")

if __name__ == "__main__":
    main()