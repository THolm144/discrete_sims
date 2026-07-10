#!/usr/bin/env python3
"""
unfold_profile_analysis.py
===========================
Alternative longitudinal shower-profile reconstruction using Tikhonov Regularization
(Regularized Matrix Inversion) with dynamic penalty scaling and ratio subpanels.
"""

import os
import sys
import datetime
import warnings
from pathlib import Path
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS 
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS = 299.792
REFRACTIVE_INDEX = 1.60
V_LIGHT_MM_NS = C_LIGHT_MM_NS / REFRACTIVE_INDEX



# Fallback base offset if dynamic energy key isn't matched
DEFAULT_T0_OFFSET_NS = -0.32  

# Module-specific, energy-dependent timing offsets (in ns)
CALIBRATION_T0_OFFSETS = {
    "radi_cal_energy": {25.0: -0.320, 50.0: -0.300, 100.0: -0.330, 200.0: -0.320},
    "radi_cal_triple": {25.0: -0.370, 50.0: -0.270, 100.0: -0.250, 200.0: -0.190},
    "rc_hex":          {25.0: -0.290, 50.0: -0.310, 100.0: -0.180, 200.0: -0.085},
    "rc_hex_triple":   {25.0: -0.370, 50.0: -0.300, 100.0: -0.220, 200.0: -0.050},
}

# Module-specific, energy-dependent bounce factors
CALIBRATION_BOUNCE_FACTORS = {
    "radi_cal_energy": {25.0: 1.17, 50.0: 1.20, 100.0: 1.23, 200.0: 1.27},
    "radi_cal_triple": {25.0: 1.45, 50.0: 1.50, 100.0: 1.55, 200.0: 1.65},
    "rc_hex":          {25.0: 1.02, 50.0: 1.02, 100.0: 1.01, 200.0: 1.01},
    "rc_hex_triple":   {25.0: 1.19, 50.0: 1.23, 100.0: 1.33, 200.0: 1.73},
}


_GT_LO_NS = 0.0
_GT_HI_NS = 50.0
_TYVEK_THICK_MM = 0.2032
_W_THICK_MM = 2.5
_N_LYSO = 29
_N_W = 28

ARRIVAL_QUANTILE = 0.10
MIN_PHOTONS_PER_FACE = 1

_KNOWN_MODULE_LYSO_THICK = {
    "radi_cal_energy": 1.5,
    "radi_cal_triple": 4.5,
    "rc_hex":          1.5,
    "rc_hex_triple":   4.5,
}

_SQUARE_HOLE_OFFSET = 3.7032
SQUARE_CAP_XY = np.array([
    [ _SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],
    [-_SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],
    [-_SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],
    [ _SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],
])

HEX_CAP_R_MM = 3.5
HEX_CAP_XY = np.array([
    [HEX_CAP_R_MM * np.cos(np.pi/2 + i*(np.pi/3)), HEX_CAP_R_MM * np.sin(np.pi/2 + i*(np.pi/3))]
    for i in range(6)
])

TARGET_SWEEPS = {
    "radi_cal_energy": Path("/home/uakgun/gate_sims/discrete_sims/radi_cal_energy/runs/radi_cal_energy/sweep_20260708_154719"),
    "radi_cal_triple": Path("/home/uakgun/gate_sims/discrete_sims/radi_cal_triple/runs/radi_cal_triple/sweep_20260708_154742"),
    "rc_hex": Path("/home/uakgun/gate_sims/discrete_sims/rc_hex/runs/rc_hex/sweep_20260702_164238/sweep_20260702_164238"),
    "rc_hex_triple": Path("/home/uakgun/gate_sims/discrete_sims/rc_hex_triple/runs/rc_hex_triple/sweep_20260708_154836/sweep_20260708_154836"),
}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS 
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
    if hi <= lo: hi = lo + 1.0

    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    mids = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(counts.astype(float), sigma=2.0)
    peak_idx = int(np.argmax(smoothed))
    mu0, A0 = float(mids[peak_idx]), float(smoothed[peak_idx])

    try:
        popt, _ = curve_fit(standard_gaussian, mids, counts, p0=[A0, mu0, spread],
                            bounds=([0.0, lo, 1e-6], [A0 * 10.0 + 1.0, hi, (hi - lo)]), maxfev=10000)
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return A0, mu0, spread

def clean_around_mode(arr, window_ps=500.0):
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
# TIKHONOV UNFOLDING MACHINERY
# ─────────────────────────────────────────────────────────────────────────────
def extended_response_matrix(n_reco, pad_layers, sigma_bins, mod):
    module_name = mod
    R_sliced = np.load(f"response_matrices/{module_name}_response_matrix.npy")
    return R_sliced 

def tikhonov_deconvolve(observed, R, alpha=0.1, sys_err=0.03):
    n_reco, n_true = R.shape
    total = np.sum(observed)
    if total <= 0:
        return np.zeros(n_true)

    variance = observed + (sys_err * observed) ** 2 + 1e-6
    V_inv = np.diag(1.0 / variance)

    D = np.zeros((n_true - 2, n_true))
    for i in range(n_true - 2):
        D[i, i]     = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    lhs = R.T @ V_inv @ R + alpha * (D.T @ D)
    rhs = R.T @ V_inv @ observed

    try:
        x = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    x[x < 0] = 0.0
    return x

def bootstrap_unfold_tikhonov(raw_z_emits, lyso_bounds, mod, sigma_layer, n_boot=40, seed=0, pad_layers=5, alpha=0.1):
    n_bins = len(lyso_bounds)
    edges = np.array([b[0] for b in lyso_bounds] + [lyso_bounds[-1][1]])
    raw_z_emits = np.asarray(raw_z_emits)
    
    R_sliced = extended_response_matrix(n_bins, pad_layers, sigma_layer, mod)

    rng = np.random.default_rng(seed)
    n_events = len(raw_z_emits)
    unfolded_reps = []
    raw_reps = []

    if n_events == 0:
        return np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins), R_sliced

    for _ in range(n_boot):
        sample_idx = rng.integers(0, n_events, size=n_events)
        sample = raw_z_emits[sample_idx]
        counts, _ = np.histogram(sample, bins=edges)
        raw_reps.append(counts.astype(float))
        
        x_unf_ext = tikhonov_deconvolve(
            counts.astype(float), R_sliced, alpha=alpha, sys_err=0.03
        )
        unfolded_reps.append(x_unf_ext[pad_layers : pad_layers + n_bins])

    unfolded_reps = np.array(unfolded_reps)
    raw_reps = np.array(raw_reps)
    return unfolded_reps.mean(axis=0), unfolded_reps.std(axis=0), raw_reps.mean(axis=0), R_sliced


# ─────────────────────────────────────────────────────────────────────────────
# DATA EXTRACTION 
# ─────────────────────────────────────────────────────────────────────────────
def extract_profile_data_unfold(batch_dir: Path, is_hex: bool, module_name: str, t0_offset: float, bounce_factor: float):
    hit_files = sorted(list(batch_dir.rglob("detector_hits_*.root")))
    if not hit_files: return None

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
        except Exception: continue

    if detected_z_sensor is None: return None

    lyso_thick = _KNOWN_MODULE_LYSO_THICK[module_name]
    
    gap_thick_mm = lyso_thick + 2 * _TYVEK_THICK_MM
    calor_thick_mm = (_N_LYSO * gap_thick_mm) + (_N_W * _W_THICK_MM)
    lyso_bounds = get_lyso_layer_bounds(lyso_thick, calor_thick_mm)

    cap_xy_map = HEX_CAP_XY if is_hex else SQUARE_CAP_XY
    t_indices = {1, 3, 5} if is_hex else {0, 1}
    e_indices = {0, 2, 4} if is_hex else {2, 3}

    up_first_raw, down_first_raw = {}, {}
    up_times_raw, dw_times_raw = {}, {}

    total_raw_entries = 0
    total_e_up, total_e_dw = 0, 0
    total_t_up, total_t_dw = 0, 0

    for idx, fpath in enumerate(hit_files):
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                tree = f[tk]
                if tree.num_entries == 0: continue
                
                total_raw_entries += tree.num_entries
                
                x = tree["Position_X"].array(library="np")
                y = tree["Position_Y"].array(library="np")
                z = tree["Position_Z"].array(library="np")
                gt = tree["GlobalTime"].array(library="np")
                lt = tree["LocalTime"].array(library="np")
                ev = tree["EventID"].array(library="np")
                pn = tree["ParticleName"].array(library="np")
        except Exception: continue

        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(np.hypot(dx, dy), axis=1)

        near_up = np.abs(z + detected_z_sensor) < 2.5
        near_dw = np.abs(z - detected_z_sensor) < 2.5
        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")
        is_prompt = (gt >= _GT_LO_NS) & (gt <= _GT_HI_NS)

        is_e = np.isin(channels, list(e_indices))
        m_e_up, m_e_dw = is_e & is_prompt & near_up, is_e & is_prompt & near_dw

        is_t = np.isin(channels, list(t_indices))
        m_t_up, m_t_dw = is_t & is_optical & near_up, is_t & is_optical & near_dw

        total_e_up += np.sum(m_e_up)
        total_e_dw += np.sum(m_e_dw)
        total_t_up += np.sum(m_t_up)
        total_t_dw += np.sum(m_t_dw)

        for eid, ti in zip(ev[m_e_up], gt[m_e_up]):
            key = (fpath, int(eid))
            if key not in up_first_raw or ti < up_first_raw[key]: up_first_raw[key] = float(ti)
            
        for eid, ti in zip(ev[m_e_dw], gt[m_e_dw]):
            key = (fpath, int(eid))
            if key not in down_first_raw or ti < down_first_raw[key]: down_first_raw[key] = float(ti)

        for e, t in zip(ev[m_t_up], lt[m_t_up] * 1000.0):
            up_times_raw.setdefault((fpath, int(e)), []).append(t)
            
        for e, t in zip(ev[m_t_dw], lt[m_t_dw] * 1000.0):
            dw_times_raw.setdefault((fpath, int(e)), []).append(t)

    # ─── RESTORE DICTIONARY KEYS BACK TO ORIGINAL EXPECTATIONS ────────────────
    up_first = {}
    down_first = {}
    up_times_by_ev = {}
    dw_times_by_ev = {}

    for (fpath, eid), ti in up_first_raw.items():
        up_first[(fpath.parent.name, eid)] = ti

    for (fpath, eid), ti in down_first_raw.items():
        down_first[(fpath.parent.name, eid)] = ti

    for (fpath, eid), t_list in up_times_raw.items():
        up_times_by_ev.setdefault((fpath.parent.name, eid), []).extend(t_list)

    for (fpath, eid), t_list in dw_times_raw.items():
        dw_times_by_ev.setdefault((fpath.parent.name, eid), []).extend(t_list)

    # ─── CALCULATE COINCIDENCES ON ALIGNED MATRIX DICTIONARIES ────────────────
    common_e_keys = set(up_first) & set(down_first)
    raw_z_list = []  
    out_of_bounds = 0
    
    for k in common_e_keys:
        delta_t_corrected = (down_first[k] - up_first[k]) - t0_offset
        # Dynamic effective velocity calculation based on local bounce factor
        v_eff_local = (C_LIGHT_MM_NS / REFRACTIVE_INDEX) * bounce_factor
        z_est = v_eff_local * delta_t_corrected / 2.0
        
        if -calor_thick_mm / 2 - 60.0 <= z_est <= calor_thick_mm / 2 + 60.0:
            raw_z_list.append(z_est)
        else:
            out_of_bounds += 1
            
    raw_z_emits = np.array(raw_z_list)

    # ─── EXTRACT TIMING RESOLUTION STRUCTS ────────────────────────────────────
    common_t_evs = set(up_times_by_ev.keys()) & set(dw_times_by_ev.keys())
    all_bm_raw_ps = []
    for e in common_t_evs:
        if len(up_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE and len(dw_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE:
            t_up_q = np.quantile(up_times_by_ev[e], ARRIVAL_QUANTILE)
            t_dw_q = np.quantile(dw_times_by_ev[e], ARRIVAL_QUANTILE)
            all_bm_raw_ps.append((t_dw_q - t_up_q) / 2.0)

    clean_bm = clean_around_mode(np.array(all_bm_raw_ps), window_ps=500.0)
    _, _, sigma_t_ps = fit_gaussian_to_peak(clean_bm)

    pitch_mm = gap_thick_mm + _W_THICK_MM
    sigma_z_mm = v_eff_local * (sigma_t_ps / 1000.0)
    sigma_layer = sigma_z_mm / pitch_mm if pitch_mm > 0 else 1.0

    print(f"    » [{module_name} @ {batch_dir.name}] "
          f"Total Raw Hits: {total_raw_entries:,} | "
          f"Filtered E (Up/Dn): {total_e_up}/{total_e_dw} | "
          f"Filtered T (Up/Dn): {total_t_up}/{total_t_dw} | "
          f"Coincidences: {len(common_e_keys)}")

    return {
        "raw_z_emits": raw_z_emits,
        "n_e_coincidences": len(common_e_keys),
        "sigma_t_ps": sigma_t_ps,
        "sigma_layer": sigma_layer,
        "pitch_mm": pitch_mm,
        "lyso_thick": lyso_thick,
        "calor_thick": calor_thick_mm,
        "lyso_bounds": lyso_bounds,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).resolve().parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_out = base_dir / "tikhonov_profile_analysis" / f"tikhonov_summary_{timestamp}"
    analysis_out.mkdir(parents=True, exist_ok=True)

    print("Spawning Tikhonov Regularized Unfolded Profile Extractor...")
    print(f"Targeting outputs to: {analysis_out.relative_to(base_dir)}\n")

    try:
        import analysis_utils as utils
    except ImportError:
        utils = None
        print("[Warning] analysis_utils not found. Sim truth overlays will be disabled.")

    mod_colors = {"radi_cal_energy": "#1976d2", "radi_cal_triple": "#388e3c", "rc_hex": "#d32f2f", "rc_hex_triple": "#7b1fa2"}
    layers = np.arange(1, _N_LYSO + 1)

    for mod, target_sweep in TARGET_SWEEPS.items():
        if not target_sweep.exists():
            print(f"  Skipping '{mod}' - Target sweep not found: {target_sweep}")
            continue

        print(f"Processing Tikhonov Profile Data for '{mod}'...")
        is_hex = "hex" in mod
        energy_dirs = sorted([d for d in target_sweep.iterdir() if d.is_dir() and "GeV" in d.name],
                             key=lambda p: extract_numerical_energy(p.name))
        if not energy_dirs:
            continue

        n_energies = len(energy_dirs)
        ncols = 2 if n_energies >= 2 else 1
        nrows = int(np.ceil(n_energies / ncols))

        fig_unf = plt.figure(figsize=(7.2 * ncols, 6.0 * nrows))
        gs_unf = gridspec.GridSpec(2 * nrows, ncols, height_ratios=[3, 1] * nrows, hspace=0.28, wspace=0.24)

        fig_raw = plt.figure(figsize=(7.2 * ncols, 6.0 * nrows))
        gs_raw = gridspec.GridSpec(2 * nrows, ncols, height_ratios=[3, 1] * nrows, hspace=0.28, wspace=0.24)

        for idx, edir in enumerate(energy_dirs):
            ekey = edir.name
            print(f"    Extracting + unfolding {ekey}")
            
            # Dynamic energy calibration parameter mapping
            energy_val = extract_numerical_energy(ekey)
            current_t0_offset = CALIBRATION_T0_OFFSETS.get(mod, {}).get(energy_val, DEFAULT_T0_OFFSET_NS)
            bounce_factor = CALIBRATION_BOUNCE_FACTORS.get(mod, {}).get(energy_val, 1.0)

            res = extract_profile_data_unfold(edir, is_hex, mod, t0_offset=current_t0_offset, bounce_factor=bounce_factor)

            r_coord = idx // ncols
            c_coord = idx % ncols
            
            ax_main_unf = fig_unf.add_subplot(gs_unf[2 * r_coord, c_coord])
            ax_ratio_unf = fig_unf.add_subplot(gs_unf[2 * r_coord + 1, c_coord], sharex=ax_main_unf)

            ax_main_raw = fig_raw.add_subplot(gs_raw[2 * r_coord, c_coord])
            ax_ratio_raw = fig_raw.add_subplot(gs_raw[2 * r_coord + 1, c_coord], sharex=ax_main_raw)

            if res is None or len(res["raw_z_emits"]) < 5:
                for ax_m, ax_r in [(ax_main_unf, ax_ratio_unf), (ax_main_raw, ax_ratio_raw)]:
                    ax_m.text(0.5, 0.5, "Insufficient Data", ha="center", va="center")
                    ax_m.set_title(ekey, fontsize=11, fontweight="bold")
                    ax_r.axis('off')
                continue

            lyso_bounds = res["lyso_bounds"]
            sigma_layer = res["sigma_layer"]
            calor_thick = res["calor_thick"]
            
            edges = np.array([b[0] for b in lyso_bounds] + [lyso_bounds[-1][1]])

            alpha_tikhonov = 0.05

            unfolded_mean, unfolded_std, raw_mean, _ = bootstrap_unfold_tikhonov(
                res["raw_z_emits"], lyso_bounds, mod, sigma_layer,
                n_boot=40, pad_layers=5, alpha=alpha_tikhonov
            )

            def safe_norm(v):
                s = np.sum(v)
                return v / s if s > 0 else v

            raw_norm = safe_norm(raw_mean)
            unf_norm = safe_norm(unfolded_mean)
            unf_err_norm = unfolded_std / np.sum(unfolded_mean) if np.sum(unfolded_mean) > 0 else unfolded_std
            raw_err_norm = np.sqrt(raw_mean) / np.sum(raw_mean) if np.sum(raw_mean) > 0 else np.zeros_like(raw_mean)

            truth_curve = None
            run_dirs = sorted(list(set(fp.parent for fp in edir.rglob("detector_hits_*.root"))))
            if utils and run_dirs:
                try:
                    long_arr, _ = utils.load_calorimeter_mhd(run_dirs, long_glob="run_Dose_edep.mhd", trans_glob="transverse_shower_max_edep.mhd")
                    if long_arr is not None:
                        dz_mm, avg = 0.1, long_arr / max(len(run_dirs), 1)
                        layer_edeps = []
                        for (z_start, z_end) in lyso_bounds:
                            z_offset_start = z_start - (-calor_thick / 2)
                            z_offset_end = z_end - (-calor_thick / 2)
                            i0 = max(0, min(int(round(z_offset_start / dz_mm)), len(avg)))
                            i1 = max(0, min(int(round(z_offset_end / dz_mm)), len(avg)))
                            layer_edeps.append(float(np.sum(avg[i0:i1])))
                        truth_curve = np.array(layer_edeps)
                except Exception:
                    truth_curve = None

            raw_norm_disp = raw_norm[::-1]
            raw_err_disp = raw_err_norm[::-1]
            unf_norm_disp = unf_norm[::-1]
            unf_err_disp = unf_err_norm[::-1]

            if truth_curve is not None and np.sum(truth_curve) > 0:
                truth_norm_disp = truth_curve / np.sum(truth_curve)
                
                for ax_m in [ax_main_unf, ax_main_raw]:
                    ax_m.bar(layers, truth_norm_disp, color="#00bcd4", alpha=0.25, edgecolor="#00838f",
                               linewidth=0.8, width=0.8, label="Sim Truth (DoseActor)")

                sigma_bins_unf = np.where(unf_err_disp > 0, unf_err_disp, 1e-4)
                chi2_unf = np.sum(((unf_norm_disp - truth_norm_disp) / sigma_bins_unf) ** 2)
                reduced_chi2_unf = chi2_unf / len(unf_norm_disp)
                mae_unf = np.mean(np.abs(unf_norm_disp - truth_norm_disp)) * 100
                fit_stats_unf = f" (χ²/ndf={reduced_chi2_unf:.2f}, MAE={mae_unf:.1f}%)"

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_unf = unf_norm_disp / truth_norm_disp
                    ratio_err_unf = unf_err_disp / truth_norm_disp
                ax_ratio_unf.errorbar(layers, ratio_unf, yerr=ratio_err_unf, color=mod_colors[mod],
                                  fmt='o-', markersize=3.5, linewidth=1.2, capsize=1.5, elinewidth=0.8)
                ax_ratio_unf.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
                ax_ratio_unf.set_ylabel("Unf / Truth", fontsize=8)
                ax_ratio_unf.set_ylim(0.4, 1.6)

                sigma_bins_raw = np.where(raw_err_disp > 0, raw_err_disp, 1e-4)
                chi2_raw = np.sum(((raw_norm_disp - truth_norm_disp) / sigma_bins_raw) ** 2)
                reduced_chi2_raw = chi2_raw / len(raw_norm_disp)
                mae_raw = np.mean(np.abs(raw_norm_disp - truth_norm_disp)) * 100
                fit_stats_raw = f" (χ²/ndf={reduced_chi2_raw:.2f}, MAE={mae_raw:.1f}%)"

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_raw = raw_norm_disp / truth_norm_disp
                    ratio_err_raw = raw_err_disp / truth_norm_disp
                ax_ratio_raw.errorbar(layers, ratio_raw, yerr=ratio_err_raw, color="#e65100",
                                  fmt='o-', markersize=3.5, linewidth=1.2, capsize=1.5, elinewidth=0.8)
                ax_ratio_raw.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
                ax_ratio_raw.set_ylabel("Raw / Truth", fontsize=8)
                ax_ratio_raw.set_ylim(0.4, 1.6)
            else:
                for ax_r in [ax_ratio_unf, ax_ratio_raw]:
                    ax_r.text(0.5, 0.5, "No Reference Truth", ha="center", va="center", alpha=0.4, transform=ax_r.transAxes)
                    ax_r.set_ylim(0, 2)
                fit_stats_unf = ""
                fit_stats_raw = ""

            # ─── POPULATE UNIFOLDED FIGURE PANEL ─────────────────────────────
            ax_main_unf.plot(layers, raw_norm_disp, color="gray", linewidth=1.2, linestyle=":",
                    marker=".", markersize=3.5, alpha=0.7, label="Raw ΔT Profile (blurred)")
            ax_main_unf.errorbar(layers, unf_norm_disp, yerr=unf_err_disp, color=mod_colors[mod],
                        linewidth=1.8, marker="o", markersize=4.0, capsize=2.5, capthick=0.9,
                        label=f"Tikhonov-Unfolded (α={alpha_tikhonov:.3f}){fit_stats_unf}")

            ax_main_unf.set_title(f"{ekey} (N={res['n_e_coincidences']})", fontsize=11, fontweight="bold")
            ax_main_unf.set_ylabel("Normalized Fraction", fontsize=9)
            ax_main_unf.set_xlim(0, _N_LYSO + 1)
            ax_main_unf.tick_params(labelbottom=False)
            ax_main_unf.grid(True, linestyle=":", alpha=0.4)
            ax_main_unf.legend(loc="upper right", fontsize=7.2)

            ax_ratio_unf.set_xlabel("LYSO Layer Number", fontsize=9)
            ax_ratio_unf.set_xlim(0, _N_LYSO + 1)
            ax_ratio_unf.grid(True, linestyle=":", alpha=0.4)

            # ─── POPULATE RAW ONLY FIGURE PANEL ──────────────────────────────
            ax_main_raw.errorbar(layers, raw_norm_disp, yerr=raw_err_disp, color="#e65100",
                        linewidth=1.8, marker="s", markersize=4.0, capsize=2.5, capthick=0.9,
                        label=f"Raw Data Profile{fit_stats_raw}")

            ax_main_raw.set_title(f"{ekey} (N={res['n_e_coincidences']})", fontsize=11, fontweight="bold")
            ax_main_raw.set_ylabel("Normalized Fraction", fontsize=9)
            ax_main_raw.set_xlim(0, _N_LYSO + 1)
            ax_main_raw.tick_params(labelbottom=False)
            ax_main_raw.grid(True, linestyle=":", alpha=0.4)
            ax_main_raw.legend(loc="upper right", fontsize=7.2)

            ax_ratio_raw.set_xlabel("LYSO Layer Number", fontsize=9)
            ax_ratio_raw.set_xlim(0, _N_LYSO + 1)
            ax_ratio_raw.grid(True, linestyle=":", alpha=0.4)

        for dummy_idx in range(n_energies, nrows * ncols):
            r_coord = dummy_idx // ncols
            c_coord = dummy_idx % ncols
            for fig_obj, gs_obj in [(fig_unf, gs_unf), (fig_raw, gs_raw)]:
                fig_obj.add_subplot(gs_obj[2 * r_coord, c_coord]).axis('off')
                fig_obj.add_subplot(gs_obj[2 * r_coord + 1, c_coord]).axis('off')

        fig_unf.suptitle(f"Tikhonov Unfolded Longitudinal Profile — {mod}", fontsize=13, fontweight="bold", y=0.99)
        out_path_unf = analysis_out / f"{mod}_tikhonov_profile.png"
        fig_unf.savefig(out_path_unf, dpi=200, bbox_inches='tight')
        plt.close(fig_unf)
        print(f"    Saved Tikhonov Unfolded: {out_path_unf.name}")

        fig_raw.suptitle(f"Raw Longitudinal Profile (No Deconvolution) — {mod}", fontsize=13, fontweight="bold", y=0.99)
        out_path_raw = analysis_out / f"{mod}_raw_profile.png"
        fig_raw.savefig(out_path_raw, dpi=200, bbox_inches='tight')
        plt.close(fig_raw)
        print(f"    Saved Raw Profile: {out_path_raw.name}")

if __name__ == "__main__":
    main()