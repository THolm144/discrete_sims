#!/usr/bin/env python3
"""
unfold_profile_analysis.py
===========================
Alternative longitudinal shower-profile reconstruction using Tikhonov Regularization,
Richardson-Lucy, and D'Agostini Bayesian unfolding to preserve asymmetric shower tails.
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
# UNFOLDING MACHINERY
# ─────────────────────────────────────────────────────────────────────────────
def extended_response_matrix(n_reco, pad_layers, sigma_bins, mod):
    module_name = mod
    R_sliced = np.load(f"response_matrices/{module_name}_response_matrix.npy")
    return R_sliced 

def tikhonov_deconvolve(observed, R, alpha=0.1, sys_err=0.03):
    n_reco, n_true = R.shape
    total = np.sum(observed)
    if total <= 0: return np.zeros(n_true)

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

def richardson_lucy_deconvolve(observed, R, iterations=15):
    eps = 1e-12
    eff = np.sum(R, axis=0)
    eff[eff == 0] = 1.0
    
    x = np.ones(R.shape[1]) * (np.sum(observed) / R.shape[1])
    for _ in range(iterations):
        reco_pred = R @ x
        ratio = observed / (reco_pred + eps)
        x = (x / eff) * (R.T @ ratio)
    return x

def dagostini_unfold(observed, R, iterations=5):
    eps = 1e-12
    n_reco, n_true = R.shape
    
    eff = np.sum(R, axis=0)
    eff[eff == 0] = 1.0
    P_E_given_C = R / eff
    
    prior = np.ones(n_true) / n_true
    n_obs_total = np.sum(observed)
    if n_obs_total <= 0: return np.zeros(n_true)
        
    x_unf = np.zeros(n_true)
    for _ in range(iterations):
        denom = P_E_given_C @ prior
        P_C_given_E = (P_E_given_C * prior) / (denom[:, np.newaxis] + eps)
        
        n_true_est = (P_C_given_E.T @ observed) / eff
        x_unf = n_true_est
        
        prior = n_true_est / (np.sum(n_true_est) + eps)
        
    return x_unf

def bootstrap_unfold(raw_z_emits, lyso_bounds, mod, sigma_layer, algo="tikhonov", n_boot=40, seed=0, pad_layers=5, **kwargs):
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
        
        if algo == "tikhonov":
            x_unf_ext = tikhonov_deconvolve(counts.astype(float), R_sliced, alpha=kwargs.get("alpha", 0.05), sys_err=0.03)
        elif algo == "rl":
            x_unf_ext = richardson_lucy_deconvolve(counts.astype(float), R_sliced, iterations=kwargs.get("iterations", 15))
        elif algo == "dagostini":
            x_unf_ext = dagostini_unfold(counts.astype(float), R_sliced, iterations=kwargs.get("iterations", 5))
            
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

    up_first = {}
    down_first = {}
    up_times_by_ev = {}
    dw_times_by_ev = {}

    for (fpath, eid), ti in up_first_raw.items(): up_first[(fpath.parent.name, eid)] = ti
    for (fpath, eid), ti in down_first_raw.items(): down_first[(fpath.parent.name, eid)] = ti
    for (fpath, eid), t_list in up_times_raw.items(): up_times_by_ev.setdefault((fpath.parent.name, eid), []).extend(t_list)
    for (fpath, eid), t_list in dw_times_raw.items(): dw_times_by_ev.setdefault((fpath.parent.name, eid), []).extend(t_list)

    common_e_keys = set(up_first) & set(down_first)
    raw_z_list = []  
    out_of_bounds = 0
    
    for k in common_e_keys:
        delta_t_corrected = (down_first[k] - up_first[k]) - t0_offset
        v_eff_local = (C_LIGHT_MM_NS / REFRACTIVE_INDEX) * bounce_factor
        z_est = v_eff_local * delta_t_corrected / 2.0
        
        if -calor_thick_mm / 2 - 60.0 <= z_est <= calor_thick_mm / 2 + 60.0:
            raw_z_list.append(z_est)
        else:
            out_of_bounds += 1
            
    raw_z_emits = np.array(raw_z_list)

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
    sigma_z_mm = ((C_LIGHT_MM_NS / REFRACTIVE_INDEX) * bounce_factor) * (sigma_t_ps / 1000.0)
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

def safe_norm(v):
    s = np.sum(v)
    return v / s if s > 0 else v

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).resolve().parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_out = base_dir / "unfolded_profile_analysis" / f"unfold_summary_{timestamp}"
    analysis_out.mkdir(parents=True, exist_ok=True)

    print("Spawning Multi-Algorithm Unfolded Profile Extractor...")
    print(f"Targeting outputs to: {analysis_out.relative_to(base_dir)}\n")

    try:
        import analysis_utils as utils
    except ImportError:
        utils = None
        print("[Warning] analysis_utils not found. Sim truth overlays will be disabled.")

    mod_colors = {"radi_cal_energy": "#1976d2", "radi_cal_triple": "#388e3c", "rc_hex": "#d32f2f", "rc_hex_triple": "#7b1fa2"}
    layers = np.arange(1, _N_LYSO + 1)

    algo_configs = {
        "tikhonov": {"title": "Tikhonov Unfolded", "kwargs": {"alpha": 0.050}},
        "rl": {"title": "Richardson-Lucy Unfolded", "kwargs": {"iterations": 15}},
        "dagostini": {"title": "D'Agostini (Bayesian) Unfolded", "kwargs": {"iterations": 5}}
    }
    algos_to_run = list(algo_configs.keys()) + ["raw"]

    for mod, target_sweep in TARGET_SWEEPS.items():
        if not target_sweep.exists():
            print(f"  Skipping '{mod}' - Target sweep not found.")
            continue

        print(f"Processing Profile Data for '{mod}'...")
        is_hex = "hex" in mod
        energy_dirs = sorted([d for d in target_sweep.iterdir() if d.is_dir() and "GeV" in d.name],
                             key=lambda p: extract_numerical_energy(p.name))
        if not energy_dirs:
            continue

        n_energies = len(energy_dirs)
        ncols = 2 if n_energies >= 2 else 1
        nrows = int(np.ceil(n_energies / ncols))

        figs = {}
        gss = {}
        for a in algos_to_run:
            figs[a] = plt.figure(figsize=(7.2 * ncols, 6.0 * nrows))
            gss[a] = gridspec.GridSpec(2 * nrows, ncols, height_ratios=[3, 1] * nrows, hspace=0.28, wspace=0.24)

        for idx, edir in enumerate(energy_dirs):
            ekey = edir.name
            print(f"    Extracting + unfolding {ekey}")
            
            energy_val = extract_numerical_energy(ekey)
            current_t0_offset = CALIBRATION_T0_OFFSETS.get(mod, {}).get(energy_val, DEFAULT_T0_OFFSET_NS)
            bounce_factor = CALIBRATION_BOUNCE_FACTORS.get(mod, {}).get(energy_val, 1.0)

            res = extract_profile_data_unfold(edir, is_hex, mod, t0_offset=current_t0_offset, bounce_factor=bounce_factor)

            r_coord = idx // ncols
            c_coord = idx % ncols
            
            axes = {}
            for a in algos_to_run:
                ax_m = figs[a].add_subplot(gss[a][2 * r_coord, c_coord])
                ax_r = figs[a].add_subplot(gss[a][2 * r_coord + 1, c_coord], sharex=ax_m)
                axes[a] = (ax_m, ax_r)

            if res is None or len(res["raw_z_emits"]) < 5:
                for a in algos_to_run:
                    ax_m, ax_r = axes[a]
                    ax_m.text(0.5, 0.5, "Insufficient Data", ha="center", va="center")
                    ax_m.set_title(ekey, fontsize=11, fontweight="bold")
                    ax_r.axis('off')
                continue

            lyso_bounds = res["lyso_bounds"]
            sigma_layer = res["sigma_layer"]
            calor_thick = res["calor_thick"]
            
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
                    
            truth_norm_disp = truth_curve / np.sum(truth_curve) if truth_curve is not None and np.sum(truth_curve) > 0 else None

            raw_ref_mean = None
            raw_ref_err = None

            for algo in algos_to_run:
                ax_m, ax_r = axes[algo]
                
                if algo == "raw":
                    algo_mean = raw_ref_mean
                    algo_std = raw_ref_err
                    algo_label = "Raw Data Profile"
                    color = "#e65100"
                    marker = "s"
                else:
                    unfolded_mean, unfolded_std, raw_mean, _ = bootstrap_unfold(
                        res["raw_z_emits"], lyso_bounds, mod, sigma_layer,
                        algo=algo, n_boot=40, pad_layers=5, **algo_configs[algo]["kwargs"]
                    )
                    
                    if raw_ref_mean is None:
                        raw_ref_mean = safe_norm(raw_mean)[::-1]
                        raw_ref_err = (np.sqrt(raw_mean) / np.sum(raw_mean))[::-1] if np.sum(raw_mean) > 0 else np.zeros_like(raw_mean)

                    algo_mean = safe_norm(unfolded_mean)[::-1]
                    algo_std = (unfolded_std / np.sum(unfolded_mean))[::-1] if np.sum(unfolded_mean) > 0 else unfolded_std[::-1]
                    
                    cfg_k, cfg_v = list(algo_configs[algo]["kwargs"].items())[0]
                    algo_label = f"{algo_configs[algo]['title']} ({cfg_k}={cfg_v})"
                    color = mod_colors[mod]
                    marker = "o"

                if truth_norm_disp is not None:
                    ax_m.bar(layers, truth_norm_disp, color="#00bcd4", alpha=0.25, edgecolor="#00838f", linewidth=0.8, width=0.8, label="Sim Truth (DoseActor)")
                    sigma_bins = np.where(algo_std > 0, algo_std, 1e-4)
                    chi2 = np.sum(((algo_mean - truth_norm_disp) / sigma_bins) ** 2)
                    reduced_chi2 = chi2 / len(algo_mean)
                    mae = np.mean(np.abs(algo_mean - truth_norm_disp)) * 100
                    fit_stats = f" (χ²/ndf={reduced_chi2:.2f}, MAE={mae:.1f}%)"

                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = algo_mean / truth_norm_disp
                        ratio_err = algo_std / truth_norm_disp
                    ax_r.errorbar(layers, ratio, yerr=ratio_err, color=color, fmt='o-', markersize=3.5, linewidth=1.2, capsize=1.5, elinewidth=0.8)
                    ax_r.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
                    ax_r.set_ylabel("Data / Truth", fontsize=8)
                    ax_r.set_ylim(0.4, 1.6)
                else:
                    fit_stats = ""
                    ax_r.text(0.5, 0.5, "No Reference Truth", ha="center", va="center", alpha=0.4, transform=ax_r.transAxes)
                    ax_r.set_ylim(0, 2)

                if algo != "raw":
                    ax_m.plot(layers, raw_ref_mean, color="gray", linewidth=1.2, linestyle=":", marker=".", markersize=3.5, alpha=0.7, label="Raw ΔT Profile (blurred)")
                
                ax_m.errorbar(layers, algo_mean, yerr=algo_std, color=color, linewidth=1.8, marker=marker, markersize=4.0, capsize=2.5, capthick=0.9, label=f"{algo_label}{fit_stats}")

                ax_m.set_title(f"{ekey} (N={res['n_e_coincidences']})", fontsize=11, fontweight="bold")
                ax_m.set_ylabel("Normalized Fraction", fontsize=9)
                ax_m.set_xlim(0, _N_LYSO + 1)
                ax_m.tick_params(labelbottom=False)
                ax_m.grid(True, linestyle=":", alpha=0.4)
                ax_m.legend(loc="upper right", fontsize=7.2)

                ax_r.set_xlabel("LYSO Layer Number", fontsize=9)
                ax_r.set_xlim(0, _N_LYSO + 1)
                ax_r.grid(True, linestyle=":", alpha=0.4)

        for dummy_idx in range(n_energies, nrows * ncols):
            r_coord = dummy_idx // ncols
            c_coord = dummy_idx % ncols
            for a in algos_to_run:
                figs[a].add_subplot(gss[a][2 * r_coord, c_coord]).axis('off')
                figs[a].add_subplot(gss[a][2 * r_coord + 1, c_coord]).axis('off')

        for a in algos_to_run:
            title_prefix = algo_configs[a]["title"] if a != "raw" else "Raw (No Deconvolution)"
            figs[a].suptitle(f"{title_prefix} Longitudinal Profile — {mod}", fontsize=13, fontweight="bold", y=0.99)
            out_path = analysis_out / f"{mod}_{a}_profile.png"
            figs[a].savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(figs[a])
            print(f"    Saved {a.capitalize()} Profile: {out_path.name}")

if __name__ == "__main__":
    main()