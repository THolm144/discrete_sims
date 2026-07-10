#!/usr/bin/env python3
"""
unfold_profile_analysis.py
===========================
Alternative longitudinal shower-profile reconstruction using Tikhonov Regularization,
Richardson-Lucy, D'Agostini Bayesian unfolding, and a hybrid R-L + Tikhonov smoothing pipeline.
"""

import os
import sys
import json
import datetime
import warnings
from pathlib import Path
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

import response_kernel as rk

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS 
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS = 299.792
REFRACTIVE_INDEX = 1.60
V_LIGHT_MM_NS = C_LIGHT_MM_NS / REFRACTIVE_INDEX

DEFAULT_T0_OFFSET_NS = -0.32  

CALIBRATION_T0_OFFSETS = {
    "radi_cal_energy": {25.0: -0.320, 50.0: -0.300, 100.0: -0.330, 200.0: -0.320},
    "radi_cal_triple": {25.0: -0.370, 50.0: -0.270, 100.0: -0.250, 200.0: -0.190},
    "rc_hex":          {25.0: -0.290, 50.0: -0.310, 100.0: -0.180, 200.0: -0.085},
    "rc_hex_triple":   {25.0: -0.370, 50.0: -0.300, 100.0: -0.220, 200.0: -0.050},
}

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

# Cache of loaded response-kernel params so we don't re-read the JSON file
# for every energy point / bootstrap call.
_RESPONSE_PARAMS_CACHE = {}

def load_response_params(mod, response_dir=None):
    """Loads the fitted kernel parameter vector for `mod`, cached per module."""
    if mod in _RESPONSE_PARAMS_CACHE:
        return _RESPONSE_PARAMS_CACHE[mod]
    response_dir = response_dir or (Path(__file__).resolve().parent / "response_matrices")
    meta_path = response_dir / f"{mod}_response_params.json"
    with open(meta_path) as f:
        meta = json.load(f)
    _RESPONSE_PARAMS_CACHE[mod] = meta
    return meta

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
    lo, hi = float(np.min(data)), float(np.max(data))
    if hi <= lo: hi = lo + 1.0

    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    mids = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(counts.astype(float), sigma=2.0)
    peak_idx = int(np.argmax(smoothed))
    mu0, A0 = float(mids[peak_idx]), float(smoothed[peak_idx])

    try:
        popt, _ = curve_fit(standard_gaussian, mids, counts, p0=[A0, mu0, max(np.std(data), 1.0)],
                            bounds=([0.0, lo, 1e-6], [A0 * 10.0 + 1.0, hi, (hi - lo)]), maxfev=10000)
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return A0, mu0, max(np.std(data), 1.0)

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
def extended_response_matrix(n_reco, pad_layers, mod, energy_val):
    """
    Builds R on the fly at the requested energy, from the module's fitted
    kernel parameters (response_matrices/<mod>_response_params.json), rather
    than loading a frozen matrix. This is what makes the deconvolution
    kernel track the actual energy-dependent timing resolution instead of
    using one static shape for every energy point.
    """
    meta = load_response_params(mod)
    if meta["n_reco"] != n_reco or meta["pad_layers"] != pad_layers:
        raise ValueError(
            f"[{mod}] response params were fit with n_reco={meta['n_reco']}, "
            f"pad_layers={meta['pad_layers']} but this call wants "
            f"n_reco={n_reco}, pad_layers={pad_layers}. Re-run "
            f"build_empirical_response_matrix.py or fix the mismatch."
        )
    return rk.build_kernel_matrix(n_reco, pad_layers, meta["params_vector"], energy_val=energy_val)

def tikhonov_deconvolve(observed, R, alpha=0.1, sys_err=0.03):
    n_reco, n_true = R.shape
    if np.sum(observed) <= 0: return np.zeros(n_true)

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

def tikhonov_smooth(x_unf, alpha=5.0):
    """Applies a Whittaker-Eilers (Tikhonov) smoothing penalty directly to an unfolded array."""
    n = len(x_unf)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i]     = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
        
    lhs = np.eye(n) + alpha * (D.T @ D)
    try:
        x_smooth = np.linalg.solve(lhs, x_unf)
    except np.linalg.LinAlgError:
        x_smooth = np.linalg.lstsq(lhs, x_unf, rcond=None)[0]
        
    x_smooth[x_smooth < 0] = 0.0
    return x_smooth

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
    if np.sum(observed) <= 0: return np.zeros(n_true)
        
    x_unf = np.zeros(n_true)
    for _ in range(iterations):
        denom = P_E_given_C @ prior
        P_C_given_E = (P_E_given_C * prior) / (denom[:, np.newaxis] + eps)
        
        n_true_est = (P_C_given_E.T @ observed) / eff
        x_unf = n_true_est
        prior = n_true_est / (np.sum(n_true_est) + eps)
        
    return x_unf

def bootstrap_unfold(raw_z_emits, lyso_bounds, mod, energy_val, algo="tikhonov", n_boot=40, seed=0, pad_layers=5, **kwargs):
    n_bins = len(lyso_bounds)
    edges = np.array([b[0] for b in lyso_bounds] + [lyso_bounds[-1][1]])
    raw_z_emits = np.asarray(raw_z_emits)

    # Built ONCE per energy point (not per bootstrap replicate) at the
    # correct energy -- this is the piece that used to be a frozen,
    # energy-independent .npy.
    R_sliced = extended_response_matrix(n_bins, pad_layers, mod, energy_val)
    rng = np.random.default_rng(seed)
    n_events = len(raw_z_emits)
    
    unfolded_reps, raw_reps = [], []
    if n_events == 0:
        return np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins), R_sliced

    for _ in range(n_boot):
        sample_idx = rng.integers(0, n_events, size=n_events)
        counts, _ = np.histogram(raw_z_emits[sample_idx], bins=edges)
        raw_reps.append(counts.astype(float))
        
        if algo == "tikhonov":
            x_unf_ext = tikhonov_deconvolve(counts.astype(float), R_sliced, alpha=kwargs.get("alpha", 0.05))
        elif algo == "rl":
            x_unf_ext = richardson_lucy_deconvolve(counts.astype(float), R_sliced, iterations=kwargs.get("iterations", 15))
        elif algo == "dagostini":
            x_unf_ext = dagostini_unfold(counts.astype(float), R_sliced, iterations=kwargs.get("iterations", 5))
        elif algo == "rl_tikhonov":
            x_rl = richardson_lucy_deconvolve(counts.astype(float), R_sliced, iterations=kwargs.get("iterations", 15))
            x_unf_ext = tikhonov_smooth(x_rl, alpha=kwargs.get("alpha", 0.5))
        elif algo == "raw":
            x_unf_ext = np.pad(counts.astype(float), (pad_layers, pad_layers)) 
            
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

    up_first, down_first = {}, {}
    up_times_by_ev, dw_times_by_ev = {}, {}

    for idx, fpath in enumerate(hit_files):
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                tree = f[tk]
                if tree.num_entries == 0: continue
                
                x, y, z = tree["Position_X"].array(library="np"), tree["Position_Y"].array(library="np"), tree["Position_Z"].array(library="np")
                gt, lt = tree["GlobalTime"].array(library="np"), tree["LocalTime"].array(library="np")
                ev, pn = tree["EventID"].array(library="np"), tree["ParticleName"].array(library="np")
        except Exception: continue

        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(np.hypot(dx, dy), axis=1)

        near_up = np.abs(z + detected_z_sensor) < 2.5
        near_dw = np.abs(z - detected_z_sensor) < 2.5
        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")
        is_prompt = (gt >= _GT_LO_NS) & (gt <= _GT_HI_NS)

        m_e_up = np.isin(channels, list(e_indices)) & is_prompt & near_up
        m_e_dw = np.isin(channels, list(e_indices)) & is_prompt & near_dw
        m_t_up = np.isin(channels, list(t_indices)) & is_optical & near_up
        m_t_dw = np.isin(channels, list(t_indices)) & is_optical & near_dw

        for eid, ti in zip(ev[m_e_up], gt[m_e_up]):
            key = (fpath.parent.name, int(eid))
            if key not in up_first or ti < up_first[key]: up_first[key] = float(ti)
            
        for eid, ti in zip(ev[m_e_dw], gt[m_e_dw]):
            key = (fpath.parent.name, int(eid))
            if key not in down_first or ti < down_first[key]: down_first[key] = float(ti)

        for eid, ti in zip(ev[m_t_up], lt[m_t_up] * 1000.0):
            up_times_by_ev.setdefault((fpath.parent.name, int(eid)), []).append(ti)
            
        for eid, ti in zip(ev[m_t_dw], lt[m_t_dw] * 1000.0):
            dw_times_by_ev.setdefault((fpath.parent.name, int(eid)), []).append(ti)

    common_e_keys = set(up_first) & set(down_first)
    raw_z_list = []  
    v_eff_local = (C_LIGHT_MM_NS / REFRACTIVE_INDEX) * bounce_factor
    
    for k in common_e_keys:
        z_est = v_eff_local * ((down_first[k] - up_first[k]) - t0_offset) / 2.0
        if -calor_thick_mm / 2 - 60.0 <= z_est <= calor_thick_mm / 2 + 60.0:
            raw_z_list.append(z_est)

    common_t_evs = set(up_times_by_ev.keys()) & set(dw_times_by_ev.keys())
    all_bm_raw_ps = []
    for e in common_t_evs:
        if len(up_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE and len(dw_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE:
            all_bm_raw_ps.append((np.quantile(dw_times_by_ev[e], ARRIVAL_QUANTILE) - np.quantile(up_times_by_ev[e], ARRIVAL_QUANTILE)) / 2.0)

    clean_bm = clean_around_mode(np.array(all_bm_raw_ps), window_ps=500.0)
    _, _, sigma_t_ps = fit_gaussian_to_peak(clean_bm)

    pitch_mm = gap_thick_mm + _W_THICK_MM
    sigma_layer = (v_eff_local * (sigma_t_ps / 1000.0)) / pitch_mm if pitch_mm > 0 else 1.0

    print(f"    » [{module_name} @ {batch_dir.name}] Coincidences: {len(common_e_keys)}")

    return {
        "raw_z_emits": np.array(raw_z_list),
        "n_e_coincidences": len(common_e_keys),
        "sigma_layer": sigma_layer,
        "lyso_bounds": lyso_bounds,
        "calor_thick": calor_thick_mm
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
    try:
        import analysis_utils as utils
    except ImportError:
        utils = None
        print("[Warning] analysis_utils not found. Sim truth overlays will be disabled.")

    mod_colors = {"radi_cal_energy": "#1976d2", "radi_cal_triple": "#388e3c", "rc_hex": "#d32f2f", "rc_hex_triple": "#7b1fa2"}
    combo_colors = {"tikhonov": "#7b1fa2", "rl": "#d32f2f", "rl_tikhonov": "#ff9800", "raw": "gray"}
    layers = np.arange(1, _N_LYSO + 1)

    algo_configs = {
        "tikhonov": {"title": "Tikhonov Unfolded", "kwargs": {"alpha": 0.050}},
        "rl": {"title": "Richardson-Lucy Unfolded", "kwargs": {"iterations": 15}},
        "dagostini": {"title": "D'Agostini (Bayesian) Unfolded", "kwargs": {"iterations": 5}},
        "rl_tikhonov": {"title": "Hybrid: R-L + Tikhonov Smoothing", "kwargs": {"iterations": 15, "alpha": 0.5}}
    }
    algos_to_run = list(algo_configs.keys()) + ["raw", "combined"]

    for mod, target_sweep in TARGET_SWEEPS.items():
        if not target_sweep.exists(): continue
        print(f"Processing Profile Data for '{mod}'...")
        is_hex = "hex" in mod
        energy_dirs = sorted([d for d in target_sweep.iterdir() if d.is_dir() and "GeV" in d.name], key=lambda p: extract_numerical_energy(p.name))
        if not energy_dirs: continue

        n_energies = len(energy_dirs)
        ncols = 2 if n_energies >= 2 else 1
        nrows = int(np.ceil(n_energies / ncols))

        figs, gss = {}, {}
        for a in algos_to_run:
            figs[a] = plt.figure(figsize=(7.2 * ncols, 6.0 * nrows))
            gss[a] = gridspec.GridSpec(2 * nrows, ncols, height_ratios=[3, 1] * nrows, hspace=0.28, wspace=0.24)

        for idx, edir in enumerate(energy_dirs):
            ekey = edir.name
            print(f"    Extracting + unfolding {ekey}")
            energy_val = extract_numerical_energy(ekey)
            res = extract_profile_data_unfold(
                edir, is_hex, mod,
                t0_offset=CALIBRATION_T0_OFFSETS.get(mod, {}).get(energy_val, DEFAULT_T0_OFFSET_NS),
                bounce_factor=CALIBRATION_BOUNCE_FACTORS.get(mod, {}).get(energy_val, 1.0)
            )

            r_coord, c_coord = idx // ncols, idx % ncols
            axes = {a: (figs[a].add_subplot(gss[a][2 * r_coord, c_coord]), figs[a].add_subplot(gss[a][2 * r_coord + 1, c_coord])) for a in algos_to_run}

            if res is None or len(res["raw_z_emits"]) < 5:
                for a in algos_to_run:
                    axes[a][0].text(0.5, 0.5, "Insufficient Data", ha="center", va="center")
                    axes[a][0].set_title(ekey, fontsize=11, fontweight="bold")
                    axes[a][1].axis('off')
                continue

            truth_curve = None
            run_dirs = sorted(list(set(fp.parent for fp in edir.rglob("detector_hits_*.root"))))
            if utils and run_dirs:
                try:
                    long_arr, _ = utils.load_calorimeter_mhd(run_dirs, long_glob="run_Dose_edep.mhd", trans_glob="transverse_shower_max_edep.mhd")
                    if long_arr is not None:
                        dz_mm, avg = 0.1, long_arr / max(len(run_dirs), 1)
                        layer_edeps = []
                        for (z_start, z_end) in res["lyso_bounds"]:
                            i0 = max(0, min(int(round((z_start + res["calor_thick"]/2) / dz_mm)), len(avg)))
                            i1 = max(0, min(int(round((z_end + res["calor_thick"]/2) / dz_mm)), len(avg)))
                            layer_edeps.append(float(np.sum(avg[i0:i1])))
                        truth_curve = np.array(layer_edeps)
                except Exception: pass
                    
            truth_norm_disp = truth_curve / np.sum(truth_curve) if truth_curve is not None and np.sum(truth_curve) > 0 else None

            algo_results = {}
            for algo in ["raw", "tikhonov", "rl", "dagostini", "rl_tikhonov"]:
                u_mean, u_std, r_mean, _ = bootstrap_unfold(
                    res["raw_z_emits"], res["lyso_bounds"], mod, energy_val,
                    algo=algo, n_boot=20, pad_layers=5, **(algo_configs[algo]["kwargs"] if algo != "raw" else {})
                )
                if "raw" not in algo_results:
                    algo_results["raw"] = (safe_norm(r_mean)[::-1], (np.sqrt(r_mean)/np.sum(r_mean))[::-1] if np.sum(r_mean)>0 else np.zeros_like(r_mean))
                if algo != "raw":
                    algo_results[algo] = (safe_norm(u_mean)[::-1], (u_std/np.sum(u_mean))[::-1] if np.sum(u_mean)>0 else u_std[::-1])

            for algo in algos_to_run:
                ax_m, ax_r = axes[algo]
                
                if truth_norm_disp is not None:
                    ax_m.bar(layers, truth_norm_disp, color="#00bcd4", alpha=0.25, edgecolor="#00838f", linewidth=0.8, width=0.8, label="Sim Truth (DoseActor)")
                else:
                    ax_r.text(0.5, 0.5, "No Reference Truth", ha="center", va="center", alpha=0.4, transform=ax_r.transAxes)

                algos_to_plot = ["tikhonov", "rl", "rl_tikhonov"] if algo == "combined" else [algo]
                if algo not in ["raw", "combined"]:
                    ax_m.plot(layers, algo_results["raw"][0], color="gray", linewidth=1.2, linestyle=":", marker=".", markersize=3.5, alpha=0.7, label="Raw ΔT Profile")

                for plt_algo in algos_to_plot:
                    a_mean, a_std = algo_results[plt_algo]
                    
                    if plt_algo == "raw":
                        label, c, m = "Raw Data Profile", "#e65100", "s"
                    else:
                        cfg_str = ", ".join(f"{k}={v}" for k, v in algo_configs[plt_algo]["kwargs"].items())
                        label = f"{algo_configs[plt_algo]['title']} ({cfg_str})"
                        c = combo_colors[plt_algo] if algo == "combined" else mod_colors[mod]
                        m = "o" if algo != "combined" else ({"tikhonov": "o", "rl": "^", "rl_tikhonov": "s"}.get(plt_algo, "o"))

                    fit_stats = ""
                    if truth_norm_disp is not None:
                        # Use a signal-scale-relative floor rather than an absolute
                        # constant, otherwise near-zero-variance tail bins blow up
                        # chi2/ndf into meaningless numbers (bug fixed this pass).
                        sigma_bins = np.maximum(a_std, 0.01 * np.max(a_mean) if np.max(a_mean) > 0 else 1e-4)
                        chi2_ndf = np.sum(((a_mean - truth_norm_disp) / sigma_bins)**2) / len(a_mean)
                        fit_stats = f" (χ²={chi2_ndf:.1f}, MAE={np.mean(np.abs(a_mean - truth_norm_disp))*100:.1f}%)"
                        with np.errstate(divide='ignore', invalid='ignore'):
                            ax_r.errorbar(layers, a_mean / truth_norm_disp, yerr=a_std / truth_norm_disp, color=c, fmt=f'{m}-', markersize=3.5, linewidth=1.2, capsize=1.5, elinewidth=0.8)

                    ax_m.errorbar(layers, a_mean, yerr=a_std, color=c, linewidth=1.8, marker=m, markersize=4.0, capsize=2.5, capthick=0.9, label=f"{label}{fit_stats}")

                if truth_norm_disp is not None:
                    ax_r.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
                    ax_r.set_ylabel("Data / Truth", fontsize=8)
                    ax_r.set_ylim(0.4, 1.6)

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
            for a in algos_to_run:
                figs[a].add_subplot(gss[a][2 * (dummy_idx // ncols), (dummy_idx % ncols)]).axis('off')
                figs[a].add_subplot(gss[a][2 * (dummy_idx // ncols) + 1, (dummy_idx % ncols)]).axis('off')

        for a in algos_to_run:
            title_prefix = algo_configs[a]["title"] if a in algo_configs else ("Combined Comparison" if a == "combined" else "Raw (No Deconvolution)")
            figs[a].suptitle(f"{title_prefix} Longitudinal Profile — {mod}", fontsize=13, fontweight="bold", y=0.99)
            out_path = analysis_out / f"{mod}_{a}_profile.png"
            figs[a].savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(figs[a])
            print(f"    Saved {a.capitalize()} Profile: {out_path.name}")

if __name__ == "__main__":
    main()