#!/usr/bin/env python3
"""
unified_sweep_analysis_optimized.py
=====================================
Optimized version for aggregating timing-resolution, ToF-reconstruction, 
and energy linearity results across 12 RADiCAL geometry variants.
"""
import argparse
import datetime
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL KINEMATICS CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS = 299.792

REFRACTIVE_INDEX = {
    "radi_cal_energy":        1.60,   # BCF92 baseline
    "radi_cal_triple":        1.60,
    "rc_hex":                 1.60,
    "rc_hex_triple":          1.60,
    "dsb1_radi_cal_energy":   1.55,   # DSB1
    "dsb1_radi_cal_triple":   1.55,
    "dsb1_rc_hex":            1.55,
    "dsb1_rc_hex_triple":     1.55,
    "luagce_radi_cal_energy": 1.84,   # LuAG:Ce
    "luagce_radi_cal_triple": 1.84,
    "luagce_rc_hex":          1.84,
    "luagce_rc_hex_triple":   1.84,
}

V_LIGHT_MM_NS = C_LIGHT_MM_NS / REFRACTIVE_INDEX

BOUNCE_FACTOR = {
    "radi_cal_energy":        0.92,
    "radi_cal_triple":        0.92,
    "rc_hex":                 0.92,
    "rc_hex_triple":          0.92,
    "dsb1_radi_cal_energy":   0.92,
    "dsb1_radi_cal_triple":   0.92,
    "dsb1_rc_hex":            0.92,
    "dsb1_rc_hex_triple":     0.92,
    "luagce_radi_cal_energy": 0.92,
    "luagce_radi_cal_triple": 0.92,
    "luagce_rc_hex":          0.92,
    "luagce_rc_hex_triple":   0.92,
}

T_OFFSET_NS = {
    "radi_cal_energy":        0.0,
    "radi_cal_triple":        0.0,
    "rc_hex":                 0.0,
    "rc_hex_triple":          0.0,
    "dsb1_radi_cal_energy":   0.0,
    "dsb1_radi_cal_triple":   0.0,
    "dsb1_rc_hex":            0.0,
    "dsb1_rc_hex_triple":     0.0,
    "luagce_radi_cal_energy": 0.0,
    "luagce_radi_cal_triple": 0.0,
    "luagce_rc_hex":          0.0,
    "luagce_rc_hex_triple":   0.0,
}

for mod in BOUNCE_FACTOR.keys():
    V_EFF_MM_NS = V_LIGHT_MM_NS * BOUNCE_FACTOR.get(mod, 0.92)


_GT_LO_NS = 0.0
_GT_HI_NS = 50.0
_TYVEK_THICK_MM = 0.2032
_W_THICK_MM = 2.5
_N_LYSO = 29
_N_W = 28

ARRIVAL_QUANTILE = 0.10
MIN_PHOTONS_PER_FACE = 1

# ── Geometry mappings ──────────────────────────────────────────────────────

_KNOWN_MODULE_LYSO_THICK = {
    # Baseline
    "radi_cal_energy": 1.5,
    "radi_cal_triple": 4.5,
    "rc_hex":          1.5,
    "rc_hex_triple":   4.5,
    # DSB1 Variants
    "dsb1_radi_cal_energy": 1.5,
    "dsb1_radi_cal_triple": 4.5,
    "dsb1_rc_hex":          1.5,
    "dsb1_rc_hex_triple":   4.5,
    # LuAG:Ce Variants
    "luagce_radi_cal_energy": 1.5,
    "luagce_radi_cal_triple": 4.5,
    "luagce_rc_hex":          1.5,
    "luagce_rc_hex_triple":   4.5,
}

_SQUARE_HOLE_OFFSET = 3.7032
SQUARE_CAP_XY = np.array([
    [ _SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],  # 0 (T)
    [-_SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],  # 1 (T)
    [-_SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],  # 2 (E)
    [ _SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],  # 3 (E)
])

HEX_CAP_R_MM = 3.5
HEX_CAP_XY = np.array([
    [HEX_CAP_R_MM * np.cos(np.pi / 2 + i * (np.pi / 3)), HEX_CAP_R_MM * np.sin(np.pi / 2 + i * (np.pi / 3))]
    for i in range(6)
])

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def v_eff_for_module(mod: str) -> float:
    return (C_LIGHT_MM_NS / REFRACTIVE_INDEX.get(mod, 1.60)) * BOUNCE_FACTOR.get(mod, 0.92)

def rebin_fine_profile_to_layers(fine_arr: np.ndarray, lyso_bounds: list, calor_thick_mm: float) -> np.ndarray:
    """
    Collapse a fine-resolution DoseActor voxel array (spanning the full
    calorimeter thickness, centered at z=0) down to one value per physical
    LYSO layer, using the same z-boundaries as get_lyso_layer_bounds().
    """
    n = len(fine_arr)
    if n == 0:
        return np.zeros(len(lyso_bounds))
    dz = calor_thick_mm / n
    centers = -calor_thick_mm / 2.0 + dz * (np.arange(n) + 0.5)
    binned = np.zeros(len(lyso_bounds))
    for i, (z_lo, z_hi) in enumerate(lyso_bounds):
        mask = (centers >= z_lo) & (centers < z_hi)
        binned[i] = fine_arr[mask].sum()
    return binned


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
# CORE ENGINE: DATA PARSING & COINCIDENCE FOLDING (vectorized)
# ─────────────────────────────────────────────────────────────────────────────
def _chunk_series(mask, values, ev, run_tag):
    n = int(mask.sum())
    if n == 0:
        return None
    idx = pd.MultiIndex.from_arrays([np.full(n, run_tag, dtype=object), ev[mask].astype(np.int64)])
    return pd.Series(values[mask], index=idx)

def _grouped(chunks, how):
    if not chunks:
        return {}
    s = pd.concat(chunks)
    g = s.groupby(level=[0, 1])
    if how == "min":
        s = g.min()
    elif how == "count":
        s = g.count()
    else:
        s = g.quantile(how)
    return {(k[0], int(k[1])): (int(v) if how == "count" else float(v)) for k, v in s.items()}

def analyze_energy_batch(batch_dir: Path, is_hex: bool, module_name: str, verbose_label: str = ""):
    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"))
    if not hit_files:
        if verbose_label:
            print(f"    [{verbose_label}] SKIPPED — no detector_hits_*.root files found")
        return None

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
        except Exception:
            continue

    if detected_z_sensor is None:
        return None

    lyso_thick = _KNOWN_MODULE_LYSO_THICK[module_name]
    v_light = v_eff_for_module(module_name)
    
    v_eff = v_eff_for_module(module_name)
    t_offset_ns = T_OFFSET_NS.get(module_name, 0.0)

    gap_thick_mm = lyso_thick + 2 * _TYVEK_THICK_MM
    calor_thick_mm = (_N_LYSO * gap_thick_mm) + (_N_W * _W_THICK_MM)
    lyso_bounds = get_lyso_layer_bounds(lyso_thick, calor_thick_mm)

    cap_xy_map = HEX_CAP_XY if is_hex else SQUARE_CAP_XY
    t_indices = list({1, 3, 5} if is_hex else {0, 1})
    e_indices = list({0, 2, 4} if is_hex else {2, 3})

    up_first_chunks, down_first_chunks = [], []
    up_q_chunks, dw_q_chunks = [], []
    dw_e_hit_chunks, dw_t_hit_chunks = [], []
    run_dirs = set()

    branch_list = ["Position_X", "Position_Y", "Position_Z", "GlobalTime", "LocalTime", "EventID", "ParticleName"]

    for fpath in hit_files:
        run_tag = fpath.parent.name
        run_dirs.add(fpath.parent)
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                tree = f[tk]
                if tree.num_entries == 0: continue
                arrs = tree.arrays(branch_list, library="np")
        except Exception:
            continue

        x, y, z = arrs["Position_X"], arrs["Position_Y"], arrs["Position_Z"]
        gt, lt, ev, pn = arrs["GlobalTime"], arrs["LocalTime"], arrs["EventID"], arrs["ParticleName"]

        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(np.hypot(dx, dy), axis=1)

        near_up = np.abs(z + detected_z_sensor) < 2.5
        near_dw = np.abs(z - detected_z_sensor) < 2.5
        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")
        gt = np.where(near_dw, gt + t_offset_ns, gt)

        is_e = np.isin(channels, e_indices)
        is_prompt = (gt >= _GT_LO_NS) & (gt <= _GT_HI_NS)

        m_e_up = is_e & is_prompt & near_up & is_optical
        m_e_dw = is_e & is_prompt & near_dw & is_optical

        c = _chunk_series(m_e_up, gt, ev, run_tag)
        if c is not None: up_first_chunks.append(c)
        c = _chunk_series(m_e_dw, gt, ev, run_tag)
        if c is not None:
            down_first_chunks.append(c)
            dw_e_hit_chunks.append(c)

        is_t = np.isin(channels, t_indices)
        m_t_up = is_t & is_optical & near_up
        m_t_dw = is_t & is_optical & near_dw

        c = _chunk_series(m_t_up, lt * 1000.0, ev, run_tag)
        if c is not None: up_q_chunks.append(c)
        c = _chunk_series(m_t_dw, lt * 1000.0, ev, run_tag)
        if c is not None:
            dw_q_chunks.append(c)
            dw_t_hit_chunks.append(c)

    up_first = _grouped(up_first_chunks, "min")
    down_first = _grouped(down_first_chunks, "min")
    up_q = _grouped(up_q_chunks, ARRIVAL_QUANTILE)
    dw_q = _grouped(dw_q_chunks, ARRIVAL_QUANTILE)
    dw_e_hits_per_ev = _grouped(dw_e_hit_chunks, "count")
    dw_t_hits_per_ev = _grouped(dw_t_hit_chunks, "count")

    common_t_evs = set(up_q) & set(dw_q)
    all_bm_raw_ps = np.array([(dw_q[e] - up_q[e]) / 2.0 for e in common_t_evs])
    clean_bm = clean_around_mode(all_bm_raw_ps, window_ps=500.0)
    _, _, sigma_t_ps = fit_gaussian_to_peak(clean_bm)

    common_e_keys = set(up_first) & set(down_first)
    z_lo, z_hi = -calor_thick_mm / 2 - 15.0, calor_thick_mm / 2 + 15.0
    valid_z_emits = np.array([
        z_est for z_est in (
            v_eff * (down_first[k] - up_first[k]) / 2.0 for k in common_e_keys
        ) if z_lo <= z_est <= z_hi
    ])

    profile_counts = np.zeros(_N_LYSO)
    if len(valid_z_emits) >= 5:
        kde = gaussian_kde(valid_z_emits, bw_method=0.15)
        for i, (zm, zx) in enumerate(lyso_bounds):
            profile_counts[i] = kde.evaluate((zm + zx) / 2.0)[0]
    else:
        for i, (zm, zx) in enumerate(lyso_bounds):
            profile_counts[i] = np.sum((valid_z_emits >= zm) & (valid_z_emits <= zx))

    profile_counts = profile_counts[::-1]

    if verbose_label:
        print(f"    [{verbose_label}] {len(run_dirs)} run dirs, "
              f"{len(common_t_evs)} T-coincidences, {len(common_e_keys)} E-coincidences "
              f"(sigma_t={sigma_t_ps:.1f}ps)")

    return {
        "sigma_t_ps": sigma_t_ps,
        "raw_bm_data": all_bm_raw_ps,
        "tof_profile": profile_counts,
        "lyso_thick": lyso_thick,
        "pitch_mm": gap_thick_mm + _W_THICK_MM,
        "n_t_coincidences": len(common_t_evs),
        "n_e_coincidences": len(common_e_keys),
        "dw_first_times": np.array(list(down_first.values())),
        "dw_e_total": np.array([dw_e_hits_per_ev.get(k, 0) for k in down_first.keys()]),
        "dw_t_total": np.array([dw_t_hits_per_ev.get(k, 0) for k in down_first.keys()]),
        "run_dirs": sorted(run_dirs),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def _run_job(args):
    mod, ekey, edir, is_hex = args
    res = analyze_energy_batch(edir, is_hex, mod, verbose_label=f"{mod}:{ekey}")
    return mod, ekey, res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", type=str, default=None,
                         help="Path to a previously pickled master_summary.")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of worker processes (default: os.cpu_count()-1).")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    modules = [
        "radi_cal_energy", "radi_cal_triple", "rc_hex", "rc_hex_triple",
        "dsb1_radi_cal_energy", "dsb1_radi_cal_triple", "dsb1_rc_hex", "dsb1_rc_hex_triple",
        "luagce_radi_cal_energy", "luagce_radi_cal_triple", "luagce_rc_hex", "luagce_rc_hex_triple"
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_out = base_dir / "12_sweep_analysis" / f"sweep_summary_{timestamp}"
    analysis_out.mkdir(parents=True, exist_ok=True)

    # ── SUBFOLDER GENERATION HIERARCHY ───────────────────────────────────────
    timing_dir = analysis_out / "timing_distributions"
    profile_dir = analysis_out / "longitudinal_profiles"
    energy_dir = analysis_out / "energy_performance"
    summary_dir = analysis_out / "summary_plots"

    for d in [timing_dir, profile_dir, energy_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    master_summary = {mod: {} for mod in modules}

    if args.from_cache:
        print(f"Loading cached master_summary from {args.from_cache}")
        with open(args.from_cache, "rb") as fh:
            master_summary = pickle.load(fh)
    else:
        print("Master processing engine spawned. Targeting tracking metrics...")

        jobs = []
        for mod in modules:
            mod_path = base_dir / mod / "runs" / mod
            if not mod_path.exists():
                mod_path = base_dir / mod 
                if not mod_path.exists():
                    print(f"  Skipping module '{mod}' (path not found)")
                    continue

            sweeps = sorted(mod_path.glob("sweep_*"), key=lambda p: p.name)
            if not sweeps: continue
            target_sweep = sweeps[-1]
            print(f"Queuing '{mod}' -> {target_sweep.name}")

            is_hex = "hex" in mod
            energy_dirs = sorted(target_sweep.glob("*GeV"), key=lambda p: extract_numerical_energy(p.name))
            for edir in energy_dirs:
                jobs.append((mod, edir.name, edir, is_hex))

        import os
        n_workers = min(100, max(1, (os.cpu_count() or 2) - 1)) if args.workers else max(1, (os.cpu_count() or 2) - 1)

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_run_job, job) for job in jobs]
            for fut in as_completed(futures):
                mod, ekey, res = fut.result()
                if res is not None:
                    master_summary[mod][ekey] = res

        cache_path = analysis_out / f"master_summary_{timestamp}.pkl"
        with open(cache_path, "wb") as fh:
            pickle.dump(master_summary, fh)

   # ── EXPLICIT DESIGN MAPS FOR THE 12 MODULE VARIANTS ──────────────────────
    mod_colors = {
        "radi_cal_energy":        "#f708af",  
        "radi_cal_triple":        "#f708af",  
        "rc_hex":                 "#f708af",  
        "rc_hex_triple":          "#f708af",  

        "dsb1_radi_cal_energy":   "#04207e",  
        "dsb1_radi_cal_triple":   "#04207e",  
        "dsb1_rc_hex":            "#04207e", 
        "dsb1_rc_hex_triple":     "#04207e",  

        "luagce_radi_cal_energy": "#fa0707",  
        "luagce_radi_cal_triple": "#fa0707",  
        "luagce_rc_hex":          "#fa0707",  
        "luagce_rc_hex_triple":   "#fa0707",  
    }

    mod_markers = {
        "radi_cal_energy":        "s", 
        "radi_cal_triple":        "s",  
        "rc_hex":                 "h",  
        "rc_hex_triple":          "h",  

        "dsb1_radi_cal_energy":   "s",
        "dsb1_radi_cal_triple":   "s",
        "dsb1_rc_hex":            "h",
        "dsb1_rc_hex_triple":     "h",

        "luagce_radi_cal_energy": "s",
        "luagce_radi_cal_triple": "s",
        "luagce_rc_hex":          "h",
        "luagce_rc_hex_triple":   "h",
    }

    mod_linestyles = {
        "radi_cal_energy":        ":",   
        "radi_cal_triple":        "--",  
        "rc_hex":                 ":",   
        "rc_hex_triple":          "--",  

        "dsb1_radi_cal_energy":   ":",
        "dsb1_radi_cal_triple":   "--",
        "dsb1_rc_hex":            ":",
        "dsb1_rc_hex_triple":     "--",

        "luagce_radi_cal_energy": ":",
        "luagce_radi_cal_triple": "--",
        "luagce_rc_hex":          ":",
        "luagce_rc_hex_triple":   "--",
    }

    layers = np.arange(1, _N_LYSO + 1)

    try:
        import analysis_utils as utils
    except ImportError:
        utils = None

    for mod in modules:
        if mod not in master_summary or not master_summary[mod]:
            continue

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

                if hi <= lo:
                    hi = lo + 1.0

                q75, q25 = np.percentile(clean, [75, 25])
                iqr = q75 - q25
                if iqr > 0 and len(clean) > 1:
                    fd_width = 2.0 * iqr / (len(clean) ** (1.0 / 3.0))
                else:
                    fd_width = 3.5 * np.std(clean) / (len(clean) ** (1.0 / 3.0)) if len(clean) > 1 else 5.0
                min_width = max(1.0, total_range / 50.0)
                optimal_width = max(min_width, min(fd_width, 10.0))
                plot_bins = max(3, int(np.ceil((hi - lo) / optimal_width)))
                actual_plot_width = (hi - lo) / plot_bins

                counts, edges, _ = ax.hist(clean, bins=plot_bins, range=(lo, hi),
                                            color=mod_colors.get(mod, "#f708af"), alpha=0.6, edgecolor="black", label="Data")

                def crystal_ball_binned(x, amp, mu, sigma, alpha, n):
                    zsc = (x - mu) / sigma
                    gauss = amp * np.exp(-0.5 * zsc ** 2)
                    a = (n / alpha) ** n * np.exp(-0.5 * alpha ** 2)
                    b = n / alpha - alpha
                    tail = amp * a * (b - zsc) ** (-n)
                    return np.where(zsc > -alpha, gauss, tail)

                bin_centers = (edges[:-1] + edges[1:]) / 2.0
                x_fit = np.linspace(lo, hi, 5000)

                peak_idx = np.argmax(counts)
                mu_guess = float(bin_centers[peak_idx])
                std_guess = float(np.std(clean)) if len(clean) > 1 else 10.0

                p0 = [float(counts.max()), mu_guess, std_guess * 0.6, 1.0, 3.0]
                bounds = ([0.0, lo, 0.1, 0.1, 1.05], [counts.max() * 2.0, hi, (hi - lo), 5.0, 20.0])

                try:
                    popt, _ = curve_fit(crystal_ball_binned, bin_centers, counts, p0=p0, bounds=bounds, maxfev=10000)
                    amp_f, mu_f, sigma_f, alpha_f, n_f = popt
                    master_summary[mod][ekey]["sigma_t_ps"] = sigma_f

                    y_fit = crystal_ball_binned(x_fit, amp_f, mu_f, sigma_f, alpha_f, n_f)
                    label_text = (f"Crystal Ball\n$\\mu$ = {mu_f:.1f} ps\n$\\sigma_{{core}}$ = {sigma_f:.1f} ps")
                except Exception:
                    _, mu, sigma = fit_gaussian_to_peak(clean, n_bins=40)
                    master_summary[mod][ekey]["sigma_t_ps"] = sigma

                    amplitude = (len(clean) * actual_plot_width) / (sigma * np.sqrt(2 * np.pi)) if sigma > 0 else counts.max()
                    y_fit = standard_gaussian(x_fit, amplitude, mu, sigma)
                    label_text = f"Gaussian Fallback\n$\\mu$ = {mu:.1f} ps\n$\\sigma_t$ = {sigma:.1f} ps"
                ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2.5, label=label_text)
                ax.legend(loc="upper right", fontsize=9)
        for idx in range(n_energies, len(axs_time)):
            fig_time.delaxes(axs_time[idx])

        fig_time.suptitle(f"Timing Resolution Distributions — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_time.tight_layout()
        fig_time.savefig(timing_dir / f"{mod}_timing_panels.png", dpi=200)
        plt.close(fig_time)

        # ─────────────────────────────────────────────────────────────────────
        # 2. LONGITUDINAL PROFILE RECONSTRUCTION & RL-UNFOLDING
        # ─────────────────────────────────────────────────────────────────────
        for ekey in energy_keys:
            raw_profile = master_summary[mod][ekey]["tof_profile"]
            sigma_t_ps = master_summary[mod][ekey]["sigma_t_ps"]
            pitch_mm = master_summary[mod][ekey]["pitch_mm"]
            lyso_thick = master_summary[mod][ekey]["lyso_thick"]
            run_dirs_ek = master_summary[mod][ekey]["run_dirs"]

            if np.sum(raw_profile) == 0:
                continue

            gap_thick_mm = lyso_thick + 2 * _TYVEK_THICK_MM
            calor_thick_mm = (_N_LYSO * gap_thick_mm) + (_N_W * _W_THICK_MM)
            lyso_bounds = get_lyso_layer_bounds(lyso_thick, calor_thick_mm)

            raw_norm_disp = raw_profile / np.sum(raw_profile)
            s_z = v_eff_for_module(mod) * (sigma_t_ps / 1000.0)
            sigma_layer = s_z / pitch_mm if pitch_mm > 0 else 1.0

            if utils is not None and hasattr(utils, 'rl_unfold'):
                unf_norm_disp, unf_err_disp = utils.rl_unfold(raw_norm_disp, sigma_layer)
            else:
                unf_norm_disp = raw_norm_disp
                unf_err_disp = np.zeros_like(raw_norm_disp)

            truth_curve = None
            if utils is not None:
                fine_truth, _ = utils.load_calorimeter_mhd(run_dirs_ek, long_glob="run_Dose_edep.mhd")
                if fine_truth is not None:
                    truth_curve = rebin_fine_profile_to_layers(fine_truth, lyso_bounds, calor_thick_mm)

            fig_prof, (ax_main, ax_ratio) = plt.subplots(
                2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True
            )

            if truth_curve is not None and np.sum(truth_curve) > 0:
                truth_norm_disp = truth_curve / np.sum(truth_curve)
                ax_main.bar(layers, truth_norm_disp, color="#00bcd4", alpha=0.25, edgecolor="#00838f",
                       linewidth=0.8, width=0.8, label="Sim Truth (DoseActor)")

                sigma_bins = np.where(unf_err_disp > 0, unf_err_disp, 1e-4)
                chi2 = np.sum(((unf_norm_disp - truth_norm_disp) / sigma_bins) ** 2)
                ndf = len(unf_norm_disp)
                reduced_chi2 = chi2 / ndf
                mae = np.mean(np.abs(unf_norm_disp - truth_norm_disp)) * 100
                fit_stats_label = f" (χ²/ndf={reduced_chi2:.2f}, MAE={mae:.1f}%)"

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = unf_norm_disp / truth_norm_disp
                    ratio_err = unf_err_disp / truth_norm_disp

                ax_ratio.errorbar(layers, ratio, yerr=ratio_err, color=mod_colors.get(mod, "black"),
                                  fmt='o-', markersize=3.5, linewidth=1.2, capsize=1.5, elinewidth=0.8)
                ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
                ax_ratio.set_ylabel("Unf / Truth", fontsize=8)
                ax_ratio.set_ylim(0.4, 1.6)
            else:
                ax_ratio.text(0.5, 0.5, "No Reference Truth", ha="center", va="center", alpha=0.4, transform=ax_ratio.transAxes)
                ax_ratio.set_ylim(0, 2)
                fit_stats_label = ""

            ax_main.plot(layers, raw_norm_disp, color="gray", linewidth=1.2, linestyle=":",
                    marker=".", markersize=3.5, alpha=0.7, label="Raw ΔT Profile (blurred)")

            ax_main.errorbar(layers, unf_norm_disp, yerr=unf_err_disp, color=mod_colors.get(mod, "black"),
                        linewidth=1.8, marker="o", markersize=4.0, capsize=2.5, capthick=0.9,
                        label=f"RL-Unfolded (σ_layer={sigma_layer:.2f}){fit_stats_label}")

            ax_main.set_ylabel("Normalized Intensity", fontsize=10)
            ax_main.set_title(f"Longitudinal Shower Profile — {mod} ({ekey})", fontsize=12, fontweight="bold")
            ax_main.grid(True, linestyle=":", alpha=0.6)
            ax_main.legend(fontsize=9)

            ax_ratio.set_xlabel("Layer Number", fontsize=10)
            ax_ratio.set_xticks(layers[::2])
            ax_ratio.grid(True, linestyle=":", alpha=0.6)

            fig_prof.tight_layout()
            fig_prof.savefig(profile_dir / f"{mod}_{ekey}_profile.png", dpi=200)
            plt.close(fig_prof)

        # ─────────────────────────────────────────────────────────────────────
        # 3. ENERGY LINEARITY AND RESOLUTION PANELS
        # ─────────────────────────────────────────────────────────────────────
        energies_gev, mu_e_list, res_e_list, mu_e_err, res_e_err = [], [], [], [], []

        for ekey in energy_keys:
            E_val = extract_numerical_energy(ekey)
            if E_val <= 0: continue

            e_totals = master_summary[mod][ekey].get("dw_e_total", np.array([]))
            if len(e_totals) < 5: continue

            _, mu_val, sigma_val = fit_gaussian_to_peak(e_totals, n_bins=40)

            if mu_val > 0:
                energies_gev.append(E_val)
                mu_e_list.append(mu_val)
                res_e_list.append(sigma_val / mu_val)
                mu_e_err.append(sigma_val / np.sqrt(len(e_totals)))
                res_e_err.append((sigma_val / mu_val) * (1.0 / np.sqrt(len(e_totals))))

        if len(energies_gev) >= 3:
            energies_gev = np.array(energies_gev)
            mu_e_list = np.array(mu_e_list)
            res_e_list = np.array(res_e_list)

            fig_er, (ax_lin, ax_res) = plt.subplots(1, 2, figsize=(14, 6))

            def linear_func(x, m, b): return m * x + b
            popt_lin, _ = curve_fit(linear_func, energies_gev, mu_e_list)

            ax_lin.errorbar(energies_gev, mu_e_list, yerr=mu_e_err, fmt=mod_markers.get(mod, 'o'),
                            color=mod_colors.get(mod, 'black'), label=f"Simulated Data ({mod})")

            x_lin_smooth = np.linspace(0, max(energies_gev) * 1.1, 100)
            ax_lin.plot(x_lin_smooth, linear_func(x_lin_smooth, *popt_lin),
                        color="black", linestyle="--", label=f"Fit: {popt_lin[0]:.3e} photons/GeV")

            ax_lin.set_xlabel("Beam Energy (GeV)", fontsize=11)
            ax_lin.set_ylabel("Sum Amplitude (Downstream E-Type Photons)", fontsize=11)
            ax_lin.set_title("Energy Linearity", fontsize=13, fontweight="bold")
            ax_lin.grid(True, linestyle=":", alpha=0.6)
            ax_lin.legend(fontsize=10)

            def resolution_func(E, c, s, n):
                return np.sqrt(c ** 2 + (s / np.sqrt(E)) ** 2 + (n / E) ** 2)

            try:
                popt_res, _ = curve_fit(resolution_func, energies_gev, res_e_list,
                                        p0=[0.05, 0.2, 0.05], bounds=(0, [2.0, 10.0, 10.0]))
                c_f, s_f, n_f = popt_res
                fit_label = f"Fit: {c_f * 100:.1f}% $\\oplus$ {s_f * 100:.1f}%/$\\sqrt{{E}}$ $\\oplus$ {n_f * 100:.1f}%/E"
            except Exception:
                popt_res = [0.0, 0.0, 0.0]
                fit_label = "Fit failed"

            ax_res.errorbar(energies_gev, res_e_list, yerr=res_e_err, fmt=mod_markers.get(mod, 'o'),
                            color=mod_colors.get(mod, 'black'), label="Simulated Resolution")

            x_res_smooth = np.linspace(min(energies_gev) * 0.8, max(energies_gev) * 1.1, 100)
            ax_res.plot(x_res_smooth, resolution_func(x_res_smooth, *popt_res),
                        color="black", linestyle="--", label=fit_label)

            ax_res.set_xlabel("Beam Energy (GeV)", fontsize=11)
            ax_res.set_ylabel(r"$\sigma_E / E_{meas}$", fontsize=11)
            ax_res.set_title("Energy Resolution", fontsize=13, fontweight="bold")
            ax_res.grid(True, linestyle=":", alpha=0.6)
            ax_res.legend(fontsize=10)

            fig_er.suptitle(f"Calorimeter Energy Performance — {mod}", fontsize=15, fontweight="bold")
            fig_er.tight_layout()
            fig_er.savefig(energy_dir / f"{mod}_energy_performance.png", dpi=200)
            plt.close(fig_er)

    # ─────────────────────────────────────────────────────────────────────
    # 4. UNIFIED OVERALL PERFORMANCE HORIZON COMPARISON GRAPH
    # ─────────────────────────────────────────────────────────────────────
    fig_perf, ax_perf = plt.subplots(figsize=(10, 7))
    any_points = False

    for mod in modules:
        if mod not in master_summary or not master_summary[mod]:
            continue

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
            ax_perf.errorbar(
                x_energy, y_res, yerr=y_err, 
                marker=mod_markers.get(mod, 'o'), 
                color=mod_colors.get(mod, 'black'),
                linestyle=mod_linestyles.get(mod, '-'),  
                linewidth=2, 
                markersize=7, 
                capsize=4, 
                capthick=1.5, 
                label=mod
            )

    ax_perf.set_xlabel("Incident Particle Beam Energy (GeV)", fontweight="bold")
    ax_perf.set_ylabel(r"BestMinus Timing Resolution $\sigma_t$ (ps)", fontweight="bold")
    ax_perf.set_title("Unified Performance Horizon — Timing Resolution vs Energy", fontsize=12, fontweight="bold")
    ax_perf.grid(True, linestyle=":", alpha=0.6)
    ax_perf.set_xscale("log")
    ax_perf.set_xticks([25, 50, 100, 200])
    ax_perf.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    if any_points:
        ax_perf.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9, frameon=True)
    else:
        ax_perf.text(0.5, 0.5, "No modules had sufficient statistics", ha='center', va='center', transform=ax_perf.transAxes)

    fig_perf.tight_layout()

    key_text = (
        "VISUAL ENCODING KEY\n"
        "───────────────────\n"
        "• Colors (Material Class):\n"
        "  Pinks   = BCF92 Baseline\n"
        "  Navys  = DSB1 Variants\n"
        "  Reds    = LuAG:Ce Variants\n\n"
        "• Line Styles (Thickness):\n"
        "  Dot  = Single (1.5mm LYSO )\n"
        "  Dash (--) = Triple (4.5mm LYSO)\n\n"
        "Marker Styles: Shape\n"
        "Square: square \n" 
        "Hexagon: hexagon"
    )

    fig_perf.text(
        1.02, 0.15, key_text, 
        fontsize=9, 
        family='monospace', 
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', edgecolor='#d3d3d3', alpha=0.9)
    )
    fig_perf.savefig(summary_dir / "timing_resolution_vs_energy.png", dpi=220, bbox_inches="tight")
    plt.close(fig_perf)

    # ─────────────────────────────────────────────────────────────────────
    # 5. EXPORT MASTER MATRIX TEXT REPORT
    # ─────────────────────────────────────────────────────────────────────
    sheet_path = analysis_out / "timing_vs_energy_report.txt"
    with open(sheet_path, "w") as f:
        f.write(f"{'=' * 80}\n")
        f.write(" RADiCAL SIMULATION UNIFIED RUN SUMMARY SHEET\n")
        f.write(f" Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 80}\n\n")

        for mod in modules:
            if mod not in master_summary or not master_summary[mod]:
                continue

            f.write(f"MODULE: {mod}\n")
            f.write(f"{'-' * 65}\n")
            f.write(f"  {'Energy':<12} | {'sigma_t (ps)':<16} | {'sigma_z (mm)':<14} | {'sigma_layer':<12} | {'N events (T/E)':<15}\n")
            f.write(f"{'-' * 65}\n")

            energy_keys = sorted(master_summary[mod].keys(), key=extract_numerical_energy)
            for ekey in energy_keys:
                s_t = master_summary[mod][ekey]["sigma_t_ps"]
                s_z = v_eff_for_module(mod) * (s_t / 1000.0)
                pitch = master_summary[mod][ekey]["pitch_mm"]
                s_layer = s_z / pitch if pitch > 0 else 0
                n_t = master_summary[mod][ekey]["n_t_coincidences"]
                n_e = master_summary[mod][ekey]["n_e_coincidences"]
                f.write(f"  {ekey:<12} | {s_t:<16.1f} | {s_z:<14.2f} | {s_layer:<12.2f} | {n_t}/{n_e}\n")
            f.write("\n")

    print(f"\nDone. Outputs written to {analysis_out}")


if __name__ == "__main__":
    main()