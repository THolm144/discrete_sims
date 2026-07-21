#!/usr/bin/env python3
"""
unified_sweep_analysis_optimized.py
=====================================
Optimized version for aggregating timing-resolution, ToF-reconstruction, 
and energy linearity results for RADiCAL geometry variants with 4 T-type fibers.
"""
import argparse
import datetime
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import itertools
import re
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
import ROOT
import uuid

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
    "dsb1_radi_cal_4t":       1.55,   # DSB1 with 4 T-type fibers
    "dsb1_radi_cal_triple":   1.55,
    "dsb1_rc_hex":            1.55,
    "dsb1_rc_hex_triple":     1.55,
    "luagce_radi_cal_energy": 1.84,   # LuAG:Ce
    "luagce_radi_cal_triple": 1.84,
    "luagce_rc_hex":          1.84,
    "luagce_rc_hex_triple":   1.84,
}

BOUNCE_FACTOR = {
    "radi_cal_energy":        0.92,
    "radi_cal_triple":        0.92,
    "rc_hex":                 0.92,
    "rc_hex_triple":          0.92,
    "dsb1_radi_cal_energy":   0.92,
    "dsb1_radi_cal_4t":       0.92,
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
    "dsb1_radi_cal_4t":       0.0,
    "dsb1_radi_cal_triple":   0.0,
    "dsb1_rc_hex":            0.0,
    "dsb1_rc_hex_triple":     0.0,
    "luagce_radi_cal_energy": 0.0,
    "luagce_radi_cal_triple": 0.0,
    "luagce_rc_hex":          0.0,
    "luagce_rc_hex_triple":   0.0,
}

_GT_LO_NS = 0.0
_GT_HI_NS = 1000.0
_TYVEK_THICK_MM = 0.2032
_W_THICK_MM = 2.5
_N_LYSO = 29
_N_W = 28

ARRIVAL_QUANTILE = 0.10
MIN_PHOTONS_PER_FACE = 1

# ── Geometry mappings ──────────────────────────────────────────────────────

_KNOWN_MODULE_LYSO_THICK = {
    "radi_cal_energy":      1.5,
    "radi_cal_triple":      4.5,
    "rc_hex":               1.5,
    "rc_hex_triple":        4.5,
    "dsb1_radi_cal_energy": 1.5,
    "dsb1_radi_cal_4t":     1.5,
    "dsb1_radi_cal_triple": 4.5,
    "dsb1_rc_hex":          1.5,
    "dsb1_rc_hex_triple":   4.5,
    "luagce_radi_cal_energy": 1.5,
    "luagce_radi_cal_triple": 4.5,
    "luagce_rc_hex":        1.5,
    "luagce_rc_hex_triple": 4.5,
}

_SQUARE_HOLE_OFFSET = 3.7032
SQUARE_CAP_XY = np.array([
    [ _SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],  # 0
    [-_SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],  # 1
    [-_SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],  # 2
    [ _SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],  # 3
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
    return (C_LIGHT_MM_NS / REFRACTIVE_INDEX.get(mod, 1.55)) * BOUNCE_FACTOR.get(mod, 0.92)

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

def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def robust_resolution(data, nsig=2.0, max_iters=4):
    """
    Computes fractional resolution (sigma/mean in %) with uncertainty.
    Uses ROOT's Log-Likelihood Minuit fit to replicate standard C++ processing.
    """
    N = len(data)
    if N < 2:
        return -1.0, 1e9  

    median = np.median(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    sg_robust = iqr / 1.349 
    if sg_robust == 0:  
        sg_robust = np.std(data, ddof=1)

    fallback_res = 100.0 * sg_robust / median if median > 0 else -1.0
    fallback_err = fallback_res / np.sqrt(2.0 * N) if (N > 1 and fallback_res > 0) else 1e9

    if median <= 0 or sg_robust <= 0:
        return fallback_res, fallback_err

    unique_id = uuid.uuid4().hex
    hname = f"h_{unique_id}"
    fname = f"f_{unique_id}"

    hist_min = max(0, int(np.floor(median - 5 * sg_robust)))
    hist_max = int(np.ceil(median + 5 * sg_robust))
    nbins = max(10, hist_max - hist_min + 1)
    h = ROOT.TH1D(hname, "temp_hist", nbins, hist_min - 0.5, hist_max + 0.5)
    h.SetDirectory(0)

    for val in data:
        h.Fill(val)

    mu = h.GetMean()
    sg = h.GetRMS()
    g = ROOT.TF1(fname, "gaus", mu - nsig * sg, mu + nsig * sg)

    sigma_err = 0.0
    for _ in range(max_iters):
        g.SetRange(mu - nsig * sg, mu + nsig * sg)
        h.Fit(g, "RQL0")

        mu = g.GetParameter(1)
        sg = g.GetParameter(2)
        sigma_err = g.GetParError(2)

        if sg <= 0:
            break

    fit_ok = (mu > 0) and (sg > 0) and (sigma_err > 0) and (sigma_err / sg < 0.25)

    if fit_ok:
        res = 100.0 * sg / mu
        err = 100.0 * sigma_err / mu
        return res, err
    else:
        return fallback_res, fallback_err

def standard_gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian_to_peak(data, n_bins=40):
    if len(data) < 8:
        mu0 = float(np.median(data)) if len(data) else 0.0
        sg0 = float(np.std(data)) if len(data) else 0.0
        return 0.0, mu0, sg0, -1.0
    spread = max(np.std(data), 1.0)
    lo, hi = float(np.min(data)), float(np.max(data))
    if hi <= lo:
        hi = lo + 1.0

    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi), density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(counts.astype(float), sigma=2.0)
    peak_idx = int(np.argmax(smoothed))
    mu0, A0 = float(mids[peak_idx]), float(smoothed[peak_idx])

    try:
        popt, pcov = curve_fit(
            standard_gaussian, mids, counts,
            p0=[A0, mu0, spread],
            bounds=([0.0, lo, 1e-6], [A0 * 10.0 + 1.0, hi, (hi - lo)]),
            maxfev=10000,
        )
        sigma_err = float(np.sqrt(pcov[2, 2])) if np.isfinite(pcov[2, 2]) and pcov[2, 2] > 0 else -1.0
        return float(popt[0]), float(popt[1]), float(popt[2]), sigma_err
    except Exception:
        return A0, mu0, spread, -1.0

def clean_around_mode(arr, window_ps=500.0):
    if len(arr) == 0:
        return arr
    counts, edges = np.histogram(arr, bins=40, density=True)
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

def analyze_energy_batch(batch_dir: Path, is_hex: bool = False, module_name: str = "dsb1_radi_cal_4t", verbose_label: str = "", all_t_type: bool = True):
    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"))
    if not hit_files:
        if verbose_label:
            print(f"    [{verbose_label}] SKIPPED — no detector_hits_*.root files found")
        return None

    # --- Sensor Coordinate Discovery ---
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

    # --- Geometry & Constants Configuration ---
    lyso_thick = _KNOWN_MODULE_LYSO_THICK.get(module_name, 1.5)
    v_eff = v_eff_for_module(module_name)
    t_offset_ns = T_OFFSET_NS.get(module_name, 0.0)

    gap_thick_mm = lyso_thick + 2 * _TYVEK_THICK_MM
    calor_thick_mm = (_N_LYSO * gap_thick_mm) + (_N_W * _W_THICK_MM)
    lyso_bounds = get_lyso_layer_bounds(lyso_thick, calor_thick_mm)

    cap_xy_map = HEX_CAP_XY if is_hex else SQUARE_CAP_XY

    # --- Fiber Channel Mapping for 4 T-Type Fibers ---
    if all_t_type:
        t_indices = list(range(len(cap_xy_map)))  # All 4 channels [0, 1, 2, 3] are T-type
        e_indices = []
    else:
        t_indices = list({1, 3, 5} if is_hex else {0, 1})
        e_indices = list({0, 2, 4} if is_hex else {2, 3})

    # --- Chunk Initializations ---
    up_first_chunks, down_first_chunks = [], []
    up_q_chunks, dw_q_chunks = [], []

    up_e_hit_chunks, dw_e_hit_chunks = [], []
    up_t_hit_chunks, dw_t_hit_chunks = [], []

    down_first_t_chunks = []
    run_dirs = set()

    branch_list = ["Position_X", "Position_Y", "Position_Z", "GlobalTime", "TrackCreatorProcess", "LocalTime", "EventID", "ParticleName"]

    # --- Processing Loop ---
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
        is_prompt = (gt >= _GT_LO_NS) & (gt <= _GT_HI_NS)

        # 1. E-Type Channel Processing (Empty if 4 T-Type)
        if len(e_indices) > 0:
            is_e = np.isin(channels, e_indices)
            m_e_up = is_e & is_prompt & near_up & is_optical
            m_e_dw = is_e & is_prompt & near_dw & is_optical

            c = _chunk_series(m_e_up, gt, ev, run_tag)
            if c is not None:
                up_first_chunks.append(c)
                up_e_hit_chunks.append(c)

            c = _chunk_series(m_e_dw, gt, ev, run_tag)
            if c is not None:
                down_first_chunks.append(c)
                dw_e_hit_chunks.append(c)

        # 2. T-Type Channel Processing (All 4 Fibers)
        is_t = np.isin(channels, t_indices)

        m_t_up = is_t & is_optical & near_up
        m_t_dw = is_t & is_optical & near_dw
        c = _chunk_series(m_t_up, lt * 1000.0, ev, run_tag)
        if c is not None: up_q_chunks.append(c)
        c = _chunk_series(m_t_dw, lt * 1000.0, ev, run_tag)
        if c is not None: dw_q_chunks.append(c)

        m_t_up_prompt = is_t & is_optical & near_up & is_prompt
        m_t_dw_prompt = is_t & is_optical & near_dw & is_prompt

        c = _chunk_series(m_t_up_prompt, gt, ev, run_tag)
        if c is not None: up_t_hit_chunks.append(c)
        c = _chunk_series(m_t_dw_prompt, gt, ev, run_tag)
        if c is not None:
            dw_t_hit_chunks.append(c)
            down_first_t_chunks.append(c)

    # --- Aggregations & Grouping ---
    up_q = _grouped(up_q_chunks, ARRIVAL_QUANTILE)
    dw_q = _grouped(dw_q_chunks, ARRIVAL_QUANTILE)

    up_t_hits_per_ev = _grouped(up_t_hit_chunks, "count")
    dw_t_hits_per_ev = _grouped(dw_t_hit_chunks, "count")

    down_first_t = _grouped(down_first_t_chunks, "min")

    # Fallback to T-type for first arrival times if E-type is empty
    up_first = _grouped(up_first_chunks if up_first_chunks else up_t_hit_chunks, "min")
    down_first = _grouped(down_first_chunks if down_first_chunks else down_first_t_chunks, "min")

    # --- Time-of-Flight & Profiles ---
    common_t_evs = set(up_q) & set(dw_q)
    all_bm_raw_ps = np.array([(dw_q[e] - up_q[e]) / 2.0 for e in common_t_evs])
    clean_bm = clean_around_mode(all_bm_raw_ps, window_ps=500.0)
    sigma_t_ps = 0.0
    if len(clean_bm) > 3:
        _, _, sigma_t_ps = fit_gaussian_to_peak(clean_bm)[:3]

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
              f"{len(common_t_evs)} T-coincidences (sigma_t={sigma_t_ps:.1f}ps)")

    # --- Explicit Event Alignment and Total Yields ---
    master_t_events = sorted(list(down_first_t.keys()))
    dw_t_total = np.array([dw_t_hits_per_ev.get(k, 0) + up_t_hits_per_ev.get(k, 0) for k in master_t_events])

    return {
        "sigma_t_ps": sigma_t_ps,
        "raw_bm_data": all_bm_raw_ps,
        "tof_profile": profile_counts,
        "lyso_thick": lyso_thick,
        "pitch_mm": gap_thick_mm + _W_THICK_MM,
        "n_t_coincidences": len(common_t_evs),
        "n_e_coincidences": len(common_e_keys),

        "dw_first_times": np.array([down_first_t[k] for k in master_t_events if k in down_first_t]),

        # Combined 4 T-fiber photon yields
        "dw_e_total": dw_t_total,       # Direct mapping for standard analysis scripts
        "dw_t_total": dw_t_total,
        "run_dirs": sorted(run_dirs),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analyze 4 T-Fiber RADiCAL Energy Sweep Data")
    parser.add_argument("--runs-dir", type=str, default="./runs", help="Base path containing simulation sweep runs")
    parser.add_argument("--module", type=str, default="dsb1_radi_cal_4t", help="Module geometry tag")
    parser.add_argument("--output-dir", type=str, default="./analysis_output", help="Directory to save output plots/results")
    args = parser.parse_args()

    base_path = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========================================================================")
    print(f" Starting Analysis for 4-T Fiber Sweep in: {base_path.resolve()}")
    print("========================================================================")

    # Locate the latest sweep directory or search directly
    sweep_dirs = sorted(base_path.glob("**/sweep_*"))
    if sweep_dirs:
        target_sweep = sweep_dirs[-1]
        print(f" Found latest sweep directory: {target_sweep}")
    else:
        target_sweep = base_path
        print(f" Scanning directory: {target_sweep}")

    # Discover energy subfolders (e.g. 30GeV, 50GeV, 70GeV, 90GeV)
    energy_dirs = sorted([d for d in target_sweep.iterdir() if d.is_dir() and "GeV" in d.name],
                         key=lambda x: extract_numerical_energy(x.name))

    if not energy_dirs:
        print(f" Error: No *GeV directories found inside {target_sweep}")
        return

    energies_gev = []
    mean_yields = []
    res_percent = []
    res_err_percent = []
    timing_res_ps = []

    for edir in energy_dirs:
        e_val = extract_numerical_energy(edir.name)
        energies_gev.append(e_val)
        print(f"\n [+] Processing [{edir.name}]...")

        res_dict = analyze_energy_batch(edir, is_hex=False, module_name=args.module, verbose_label=edir.name, all_t_type=True)
        if res_dict is None or len(res_dict["dw_t_total"]) == 0:
            print(f"   [!] Warning: No valid hits retrieved for {edir.name}")
            mean_yields.append(0.0)
            res_percent.append(0.0)
            res_err_percent.append(0.0)
            timing_res_ps.append(0.0)
            continue

        photon_counts = res_dict["dw_t_total"]
        mean_N = np.mean(photon_counts)
        res, err = robust_resolution(photon_counts)
        st_ps = res_dict["sigma_t_ps"]

        mean_yields.append(mean_N)
        res_percent.append(res)
        res_err_percent.append(err)
        timing_res_ps.append(st_ps)

        print(f"     -> Events: {len(photon_counts)}")
        print(f"     -> Mean Light Yield: {mean_N:.2f} optical photons (4 T-fibers)")
        print(f"     -> Energy Resolution: {res:.2f}% ± {err:.2f}%")
        print(f"     -> Timing Resolution (10% quantile): {st_ps:.1f} ps")

    # Save summary dataframe
    df_summary = pd.DataFrame({
        "Energy_GeV": energies_gev,
        "Mean_Photons_4T": mean_yields,
        "Energy_Resolution_Percent": res_percent,
        "Energy_Resolution_Err_Percent": res_err_percent,
        "Timing_Resolution_ps": timing_res_ps,
    })
    csv_path = out_dir / "sweep_4T_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n [✓] Results saved to: {csv_path}")

    # Plot Energy Resolution & Light Yield
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Beam Energy [GeV]', fontsize=12)
    ax1.set_ylabel(r'Energy Resolution $\sigma_E / E$ [%]', color=color, fontsize=12)
    ax1.errorbar(energies_gev, res_percent, yerr=res_err_percent, fmt='o-', color=color, lw=2, capsize=4, label='Resolution')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Photon Yield [4 T-Fibers]', color=color, fontsize=12)
    ax2.plot(energies_gev, mean_yields, 's--', color=color, lw=2, label='Light Yield')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'4 T-Fiber Dynamic Optical Sweep ({args.module})', fontsize=14)
    fig.tight_layout()
    plot_path = out_dir / "energy_resolution_linearity_4T.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" [✓] Summary plot generated: {plot_path}")

if __name__ == "__main__":
    main()