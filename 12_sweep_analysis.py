"""
unified_profile_analysis.py
================================================================================
Advanced spatial, temporal, and prompt reconstruction analysis for 12 RADiCAL 
geometry variants, utilizing identical data-loading as the baseline script.
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

_GT_LO_NS = 0.0
_GT_HI_NS = 50.0
_TYVEK_THICK_MM = 0.2032
_W_THICK_MM = 2.5
_N_LYSO = 29
_N_W = 28

ARRIVAL_QUANTILE = 0.10

# ── Geometry mappings ──────────────────────────────────────────────────────

_KNOWN_MODULE_LYSO_THICK = {
    "radi_cal_energy": 1.5,
    "radi_cal_triple": 4.5,
    "rc_hex":          1.5,
    "rc_hex_triple":   4.5,
    "dsb1_radi_cal_energy": 1.5,
    "dsb1_radi_cal_triple": 4.5,
    "dsb1_rc_hex":          1.5,
    "dsb1_rc_hex_triple":   4.5,
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
# HELPERS (Retained exactly from baseline)
# ─────────────────────────────────────────────────────────────────────────────
def v_eff_for_module(mod: str) -> float:
    return (C_LIGHT_MM_NS / REFRACTIVE_INDEX.get(mod, 1.60)) * BOUNCE_FACTOR.get(mod, 0.92)

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

    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi), density=True)
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
# CORE ENGINE: DATA PARSING & VECTORIZED EXTRACTIONS (Identical path matching)
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

def analyze_profile_batch(batch_dir: Path, is_hex: bool, module_name: str, verbose_label: str = ""):
    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"))
    if not hit_files:
        if verbose_label:
            print(f"    [{verbose_label}] SKIPPED — no detector_hits_*.root files found")
        return None

    # Dynamic z-sensor position extraction exactly matching baseline
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

    # Pre-allocating structures for downstream time profiling (Histograms bypass heavy memory overhead)
    gt_bins = np.linspace(0.0, 100.0, 501)
    lt_bins = np.linspace(0.0, 25.0, 501)
    gt_counts = np.zeros(500)
    lt_counts = np.zeros(500)

    # Calculate expected arrival times from each layer to downstream SiPM (Graph 4)
    expected_times = []
    for z_lo, z_hi in lyso_bounds:
        z_center = (z_lo + z_hi) / 2.0
        dist_to_downstream = np.abs(detected_z_sensor - z_center)
        t_expected = dist_to_downstream / v_eff # (mm) / (mm/ns) = ns
        expected_times.append(t_expected)

    # Prompt photon selection window (LocalTime tolerance window around calculated target flight time)
    half_window_ns = 0.15 # ±150 ps
    prompt_counts = np.zeros(_N_LYSO)
    total_events_processed = 0

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

        total_events_processed += len(np.unique(ev))

        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(np.hypot(dx, dy), axis=1)

        near_up = np.abs(z + detected_z_sensor) < 2.5
        near_dw = np.abs(z - detected_z_sensor) < 2.5
        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")
        gt = np.where(near_dw, gt + t_offset_ns, gt)

        # Baseline logic tracking for Energy hits & dual-ended timing
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

        # ── EXTRACTIONS FOR NEW GRAPHS ───────────────────────────────────────
        m_dw_opt = near_dw & is_optical
        gt_downstream_opt = gt[m_dw_opt]
        lt_downstream_opt = lt[m_dw_opt]

        # Histogramming downstream optical strikes in-place
        hist_gt, _ = np.histogram(gt_downstream_opt, bins=gt_bins)
        gt_counts += hist_gt

        hist_lt, _ = np.histogram(lt_downstream_opt, bins=lt_bins)
        lt_counts += hist_lt

        # Layer-reconstruction logic (Graph 4)
        for layer_idx, t_exp in enumerate(expected_times):
            prompt_mask = (lt_downstream_opt >= (t_exp - half_window_ns)) & (lt_downstream_opt <= (t_exp + half_window_ns))
            prompt_counts[layer_idx] += np.sum(prompt_mask)

    # Finalize coincidence maps
    up_q = _grouped(up_q_chunks, ARRIVAL_QUANTILE)
    dw_q = _grouped(dw_q_chunks, ARRIVAL_QUANTILE)
    down_first = _grouped(down_first_chunks, "min")
    dw_e_hits_per_ev = _grouped(dw_e_hit_chunks, "count")

    # Double-ended timing calculation (Graph 5)
    common_t_evs = set(up_q) & set(dw_q)
    t_two_end_list = []
    for e in common_t_evs:
        # Coincidence time = (t_up + t_down) / 2 (ps)
        t_two_end_list.append((dw_q[e] + up_q[e]) / 2.0)
    t_two_end_raw = np.array(t_two_end_list)

    if verbose_label:
        print(f"    [{verbose_label}] {len(run_dirs)} run dirs, "
              f"{len(common_t_evs)} double-ended coincidences, "
              f"Mean Prompt strikes: {np.mean(prompt_counts):.1f}")

    return {
        "dw_e_total": np.array([dw_e_hits_per_ev.get(k, 0) for k in down_first.keys()]),
        "gt_counts": gt_counts,
        "gt_bins": gt_bins,
        "lt_counts": lt_counts,
        "lt_bins": lt_bins,
        "prompt_profile": prompt_counts / max(1, total_events_processed), # Mean hits per event
        "t_two_end_raw": t_two_end_raw,
        "n_t_coincidences": len(common_t_evs),
        "run_dirs": sorted(run_dirs),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def _run_job(args):
    mod, ekey, edir, is_hex = args
    res = analyze_profile_batch(edir, is_hex, mod, verbose_label=f"{mod}:{ekey}")
    return mod, ekey, res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", type=str, default=None,
                         help="Path to a previously pickled master_summary.")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of worker processes.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    modules = [
        "radi_cal_energy", "radi_cal_triple", "rc_hex", "rc_hex_triple",
        "dsb1_radi_cal_energy", "dsb1_radi_cal_triple", "dsb1_rc_hex", "dsb1_rc_hex_triple",
        "luagce_radi_cal_energy", "luagce_radi_cal_triple", "luagce_rc_hex", "luagce_rc_hex_triple"
    ]

    # Setup specialized profile_analysis directories
    out_dir = base_dir / "profile_analysis"
    energy_dir = out_dir / "energy_performance"
    global_dir = out_dir / "globaltime"
    local_dir = out_dir / "localtime"
    prompt_dir = out_dir / "prompt_photon_reconstruction"
    two_end_dir = out_dir / "two_end_timing"

    for d in [energy_dir, global_dir, local_dir, prompt_dir, two_end_dir]:
        d.mkdir(parents=True, exist_ok=True)

    master_summary = {mod: {} for mod in modules}

    if args.from_cache:
        print(f"Loading cached master_summary from {args.from_cache}")
        with open(args.from_cache, "rb") as fh:
            master_summary = pickle.load(fh)
    else:
        print("Master profile processing engine spawned...")
        jobs = []
        for mod in modules:
            mod_path = base_dir / mod / "runs" / mod
            if not mod_path.exists():
                mod_path = base_dir / mod 
                if not mod_path.exists():
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

        # Cache summary to prevent duplicate slow runs
        with open(out_dir / "master_profile_summary.pkl", "wb") as fh:
            pickle.dump(master_summary, fh)

    # Stylistic configuration maps matching your baseline
    mod_colors = {
        "radi_cal_energy": "#f708af", "radi_cal_triple": "#f708af", "rc_hex": "#f708af", "rc_hex_triple": "#f708af",
        "dsb1_radi_cal_energy": "#04207e", "dsb1_radi_cal_triple": "#04207e", "dsb1_rc_hex": "#04207e", "dsb1_rc_hex_triple": "#04207e",
        "luagce_radi_cal_energy": "#fa0707", "luagce_radi_cal_triple": "#fa0707", "luagce_rc_hex": "#fa0707", "luagce_rc_hex_triple": "#fa0707",
    }
    mod_markers = {
        "radi_cal_energy": "s", "radi_cal_triple": "s", "rc_hex": "h", "rc_hex_triple": "h",
        "dsb1_radi_cal_energy": "s", "dsb1_radi_cal_triple": "s", "dsb1_rc_hex": "h", "dsb1_rc_hex_triple": "h",
        "luagce_radi_cal_energy": "s", "luagce_radi_cal_triple": "s", "luagce_rc_hex": "h", "luagce_rc_hex_triple": "h",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH PLOTTING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for mod in modules:
        if mod not in master_summary or not master_summary[mod]:
            continue

        energy_keys = sorted(master_summary[mod].keys(), key=extract_numerical_energy)
        if not energy_keys:
            continue

        # ── GRAPH 1: ENERGY PERFORMANCE (Linearity & Resolution) ─────────────────
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

            fig_er, (ax_lin, ax_res) = plt.subplots(1, 2, figsize=(14, 5))

            def linear_func(x, m, b): return m * x + b
            popt_lin, _ = curve_fit(linear_func, energies_gev, mu_e_list)

            ax_lin.errorbar(energies_gev, mu_e_list, yerr=mu_e_err, fmt=mod_markers.get(mod, 'o'),
                            color=mod_colors.get(mod, 'black'), label=f"Data ({mod})")
            x_lin_smooth = np.linspace(0, max(energies_gev) * 1.1, 100)
            ax_lin.plot(x_lin_smooth, linear_func(x_lin_smooth, *popt_lin),
                        color="black", linestyle="--", label=f"Fit: {popt_lin[0]:.3e} ph/GeV")

            ax_lin.set_xlabel("Beam Energy (GeV)", fontweight="bold")
            ax_lin.set_ylabel("Sum Amplitude (Downstream Photons)", fontweight="bold")
            ax_lin.set_title("Energy Linearity", fontsize=12, fontweight="bold")
            ax_lin.grid(True, linestyle=":", alpha=0.6)
            ax_lin.legend(fontsize=9)

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
                            color=mod_colors.get(mod, 'black'), label="Data Resolution")
            x_res_smooth = np.linspace(min(energies_gev) * 0.8, max(energies_gev) * 1.1, 100)
            ax_res.plot(x_res_smooth, resolution_func(x_res_smooth, *popt_res),
                        color="black", linestyle="--", label=fit_label)

            ax_res.set_xlabel("Beam Energy (GeV)", fontweight="bold")
            ax_res.set_ylabel(r"$\sigma_E / E_{meas}$", fontweight="bold")
            ax_res.set_title("Energy Resolution", fontsize=12, fontweight="bold")
            ax_res.grid(True, linestyle=":", alpha=0.6)
            ax_res.legend(fontsize=9)

            fig_er.suptitle(f"Calorimeter Energy Performance — {mod}", fontsize=13, fontweight="bold")
            fig_er.tight_layout()
            fig_er.savefig(energy_dir / f"{mod}_energy_performance.png", dpi=200)
            plt.close(fig_er)

        # ── GRAPH 2: GLOBAL TIME VS PARTICLE STRIKES (Downstream) ────────────────
        fig_gt, ax_gt = plt.subplots(figsize=(8, 5.5))
        for ekey in energy_keys:
            counts = master_summary[mod][ekey]["gt_counts"]
            bins = master_summary[mod][ekey]["gt_bins"]
            bin_mids = 0.5 * (bins[:-1] + bins[1:])
            ax_gt.plot(bin_mids, counts, label=ekey, alpha=0.8, linewidth=1.5)

        ax_gt.set_xlabel("GlobalTime (ns)", fontweight="bold")
        ax_gt.set_ylabel("Optical Photon Strikes (Downstream)", fontweight="bold")
        ax_gt.set_title(f"Downstream GlobalTime Timing Spectrum — {mod}", fontsize=11, fontweight="bold")
        ax_gt.set_yscale("log")
        ax_gt.grid(True, linestyle=":", alpha=0.5)
        ax_gt.legend(title="Beam Energy", fontsize=9)
        fig_gt.tight_layout()
        fig_gt.savefig(global_dir / f"{mod}_globaltime.png", dpi=200)
        plt.close(fig_gt)

        # ── GRAPH 3: LOCAL TIME VS PARTICLE STRIKES (Downstream) ─────────────────
        fig_lt, ax_lt = plt.subplots(figsize=(8, 5.5))
        for ekey in energy_keys:
            counts = master_summary[mod][ekey]["lt_counts"]
            bins = master_summary[mod][ekey]["lt_bins"]
            bin_mids = 0.5 * (bins[:-1] + bins[1:])
            ax_lt.plot(bin_mids, counts, label=ekey, alpha=0.8, linewidth=1.5)

        ax_lt.set_xlabel("LocalTime (ns)", fontweight="bold")
        ax_lt.set_ylabel("Optical Photon Strikes (Downstream)", fontweight="bold")
        ax_lt.set_title(f"Downstream LocalTime Timing Spectrum — {mod}", fontsize=11, fontweight="bold")
        ax_lt.set_yscale("log")
        ax_lt.grid(True, linestyle=":", alpha=0.5)
        ax_lt.legend(title="Beam Energy", fontsize=9)
        fig_lt.tight_layout()
        fig_lt.savefig(local_dir / f"{mod}_localtime.png", dpi=200)
        plt.close(fig_lt)

        # ── GRAPH 4: PROMPT PHOTON LONGITUDINAL RECONSTRUCTION ──────────────────
        fig_rec, ax_rec = plt.subplots(figsize=(8.5, 5.5))
        layers_x = np.arange(1, _N_LYSO + 1)
        for ekey in energy_keys:
            profile = master_summary[mod][ekey]["prompt_profile"]
            ax_rec.plot(layers_x, profile, marker="o", markersize=4, label=ekey, alpha=0.8, linewidth=1.5)

        ax_rec.set_xlabel("LYSO Layer Number (Upstream 1 -> Downstream 29)", fontweight="bold")
        ax_rec.set_ylabel("Mean Prompt Photons / Event", fontweight="bold")
        ax_rec.set_title(f"Prompt-Photon Reconstructed Longitudinal Profile — {mod}\n(Window: $\\pm 150$ ps around layer ToF)", fontsize=11, fontweight="bold")
        ax_rec.grid(True, linestyle=":", alpha=0.5)
        ax_rec.legend(title="Beam Energy", fontsize=9)
        fig_rec.tight_layout()
        fig_rec.savefig(prompt_dir / f"{mod}_prompt_reconstruction.png", dpi=200)
        plt.close(fig_rec)

        # ── GRAPH 5: TWO-ENDED FIBER TIMING spectra & resolutions ────────────────
        ncols = 2 if len(energy_keys) >= 2 else 1
        nrows = int(np.ceil(len(energy_keys) / ncols))
        fig_two, axs_two = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_two = axs_two.flatten()

        plotted_count = 0
        t_res_x, t_res_y, t_res_yerr = [], [], []

        for idx, ekey in enumerate(energy_keys):
            ax = axs_two[idx]
            raw_t = master_summary[mod][ekey].get("t_two_end_raw", np.array([]))
            n_ev = master_summary[mod][ekey].get("n_t_coincidences", 0)

            if len(raw_t) >= 8:
                plotted_count += 1
                clean_t = clean_around_mode(raw_t, window_ps=500.0)
                
                # Align median to 0 ps to visualize the jitter width directly
                clean_t = clean_t - np.median(clean_t)
                
                _, mu_f, sigma_f = fit_gaussian_to_peak(clean_t)
                
                # Standard deviation error
                sigma_err = sigma_f / np.sqrt(2 * n_ev)
                
                t_res_x.append(extract_numerical_energy(ekey))
                t_res_y.append(sigma_f)
                t_res_yerr.append(sigma_err)

                # Visualizing the coincidence distributions
                lo, hi = -250.0, 250.0
                counts, edges, _ = ax.hist(clean_t, bins=50, range=(lo, hi),
                                           color=mod_colors.get(mod, "#f708af"), alpha=0.6, edgecolor="black", label="Data")

                bin_mids = 0.5 * (edges[:-1] + edges[1:])
                x_fit = np.linspace(lo, hi, 200)
                y_fit = counts.max() * np.exp(-0.5 * ((x_fit - mu_f) / sigma_f) ** 2)

                ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2.0,
                        label=f"Gaussian Fit\n$\\sigma_{{coinc}}$ = {sigma_f:.1f} ps")
                ax.set_title(f"Coincidence Spectrum — {ekey}", fontsize=10, fontweight="bold")
                ax.set_xlabel(r"$(t_{up} + t_{down})/2 - \mathrm{offset}$ (ps)", fontsize=9)
                ax.set_xlim(lo, hi)
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, linestyle=":", alpha=0.5)

        for idx in range(plotted_count, len(axs_two)):
            fig_two.delaxes(axs_two[idx])

        if plotted_count > 0:
            fig_two.suptitle(f"Double-Ended Coincidence Spectra — {mod}", fontsize=12, fontweight="bold", y=0.98)
            fig_two.tight_layout()
            fig_two.savefig(two_end_dir / f"{mod}_two_end_distributions.png", dpi=200)
        plt.close(fig_two)

        # Plot Coincidence Resolution vs Energy for the Module
        if len(t_res_x) >= 2:
            fig_tcurve, ax_tcurve = plt.subplots(figsize=(7, 5))
            ax_tcurve.errorbar(t_res_x, t_res_y, yerr=t_res_yerr, fmt="o-",
                               color=mod_colors.get(mod, "black"), marker=mod_markers.get(mod, "o"),
                               linewidth=2, markersize=6, capsize=4, label=f"Coincidence Resolution ({mod})")
            ax_tcurve.set_xlabel("Beam Energy (GeV)", fontweight="bold")
            ax_tcurve.set_ylabel(r"Timing Coincidence Resolution $\sigma_{coinc}$ (ps)", fontweight="bold")
            ax_tcurve.set_title(f"Two-Ended Fiber Coincidence Timing Resolution — {mod}", fontsize=11, fontweight="bold")
            ax_tcurve.set_xscale("log")
            ax_tcurve.set_xticks([25, 50, 100, 200])
            ax_tcurve.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax_tcurve.grid(True, linestyle=":", alpha=0.6)
            ax_tcurve.legend()
            fig_tcurve.tight_layout()
            fig_tcurve.savefig(two_end_dir / f"{mod}_two_end_resolution_vs_energy.png", dpi=200)
            plt.close(fig_tcurve)

    print(f"\nDone. Profile analysis reports and graphs written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()