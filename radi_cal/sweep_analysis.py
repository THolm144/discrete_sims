"""
sweep_analysis_4T.py
=====================================
Optimized version for aggregating timing and energy resolution results 
for 4-T fiber RADiCAL geometries, matching paper comparison curves.
"""
import os

# MUST be set before importing numpy/scipy/ROOT!
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import datetime
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import itertools
import re
import uuid
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
import ROOT

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL KINEMATICS & REFERENCE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS = 299.792

REFRACTIVE_INDEX = {
    "radi_cal_energy":        1.60,
    "dsb1_radi_cal_energy":   1.55,
    "dsb1_radi_cal_4t":       1.55,
    "luagce_radi_cal_energy": 1.84,
}

BOUNCE_FACTOR = 0.92
_GT_LO_NS = 0.0
_GT_HI_NS = 1000.0
_TYVEK_THICK_MM = 0.2032
_W_THICK_MM = 2.5
_N_LYSO = 29
_N_W = 28

ARRIVAL_QUANTILE = 0.10

_KNOWN_MODULE_LYSO_THICK = {
    "radi_cal_energy":      1.5,
    "dsb1_radi_cal_energy": 1.5,
    "dsb1_radi_cal_4t":     1.5,
}

_SQUARE_HOLE_OFFSET = 3.7032
SQUARE_CAP_XY = np.array([
    [ _SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],  # 0
    [-_SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],  # 1
    [-_SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],  # 2
    [ _SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],  # 3
])

# Reference curves for resolution comparison
ENERGY_REF_CURVES = {
    "paper Fig 17": {
        "c": 9.31, 
        "s": 52.04, 
        "n": 31.62, 
        "color": "gray", 
        "ls": "--"
    },
}

# Paper timing reference curve (17.52 (+) 255.58 / sqrt(E) ps)
TIMING_REF_CURVE = {
    "c": 17.52,
    "s": 255.58,
    "color": "#ff7f0e",  # Orange matching the paper plot
    "label": r"Paper: $17.52 \oplus 255.58 / \sqrt{E}$ ps"
}

def resolution_fit_func(E, c, s, n):
    """ Energy resolution parametrization: c (+) s/sqrt(E) (+) n/E in % """
    return np.sqrt(c**2 + (s / np.sqrt(E))**2 + (n / E)**2)

def timing_fit_func(E, c, s):
    """ Timing resolution parametrization: c (+) s/sqrt(E) in ps """
    return np.sqrt(c**2 + (s / np.sqrt(E))**2)

def v_eff_for_module(mod: str) -> float:
    return (C_LIGHT_MM_NS / REFRACTIVE_INDEX.get(mod, 1.55)) * BOUNCE_FACTOR

def robust_resolution(data, nsig=2.0, max_iters=4):
    """
    Python/ROOT replica of scan_resolution.C robustRes():
    Iterative +/- nsig core fit with dynamic binning (~sigma/5), falling back to
    RMS/median if the relative fit error on sigma exceeds 25%.
    """
    N = len(data)
    if N < 2:
        return -1.0, 1e9  

    median = np.median(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    sg_robust = iqr / 1.349 if iqr > 0 else np.std(data, ddof=1)

    fallback_res = 100.0 * sg_robust / median if median > 0 else -1.0
    fallback_err = fallback_res / np.sqrt(2.0 * N) if (N > 1 and fallback_res > 0) else 1e9

    if median <= 0 or sg_robust <= 0:
        return fallback_res, fallback_err

    unique_id = uuid.uuid4().hex
    hname = f"h_{unique_id}"
    fname = f"f_{unique_id}"

    # Dynamic rebinning (matching scan_resolution.C: bin_width ~ sg_robust / 5.0)
    hist_min = max(0, int(np.floor(median - 5.0 * sg_robust)))
    hist_max = int(np.ceil(median + 5.0 * sg_robust))
    bin_width = max(1.0, sg_robust / 5.0)
    nbins = max(10, int(np.ceil((hist_max - hist_min) / bin_width)))

    h = ROOT.TH1D(hname, "temp_hist", nbins, hist_min - 0.5 * bin_width, hist_max + 0.5 * bin_width)
    h.SetDirectory(0)

    for val in data:
        h.Fill(float(val))

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

    # Core fit quality check (relative error on sigma < 25%)
    fit_ok = (mu > 0) and (sg > 0) and (sigma_err > 0) and (sigma_err / sg < 0.25)
    res_tuple = (100.0 * sg / mu, 100.0 * sigma_err / mu) if fit_ok else (fallback_res, fallback_err)

    # Memory cleanup: Disown objects before explicit C++ deletion
    ROOT.SetOwnership(h, False)
    ROOT.SetOwnership(g, False)
    h.Delete()
    g.Delete()

    return res_tuple

def analyze_timing_distribution(dt_data_ps, window_ps=500.0):
    """
    Calculates both Gaussian core sigma and Empirical FWHM timing resolution
    from raw delta_t (ps) coincidence data.
    """
    if len(dt_data_ps) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    med = np.median(dt_data_ps)
    clean = dt_data_ps[np.abs(dt_data_ps - med) <= window_ps]
    if len(clean) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # 1. Dynamic histogram binning
    iqr = np.percentile(clean, 75) - np.percentile(clean, 25)
    std_rob = iqr / 1.349 if iqr > 0 else np.std(clean)
    bin_width = max(2.5, std_rob / 3.0)
    hist_lo, hist_hi = med - 4.0 * std_rob, med + 4.0 * std_rob
    nbins = max(15, int(np.ceil((hist_hi - hist_lo) / bin_width)))

    counts, edges = np.histogram(clean, bins=nbins, range=(hist_lo, hist_hi))
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    # 2. Empirical FWHM Calculation
    max_idx = np.argmax(counts)
    max_val = counts[max_idx]
    half_max = max_val / 2.0

    # Left crossing
    left_slice = counts[:max_idx]
    if len(left_slice) > 0 and np.any(left_slice < half_max):
        l_idx = np.where(left_slice < half_max)[0][-1]
        x0, y0 = bin_centers[l_idx], counts[l_idx]
        x1, y1 = bin_centers[l_idx + 1], counts[l_idx + 1]
        x_left = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
    else:
        x_left = bin_centers[0]

    # Right crossing
    right_slice = counts[max_idx:]
    if len(right_slice) > 0 and np.any(right_slice < half_max):
        r_idx_rel = np.where(right_slice < half_max)[0][0]
        r_idx = max_idx + r_idx_rel
        x0, y0 = bin_centers[r_idx - 1], counts[r_idx - 1]
        x1, y1 = bin_centers[r_idx], counts[r_idx]
        x_right = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
    else:
        x_right = bin_centers[-1]

    fwhm_ps = max(0.0, x_right - x_left)
    sigma_fwhm_ps = fwhm_ps / 2.35482
    err_fwhm_ps = sigma_fwhm_ps / np.sqrt(2.0 * len(clean))

    # 3. Gaussian Core Fit (Top 50% mask)
    def straight_gaussian(x, amp, mu, sig):
        return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    mask_top = counts >= half_max
    if np.sum(mask_top) >= 4:
        p0 = [float(max_val), bin_centers[max_idx], std_rob]
        bounds = ([0.0, hist_lo, 0.1], [max_val * 2.0, hist_hi, std_rob * 5.0])
        try:
            popt, pcov = curve_fit(straight_gaussian, bin_centers[mask_top], counts[mask_top], p0=p0, bounds=bounds)
            sigma_gaus_ps = abs(popt[2])
            err_gaus_ps = np.sqrt(pcov[2, 2]) if pcov is not None and pcov[2, 2] > 0 else sigma_gaus_ps / np.sqrt(2 * len(clean))
        except Exception:
            sigma_gaus_ps, err_gaus_ps = std_rob, std_rob / np.sqrt(2 * len(clean))
    else:
        sigma_gaus_ps, err_gaus_ps = std_rob, std_rob / np.sqrt(2 * len(clean))

    return sigma_gaus_ps, err_gaus_ps, fwhm_ps, sigma_fwhm_ps, err_fwhm_ps

def extract_numerical_energy(label: str) -> float:
    try:
        return float(''.join(c for c in label if c.isdigit() or c == '.'))
    except ValueError:
        return 0.0

def _chunk_series(mask, values, ev, run_tag):
    n = int(mask.sum())
    if n == 0: return None
    idx = pd.MultiIndex.from_arrays([np.full(n, run_tag, dtype=object), ev[mask].astype(np.int64)])
    return pd.Series(values[mask], index=idx)

def _grouped(chunks, how):
    if not chunks: return {}
    s = pd.concat(chunks)
    g = s.groupby(level=[0, 1])
    if how == "min": s = g.min()
    elif how == "count": s = g.count()
    else: s = g.quantile(how)
    return {(k[0], int(k[1])): (int(v) if how == "count" else float(v)) for k, v in s.items()}

def analyze_energy_batch(batch_dir: Path, module_name: str = "dsb1_radi_cal_energy"):
    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"))
    if not hit_files:
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

    cap_xy_map = SQUARE_CAP_XY
    t_indices = list(range(len(cap_xy_map)))  # All 4 channels treated as optical T-sensors

    up_t_hit_chunks, dw_t_hit_chunks = [], []
    up_t_q_chunks, dw_t_q_chunks = [], []
    branch_list = ["Position_X", "Position_Y", "Position_Z", "GlobalTime", "ParticleName", "EventID"]

    for fpath in hit_files:
        run_tag = fpath.parent.name
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
        gt, ev, pn = arrs["GlobalTime"], arrs["EventID"], arrs["ParticleName"]

        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(np.hypot(dx, dy), axis=1)

        near_up = np.abs(z + detected_z_sensor) < 2.5
        near_dw = np.abs(z - detected_z_sensor) < 2.5
        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")
        is_prompt = (gt >= _GT_LO_NS) & (gt <= _GT_HI_NS)

        # Flat SiPM PDE
        FLAT_PDE = 0.40 
        is_detected = np.random.rand(len(gt)) < FLAT_PDE

        is_t = np.isin(channels, t_indices)
        
        m_t_up_prompt = is_t & is_optical & near_up & is_prompt & is_detected
        m_t_dw_prompt = is_t & is_optical & near_dw & is_prompt & is_detected

        # Quantile timing mask (unfiltered prompt window for raw arrival distribution)
        m_t_up_q = is_t & is_optical & near_up & is_detected
        m_t_dw_q = is_t & is_optical & near_dw & is_detected

        # Photon count chunks
        c = _chunk_series(m_t_up_prompt, gt, ev, run_tag)
        if c is not None: up_t_hit_chunks.append(c)
        c = _chunk_series(m_t_dw_prompt, gt, ev, run_tag)
        if c is not None: dw_t_hit_chunks.append(c)

        # Quantile timing chunks (timestamps converted to picoseconds: gt * 1000.0)
        c_q = _chunk_series(m_t_up_q, gt * 1000.0, ev, run_tag)
        if c_q is not None: up_t_q_chunks.append(c_q)
        c_q = _chunk_series(m_t_dw_q, gt * 1000.0, ev, run_tag)
        if c_q is not None: dw_t_q_chunks.append(c_q)

    # Photon Counts per event
    up_t_hits_per_ev = _grouped(up_t_hit_chunks, "count")
    dw_t_hits_per_ev = _grouped(dw_t_hit_chunks, "count")
    all_events = sorted(list(set(up_t_hits_per_ev.keys()) | set(dw_t_hits_per_ev.keys())))
    dw_t_total = np.array([dw_t_hits_per_ev.get(k, 0) + up_t_hits_per_ev.get(k, 0) for k in all_events])

    # 10% Quantile Arrival Times per event
    up_q = _grouped(up_t_q_chunks, ARRIVAL_QUANTILE)
    dw_q = _grouped(dw_t_q_chunks, ARRIVAL_QUANTILE)
    common_t_evs = set(up_q.keys()) & set(dw_q.keys())
    
    # Delta T / 2 coincidence observable in ps
    delta_t_ps = np.array([(dw_q[e] - up_q[e]) / 2.0 for e in common_t_evs])

    return {"dw_t_total": dw_t_total, "delta_t_ps": delta_t_ps}


def analyze_showermax_edep_batch(batch_dir: Path):
    edep_files = sorted(batch_dir.rglob("showermax_edep_*.root"))
    if not edep_files:
        return None

    edep_sum_per_ev = {}
    for fpath in edep_files:
        run_tag = fpath.parent.name
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if k.split(";")[0].startswith("showermax_edep")), None)
                if not tk: continue
                tree = f[tk]
                if tree.num_entries == 0: continue
                arrs = tree.arrays(["EventID", "TotalEnergyDeposit"], library="np")
        except Exception:
            continue

        for e_id, dep in zip(arrs["EventID"], arrs["TotalEnergyDeposit"]):
            key = (run_tag, int(e_id))
            edep_sum_per_ev[key] = edep_sum_per_ev.get(key, 0.0) + float(dep)

    if not edep_sum_per_ev:
        return None

    return np.array(list(edep_sum_per_ev.values()))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analyze 4-T Dynamic Optical Sweep Data")
    parser.add_argument("--runs-dir", type=str, default="./runs", help="Base runs directory")
    parser.add_argument("--module", type=str, default="dsb1_radi_cal_energy", help="World module directory name")
    parser.add_argument("--output-dir", type=str, default="./analysis_output", help="Output directory for plots and CSV")
    args = parser.parse_args()

    # ROOT Headless Setup
    ROOT.gROOT.SetBatch(True)

    base_path = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    module_runs = base_path / args.module
    if module_runs.exists():
        sweep_dirs = sorted(module_runs.glob("sweep_*"))
    else:
        sweep_dirs = sorted(base_path.glob("**/sweep_*"))

    if not sweep_dirs:
        print(f"[-] Error: No sweep_* folders found inside {base_path}")
        return

    target_sweep = sweep_dirs[-1]
    print("========================================================================")
    print(f" Target Sweep Path: {target_sweep.resolve()}")
    print("========================================================================")

    energy_dirs = sorted(
        [d for d in target_sweep.iterdir() if d.is_dir() and "GeV" in d.name],
        key=lambda x: extract_numerical_energy(x.name)
    )

    energies_gev, mean_yields, res_percent, res_err_percent = [], [], [], []
    photon_counts_by_energy = []
    edep_res_percent, edep_res_err_percent = [], []

    # Timing metrics arrays
    res_gaus_ps, err_gaus_ps = [], []
    res_fwhm_ps, err_fwhm_ps = [], []
    raw_fwhm_ps = []

    for edir in energy_dirs:
        e_val = extract_numerical_energy(edir.name)
        print(f" [+] Processing [{edir.name}]...")

        res_dict = analyze_energy_batch(edir, module_name=args.module)
        if res_dict is None or len(res_dict["dw_t_total"]) == 0:
            print(f"     [!] Skipped — no data found")
            continue

        photon_counts = res_dict["dw_t_total"]
        dt_data = res_dict.get("delta_t_ps", np.array([]))

        mean_N = np.mean(photon_counts)
        res, err = robust_resolution(photon_counts)

        energies_gev.append(e_val)
        mean_yields.append(mean_N)
        res_percent.append(res)
        res_err_percent.append(err)
        photon_counts_by_energy.append(photon_counts)

        # Raw dE/dx
        edep_MeV = analyze_showermax_edep_batch(edir)
        if edep_MeV is not None and len(edep_MeV) >= 2:
            res_e, err_e = robust_resolution(edep_MeV)
            edep_res_percent.append(res_e)
            edep_res_err_percent.append(err_e)
        else:
            edep_res_percent.append(np.nan)
            edep_res_err_percent.append(np.nan)

        # Timing Resolution Calculations
        if len(dt_data) >= 5:
            sg_g, err_g, fwhm_raw, sg_fwhm, err_fwhm = analyze_timing_distribution(dt_data)
            res_gaus_ps.append(sg_g)
            err_gaus_ps.append(err_g)
            res_fwhm_ps.append(sg_fwhm)
            err_fwhm_ps.append(err_fwhm)
            raw_fwhm_ps.append(fwhm_raw)
            print(f"     -> [Timing] Events: {len(dt_data)} | Gauss Sigma: {sg_g:.2f} ± {err_g:.2f} ps "
                  f"| FWHM: {fwhm_raw:.2f} ps (Sigma_FWHM: {sg_fwhm:.2f} ± {err_fwhm:.2f} ps)")
        else:
            res_gaus_ps.append(np.nan)
            err_gaus_ps.append(np.nan)
            res_fwhm_ps.append(np.nan)
            err_fwhm_ps.append(np.nan)
            raw_fwhm_ps.append(np.nan)

        print(f"     -> Events: {len(photon_counts)} | Mean Photons: {mean_N:.1f} | Resolution: {res:.2f}% ± {err:.2f}%")

    if not energies_gev:
        print("[-] No valid data to plot.")
        return

    energies_gev = np.array(energies_gev)
    res_percent = np.array(res_percent)
    res_err_percent = np.array(res_err_percent)
    
    res_gaus_ps = np.array(res_gaus_ps)
    err_gaus_ps = np.array(err_gaus_ps)
    res_fwhm_ps = np.array(res_fwhm_ps)
    err_fwhm_ps = np.array(err_fwhm_ps)
    raw_fwhm_ps = np.array(raw_fwhm_ps)

    # Fits for Energy
    popt_sim = [15.92, 0.0, 122.8]
    try:
        popt, _ = curve_fit(
            resolution_fit_func, energies_gev, res_percent, sigma=res_err_percent, 
            p0=[12.0, 50.0, 30.0], bounds=([0.0, 0.0, 0.0], [30.0, 150.0, 100.0])
        )
        popt_sim = popt
    except Exception as e:
        print(f" [+] Fit Warning: curve_fit failed for energy ({e}).")

    # Fits for Timing Resolution
    popt_gaus, popt_fwhm = None, None
    valid_timing = np.isfinite(res_gaus_ps)
    if valid_timing.sum() >= 2:
        try:
            popt_gaus, _ = curve_fit(
                timing_fit_func, energies_gev[valid_timing], res_gaus_ps[valid_timing],
                sigma=err_gaus_ps[valid_timing], p0=[15.0, 200.0], bounds=([0.0, 0.0], [100.0, 1000.0])
            )
        except Exception as e:
            print(f" [+] Fit Warning: timing curve_fit (Gaussian) failed ({e}).")

        try:
            popt_fwhm, _ = curve_fit(
                timing_fit_func, energies_gev[valid_timing], res_fwhm_ps[valid_timing],
                sigma=err_fwhm_ps[valid_timing], p0=[15.0, 200.0], bounds=([0.0, 0.0], [100.0, 1000.0])
            )
        except Exception as e:
            print(f" [+] Fit Warning: timing curve_fit (FWHM) failed ({e}).")

    # Calibrated Energy Analysis
    mean_yields_arr = np.array(mean_yields)
    calib_slope, calib_intercept = np.polyfit(energies_gev, mean_yields_arr, 1)
    res_calib_percent, res_calib_err_percent = [], []
    for e_val, photon_counts in zip(energies_gev, photon_counts_by_energy):
        e_reco = (np.asarray(photon_counts, dtype=float) - calib_intercept) / calib_slope
        e_reco = e_reco[e_reco > 0]
        if len(e_reco) < 2:
            res_calib_percent.append(np.nan)
            res_calib_err_percent.append(np.nan)
            continue
        res_c, err_c = robust_resolution(e_reco)
        res_calib_percent.append(res_c)
        res_calib_err_percent.append(err_c)

    res_calib_percent = np.array(res_calib_percent)
    res_calib_err_percent = np.array(res_calib_err_percent)

    # Save summary DataFrame
    df_summary = pd.DataFrame({
        "Energy_GeV": energies_gev,
        "Mean_Photons": mean_yields,
        "Energy_Resolution_Percent": res_percent,
        "Energy_Resolution_Err_Percent": res_err_percent,
        "Calibrated_Energy_Resolution_Percent": res_calib_percent,
        "Calibrated_Energy_Resolution_Err_Percent": res_calib_err_percent,
        "Time_Resolution_Gauss_ps": res_gaus_ps,
        "Time_Resolution_Gauss_Err_ps": err_gaus_ps,
        "Time_Resolution_FWHM_Equivalent_ps": res_fwhm_ps,
        "Time_Resolution_FWHM_Equivalent_Err_ps": err_fwhm_ps,
        "Raw_FWHM_ps": raw_fwhm_ps,
    })
    df_summary.to_csv(out_dir / "sweep_4T_summary.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 1: ENERGY RESOLUTION
    # ─────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 6.5))
    plt.errorbar(
        energies_gev, res_percent, yerr=res_err_percent,
        fmt='s', color='m', ecolor='m', capsize=3, elinewidth=1.2,
        label=f'sim (photon count): {popt_sim[0]:.2f}% $\\oplus$ {popt_sim[1]:.1f}%/$\\sqrt{{E}}$ $\\oplus$ {popt_sim[2]:.1f}%/E'
    )
    e_smooth = np.linspace(max(0.5, min(energies_gev) * 0.8), max(energies_gev) * 1.1, 200)
    sim_curve = resolution_fit_func(e_smooth, *popt_sim)
    plt.plot(e_smooth, sim_curve, 'm--', lw=1.8)

    for label, params in ENERGY_REF_CURVES.items():
        ref_curve = resolution_fit_func(e_smooth, params["c"], params["s"], params["n"])
        plt.plot(
            e_smooth, ref_curve, ls=params.get("ls", "--"), lw=1.8, color=params.get("color", "gray"),
            label=f'{label}: {params["c"]:.2f}% $\\oplus$ {params["s"]:.1f}%/$\\sqrt{{E}}$ $\\oplus$ {params["n"]:.1f}%/E'
        )

    plt.title('Shower-max energy resolution', fontsize=16, pad=12)
    plt.xlabel(r'$E_{\mathrm{beam}}$ (GeV)', fontsize=14)
    plt.ylabel(r'$\sigma / \mathrm{mean}$ (%)', fontsize=14)
    plt.xlim(0, max(energies_gev) * 1.05)
    plt.ylim(0, max(res_percent) * 1.15)
    plt.grid(True, which='both', linestyle=':', color='gray', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11, frameon=False)
    plt.tight_layout()
    plot_path_e = out_dir / "showermax_energy_resolution_4T.png"
    plt.savefig(plot_path_e, dpi=300)
    plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 2: TIME RESOLUTION (MATCHING PAPER GRAPH WITH FITS)
    # ─────────────────────────────────────────────────────────────────────────
    if valid_timing.sum() >= 1:
        plt.figure(figsize=(9, 6.5))

        # 1. Paper Reference Curve (Orange)
        c_ref, s_ref = TIMING_REF_CURVE["c"], TIMING_REF_CURVE["s"]
        t_ref_curve = timing_fit_func(e_smooth, c_ref, s_ref)
        plt.plot(
            e_smooth, t_ref_curve, color=TIMING_REF_CURVE["color"], lw=2.0,
            label=f'{TIMING_REF_CURVE["label"]}: {c_ref:.2f} $\\oplus$ {s_ref:.2f} / $\\sqrt{{E}}$'
        )

        # 2. Gaussian Fit Data Points + Fitted Curve (Blue Dots)
        lbl_g = "Sim (Gaussian Core)"
        if popt_gaus is not None:
            lbl_g += f": {popt_gaus[0]:.2f} $\\oplus$ {popt_gaus[1]:.2f} / $\\sqrt{{E}}$"
            plt.plot(e_smooth, timing_fit_func(e_smooth, *popt_gaus), 'b--', lw=1.8)

        plt.errorbar(
            energies_gev[valid_timing], res_gaus_ps[valid_timing], yerr=err_gaus_ps[valid_timing],
            fmt='o', color='#1f77b4', ecolor='#1f77b4', capsize=3, elinewidth=1.2, label=lbl_g
        )

        # 3. Empirical FWHM Data Points + Fitted Curve (Green Squares)
        lbl_f = "Sim (FWHM / 2.355)"
        if popt_fwhm is not None:
            lbl_f += f": {popt_fwhm[0]:.2f} $\\oplus$ {popt_fwhm[1]:.2f} / $\\sqrt{{E}}$"
            plt.plot(e_smooth, timing_fit_func(e_smooth, *popt_fwhm), 'g:', lw=1.8)

        plt.errorbar(
            energies_gev[valid_timing], res_fwhm_ps[valid_timing], yerr=err_fwhm_ps[valid_timing],
            fmt='s', color='#2ca02c', ecolor='#2ca02c', capsize=3, elinewidth=1.2, label=lbl_f
        )

        plt.title('Time Resolution vs Beam Energy', fontsize=16, pad=12)
        plt.xlabel(r'Beam Energy  (GeV)', fontsize=14)
        plt.ylabel(r'Time Resolution  (ps)', fontsize=14)
        plt.xlim(min(energies_gev) * 0.8, max(energies_gev) * 1.08)
        
        # Adjust Y limits dynamically to accommodate reference + data
        max_y = max(np.max(t_ref_curve), np.max(res_gaus_ps[valid_timing])) * 1.15
        plt.ylim(0, max_y)
        plt.grid(True, which='both', linestyle=':', color='gray', alpha=0.6)
        plt.legend(loc='upper right', fontsize=11, frameon=False)
        plt.tight_layout()

        plot_path_t = out_dir / "time_resolution_4T.png"
        plt.savefig(plot_path_t, dpi=300)
        plt.close()
        print(f" [✓] Time Resolution Plot: {plot_path_t.resolve()}")

    print(f"\n [✓] Results saved to {out_dir.resolve()}")
    print(f" [✓] Summary CSV: {(out_dir / 'sweep_4T_summary.csv').resolve()}")

if __name__ == "__main__":
    main()