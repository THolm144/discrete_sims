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
import SimpleITK as sitk

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
_GT_HI_NS = 1000.0
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

def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))



def robust_resolution(data, nsig=2.0, max_iters=4):
    """
    Computes fractional resolution (sigma/mean in %) with uncertainty.
    Uses ROOT's Log-Likelihood Minuit fit (RQL0) to perfectly replicate the C++ script.
    """
    N = len(data)
    if N < 2:
        return -1.0, 1e9  
    
    # --- 1. FALLBACK METRICS ---
    # We still use your existing robust numpy fallback if the ROOT fit fails
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

    # --- 2. CREATE ROOT HISTOGRAM ---
    # Create a unique name to prevent ROOT memory warnings during loops
    unique_id = uuid.uuid4().hex
    hname = f"h_{unique_id}"
    fname = f"f_{unique_id}"
    

    # Define integer-aligned boundaries to capture discrete photon counts cleanly
    hist_min = max(0, int(np.floor(median - 5 * sg_robust)))
    hist_max = int(np.ceil(median + 5 * sg_robust))
    nbins = hist_max - hist_min + 1
    h = ROOT.TH1D(hname, "temp_hist", nbins, hist_min - 0.5, hist_max + 0.5)
    h.SetDirectory(0)  # <-- ADD THIS: Tells ROOT C++ not to own this object
    
    # Fill the ROOT histogram with the numpy data
    for val in data:
        h.Fill(val)

    # --- 3. THE C++ ITERATIVE FIT ---
    # Grab initial seeds directly from the histogram
    mu = h.GetMean()
    sg = h.GetRMS()
    
    g = ROOT.TF1(fname, "gaus", mu - nsig * sg, mu + nsig * sg)
    
    fit_success = False
    sigma_err = 0.0

    # Iterative core-fit exactly mirroring C++ coreFit()
    for _ in range(max_iters):
        g.SetRange(mu - nsig * sg, mu + nsig * sg)
        # R = Range, Q = Quiet, L = Log-Likelihood, 0 = Don't draw
        h.Fit(g, "RQL0")
        
        mu = g.GetParameter(1)
        sg = g.GetParameter(2)
        sigma_err = g.GetParError(2)
        
        if sg <= 0:
            break

    # --- 4. C++ EVALUATION LOGIC ---
    # Check if the fit relative error on sigma exceeds 25% (exactly as in robustRes)
    fit_ok = (mu > 0) and (sg > 0) and (sigma_err > 0) and (sigma_err / sg < 0.25)

   
    # --- 5. RETURN RESULT ---
    if fit_ok:
        res = 100.0 * sg / mu
        err = 100.0 * sigma_err / mu
        return res, err
    else:
        return fallback_res, fallback_err

def standard_gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian_to_peak(data, n_bins=40):
    """
    Fit a Gaussian to the histogram peak of `data`, returning (amp, mu, sigma,
    sigma_err). sigma_err is the 1-sigma parameter uncertainty on sigma taken
    from the curve_fit covariance matrix (analogous to ROOT's GetParError(2)),
    used downstream by robust_res() to judge fit reliability the same way the
    ROOT macro's robustRes() does.
    """
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


def robust_res(data, fit_mu, fit_sigma, fit_sigma_err):
    """
    Robust fractional resolution (sigma/mean) with RMS fallback — the Python
    analog of the ROOT macro's robustRes(). At low light yield the photon-count
    spectra are Poisson-like/skewed and the Gaussian peak fit can degenerate
    (small or blown-up sigma with a huge parameter error) even though the
    reported central value looks superficially plausible. Rather than judging
    the fit by the size of the resulting resolution (which is circular), judge
    it the way ROOT does: by the fit's own uncertainty on sigma. If the
    relative error on sigma exceeds 25% (fitOK == False in the macro), fall
    back to the well-defined raw RMS/mean with its analytic large-N error
    sigma_rel/sqrt(2N).

    Returns (res, err, used_fallback).
    """
    n = len(data)
    raw_mean = float(np.mean(data)) if n else 0.0
    raw_std = float(np.std(data)) if n else 0.0
    rms_res = raw_std / raw_mean if raw_mean > 0 else -1.0
    rms_err = rms_res / np.sqrt(2.0 * n) if (n > 1 and rms_res > 0) else 1e9

    fit_ok = (
        fit_mu > 0 and fit_sigma > 0 and fit_sigma_err > 0
        and (fit_sigma_err / fit_sigma) < 0.25
    )
    if fit_ok:
        return fit_sigma / fit_mu, fit_sigma_err / fit_mu, False
    else:
        return rms_res, rms_err, True

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
    
def calculate_empirical_fwhm(data, bins=100):
    """
    Calculates the Full-Width at Half-Maximum (FWHM) empirically from data 
    without applying any analytical fits. Uses linear interpolation between 
    bin centers crossing the half-maximum threshold.
    """
    if len(data) < 10:
        return 0.0

    med = float(np.median(data))
    std = float(np.std(data))
    
    # Restrict to a reasonable window to avoid extreme outliers stretching the bins
    hist_lo, hist_hi = med - 4 * std, med + 4 * std
    
    counts, edges = np.histogram(data, bins=bins, range=(hist_lo, hist_hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    
    max_idx = np.argmax(counts)
    max_val = counts[max_idx]
    half_max = max_val / 2.0
    
    # 1. Find Left Crossing
    left_slice = counts[:max_idx]
    if len(left_slice) == 0 or not np.any(left_slice < half_max):
        x_left = centers[0]
    else:
        l_idx = np.where(left_slice < half_max)[0][-1]
        x0, y0 = centers[l_idx], counts[l_idx]
        x1, y1 = centers[l_idx + 1], counts[l_idx + 1]
        x_left = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
        
    # 2. Find Right Crossing
    right_slice = counts[max_idx:]
    if len(right_slice) == 0 or not np.any(right_slice < half_max):
        x_right = centers[-1]
    else:
        r_idx_rel = np.where(right_slice < half_max)[0][0]
        r_idx = max_idx + r_idx_rel
        x0, y0 = centers[r_idx - 1], counts[r_idx - 1]
        x1, y1 = centers[r_idx], counts[r_idx]
        x_right = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
        
    return float(x_right - x_left)

# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL REFERENCE CURVES (hardcoded — no underlying data files)
# ─────────────────────────────────────────────────────────────────────────────
# These constants are read directly off the legends of the reference plots and
# are NOT derived from any sim output. They exist purely for visual comparison
# against the single simulated flavor this pipeline actually computes.

# Timing resolution: sigma_t(E) [ps] = sqrt( (stoch/sqrt(E))^2 + const^2 )
TIMING_REF_CURVES = {
    "DATA (test beam)":        {"stoch": 181.0, "const": 34.9, "color": "black",   "ls": "-"},
    "paper (arXiv:2401.01747)": {"stoch": 256.0, "const": 17.5, "color": "gray",    "ls": "--"},
}

def timing_ref_curve(E_gev, stoch, const):
    E_gev = np.asarray(E_gev, dtype=float)
    return np.sqrt((stoch / np.sqrt(E_gev)) ** 2 + const ** 2)

# Energy resolution: sigma_E/E(E) [fraction] = sqrt(c^2 + (s/sqrt(E))^2 + (n/E)^2)
# Percent constants converted to fractions (i.e. /100) so they overlay directly
# on this script's existing res_e_list (sigma/mu, dimensionless).
ENERGY_REF_CURVES = {
    "paper Fig 17": {"c": 9.31 / 100.0, "s": 52.04 / 100.0, "n": 31.62 / 100.0, "color": "gray", "ls": "--"},
}
# DATA sum_lg band from test beam, given as a flat 11-19% range (no explicit
# energy dependence was shown), plotted as a shaded horizontal band.
ENERGY_DATA_BAND_FRAC = (0.11, 0.19)

def energy_ref_curve(E_gev, c, s, n):
    E_gev = np.asarray(E_gev, dtype=float)
    return np.sqrt(c ** 2 + (s / np.sqrt(E_gev)) ** 2 + (n / E_gev) ** 2)

# Longo shower-profile parametrization: dE/dt ~ t^(alpha-1) * exp(-beta*t)
def longo_profile(t, norm, alpha, beta):
    t = np.asarray(t, dtype=float)
    t_safe = np.clip(t, 1e-6, None)
    return norm * (t_safe ** (alpha - 1.0)) * np.exp(-beta * t_safe)

def print_channel_diagnostics(label, mod, ekey, totals, mu_fit, sigma_fit):
    """
    Print raw (unfit) statistics for a per-event photon-count array alongside
    the Gaussian-peak-fit result, so it's obvious when the fit is being asked
    to do something unreasonable (very low counts, lots of zero-hit events,
    fitted resolution blowing up, etc).
    """
    if len(totals) == 0:
        print(f"    [DIAG:{label}] {mod} {ekey}: no events")
        return

    n = len(totals)
    raw_mean = float(np.mean(totals))
    raw_std = float(np.std(totals))
    frac_zero = float(np.mean(totals == 0))
    lo, hi = float(np.min(totals)), float(np.max(totals))
    raw_res = raw_std / raw_mean if raw_mean > 0 else float("inf")
    fit_res = sigma_fit / mu_fit if mu_fit > 0 else float("inf")

    flags = []
    if raw_mean < 20:
        flags.append(f"LOW-STATS(mean={raw_mean:.1f} photons)")
    if frac_zero > 0.05:
        flags.append(f"ZERO-HEAVY({frac_zero*100:.1f}% events=0 hits)")
    if fit_res > 1.0:
        flags.append(f"FIT-RES-BLOWUP({fit_res*100:.0f}%)")
    if mu_fit > 0 and mu_fit < 0.5:
        flags.append(f"FIT-MU-NEAR-ZERO({mu_fit:.3f})")
    flag_str = f"  <-- {', '.join(flags)}" if flags else ""

    print(f"    [DIAG:{label}] {mod} {ekey}: N={n}, raw mean={raw_mean:.2f}, raw std={raw_std:.2f}, "
          f"raw res={raw_res*100:.1f}%, range=[{lo:.0f},{hi:.0f}], zero-frac={frac_zero*100:.1f}%, "
          f"fit mu={mu_fit:.3f}, fit sigma={sigma_fit:.3f}, fit res={fit_res*100:.1f}%{flag_str}")

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

    # --- Chunk Initializations ---
    up_first_chunks, down_first_chunks = [], []
    up_q_chunks, dw_q_chunks = [], []
    
    # Track BOTH faces for E and T energy counting
    up_e_hit_chunks, dw_e_hit_chunks = [], []
    up_t_hit_chunks, dw_t_hit_chunks = [], []
    
    down_first_t_chunks = []
    run_dirs = set()

    branch_list = ["Position_X", "Position_Y", "Position_Z", "GlobalTime", "TrackCreatorProcess", "LocalTime", "EventID", "ParticleName"]

    # --- Main File/Data Processing Loop ---
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
        proc = arrs["TrackCreatorProcess"]

        is_wls = (proc == b"OpWLS") | (proc == "OpWLS")

        # Channel Mapping
        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(np.hypot(dx, dy), axis=1)

        # Spatial / Particle Masks
        near_up = np.abs(z + detected_z_sensor) < 2.5
        near_dw = np.abs(z - detected_z_sensor) < 2.5
        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")
        gt = np.where(near_dw, gt + t_offset_ns, gt)
        
        is_prompt = (gt >= _GT_LO_NS) & (gt <= _GT_HI_NS)

        # 1. E-Type Channel Processing
        is_e = np.isin(channels, e_indices)
        m_e_up = is_e & is_prompt & near_up & is_optical
        m_e_dw = is_e & is_prompt & near_dw & is_optical

        c = _chunk_series(m_e_up, gt, ev, run_tag)
        if c is not None: 
            up_first_chunks.append(c)
            up_e_hit_chunks.append(c)   # Save upstream E-hits
            
        c = _chunk_series(m_e_dw, gt, ev, run_tag)
        if c is not None:
            down_first_chunks.append(c)
            dw_e_hit_chunks.append(c)   # Save downstream E-hits

        
        # 2. T-Type Channel Processing
    # 2. T-Type Channel Processing
        is_t = np.isin(channels, t_indices)

        # Raw timing logic for Quantile resolution — leave unfiltered (ΔT observable)
        m_t_up = is_t & is_optical & near_up
        m_t_dw = is_t & is_optical & near_dw
        c = _chunk_series(m_t_up, lt * 1000.0, ev, run_tag)
        if c is not None: up_q_chunks.append(c)
        c = _chunk_series(m_t_dw, lt * 1000.0, ev, run_tag)
        if c is not None: dw_q_chunks.append(c)

        # Prompt-filtered PHOTON-COUNT masks — WLS-only
        m_t_up_prompt = is_t & is_optical & is_wls & near_up & is_prompt
        m_t_dw_prompt = is_t & is_optical & is_wls & near_dw & is_prompt

        c = _chunk_series(m_t_up_prompt, gt, ev, run_tag)
        if c is not None: up_t_hit_chunks.append(c)
        c = _chunk_series(m_t_dw_prompt, gt, ev, run_tag)
        if c is not None:
            dw_t_hit_chunks.append(c)
            down_first_t_chunks.append(c)

    # --- Aggregations & Grouping --- (4 spaces: after the for-loop, not inside it)
    up_first = _grouped(up_first_chunks, "min")
    down_first = _grouped(down_first_chunks, "min")
    down_first_t = _grouped(down_first_t_chunks, "min")
    
        
    up_q = _grouped(up_q_chunks, ARRIVAL_QUANTILE)
    dw_q = _grouped(dw_q_chunks, ARRIVAL_QUANTILE)
        
    # Hits per event dictionary groupings
    up_e_hits_per_ev = _grouped(up_e_hit_chunks, "count")
    dw_e_hits_per_ev = _grouped(dw_e_hit_chunks, "count")
        
    up_t_hits_per_ev = _grouped(up_t_hit_chunks, "count")
    dw_t_hits_per_ev = _grouped(dw_t_hit_chunks, "count")

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
              f"{len(common_t_evs)} T-coincidences, {len(common_e_keys)} E-coincidences "
              f"(sigma_t={sigma_t_ps:.1f}ps)")

    # --- Explicit Event Alignment and Double-Ended Reconstitution ---
    master_e_events = sorted(list(down_first.keys()))
    master_t_events = sorted(list(down_first_t.keys()))

    return {
        "sigma_t_ps": sigma_t_ps,
        "raw_bm_data": all_bm_raw_ps,
        "tof_profile": profile_counts,
        "lyso_thick": lyso_thick,
        "pitch_mm": gap_thick_mm + _W_THICK_MM,
        "n_t_coincidences": len(common_t_evs),
        "n_e_coincidences": len(common_e_keys),
        
        # Aligned primary time measurements
        "dw_first_times": np.array([down_first[k] for k in master_e_events]),
        
        # Combined dual-ended yields (Front + Back) aligned perfectly by Event ID
        "dw_e_total": np.array([dw_e_hits_per_ev.get(k, 0) + up_e_hits_per_ev.get(k, 0) for k in master_e_events]),
        "dw_t_total": np.array([dw_t_hits_per_ev.get(k, 0) + up_t_hits_per_ev.get(k, 0) for k in master_t_events]),
        
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
    # Per-module plots (timing panels, longitudinal profiles, energy
    # performance, transverse profiles) live under analysis_out/<module_name>/.
    # Global cross-module comparison plots stay in analysis_out/summary_plots/.
    summary_dir = analysis_out / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)

    def module_dir(mod_name: str) -> Path:
        d = analysis_out / mod_name
        d.mkdir(parents=True, exist_ok=True)
        return d

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

            sweeps = list(mod_path.glob("sweep_*"))
            if not sweeps: continue
            
            # Extract all digits from the folder name to compare them numerically
            def extract_timestamp_key(path):
                digits = "".join(re.findall(r"\d+", path.name))
                return int(digits) if digits else 0

            # Grab the folder representing the highest numerical timestamp
            target_sweep = max(sweeps, key=extract_timestamp_key)
            print(f"Queuing '{mod}' -> {target_sweep.name} (Latest future/past timestamp)")

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

    energy_res_by_module = {}  # mod -> (energies_gev, res_e_list, res_e_err), for global comparison plot

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
       
        # 1. TIMING HIERARCHY — STABLE FIT WITH DYNAMIC FREEDMAN-DIACONIS BINNING
        # ─────────────────────────────────────────────────────────────────────
        # Ensure the directory physically exists before saving
        mod_dir = module_dir(mod)

        print(f"\n[DEBUG] Processing module {mod}. Total energy keys to plot: {len(energy_keys)}")

        fig_time, axs_time = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_time = axs_time.flatten()

        plotted_count = 0

        for idx, ekey in enumerate(energy_keys):
            ax = axs_time[idx]
            data = master_summary[mod][ekey].get("raw_bm_data", np.array([]))

            if len(data) > 0:
                plotted_count += 1
                # Clean outlier events far from the core
                clean = clean_around_mode(data, window_ps=500.0)

                # Robustly estimate the peak location and width (ignoring extreme tails)
                median_val = float(np.median(clean))
                std_robust = float(np.std(clean)) if len(clean) > 1 else 15.0

                # Auto-focus the histogram range on the active peak region (+/- 3.5 sigma)
                hist_lo = max(float(np.min(clean)), median_val - 3.5 * std_robust)
                hist_hi = min(float(np.max(clean)), median_val + 3.5 * std_robust)

                if hist_hi <= hist_lo:
                    hist_hi = hist_lo + 50.0

                plot_bins = 80 
                # Focus our data array to just the plotting range to calculate Freedman-Diaconis binning
                focused_data = clean[(clean >= hist_lo) & (clean <= hist_hi)]
                n_points = len(focused_data)

                if n_points > 2:
                    q75, q25 = np.percentile(focused_data, [75, 25])
                    iqr = q75 - q25
                    if iqr > 0:
                        # Freedman-Diaconis bin width formula
                        fd_width = 2.0 * iqr / (n_points ** (1.0 / 3.0))
                    else:
                        fd_width = 3.5 * std_robust / (n_points ** (1.0 / 3.0))
                    
                    # Prevent bins from being ridiculously small or large
                    fd_width = max(2.5, min(fd_width, 15.0)) 
                    plot_bins = max(10, int(np.ceil((hist_hi - hist_lo) / fd_width)))
                else:
                    plot_bins = 25

                # Plot the underlying data histogram focused on the peak area
                # Plot the underlying data histogram with dynamic binning
                # Plot the underlying data histogram with dynamic binning
                counts, edges, _ = ax.hist(clean, bins=plot_bins, range=(hist_lo, hist_hi),
                                            color=mod_colors.get(mod, "#f708af"), alpha=0.6, edgecolor="black", label="Data")

                bin_centers = (edges[:-1] + edges[1:]) / 2.0
                
                # -----------------------------------------------------------------
                # NEW: EMPIRICAL FWHM CALCULATION & VISUAL MARKER
                # -----------------------------------------------------------------
                max_idx = np.argmax(counts)
                max_val = counts[max_idx]
                half_max = max_val / 2.0
                
                # Find left crossing
                left_slice = counts[:max_idx]
                if len(left_slice) > 0 and np.any(left_slice < half_max):
                    l_idx = np.where(left_slice < half_max)[0][-1]
                    x0, y0 = bin_centers[l_idx], counts[l_idx]
                    x1, y1 = bin_centers[l_idx + 1], counts[l_idx + 1]
                    x_left = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
                else:
                    x_left = bin_centers[0]
                    
                # Find right crossing
                right_slice = counts[max_idx:]
                if len(right_slice) > 0 and np.any(right_slice < half_max):
                    r_idx_rel = np.where(right_slice < half_max)[0][0]
                    r_idx = max_idx + r_idx_rel
                    x0, y0 = bin_centers[r_idx - 1], counts[r_idx - 1]
                    x1, y1 = bin_centers[r_idx], counts[r_idx]
                    x_right = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
                else:
                    x_right = bin_centers[-1]
                    
                emp_fwhm = x_right - x_left
                
                # Draw Empirical FWHM Dimension Line (Blue)
                ax.annotate('', xy=(x_left, half_max), xytext=(x_right, half_max),
                            arrowprops=dict(arrowstyle='<|-|>', color='#004488', lw=1.5, shrinkA=0, shrinkB=0))
                ax.text(bin_centers[max_idx], half_max + max_val * 0.03, f"Emp. FWHM: {emp_fwhm:.1f} ps", 
                        ha='center', va='bottom', color='#004488', fontsize=8, fontweight='bold')
                # -----------------------------------------------------------------

                # Define standard Gaussian equation inline
                def straight_gaussian(x, amp, mu, sigma):
                    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

                threshold_val = counts.max() * 0.5

                # Initialize fit parameters based strictly on the peak
                mu_guess = float(bin_centers[max_idx])
                std_guess = std_robust * 0.6

                p0_g = [float(counts.max()), mu_guess, std_guess]
                bounds_g = ([0.0, hist_lo, 0.1], [counts.max() * 2.0, hist_hi, (hist_hi - hist_lo)])

                try:
                    # 1. Create a mask to isolate bins >= 50% of the peak maximum
                    fit_mask = counts >= threshold_val
                    
                    if np.sum(fit_mask) < 10.0:
                        raise ValueError("Not enough bins in the top-half window for a stable curve_fit")

                    # 2. FIT using ONLY the top half of the peak
                    popt, _ = curve_fit(
                        straight_gaussian, 
                        bin_centers[fit_mask], 
                        counts[fit_mask], 
                        p0=p0_g, 
                        bounds=bounds_g, 
                        maxfev=10000
                    )
                    amp_f, mu_f, sigma_f = popt
                    master_summary[mod][ekey]["sigma_t_ps"] = sigma_f

                    # 3. Find boundaries for visualization and Fit FWHM marker
                    x_min = mu_f - 1.177 * sigma_f
                    x_max = mu_f + 1.177 * sigma_f

                    x_fit = np.linspace(x_min, x_max, 1000)
                    y_fit = straight_gaussian(x_fit, amp_f, mu_f, sigma_f)

                    label_text = f"Gaussian (Top Half Fit)\n$\\sigma_t$ = {sigma_f:.1f} ps"
                    
                    # Draw Fit FWHM Dimension Line (Black, dashed) slightly lower to prevent overlap
                    fit_half_max = amp_f / 2.0
                    ax.annotate('', xy=(x_min, fit_half_max * 0.95), xytext=(x_max, fit_half_max * 0.95),
                                arrowprops=dict(arrowstyle='<|-|>', color='black', lw=1.2, linestyle='--', shrinkA=0, shrinkB=0))
                    ax.text(mu_f, fit_half_max * 0.95 - max_val * 0.05, f"Fit FWHM: {(2.355*sigma_f):.1f} ps", 
                            ha='center', va='top', color='black', fontsize=8)
                    
                except Exception as e:
                    print(f"  [WARNING] Fit failed for {ekey} ({mod}): {e}. Using fallback.")
                    fit_mask = counts >= threshold_val
                    bin_lefts = edges[:-1][fit_mask]
                    bin_rights = edges[1:][fit_mask]
                    x_min, x_max = bin_lefts.min(), bin_rights.max() if fit_mask.any() else (hist_lo, hist_hi)

                    top_half_raw = clean[(clean >= x_min) & (clean <= x_max)]
                    mu_f = float(np.mean(top_half_raw)) if len(top_half_raw) > 0 else mu_guess
                    sigma_f = float(np.std(top_half_raw)) if len(top_half_raw) > 1 else std_guess
                    master_summary[mod][ekey]["sigma_t_ps"] = sigma_f

                    x_fit = np.linspace(x_min, x_max, 1000)
                    y_fit = counts.max() * np.exp(-0.5 * ((x_fit - mu_f) / sigma_f) ** 2)
                    label_text = f"RMS Fallback\n$\\sigma_{{top\\,half}}$ = {sigma_f:.1f} ps"

                # Plot the resulting fit strictly within the top-half domain
                ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=2.5, label=label_text)
                ax.set_xlim(hist_lo, hist_hi)
                ax.legend(loc="upper right", fontsize=9)
            else:
                print(f"  [WARNING] No raw data found for energy key: {ekey}")

        # Safely remove empty axes using the actual plotted count
        for idx in range(plotted_count, len(axs_time)):
            fig_time.delaxes(axs_time[idx])

        # Write out the figure if we actually plotted data
        if plotted_count > 0:
            fig_time.suptitle(f"Timing Resolution Distributions — {mod}", fontsize=14, fontweight="bold", y=0.98)
            fig_time.tight_layout()

            save_path = mod_dir / f"{mod}_timing_panels.png"
            fig_time.savefig(save_path, dpi=200)
            print(f"[SUCCESS] Saved timing plot to: {save_path.resolve()}")
        else:
            print(f"[ERROR] Did not generate plot for {mod} because 0 subplots had data.")

        plt.close(fig_time)

        # ─────────────────────────────────────────────────────────────────────
        # 2. LONGITUDINAL PROFILE RECONSTRUCTION & RL-UNFOLDING
        # ─────────────────────────────────────────────────────────────────────
        truth_curves_by_energy = {}  # E_gev -> normalized DoseActor profile (for overlay plot below)

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
            # 1. Calculate the spatial spread based on the prompt timing peak
            s_z_prompt = v_eff_for_module(mod) * (sigma_t_ps / 1000.0)
            base_sigma_layer = s_z_prompt / pitch_mm if pitch_mm > 0 else 1.0

            # 2. HEURISTIC KERNEL INFLATION
            # Scintillator decay time and extreme optical bouncing make the bulk light
            # spread much wider than the prompt peak. We multiply the kernel to match reality.
            # (Try values between 3.0 and 6.0; 4.0 is a solid starting point for LYSO/W)
            HEURISTIC_DISPERSION_FACTOR = 1.0
             
            
            sigma_layer = base_sigma_layer * HEURISTIC_DISPERSION_FACTOR

            if utils is not None and hasattr(utils, 'rl_unfold'):
                # Pass the inflated kernel to the algorithm
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
                truth_curves_by_energy[extract_numerical_energy(ekey)] = truth_norm_disp
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
            fig_prof.savefig(mod_dir / f"{mod}_{ekey}_profile.png", dpi=200)
            plt.close(fig_prof)

        # ─────────────────────────────────────────────────────────────────────
        # 2B. LONGITUDINAL SHOWER PROFILE VS ENERGY — DoseActor TRUTH OVERLAY
        # All energies overlaid on one axes, each fit with a Longo parametrization
        # ─────────────────────────────────────────────────────────────────────
        if len(truth_curves_by_energy) >= 2:
            fig_ov, ax_ov = plt.subplots(figsize=(8, 6))
            cmap = cm.get_cmap("rainbow", len(truth_curves_by_energy))
            sorted_e = sorted(truth_curves_by_energy.keys())

            for i, E_val in enumerate(sorted_e):
                prof = truth_curves_by_energy[E_val]
                color = cmap(i)

                ax_ov.step(layers, prof, where="mid", color=color, alpha=0.5, linewidth=1.0,
                           label=f"{E_val:g} GeV")

                try:
                    peak_idx = int(np.argmax(prof))
                    p0 = [float(np.max(prof)), max(peak_idx * 0.3, 1.5), 0.3]
                    popt, _ = curve_fit(
                        longo_profile, layers, prof, p0=p0,
                        bounds=([0.0, 0.5, 0.01], [np.max(prof) * 20.0, 40.0, 5.0]),
                        maxfev=10000,
                    )
                    t_smooth = np.linspace(layers.min(), layers.max(), 300)
                    ax_ov.plot(t_smooth, longo_profile(t_smooth, *popt), color=color, linewidth=1.8)
                except Exception as e:
                    print(f"  [WARNING] Longo fit failed for {mod} @ {E_val} GeV: {e}")

            ax_ov.set_xlabel("LYSO layer", fontsize=11)
            ax_ov.set_ylabel("normalized ⟨E⟩", fontsize=11)
            ax_ov.set_title(f"Longitudinal shower profile vs energy — {mod}", fontsize=13, fontweight="bold")
            ax_ov.grid(True, linestyle=":", alpha=0.6)
            ax_ov.legend(title="Longo fit: $t^{\\alpha-1}e^{-\\beta t}$", fontsize=8, title_fontsize=9)
            fig_ov.tight_layout()
            fig_ov.savefig(mod_dir / f"{mod}_longitudinal_overlay.png", dpi=200)
            plt.close(fig_ov)
            print(f"[SUCCESS] Saved longitudinal overlay plot for {mod}")
        else:
            print(f"[WARNING] Not enough DoseActor truth curves to build longitudinal overlay for {mod}")
# ─────────────────────────────────────────────────────────────────────
        # 3. ENERGY LINEARITY AND RESOLUTION PANELS (E-type channels)
        # ─────────────────────────────────────────────────────────────────────
        energies_gev, mu_e_list, res_e_list, mu_e_err, res_e_err = [], [], [], [], []

        for ekey in energy_keys:
            E_val = extract_numerical_energy(ekey)
            if E_val <= 0: continue

            e_totals = master_summary[mod][ekey].get("dw_e_total", np.array([]))
            if len(e_totals) < 5: continue

            # Convert the percentage output of robust_resolution back to a fraction
            res_val_pct, res_err_val_pct = robust_resolution(e_totals, nsig=2.0, max_iters=4)
            res_val = res_val_pct / 100.0
            res_err_val = res_err_val_pct / 100.0

            mu_val = float(np.mean(e_totals)) # Ensure mu_val is defined for downstream arrays

            if mu_val > 0.1 and res_val > 0:
                if not np.isnan(res_val) and not np.isinf(res_val) and res_val < 10.0:
                    energies_gev.append(E_val)
                    mu_e_list.append(mu_val)
                    res_e_list.append(res_val)
                    mu_e_err.append(mu_val * res_val / np.sqrt(len(e_totals)))
                    res_e_err.append(res_err_val)
                else:
                    print(f"  [FILTERED] E-type {mod} @ {E_val} GeV rejected (Unphysical resolution: {res_val*100:.1f}%)")

        # ─────────────────────────────────────────────────────────────────────
        # 3-T. SHOWER-MAX RESOLUTION INPUT (T-type channels)
        # ─────────────────────────────────────────────────────────────────────
        energies_gev_t, mu_t_list, res_t_list, mu_t_err, res_t_err = [], [], [], [], []

        for ekey in energy_keys:
            E_val = extract_numerical_energy(ekey)
            if E_val <= 0: continue

            t_totals = master_summary[mod][ekey].get("dw_t_total", np.array([]))
            t_totals = np.array(t_totals) # Ensure it's a numpy array
            
            # --- DEBUG AND FILTER ---
            if len(t_totals) > 0:
                print(f"  [{mod} @ {ekey}] RAW  -> N: {len(t_totals)}, Mean: {np.mean(t_totals):.1f}, Max: {np.max(t_totals)}")
            
            # Cut out events that recorded exactly 0 photons
            t_totals = t_totals[t_totals > 0]
            
            if len(t_totals) > 0:
                print(f"  [{mod} @ {ekey}] >0 CUT -> N: {len(t_totals)}, Mean: {np.mean(t_totals):.1f}, Median: {np.median(t_totals):.1f}")
            # ------------------------

            if len(t_totals) < 5: continue

            # Convert the percentage output of robust_resolution back to a fraction
            res_t_val_pct, res_t_err_val_pct = robust_resolution(t_totals, nsig=2.0, max_iters=4)
            res_t_val = res_t_val_pct / 100.0
            res_t_err_val = res_t_err_val_pct / 100.0

            mu_val = float(np.mean(t_totals)) # Ensure mu_val is defined for downstream arrays

            if mu_val > 0.1 and res_t_val > 0:
                if not np.isnan(res_t_val) and not np.isinf(res_t_val) and res_t_val < 10.0:
                    energies_gev_t.append(E_val)
                    mu_t_list.append(mu_val)
                    res_t_list.append(res_t_val)
                    mu_t_err.append(mu_val * res_t_val / np.sqrt(len(t_totals)))
                    res_t_err.append(res_t_err_val)
                else:
                    print(f"  [FILTERED] T-type {mod} @ {E_val} GeV rejected (Unphysical resolution: {res_t_val*100:.1f}%)")

        def resolution_func(E, c, s, n):
            return np.sqrt(c ** 2 + (s / np.sqrt(E)) ** 2 + (n / E) ** 2)

        if len(energies_gev) >= 1:
            energies_gev = np.array(energies_gev)
            mu_e_list = np.array(mu_e_list)
            res_e_list = np.array(res_e_list)
            res_e_err_arr = np.array(res_e_err)
            energy_res_by_module[mod] = (energies_gev, res_e_list, res_e_err_arr)

            fig_er, (ax_lin, ax_res) = plt.subplots(1, 2, figsize=(14, 6))

            # --- ROBUST LINEAR FIT GUARD ---
            def linear_func(x, m, b): return m * x + b
            popt_lin = None
            if len(energies_gev) >= 2:
                try:
                    popt_lin, _ = curve_fit(linear_func, energies_gev, mu_e_list)
                except Exception as e:
                    print(f"  [WARNING] Linearity fit failed for {mod}: {e}")

            ax_lin.errorbar(energies_gev, mu_e_list, yerr=mu_e_err, fmt=mod_markers.get(mod, 'o'),
                            color=mod_colors.get(mod, 'black'), label=f"Simulated Data ({mod})")

            if popt_lin is not None:
                x_lin_smooth = np.linspace(0, max(energies_gev) * 1.1, 100)
                ax_lin.plot(x_lin_smooth, linear_func(x_lin_smooth, *popt_lin),
                            color="black", linestyle="--", label=f"Fit: {popt_lin[0]:.3e} photons/GeV")

            ax_lin.set_xlabel("Beam Energy (GeV)", fontsize=11)
            ax_lin.set_ylabel("Sum Amplitude (Downstream E-Type Photons)", fontsize=11)
            ax_lin.set_title("Energy Linearity", fontsize=13, fontweight="bold")
            ax_lin.grid(True, linestyle=":", alpha=0.6)
            ax_lin.legend(fontsize=10)

            # --- ROBUST RESOLUTION FIT GUARD ---
            popt_res = None
            fit_label = "Fit failed (Not enough data points)"
            if len(energies_gev) >= 3:
                try:
                    popt_res, _ = curve_fit(resolution_func, energies_gev, res_e_list,
                                            p0=[0.05, 0.2, 0.05], bounds=(0, [2.0, 10.0, 10.0]))
                    c_f, s_f, n_f = popt_res
                    fit_label = f"Fit: {c_f * 100:.1f}% $\\oplus$ {s_f * 100:.1f}%/$\\sqrt{{E}}$ $\\oplus$ {n_f * 100:.1f}%/E"
                except Exception as e:
                    print(f"  [WARNING] Resolution fit failed for {mod}: {e}")
            else:
                fit_label = f"Fit skipped (Requires 3 points; have {len(energies_gev)})"

            ax_res.errorbar(energies_gev, res_e_list, yerr=res_e_err, fmt=mod_markers.get(mod, 'o'),
                            color=mod_colors.get(mod, 'black'), label="Simulated Resolution (E-type)")

            if popt_res is not None:
                x_res_smooth = np.linspace(min(energies_gev) * 0.8, max(energies_gev) * 1.1, 100)
                ax_res.plot(x_res_smooth, resolution_func(x_res_smooth, *popt_res),
                            color="black", linestyle="--", label=fit_label)
                
                # -- External reference overlays (only active when plotting fit curve space) --
                for ref_name, ref_p in ENERGY_REF_CURVES.items():
                    y_ref = energy_ref_curve(x_res_smooth, ref_p["c"], ref_p["s"], ref_p["n"])
                    ax_res.plot(x_res_smooth, y_ref, color=ref_p["color"], linestyle=ref_p["ls"], linewidth=1.5,
                                label=f"{ref_name}: {ref_p['c']*100:.2f}$\\oplus${ref_p['s']*100:.2f}/$\\sqrt{{E}}$"
                                      f"$\\oplus${ref_p['n']*100:.2f}/E")
            
            ax_res.axhspan(ENERGY_DATA_BAND_FRAC[0], ENERGY_DATA_BAND_FRAC[1], color="lightgray", alpha=0.4,
                           label=f"DATA sum$_{{lg}}$: {ENERGY_DATA_BAND_FRAC[0]*100:.0f}-{ENERGY_DATA_BAND_FRAC[1]*100:.0f}%")

            ax_res.set_xlabel("Beam Energy (GeV)", fontsize=11)
            ax_res.set_ylabel(r"$\sigma_E / E_{meas}$", fontsize=11)
            ax_res.set_title("Energy Resolution (E-type channels)", fontsize=13, fontweight="bold")
            ax_res.grid(True, linestyle=":", alpha=0.6)
            ax_res.legend(fontsize=10)

            fig_er.suptitle(f"Calorimeter Energy Performance — {mod}", fontsize=15, fontweight="bold")
            fig_er.tight_layout()
            fig_er.savefig(mod_dir / f"{mod}_energy_performance.png", dpi=200)
            plt.close(fig_er)
        else:
            print(f"  [WARNING] Not enough E-type energy points for {mod} energy_performance plot.")

        # ─────────────────────────────────────────────────────────────────
        # 3B. SHOWER-MAX ENERGY RESOLUTION (standalone, paper-style layout)
        # ─────────────────────────────────────────────────────────────────
        if len(energies_gev_t) >= 1:
            energies_gev_t = np.array(energies_gev_t)
            res_t_list = np.array(res_t_list)
            res_t_err_arr = np.array(res_t_err)

            popt_res_t = None
            c_ft, s_ft, n_ft = 0.0, 0.0, 0.0
            if len(energies_gev_t) >= 3:
                try:
                    popt_res_t, _ = curve_fit(resolution_func, energies_gev_t, res_t_list,
                                              p0=[0.05, 0.2, 0.05], bounds=(0, [2.0, 10.0, 10.0]))
                    c_ft, s_ft, n_ft = popt_res_t
                except Exception as e:
                    print(f"  [WARNING] T-type resolution fit failed for {mod}: {e}")

            fig_sm, ax_sm = plt.subplots(figsize=(8, 6))

            ax_sm.errorbar(energies_gev_t, res_t_list * 100.0, yerr=res_t_err_arr * 100.0,
                           fmt=mod_markers.get(mod, 'o'), color=mod_colors.get(mod, 'blue'),
                           markersize=7, capsize=4, capthick=1.2,
                           label="sim (photon count, T-type channels)")

            if popt_res_t is not None:
                x_sm_smooth = np.linspace(min(energies_gev_t) * 0.8, max(energies_gev_t) * 1.1, 200)
                ax_sm.plot(x_sm_smooth, resolution_func(x_sm_smooth, *popt_res_t) * 100.0,
                           color=mod_colors.get(mod, 'blue'), linestyle=':', linewidth=1.5)

                for ref_name, ref_p in ENERGY_REF_CURVES.items():
                    y_ref = energy_ref_curve(x_sm_smooth, ref_p["c"], ref_p["s"], ref_p["n"]) * 100.0
                    ax_sm.plot(x_sm_smooth, y_ref, color=ref_p["color"], linestyle=ref_p["ls"], linewidth=1.5)

            fit_text = ""
            if popt_res_t is not None:
                fit_text += (
                    f"sim (photon count, T-type channels): "
                    f"{c_ft*100:.2f} $\\oplus$ {s_ft*100:.2f}/$\\sqrt{{E}}$ $\\oplus$ {n_ft*100:.2f}/E\n"
                )
            else:
                fit_text += f"sim (photon count, T-type channels): Fit skipped (< 3 points)\n"
                
            for ref_name, ref_p in ENERGY_REF_CURVES.items():
                fit_text += (
                    f"{ref_name}: {ref_p['c']*100:.2f} $\\oplus$ {ref_p['s']*100:.2f}/$\\sqrt{{E}}$ "
                    f"$\\oplus$ {ref_p['n']*100:.2f}/E\n"
                )
            ax_sm.text(0.98, 0.97, fit_text.strip(), transform=ax_sm.transAxes,
                       ha='right', va='top', fontsize=9, color=mod_colors.get(mod, 'blue'))

            ax_sm.set_xlabel("E$_{beam}$ (GeV)", fontsize=11)
            ax_sm.set_ylabel(r"$\sigma$/mean (%)", fontsize=11)
            ax_sm.set_title(f"Shower-max energy resolution (T-type) — {mod}", fontsize=13, fontweight="bold")
            ax_sm.grid(True, linestyle=":", alpha=0.6)
            ax_sm.legend(fontsize=9, loc='lower right')

            fig_sm.tight_layout()
            fig_sm.savefig(mod_dir / f"{mod}_showermax_energy_resolution.png", dpi=200)
            plt.close(fig_sm)
        else:
            print(f"  [WARNING] Not enough T-type energy points for {mod} shower-max plot.")

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

    # -- External reference overlays: DATA (test beam) + published paper --
    if any_points:
        x_ref = np.linspace(4.0, 200.0, 200)
        for ref_name, ref_p in TIMING_REF_CURVES.items():
            y_ref = timing_ref_curve(x_ref, ref_p["stoch"], ref_p["const"])
            ax_perf.plot(x_ref, y_ref, color=ref_p["color"], linestyle=ref_p["ls"], linewidth=2.0,
                         label=f"{ref_name}: {ref_p['stoch']:.0f}/$\\sqrt{{E}}$ $\\oplus$ {ref_p['const']:.1f} ps")

    ax_perf.set_xlabel("Incident Particle Beam Energy (GeV)", fontweight="bold")
    ax_perf.set_ylabel(r"Gaussian Timing Resolution $\sigma_t$ (ps)", fontweight="bold")
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
    # 4B. UNIFIED PERFORMANCE HORIZON — EMPIRICAL FWHM (NO FIT)
    # ─────────────────────────────────────────────────────────────────────
    fig_fwhm, ax_fwhm = plt.subplots(figsize=(10, 7))
    any_fwhm_points = False

    for mod in modules:
        if mod not in master_summary or not master_summary[mod]:
            continue

        energy_keys = sorted(master_summary[mod].keys(), key=extract_numerical_energy)
        if not energy_keys:
            continue

        x_energy_fwhm, y_fwhm = [], []
        for ekey in energy_keys:
            raw_data = master_summary[mod][ekey].get("raw_bm_data", np.array([]))
            n_ev = master_summary[mod][ekey].get("n_t_coincidences", 0)
            
            if n_ev < 8 or len(raw_data) < 10:
                continue
                
            # Clean outliers using your existing methodology
            clean_data = clean_around_mode(raw_data, window_ps=500.0)
            fwhm_val = calculate_empirical_fwhm(clean_data, bins=80)
            
            if fwhm_val > 0:
                x_energy_fwhm.append(extract_numerical_energy(ekey))
                y_fwhm.append(fwhm_val)

        if x_energy_fwhm:
            any_fwhm_points = True
            ax_fwhm.plot(
                x_energy_fwhm, y_fwhm, 
                marker=mod_markers.get(mod, 'o'), 
                color=mod_colors.get(mod, 'black'),
                linestyle=mod_linestyles.get(mod, '-'),  
                linewidth=2, 
                markersize=7, 
                label=mod
            )

    ax_fwhm.set_xlabel("Incident Particle Beam Energy (GeV)", fontweight="bold")
    ax_fwhm.set_ylabel("Empirical FWHM (ps)", fontweight="bold")
    ax_fwhm.set_title("Unified Performance Horizon — Empirical FWHM vs Energy", fontsize=12, fontweight="bold")
    ax_fwhm.grid(True, linestyle=":", alpha=0.6)
    ax_fwhm.set_xscale("log")
    ax_fwhm.set_xticks([25, 50, 100, 200])
    ax_fwhm.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    if any_fwhm_points:
        ax_fwhm.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9, frameon=True)
    else:
        ax_fwhm.text(0.5, 0.5, "No modules had sufficient statistics", ha='center', va='center', transform=ax_fwhm.transAxes)

    fig_fwhm.tight_layout()

    # Re-use the existing visual encoding key text
    fig_fwhm.text(
        1.02, 0.15, key_text, 
        fontsize=9, 
        family='monospace', 
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', edgecolor='#d3d3d3', alpha=0.9)
    )
    
    fwhm_save_path = summary_dir / "timing_fwhm_vs_energy.png"
    fig_fwhm.savefig(fwhm_save_path, dpi=220, bbox_inches="tight")
    plt.close(fig_fwhm)
    print(f"[SUCCESS] Saved Empirical FWHM vs Energy plot to: {fwhm_save_path.resolve()}")

    # ─────────────────────────────────────────────────────────────────────
    # 4C. UNIFIED PERFORMANCE HORIZON — ENERGY RESOLUTION VS ENERGY
    # ─────────────────────────────────────────────────────────────────────
    fig_eres, ax_eres = plt.subplots(figsize=(10, 7))
    any_eres_points = False

    for mod in modules:
        if mod not in energy_res_by_module:
            continue
        x_e, y_e, yerr_e = energy_res_by_module[mod]
        any_eres_points = True
        ax_eres.errorbar(
            x_e, y_e * 100.0, yerr=yerr_e * 100.0,
            marker=mod_markers.get(mod, 'o'),
            color=mod_colors.get(mod, 'black'),
            linestyle=mod_linestyles.get(mod, '-'),
            linewidth=2, markersize=7, capsize=4, capthick=1.5,
            label=mod
        )

    if any_eres_points:
        x_ref = np.linspace(4.0, 200.0, 200)
        for ref_name, ref_p in ENERGY_REF_CURVES.items():
            y_ref = energy_ref_curve(x_ref, ref_p["c"], ref_p["s"], ref_p["n"]) * 100.0
            ax_eres.plot(x_ref, y_ref, color=ref_p["color"], linestyle=ref_p["ls"], linewidth=2.0,
                         label=f"{ref_name}: {ref_p['c']*100:.2f}$\\oplus${ref_p['s']*100:.2f}/$\\sqrt{{E}}$"
                               f"$\\oplus${ref_p['n']*100:.2f}/E")
        ax_eres.axhspan(ENERGY_DATA_BAND_FRAC[0] * 100.0, ENERGY_DATA_BAND_FRAC[1] * 100.0,
                        color="lightgray", alpha=0.4,
                        label=f"DATA sum$_{{lg}}$: {ENERGY_DATA_BAND_FRAC[0]*100:.0f}-{ENERGY_DATA_BAND_FRAC[1]*100:.0f}%")

    ax_eres.set_xlabel("Incident Particle Beam Energy (GeV)", fontweight="bold")
    ax_eres.set_ylabel(r"$\sigma_E / E_{meas}$ (%)", fontweight="bold")
    ax_eres.set_title("Unified Performance Horizon — Energy Resolution vs Energy", fontsize=12, fontweight="bold")
    ax_eres.grid(True, linestyle=":", alpha=0.6)
    ax_eres.set_xscale("log")
    ax_eres.set_xticks([25, 50, 100, 200])
    ax_eres.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    if any_eres_points:
        ax_eres.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9, frameon=True)
    else:
        ax_eres.text(0.5, 0.5, "No modules had sufficient statistics", ha='center', va='center', transform=ax_eres.transAxes)

    fig_eres.tight_layout()
    fig_eres.text(
        1.02, 0.15, key_text,
        fontsize=9,
        family='monospace',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', edgecolor='#d3d3d3', alpha=0.9)
    )
    eres_save_path = summary_dir / "energy_resolution_vs_energy.png"
    fig_eres.savefig(eres_save_path, dpi=220, bbox_inches="tight")
    plt.close(fig_eres)
    print(f"[SUCCESS] Saved Energy Resolution vs Energy plot to: {eres_save_path.resolve()}")

    # ─────────────────────────────────────────────────────────────────────
    # 5. Plot Transverse Shower Profiles for Each Module and Energy
    # ─────────────────────────────────────────────────────────────────────
    def plot_transverse_profile(transverse_data, module_name):
        from matplotlib.colors import LogNorm
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(transverse_data, cmap='inferno', norm = LogNorm(), origin='lower', interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Energy Deposited / Dose", rotation=270, labelpad=15)
        ax.set_title(f"Transverse Shower Profile — {module_name}", fontweight='bold')
        ax.set_xlabel("X Bins")
        ax.set_ylabel("Y Bins")
        plt.tight_layout()

    for mod in modules:
        if mod not in master_summary or not master_summary[mod]:
            continue

        mod_path = base_dir / mod / "runs" / mod
        if not mod_path.exists():
            mod_path = base_dir / mod
        sweeps = sorted(mod_path.glob("sweep_*"), key=lambda p: p.name)
        if not sweeps:
            print(f"[WARNING] No sweep directories found for {mod}. Skipping transverse profile.")
            continue
        target_sweep = sweeps[-1]

        mhd_files = sorted(target_sweep.glob("*GeV/run_*/transverse_shower_max_edep.mhd"))
        if not mhd_files:
            print(f"[WARNING] No transverse_shower_max_edep.mhd under {target_sweep}. Skipping {mod}.")
            continue

        accum = None
        for mhd_file in mhd_files:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mhd_file)))
            accum = arr.astype(np.float64) if accum is None else accum + arr

        accum_squeezed = np.squeeze(accum)

        plot_transverse_profile(accum_squeezed, mod)
        save_path = module_dir(mod) / f"{mod}_transverse_profile.png"
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[SUCCESS] Saved transverse profile plot for {mod} to: {save_path.resolve()}")


    
    # ─────────────────────────────────────────────────────────────────────
    # 6. EXPORT MASTER MATRIX TEXT REPORT
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