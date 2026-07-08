#!/usr/bin/env python3
import os
import sys
import datetime
import warnings
from pathlib import Path
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from scipy.optimize import minimize
from scipy.stats import gamma
import scipy.special as sp

# Add these alongside your other constants
_SHOWER_FIRST = 9  # T-type active region start (LYSO layer 9)
_SHOWER_LAST = 13  # T-type active region end (LYSO layer 13)


# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL KINEMATICS & FIBER CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS = 299.792
REFRACTIVE_INDEX = 1.60
V_LIGHT_MM_NS = C_LIGHT_MM_NS / REFRACTIVE_INDEX
BOUNCE_FACTOR = 0.92
V_EFF_MM_NS = V_LIGHT_MM_NS * BOUNCE_FACTOR

# Assumed effective attenuation length of the BCF-92 WLS fiber inside the calorimeter
# (Can be tuned or calibrated against Monte Carlo truth)
_LAMBDA_EFF_MM = 3500.0  

_GT_LO_NS = 0.0
_GT_HI_NS = 50.0
_TYVEK_THICK_MM = 0.2032
_W_THICK_MM = 2.5
_N_LYSO = 29
_N_W = 28

ARRIVAL_QUANTILE = 0.10
MIN_PHOTONS_PER_FACE = 1

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

_KNOWN_Z_PLANES = {91.65: 1.5, 135.15: 4.5}

# ─────────────────────────────────────────────────────────────────────────────
# TARGET DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SWEEPS = {
    "radi_cal_energy": Path("/home/uakgun/env/THOMAS/discrete_sims/radi_cal_energy/runs/radi_cal_energy/sweep_20260707_170533"),
    "radi_cal_triple": Path("/home/uakgun/env/THOMAS/discrete_sims/radi_cal_triple/runs/radi_cal_triple/sweep_20260706_171040"),
    "rc_hex": Path("/home/uakgun/env/THOMAS/discrete_sims/rc_hex/runs/rc_hex/sweep_20260706_171127"),
    "rc_hex_triple": Path("/home/uakgun/env/THOMAS/discrete_sims/rc_hex_triple/runs/rc_hex_triple/sweep_20260706_171101")
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def gamma_profile(z, E0, a, b):
    """
    Standard parameterization of EM shower longitudinal profile.
    Uses scipy.stats.gamma.pdf for numerical stability against overflow.
    """
    # SciPy's gamma.pdf is exactly our formula when scale = 1/b.
    # It calculates the curve safely without overflowing at high values.
    return E0 * gamma.pdf(z, a, scale=1.0/b)

def fit_gamma_to_observables(Z_cg_meas, Q_T_frac, z_T_start, z_T_end):
    """
    Solves for Gamma parameters (a, b) with strict physical boundaries.
    """
    def loss_function(params):
        a, b = params
        
        # 1. Match Center of Gravity (Mean = a/b)
        cg_theory = a / b
        cg_penalty = (cg_theory - Z_cg_meas)**2

        # 2. Match T-type hardware depth fraction
        integral_theory = gamma.cdf(z_T_end, a, scale=1.0/b) - gamma.cdf(z_T_start, a, scale=1.0/b)
        t_frac_penalty = ((integral_theory - Q_T_frac) * 1000)**2

        return cg_penalty + t_frac_penalty

    # Set strict physical bounds for an EM shower
    # 'a' (shape) typically between 2 and 20. 'b' (scale) typically between 0.1 and 1.0
    bounds = [(1.1, 30.0), (0.01, 5.0)] 
    
    # Keep the initial guesses sane and within bounds
    a_guess = 3.0
    b_guess = max(0.01, min(a_guess / max(Z_cg_meas, 1.0), 4.9))
    
    # Use L-BFGS-B to enforce the boundaries
    res = minimize(loss_function, [a_guess, b_guess], bounds=bounds, method='L-BFGS-B')
    
    return res.x[0], res.x[1]


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
# DATA PARSING
# ─────────────────────────────────────────────────────────────────────────────
def extract_profile_data(batch_dir: Path, is_hex: bool):
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

    lyso_thick = min(_KNOWN_Z_PLANES.items(), key=lambda kv: abs(kv[0] - detected_z_sensor))[1]
    gap_thick_mm = lyso_thick + 2 * _TYVEK_THICK_MM
    calor_thick_mm = (_N_LYSO * gap_thick_mm) + (_N_W * _W_THICK_MM)
    lyso_bounds = get_lyso_layer_bounds(lyso_thick, calor_thick_mm)

    cap_xy_map = HEX_CAP_XY if is_hex else SQUARE_CAP_XY
    t_indices = {1, 3, 5} if is_hex else {0, 1}
    e_indices = {0, 2, 4} if is_hex else {2, 3}

    up_first, down_first = {}, {}
    up_times_by_ev, dw_times_by_ev = {}, {}
    q_up_by_ev, q_dw_by_ev = {}, {}

    for fpath in hit_files:
        run_tag = fpath.parent.name
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                tree = f[tk]
                if tree.num_entries == 0: continue
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

        # ToF Reconstructions
        for eid, ti in zip(ev[m_e_up], gt[m_e_up]):
            key = (run_tag, int(eid))
            if key not in up_first or ti < up_first[key]: up_first[key] = float(ti)
        for eid, ti in zip(ev[m_e_dw], gt[m_e_dw]):
            key = (run_tag, int(eid))
            if key not in down_first or ti < down_first[key]: down_first[key] = float(ti)

        # Asymmetry Reconstructions (Charge Integration)
        ev_up, counts_up = np.unique(ev[m_e_up & is_optical], return_counts=True)
        ev_dw, counts_dw = np.unique(ev[m_e_dw & is_optical], return_counts=True)
        
        for eid, count in zip(ev_up, counts_up):
            q_up_by_ev[(run_tag, int(eid))] = q_up_by_ev.get((run_tag, int(eid)), 0) + count
        for eid, count in zip(ev_dw, counts_dw):
            q_dw_by_ev[(run_tag, int(eid))] = q_dw_by_ev.get((run_tag, int(eid)), 0) + count

        # Timing Resolution
        is_t = np.isin(channels, list(t_indices))
        m_t_up, m_t_dw = is_t & is_optical & near_up, is_t & is_optical & near_dw
        
        for e, t in zip(ev[m_t_up], lt[m_t_up] * 1000.0):
            up_times_by_ev.setdefault((run_tag, int(e)), []).append(t)
        for e, t in zip(ev[m_t_dw], lt[m_t_dw] * 1000.0):
            dw_times_by_ev.setdefault((run_tag, int(e)), []).append(t)

    # 1. Delta-T ToF Layer Extraction
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

    # 2. Timing Resolution Calculation
    common_t_evs = set(up_times_by_ev.keys()) & set(dw_times_by_ev.keys())
    all_bm_raw_ps = []
    for e in common_t_evs:
        if len(up_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE and len(dw_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE:
            t_up_q = np.quantile(up_times_by_ev[e], ARRIVAL_QUANTILE)
            t_dw_q = np.quantile(dw_times_by_ev[e], ARRIVAL_QUANTILE)
            all_bm_raw_ps.append((t_dw_q - t_up_q) / 2.0)

    clean_bm = clean_around_mode(np.array(all_bm_raw_ps), window_ps=500.0)
    _, _, sigma_t_ps = fit_gaussian_to_peak(clean_bm)

    # 3. Asymmetry Z_cg Calculation
    common_q_evs = set(q_up_by_ev.keys()) & set(q_dw_by_ev.keys())
    z_cg_arr = []
    for k in common_q_evs:
        q_u, q_d = q_up_by_ev[k], q_dw_by_ev[k]
        if q_u > 0 and q_d > 0:
            # Reconstruct Z relative to center. If shower is upstream (negative Z), 
            # Q_up > Q_down, so ln() > 0. We negate to match detector coordinates.
            z = - (_LAMBDA_EFF_MM / 2.0) * np.log(q_u / q_d)
            z_cg_arr.append(z)
    # Calculate aggregate hits for macroscopic fitting
    total_e_up = np.sum(list(q_up_by_ev.values()))
    total_e_dw = np.sum(list(q_dw_by_ev.values()))
    
    # We also need the total hits on the T-type depth markers
    total_t_hits = 0
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                tree = f[tk]
                if tree.num_entries == 0: continue
                x, y = tree["Position_X"].array(library="np"), tree["Position_Y"].array(library="np")
                pn = tree["ParticleName"].array(library="np")
                
                dx = x[:, np.newaxis] - cap_xy_map[:, 0]
                dy = y[:, np.newaxis] - cap_xy_map[:, 1]
                channels = np.argmin(np.hypot(dx, dy), axis=1)
                
                is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")
                is_t = np.isin(channels, list(t_indices))
                total_t_hits += np.sum(is_t & is_optical)
        except Exception: continue
    return {
        "tof_profile": profile_counts,
        "n_e_coincidences": len(common_e_keys),
        "sigma_t_ps": sigma_t_ps,
        "pitch_mm": gap_thick_mm + _W_THICK_MM,
        "lyso_thick": lyso_thick,
        "dw_first_times": np.array(list(down_first.values())),
        "z_cg_dist": np.array(z_cg_arr),
        "calor_thick": calor_thick_mm,
        "lyso_bounds": lyso_bounds,
        "total_e_up": total_e_up,
        "total_e_dw": total_e_dw,
        "total_t_hits": total_t_hits
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).resolve().parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_out = base_dir / "profile_analysis" / f"profile_summary_{timestamp}"
    analysis_out.mkdir(parents=True, exist_ok=True)

    print("Spawning Longitudinal Profile Extractor...")
    print(f"Targeting outputs to: {analysis_out.relative_to(base_dir)}\n")

    try:
        import analysis_utils as utils
    except ImportError:
        utils = None
        print("[Warning] analysis_utils not found. Sim truth overlays will be disabled.")

    mod_colors = {"radi_cal_energy": "#1976d2", "radi_cal_triple": "#388e3c", "rc_hex": "#d32f2f", "rc_hex_triple": "#7b1fa2"}
    layers = np.arange(1, _N_LYSO + 1)
    WLS_LAYERS = {8, 9, 10, 11}

    for mod, target_sweep in TARGET_SWEEPS.items():
        if not target_sweep.exists():
            print(f"  Skipping '{mod}' - Target sweep not found: {target_sweep}")
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

        fig_tof, axs_tof = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_tof = axs_tof.flatten()
        
        fig_dist, axs_dist = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_dist = axs_dist.flatten()

        # New Figure for Parametric Reconstruction
        fig_gamma, axs_gamma = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_gamma = axs_gamma.flatten()

        # New Figure for Asymmetry Z_cg
        fig_cg, axs_cg = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        axs_cg = axs_cg.flatten()

        for idx, edir in enumerate(energy_dirs):
            ekey = edir.name
            print(f"    Extracting {ekey}")
            res = extract_profile_data(edir, is_hex)
            if res is None: continue

            ax_t = axs_tof[idx]
            ax_d = axs_dist[idx]
            ax_c = axs_cg[idx]
            
            prof, s_t, pitch = res["tof_profile"], res["sigma_t_ps"], res["pitch_mm"]
            lyso_thick, calor_thick = res["lyso_thick"], res["calor_thick"]
            z_cg_dist = res["z_cg_dist"]

            sigma_z = V_EFF_MM_NS * (s_t / 1000.0)
            sigma_layer = sigma_z / pitch
            total_hits = np.sum(prof)
            norm_prof = prof / total_hits if total_hits > 0 else prof

            run_dirs = sorted(list(set(fp.parent for fp in edir.rglob("detector_hits_*.root"))))
            truth_curve, bounds = None, None

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

            # 1. TOF PROFILE PANELS
            if truth_curve is not None and np.sum(truth_curve) > 0:
                norm_truth = truth_curve / np.sum(truth_curve)
                ax_t.bar(layers, norm_truth, color="#00bcd4", alpha=0.35, edgecolor="#00838f",
                       linewidth=0.8, width=0.8, label="Sim Truth (DoseActor)")

            ax_t.errorbar(layers, norm_prof, xerr=sigma_layer, color=mod_colors[mod],
                        linewidth=2, marker="o", markersize=4, capsize=3, capthick=1.0,
                        label=f"ΔT Coincidence (N={res['n_e_coincidences']})")
            ax_t.set_title(f"{ekey}", fontsize=11, fontweight="bold")
            ax_t.set_xlabel("LYSO Layer Number", fontsize=9)
            ax_t.set_ylabel("Normalized Density Fraction", fontsize=9)
            ax_t.set_xlim(0, _N_LYSO + 1)
            ax_t.grid(True, linestyle=":", alpha=0.5)
            ax_t.legend(loc="upper right", fontsize=8)

            # 2. DOWNSTREAM PROMPT STRIKES PANELS
            first_times = res["dw_first_times"]
            if len(first_times) > 0:
                distances_mm = first_times * V_EFF_MM_NS
                ax_d.hist(distances_mm, bins=80, range=(120.0, 350.0), color="#ff9800", 
                          alpha=0.6, edgecolor="black", linewidth=0.5, label=f"Prompt Strikes")
                ax_d.set_title(f"Prompt Strikes vs Distance: {ekey}", fontsize=11, fontweight="bold")
                ax_d.set_xlabel(f"Kinematic Pseudo-Distance (mm)", fontsize=9)
            else:
                ax_d.text(0.5, 0.5, "No Data", ha='center', va='center')

            # 3. ASYMMETRY Z_CG PANELS
            if len(z_cg_dist) > 0:
                ax_c.hist(z_cg_dist, bins=50, range=(-calor_thick/2 - 10, calor_thick/2 + 10), 
                          color="#9c27b0", alpha=0.7, edgecolor="black", linewidth=0.8, 
                          label=f"Experimental Z_cg (N={len(z_cg_dist)})")
                
                cg_mean = np.mean(z_cg_dist)
                ax_c.axvline(cg_mean, color="black", linestyle="--", linewidth=1.5, label=f"Mean Z_cg: {cg_mean:.1f} mm")

                if truth_curve is not None and bounds is not None:
                    # Calculate Sim Truth Center of Gravity for context
                    layer_centers = [(z_start + z_end)/2.0 for (z_start, z_end) in bounds]
                    truth_cg = np.average(layer_centers, weights=truth_curve)
                    ax_c.axvline(truth_cg, color="#00bcd4", linestyle="-", linewidth=2.0, label=f"True Z_cg: {truth_cg:.1f} mm")

                ax_c.set_title(f"Signal Asymmetry Z_cg: {ekey}", fontsize=11, fontweight="bold")
                ax_c.set_xlabel("Reconstructed Depth Z (mm)", fontsize=9)
                ax_c.set_ylabel("Events", fontsize=9)
                ax_c.legend(loc="upper right", fontsize=8)
                ax_c.grid(True, linestyle=":", alpha=0.5)
            else:
                ax_c.text(0.5, 0.5, "No Asymmetry Data", ha='center', va='center')

            ax_g = axs_gamma[idx]
            
            # --- PARAMETRIC GAMMA RECONSTRUCTION ---
            t_up, t_dw, t_hits = res["total_e_up"], res["total_e_dw"], res["total_t_hits"]
            bounds = res["lyso_bounds"]
            
            if t_up > 0 and t_dw > 0:
                # 1. Shift coordinate system so Face = 0 mm
                z_cg_center = - (_LAMBDA_EFF_MM / 2.0) * np.log(t_up / t_dw)
                z_cg_face = z_cg_center + (calor_thick / 2.0)
                
                # 2. Extract hardware anchors
                q_tot_e = t_up + t_dw
                q_t_frac = t_hits / q_tot_e if q_tot_e > 0 else 0
                
                # Get the Z boundaries of the T-type region relative to the face
                z_t_start = bounds[_SHOWER_FIRST - 1][0] + (calor_thick / 2.0)
                z_t_end = bounds[_SHOWER_LAST - 1][1] + (calor_thick / 2.0)
                
                # 3. Fit
                a_opt, b_opt = fit_gamma_to_observables(z_cg_face, q_t_frac, z_t_start, z_t_end)
                
                # 4. Plot
                z_plot = np.linspace(0, calor_thick, 200)
                norm_E0 = np.sum(truth_curve) if truth_curve is not None else 1.0
                
                # Scale the continuous PDF by the layer thickness so it matches the binned bars
                recon_curve = gamma_profile(z_plot, norm_E0 * res["pitch_mm"], a_opt, b_opt)
                
                if truth_curve is not None:
                    truth_centers = [(zs + ze)/2.0 + (calor_thick/2.0) for (zs, ze) in bounds]
                    ax_g.bar(truth_centers, truth_curve, width=(bounds[0][1]-bounds[0][0]), 
                             color="#00bcd4", alpha=0.5, edgecolor="#00838f", label="MC Truth")
                
                ax_g.plot(z_plot, recon_curve, color="#d32f2f", linewidth=2.5, 
                          label=f"Reconstructed Fit\n$a={a_opt:.2f}, b={b_opt:.3f}$")
                ax_g.axvline(z_cg_face, color="black", linestyle="--", label=f"$Z_{{cg}} = {z_cg_face:.1f}$ mm")
                ax_g.axvspan(z_t_start, z_t_end, color="#ffeb3b", alpha=0.2, label="Hardware Depth Marker")
                
                ax_g.set_title(f"Parametric Profile: {ekey}", fontsize=11, fontweight="bold")
                ax_g.set_xlabel("Depth from Calorimeter Face (mm)", fontsize=9)
                ax_g.set_xlim(0, calor_thick)
                ax_g.legend(loc="upper right", fontsize=8)
                ax_g.grid(True, linestyle=":", alpha=0.5)


        for idx in range(n_energies, len(axs_tof)):
            fig_tof.delaxes(axs_tof[idx])
            fig_dist.delaxes(axs_dist[idx])
            fig_cg.delaxes(axs_cg[idx])

        fig_tof.suptitle(f"Continuous E-Type ToF Reconstructions — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_tof.tight_layout()
        fig_tof.savefig(analysis_out / f"{mod}_tof_panels.png", dpi=200)
        plt.close(fig_tof)

        fig_dist.suptitle(f"Downstream Prompt Strikes vs Distance (Cropped) — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_dist.tight_layout()
        fig_dist.savefig(analysis_out / f"{mod}_dw_prompt_distance_cropped.png", dpi=200)
        plt.close(fig_dist)

        fig_cg.suptitle(f"E-Type Signal Asymmetry Center of Gravity — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_cg.tight_layout()
        fig_cg.savefig(analysis_out / f"{mod}_asymmetry_cg.png", dpi=200)
        plt.close(fig_cg)

        fig_gamma.suptitle(f"Continuous Parametric Gamma Fit — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_gamma.tight_layout()
        fig_gamma.savefig(analysis_out / f"{mod}_parametric_reconstruction.png", dpi=200)
        plt.close(fig_gamma)

if __name__ == "__main__":
    main()