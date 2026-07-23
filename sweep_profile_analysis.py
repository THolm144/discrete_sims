"""
unified_profile_analysis.py
================================================================================
Advanced spatial, temporal, and prompt reconstruction analysis for RADiCal 
geometry variants, utilizing native Gate DoseActor (.mhd) truth extraction 
and natural timestamp/numerical directory sorting.
"""
import argparse
import pickle
import warnings
import re
from pathlib import Path
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL & GEOMETRICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS = 299.792

SIGMA_NS = 0.02

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
    "radi_cal_energy":        1.0,
    "radi_cal_triple":        1.0,
    "rc_hex":                 1.0,
    "rc_hex_triple":          1.0,
    "dsb1_radi_cal_energy":   1.0,
    "dsb1_radi_cal_triple":   1.0,
    "dsb1_rc_hex":            1.0,
    "dsb1_rc_hex_triple":     1.0,
    "luagce_radi_cal_energy": 1.0,
    "luagce_radi_cal_triple": 1.0,
    "luagce_rc_hex":          1.0,
    "luagce_rc_hex_triple":   1.0,
}

# Map the effective attenuation length (in mm) to each module type
# Map the validated simulated effective attenuation length (in mm) to each module type
EFFECTIVE_ATT_LENGTH = {
    "radi_cal_energy":        2428.38,   # BCF92 simulated waveguide lambda_eff
    "radi_cal_triple":        2428.38,
    "rc_hex":                 2428.38,
    "rc_hex_triple":          2428.38,
    "dsb1_radi_cal_energy":   2890.35,   # DSB1 simulated waveguide lambda_eff
    "dsb1_radi_cal_triple":   2890.35,
    "dsb1_rc_hex":            2890.35,
    "dsb1_rc_hex_triple":     2890.35,

    # LuAG:Ce: Use 140.0 mm if you updated your configuration to the real 200 mm bulk.
    # If you are still using the old 5000 mm bulk configurations, change this to 10200.26.
    "luagce_radi_cal_energy": 10200.26,     
    "luagce_radi_cal_triple": 10200.26,
    "luagce_rc_hex":          10200.26,
    "luagce_rc_hex_triple":   10200.26,
}

T_OFFSET_NS = {mod: 0.0 for mod in REFRACTIVE_INDEX.keys()}

_GT_LO_NS = 0.0
_GT_HI_NS = 50.0
_TYVEK_THICK_MM = 0.2032
_W_THICK_MM = 2.5
_N_LYSO = 29
_N_W = 28

ARRIVAL_QUANTILE = 0.10

_KNOWN_MODULE_LYSO_THICK = {mod: (4.5 if "triple" in mod else 1.5) for mod in REFRACTIVE_INDEX.keys()}

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
# DIRECTORY & STAMP SORTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def get_natural_sort_key(path: Path):
    """
    Extracts all numeric sequences from a file or folder path.
    Ensures sweep_12 sorts after sweep_2, and timestamped folders sort chronologically.
    """
    numbers = [int(s) for s in re.findall(r'\d+', path.name)]
    return numbers if numbers else [path.stat().st_mtime]

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

# ─────────────────────────────────────────────────────────────────────────────
# DOSEACTOR MHD/RAW PARSER ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def load_mhd_z_profile(mhd_path: Path):
    """
    Parses a Gate DoseActor .mhd header and loads its binary .raw counterpart,
    projecting the 3D grid into a 1D longitudinal Z-profile.
    """
    if not mhd_path.exists():
        return None

    meta = {}
    try:
        with open(mhd_path, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=", 1)
                    meta[k.strip()] = v.strip()

        raw_file = meta.get("ElementDataFile")
        if not raw_file:
            return None

        raw_path = mhd_path.parent / raw_file
        if not raw_path.exists():
            return None

        dim_size = [int(x) for x in meta.get("DimSize", "1 1 1").split()]
        dtype = np.float32 if meta.get("ElementType") == "MET_FLOAT" else np.float64

        # Load the raw binary matrix
        data = np.fromfile(raw_path, dtype=dtype)

        # Squeeze/reshape array depending on its dimensions
        if len(dim_size) == 3:
            # Gate 3D DoseActor matrices are saved in C-contiguous format: (Z, Y, X)
            data = data.reshape((dim_size[2], dim_size[1], dim_size[0]))
            # Project/sum over lateral axes (X and Y) to extract the longitudinal profile
            z_profile = np.sum(data, axis=(1, 2))
            return z_profile
        elif len(dim_size) == 1:
            return data
        else:
            return data
    except Exception:
        return None

def rebin_fine_profile_to_layers(fine_profile, lyso_bounds, calor_thick_mm):
    """
    Maps and aggregates a fine-grained longitudinal Z-profile to the 29 physical LYSO layers.
    """
    n_bins = len(fine_profile)
    # Gate centers DoseActor coordinate grid symmetrically around Z = 0
    z_edges = np.linspace(-calor_thick_mm / 2.0, calor_thick_mm / 2.0, n_bins + 1)
    z_mids = 0.5 * (z_edges[:-1] + z_edges[1:])

    layer_profile = np.zeros(len(lyso_bounds))
    for idx, (z_lo, z_hi) in enumerate(lyso_bounds):
        mask = (z_mids >= z_lo) & (z_mids <= z_hi)
        layer_profile[idx] = np.sum(fine_profile[mask])

    return layer_profile

# ─────────────────────────────────────────────────────────────────────────────
# COINCIDENCE FIT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def standard_gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian_to_peak(data, n_bins=40):
    if len(data) < 8:
        return 0.0, float(np.median(data)) if len(data) else 0.0, float(np.std(data)) if len(data) else 0.0
    spread = max(np.std(data), 1.0)
    lo, hi = float(np.min(data)), float(np.max(data))
    if hi <= lo: hi = lo + 1.0

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
    if len(arr) == 0: return arr
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
# CORE ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def _chunk_series(mask, values, ev, run_tag):
    n = int(mask.sum())
    if n == 0: return None
    idx = pd.MultiIndex.from_arrays([np.full(n, run_tag, dtype=object), ev[mask].astype(np.int64)])
    return pd.Series(values[mask], index=idx)

def _grouped(chunks, how):
    if not chunks: return {}
    s = pd.concat(chunks)
    g = s.groupby(level=[0, 1])
    if how == "min":
        s = g.min()
    elif how == "count":
        s = g.count()
    else:
        s = g.quantile(how)
    return {(k[0], int(k[1])): (int(v) if how == "count" else float(v)) for k, v in s.items()}

def get_bar_colors(ekey, idx):
    # Cohesive color families (Darker for target hits, lighter for bounced)
    energy_colors = {
        "25GeV":  {"target": "#1f77b4", "bounced": "#aec7e8"}, # Blue
        "50GeV":  {"target": "#ff7f0e", "bounced": "#ffbb78"}, # Orange
        "100GeV": {"target": "#2ca02c", "bounced": "#98df8a"}  # Green
    }
    if ekey in energy_colors:
        return energy_colors[ekey]["target"], energy_colors[ekey]["bounced"]
    
    # Fallback palette builder
    import matplotlib.colors as mcolors
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    base_col = base_colors[idx % len(base_colors)]
    rgb = mcolors.to_rgb(base_col)
    light_col = tuple(0.4 * c + 0.6 for c in rgb) # blend with white
    return base_col, light_col

def get_layer_idx_from_z(z_vals, lyso_bounds):
    """Map absolute z coordinates (mm) to layer indices (0..29). Returns -1 for out-of-bounds hits."""
    layer_idx = np.full(len(z_vals), -1, dtype=int)
    for i, (z_lo, z_hi) in enumerate(lyso_bounds):
        in_layer = (z_vals >= (z_lo - 0.5)) & (z_vals <= (z_hi + 0.5))
        layer_idx[in_layer] = i
    return layer_idx


def analyze_profile_batch(batch_dir: Path, is_hex: bool, module_name: str, verbose_label: str = ""):
    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"), key=get_natural_sort_key)
    if not hit_files:
        return None

    # Detect sensor z boundaries
    detected_z_min, detected_z_max = None, None
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                z_arr = f[tk]["Position_Z"].array(library="np")
                if len(z_arr) > 0:
                    detected_z_min = float(np.min(z_arr))
                    detected_z_max = float(np.max(z_arr))
                    break
        except Exception:
            continue

    if detected_z_min is None or detected_z_max is None:
        return None

    # 1. Module Geometry & Bounds
    lyso_thick = _KNOWN_MODULE_LYSO_THICK[module_name]
    v_eff = v_eff_for_module(module_name)  # mm/ns (~140 - 160 mm/ns)

    gap_thick_mm = lyso_thick + 2 * _TYVEK_THICK_MM
    calor_thick_mm = (_N_LYSO * gap_thick_mm) + (_N_W * _W_THICK_MM)
    lyso_bounds = get_lyso_layer_bounds(lyso_thick, calor_thick_mm)

    z_min_calor = lyso_bounds[0][0]
    z_max_calor = lyso_bounds[-1][1]

    cap_xy_map = HEX_CAP_XY if is_hex else SQUARE_CAP_XY
    t_indices = list({1, 3, 5} if is_hex else {0, 1})
    e_indices = list({0, 2, 4} if is_hex else {2, 3})

    run_dirs = set(fpath.parent for fpath in hit_files)

    # 2. Extract DoseActor Truth Data (.mhd/.raw files)
    truth_profiles = []
    for rdir in run_dirs:
        mhd_files = list(rdir.glob("run_Dose_edep.mhd")) or list(rdir.glob("*Dose_edep.mhd"))
        if mhd_files:
            fine_profile = load_mhd_z_profile(mhd_files[0])
            if fine_profile is not None:
                rebinned = rebin_fine_profile_to_layers(fine_profile, lyso_bounds, calor_thick_mm)
                truth_profiles.append(rebinned)

    # 3. Accumulation Arrays for 4-Panel Subplots
    profile_up_recon = np.zeros(_N_LYSO)
    profile_dw_recon = np.zeros(_N_LYSO)
    total_events_processed = 0

    # 4. Conversion Constants for ToF Equations
    denom_up = (1.0 / C_LIGHT_MM_NS) + (1.0 / v_eff)
    denom_dw = (1.0 / v_eff) - (1.0 / C_LIGHT_MM_NS)
    t_max_dw = calor_thick_mm / v_eff
    lambda_eff = 120.0  # mm (Optical attenuation length)

    # 5. Process Hits File-by-File
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                tree = f[tk]
                if tree.num_entries == 0: continue
                arrs = tree.arrays(["Position_X", "Position_Y", "Position_Z", "GlobalTime", "LocalTime", "EventID", "ParticleName"], library="np")
        except Exception:
            continue

        x, y, z = arrs["Position_X"], arrs["Position_Y"], arrs["Position_Z"]
        gt_raw, ev, pn = arrs["GlobalTime"], arrs["EventID"], arrs["ParticleName"]

        unique_evs = np.unique(ev)
        total_events_processed += len(unique_evs)

        # Channel Mapping
        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(np.hypot(dx, dy), axis=1)

        near_up = np.abs(z - detected_z_min) < 5.0
        near_dw = np.abs(z - detected_z_max) < 5.0

        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")
        is_readout_channel = np.isin(channels, t_indices + e_indices)

        m_up = near_up & is_optical & is_readout_channel
        m_dw = near_dw & is_optical & is_readout_channel

        # --- A. UPSTREAM SINGLE-ENDED READOUT ---
        if np.any(m_up):
            gt_up = gt_raw[m_up]
            ev_up = ev[m_up]

            # Compute t0 per event (earliest hit arrival time)
            t0_map_up = pd.DataFrame({'ev': ev_up, 'gt': gt_up}).groupby('ev')['gt'].min().to_dict()
            t0_up = np.array([t0_map_up[e] for e in ev_up])

            dt_up = gt_up - t0_up
            z_local_up = dt_up / denom_up
            z_abs_up = z_min_calor + z_local_up

            layers_up = get_layer_idx_from_z(z_abs_up, lyso_bounds)
            valid_up = (layers_up != -1) & (z_local_up >= 0) & (z_local_up <= calor_thick_mm)

            if np.any(valid_up):
                v_layers = layers_up[valid_up]
                v_z_local = z_local_up[valid_up]
                weights_up = np.exp(v_z_local / lambda_eff)  # Inverse LCE correction
                np.add.at(profile_up_recon, v_layers, weights_up)

        # --- B. DOWNSTREAM SINGLE-ENDED READOUT ---
        if np.any(m_dw):
            gt_dw = gt_raw[m_dw]
            ev_dw = ev[m_dw]

            # Compute t0 per event
            t0_map_dw = pd.DataFrame({'ev': ev_dw, 'gt': gt_dw}).groupby('ev')['gt'].min().to_dict()
            t0_dw = np.array([t0_map_dw[e] for e in ev_dw])

            dt_dw = gt_dw - t0_dw
            z_local_dw = (t_max_dw - dt_dw) / denom_dw
            z_abs_dw = z_min_calor + z_local_dw

            layers_dw = get_layer_idx_from_z(z_abs_dw, lyso_bounds)
            valid_dw = (layers_dw != -1) & (z_local_dw >= 0) & (z_local_dw <= calor_thick_mm)

            if np.any(valid_dw):
                v_layers = layers_dw[valid_dw]
                v_z_local = z_local_dw[valid_dw]
                weights_dw = np.exp((calor_thick_mm - v_z_local) / lambda_eff)  # Inverse LCE correction
                np.add.at(profile_dw_recon, v_layers, weights_dw)

    # 6. Normalize per Event
    events_denom = max(1, total_events_processed)
    norm_up = profile_up_recon / events_denom
    norm_dw = profile_dw_recon / events_denom
    combined_profile = norm_up + norm_dw

    mean_truth_profile = (
        np.mean(truth_profiles, axis=0) / max(1, total_events_processed / len(run_dirs))
        if truth_profiles else np.zeros(_N_LYSO)
    )

    if verbose_label:
        print(f"    [{verbose_label}] Processed {total_events_processed} events across {len(run_dirs)} runs.")

    return {
        "profile_up_recon": norm_up,              # Top-Left Subplot
        "profile_dw_recon": norm_dw,              # Top-Right Subplot
        "combined_profile": combined_profile,      # Bottom-Left Subplot
        "truth_layer_profile": mean_truth_profile, # Bottom-Right Subplot
        "total_events": total_events_processed,
        "run_dirs": sorted(run_dirs),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).resolve().parent
    modules = [
        #"radi_cal_energy", "radi_cal_triple", "rc_hex", "rc_hex_triple",
        "dsb1_radi_cal_energy", "dsb1_radi_cal_triple", "dsb1_rc_hex", "dsb1_rc_hex_triple",
        "luagce_radi_cal_energy", "luagce_radi_cal_triple", "luagce_rc_hex", "luagce_rc_hex_triple"
    ]

    out_dir = base_dir / "profile_analysis"
    energy_dir = out_dir / "energy_performance"
    global_dir = out_dir / "globaltime"
    local_dir = out_dir / "localtime"
    prompt_dir = out_dir / "prompt_photon_reconstruction"
    two_end_dir = out_dir / "two_end_timing"

    for d in [energy_dir, global_dir, local_dir, prompt_dir, two_end_dir]:
        d.mkdir(parents=True, exist_ok=True)

    master_summary = {mod: {} for mod in modules}

    print("Master profile processing engine spawned...")
    for mod in modules:
        mod_path = base_dir / mod / "runs" / mod
        if not mod_path.exists():
            mod_path = base_dir / mod 
            if not mod_path.exists(): continue

        # UPGRADE: Natural numeric sorting guarantees sweep_12 is identified as the latest directory
        sweeps = sorted(mod_path.glob("sweep_*"), key=get_natural_sort_key)
        if not sweeps: continue
        target_sweep = sweeps[-1]
        print(f"Analyzing Sweep -> {mod}/{target_sweep.name}")

        is_hex = "hex" in mod
        energy_dirs = sorted(target_sweep.glob("*GeV"), key=get_natural_sort_key)
        for edir in energy_dirs:
            res = analyze_profile_batch(edir, is_hex, mod, verbose_label=f"{mod}:{edir.name}")
            if res is not None:
                master_summary[mod][edir.name] = res

    # Visual settings
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
    # GRAPH GENERATION PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    for mod in modules:
        if mod not in master_summary or not master_summary[mod]: continue
        energy_keys = sorted(master_summary[mod].keys(), key=extract_numerical_energy)
        if not energy_keys: continue

        # ── GRAPH 1: ENERGY PERFORMANCE (Using True DoseActor Energy) ─────────
        energies_gev, mu_e_list, res_e_list, mu_e_err, res_e_err = [], [], [], [], []

        for ekey in energy_keys:
            E_val = extract_numerical_energy(ekey)
            if E_val <= 0: continue

            active_edeps = master_summary[mod][ekey].get("active_edep_total", np.array([]))
            if len(active_edeps) < 2: continue  # MHD statistics might be lower (one per run)

            # Standard fitting for dose totals
            _, mu_val, sigma_val = fit_gaussian_to_peak(active_edeps, n_bins=10)

            if mu_val > 0:
                energies_gev.append(E_val)
                mu_e_list.append(mu_val)
                res_e_list.append(sigma_val / mu_val)
                mu_e_err.append(sigma_val / np.sqrt(len(active_edeps)))
                res_e_err.append((sigma_val / mu_val) * (1.0 / np.sqrt(len(active_edeps))))

        if len(energies_gev) >= 3:
            energies_gev = np.array(energies_gev)
            mu_e_list = np.array(mu_e_list)
            res_e_list = np.array(res_e_list)

            fig_er, (ax_lin, ax_res) = plt.subplots(1, 2, figsize=(14, 5))

            def linear_func(x, m, b): return m * x + b
            popt_lin, _ = curve_fit(linear_func, energies_gev, mu_e_list)

            ax_lin.errorbar(energies_gev, mu_e_list, yerr=mu_e_err, fmt=mod_markers.get(mod, 'o'),
                            color=mod_colors.get(mod, 'black'), label=f"DoseActor ({mod})")
            x_lin_smooth = np.linspace(0, max(energies_gev) * 1.1, 100)
            ax_lin.plot(x_lin_smooth, linear_func(x_lin_smooth, *popt_lin),
                        color="black", linestyle="--", label=f"Fit: {popt_lin[0]:.3e} MeV/GeV")

            ax_lin.set_xlabel("Beam Energy (GeV)", fontweight="bold")
            ax_lin.set_ylabel("Integrated Dose Energy (MeV)", fontweight="bold")
            ax_lin.set_title("Calorimeter Energy Linearity", fontsize=11, fontweight="bold")
            ax_lin.grid(True, linestyle=":", alpha=0.6)
            ax_lin.legend(fontsize=9)

            def resolution_func(E, c, s, n):
                return np.sqrt(c ** 2 + (s / np.sqrt(E)) ** 2 + (n / E) ** 2)

            try:
                popt_res, _ = curve_fit(resolution_func, energies_gev, res_e_list,
                                        p0=[0.02, 0.15, 0.01], bounds=(0, [2.0, 10.0, 10.0]))
                c_f, s_f, n_f = popt_res
                fit_label = f"Fit: {c_f * 100:.2f}% $\\oplus$ {s_f * 100:.2f}%/$\\sqrt{{E}}$ $\\oplus$ {n_f * 100:.2f}%/E"
            except Exception:
                popt_res = [0.0, 0.0, 0.0]
                fit_label = "Fit failed"

            ax_res.errorbar(energies_gev, res_e_list, yerr=res_e_err, fmt=mod_markers.get(mod, 'o'),
                            color=mod_colors.get(mod, 'black'), label="Resolution")
            x_res_smooth = np.linspace(min(energies_gev) * 0.8, max(energies_gev) * 1.1, 100)
            ax_res.plot(x_res_smooth, resolution_func(x_res_smooth, *popt_res),
                        color="black", linestyle="--", label=fit_label)

            ax_res.set_xlabel("Beam Energy (GeV)", fontweight="bold")
            ax_res.set_ylabel(r"$\sigma_E / E_{meas}$", fontweight="bold")
            ax_res.set_title("Calorimeter Energy Resolution", fontsize=11, fontweight="bold")
            ax_res.grid(True, linestyle=":", alpha=0.6)
            ax_res.legend(fontsize=9)

            fig_er.suptitle(f"Active Gate DoseActor Calorimetry — {mod}", fontsize=12, fontweight="bold")
            fig_er.tight_layout()
            fig_er.savefig(energy_dir / f"{mod}_energy_performance.png", dpi=200)
            plt.close(fig_er)

        # ── GRAPH 2: GLOBAL TIME VS STRIP STRIKES ─────────────────────────────
        fig_gt, ax_gt = plt.subplots(figsize=(8, 5))
        for ekey in energy_keys:
            counts = master_summary[mod][ekey]["gt_counts"]
            bins = master_summary[mod][ekey]["gt_bins"]
            ax_gt.plot(0.5 * (bins[:-1] + bins[1:]), counts, label=ekey, alpha=0.8, linewidth=1.5)

        ax_gt.set_xlabel("GlobalTime (ns)", fontweight="bold")
        ax_gt.set_ylabel("Optical Photon Strikes (Downstream)", fontweight="bold")
        ax_gt.set_title(f"Downstream GlobalTime Spectrum — {mod}", fontsize=11, fontweight="bold")
        ax_gt.set_yscale("linear")
        ax_gt.grid(True, linestyle=":", alpha=0.5)
        ax_gt.legend(title="Beam Energy")
        fig_gt.tight_layout()
        fig_gt.savefig(global_dir / f"{mod}_globaltime.png", dpi=200)
        plt.close(fig_gt)

       # ── GRAPH 3: LOCAL TIME (Pure Optical Travel Time) ───────────────────
        fig_lt, ax_lt = plt.subplots(figsize=(8, 5))

        for idx, ekey in enumerate(energy_keys):
            lt_counts = master_summary[mod][ekey]["lt_counts"]
            lt_bins = master_summary[mod][ekey]["lt_bins"]
            
            # Convert bin edges to bin centers
            bin_centers = 0.5 * (lt_bins[:-1] + lt_bins[1:])

            # 1. Use scipy.signal.find_peaks to isolate prominent local peaks
            # We filter for peaks with a prominence of at least 10% of the maximum height
            peaks, _ = find_peaks(lt_counts, prominence=np.max(lt_counts) * 0.1)
            
            if len(peaks) == 0:
                # Fallback if the peak is incredibly sharp and misses prominence criteria
                primary_peak_idx = np.argmax(lt_counts)
            else:
                # Sort found peaks by count height and grab the absolute largest (the true wavefront)
                primary_peak_idx = peaks[np.argsort(lt_counts[peaks])[-1]]

            peak_time = bin_centers[primary_peak_idx]
            peak_intensity = lt_counts[primary_peak_idx]

            # 2. Plot the main distribution curve (now linear!)
            line, = ax_lt.plot(bin_centers, lt_counts, label=f"{ekey} (Peak: {peak_time:.3f} ns)", alpha=0.85)
            color = line.get_color()

            # 3. Mark the peak point with a distinct star
            ax_lt.scatter(peak_time, peak_intensity, marker="*", color=color, s=120, edgecolor="black", zorder=5)

            # 4. Annotate the peak value with alternating offsets to prevent overlapping text
            x_offset = 20 if idx % 2 == 0 else -95
            y_offset = 15 if idx % 2 == 0 else -25

            ax_lt.annotate(
                f"{ekey}: {peak_time:.3f} ns",
                xy=(peak_time, peak_intensity),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8, connectionstyle="arc3,rad=0.1"),
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8)
            )

        ax_lt.set_xlabel("Local Arrival Time (ns)", fontweight="bold")
        ax_lt.set_ylabel("Photon Strikes", fontweight="bold")
        ax_lt.set_title(f"Local Arrival Time Distribution (Linear) — {mod}", fontsize=11, fontweight="bold")
        ax_lt.set_yscale("linear") # Explicitly linear!
        ax_lt.grid(True, linestyle=":", alpha=0.5)
        ax_lt.legend(title="Beam Components", loc="upper right")

        fig_lt.tight_layout()
        fig_lt.savefig(prompt_dir / f"{mod}_localtime_spectra.png", dpi=200)
        plt.close(fig_lt)

        # ── GRAPH 4: PROMPT PHOTON LONGITUDINAL RECONSTRUCTION ────────────────
        # Using a 2x2 grid to prevent the subplots from becoming squashed horizontally
        fig_rec, axs = plt.subplots(2, 2, figsize=(15, 10.5))
        ax_target = axs[0, 0]
        ax_bounced = axs[0, 1]
        ax_both = axs[1, 0]
        ax_truth = axs[1, 1]

        # Create secondary y-axes to cleanly overlay truth energy dots onto photon count plots
        ax_target_twin = ax_target.twinx()
        ax_bounced_twin = ax_bounced.twinx()
        ax_both_twin = ax_both.twinx()

        layers_x = np.arange(1, _N_LYSO + 1)

        # Set up spacing parameters to stack and group the bars side-by-side
        n_energies = len(energy_keys)
        total_width = 0.8
        width = total_width / max(1, n_energies)

        for idx, ekey in enumerate(energy_keys):
            # Calculate grouped horizontal shift
            offset = (idx - (n_energies - 1) / 2.0) * width
            x_coords = layers_x + offset

            # Fetch the profiles
            target_profile = master_summary[mod][ekey].get("prompt_profile_target", np.zeros(_N_LYSO))
            bounced_profile = master_summary[mod][ekey].get("prompt_profile_bounced", np.zeros(_N_LYSO))
            truth_prof = master_summary[mod][ekey]["truth_layer_profile"]
            
            col_target, col_bounced = get_bar_colors(ekey, idx)

            # Panel 1: Target-Only Photons (Reconstructed) + Truth Overlay
            ax_target.bar(x_coords, target_profile, width=width, color=col_target,
                          edgecolor="black", linewidth=0.3, alpha=0.9, label=ekey)
            ax_target_twin.plot(layers_x, truth_prof, marker="o", linestyle="None", 
                                color=col_target, markersize=5, alpha=0.75, 
                                markeredgecolor="black", markeredgewidth=0.5)

            # Panel 2: Bounced-Only Photons (Reconstructed) + Truth Overlay
            ax_bounced.bar(x_coords, bounced_profile, width=width, color=col_bounced,
                           edgecolor="black", linewidth=0.3, alpha=0.9, label=ekey)
            ax_bounced_twin.plot(layers_x, truth_prof, marker="o", linestyle="None", 
                                 color=col_target, markersize=5, alpha=0.75, 
                                 markeredgecolor="black", markeredgewidth=0.5)

            # Panel 3: Combined Profile (Stacked Reconstructed) + Truth Overlay
            ax_both.bar(x_coords, target_profile, width=width, color=col_target, 
                        edgecolor="black", linewidth=0.3, alpha=0.9, label=f"{ekey} (Target)")
            ax_both.bar(x_coords, bounced_profile, width=width, bottom=target_profile, 
                        color=col_bounced, edgecolor="black", linewidth=0.3, alpha=0.6, label=f"{ekey} (Bounced)")
            ax_both_twin.plot(layers_x, truth_prof, marker="o", linestyle="None", 
                              color=col_target, markersize=5, alpha=0.75, 
                              markeredgecolor="black", markeredgewidth=0.5)

            # Panel 4: Simulated Truth (MHD Dose)
            ax_truth.plot(layers_x, truth_prof, marker="s", markersize=4, label=ekey, alpha=0.8)

        # Apply standard formatting and axis labels
        for ax in [ax_target, ax_bounced, ax_both]:
            ax.set_xlabel("LYSO Layer Number", fontweight="bold")
            ax.set_xlim(0, _N_LYSO + 1)
            ax.grid(True, linestyle=":", alpha=0.5)

        # Label the secondary twin axes cleanly to denote the truth dots
        for twin_ax in [ax_target_twin, ax_bounced_twin, ax_both_twin]:
            twin_ax.set_ylabel("Truth Energy Deposition [Dots] (MeV)", color="dimgray", fontsize=9)
            twin_ax.tick_params(axis='y', labelcolor="dimgray")

        # Panel 1 titles & legends
        ax_target.set_ylabel("Target Prompt Photon Strikes", fontweight="bold")
        ax_target.set_title("Reconstructed: Target-Only", fontsize=11, fontweight="bold")
        ax_target.legend(title="Beam Energy", fontsize=8, loc="upper left")

        # Panel 2 titles & legends
        ax_bounced.set_ylabel("Bounced Prompt Photon Strikes", fontweight="bold")
        ax_bounced.set_title("Reconstructed: Bounced-Only", fontsize=11, fontweight="bold")
        ax_bounced.legend(title="Beam Energy", fontsize=8, loc="upper left")

        # Panel 3 titles & legends
        ax_both.set_ylabel("Total Prompt Photon Strikes", fontweight="bold")
        ax_both.set_title("Reconstructed: Combined Profile", fontsize=11, fontweight="bold")
        handles, labels = ax_both.get_legend_handles_labels()
        ax_both.legend(handles, labels, title="Beam Components", fontsize=8, loc="upper left")

        # Panel 4 titles & legends
        ax_truth.set_xlabel("LYSO Layer Number", fontweight="bold")
        ax_truth.set_ylabel("Mean Active Energy Deposited (MeV / Event)", fontweight="bold")
        ax_truth.set_title("Simulated Truth Shower Profile (DoseActor MHD)", fontsize=11, fontweight="bold")
        ax_truth.set_xlim(0, _N_LYSO + 1)
        ax_truth.grid(True, linestyle=":", alpha=0.5)
        ax_truth.legend(title="Beam Energy", fontsize=8, loc="upper left")

        fig_rec.suptitle(f"Longitudinal Shower Profiles — {mod}", fontsize=13, fontweight="bold")
        fig_rec.tight_layout()
        fig_rec.savefig(prompt_dir / f"{mod}_prompt_reconstruction_vs_truth.png", dpi=200)
        plt.close(fig_rec)

        # ── GRAPH 5: TWO-ENDED FIBER TIMING ───────────────────────────────────
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
                clean_t = clean_t - np.median(clean_t) # Zero alignment
                _, mu_f, sigma_f = fit_gaussian_to_peak(clean_t)

                t_res_x.append(extract_numerical_energy(ekey))
                t_res_y.append(sigma_f)
                t_res_yerr.append(sigma_f / np.sqrt(2 * n_ev))

                lo, hi = -250.0, 250.0
                counts, edges, _ = ax.hist(clean_t, bins=50, range=(lo, hi),
                                           color=mod_colors.get(mod, "#f708af"), alpha=0.6, edgecolor="black")

                bin_mids = 0.5 * (edges[:-1] + edges[1:])
                x_fit = np.linspace(lo, hi, 200)
                y_fit = counts.max() * np.exp(-0.5 * ((x_fit - mu_f) / sigma_f) ** 2)

                ax.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=1.8,
                        label=f"Gaussian Fit\n$\\sigma_{{coinc}}$ = {sigma_f:.1f} ps")
                ax.set_title(f"Coincidence Spectrum — {ekey}", fontsize=10, fontweight="bold")
                ax.set_xlabel(r"$(t_{up} + t_{down})/2 - \mathrm{offset}$ (ps)", fontsize=9)
                ax.set_xlim(lo, hi)
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, linestyle=":", alpha=0.5)

        for idx in range(plotted_count, len(axs_two)):
            fig_two.delaxes(axs_two[idx])

        if plotted_count > 0:
            fig_two.suptitle(f"Double-Ended Coincidence Spectra — {mod}", fontsize=11, fontweight="bold", y=0.98)
            fig_two.tight_layout()
            fig_two.savefig(two_end_dir / f"{mod}_two_end_distributions.png", dpi=200)
        plt.close(fig_two)

        # Plot Coincidence Jitter Curve
        if len(t_res_x) >= 2:
            fig_tcurve, ax_tcurve = plt.subplots(figsize=(7, 5))
            ax_tcurve.errorbar(t_res_x, t_res_y, yerr=t_res_yerr, fmt="o-",
                               color=mod_colors.get(mod, "black"), marker=mod_markers.get(mod, "o"),
                               linewidth=2, markersize=6, capsize=4, label=f"Coincidence Resolution ({mod})")
            ax_tcurve.set_xlabel("Beam Energy (GeV)", fontweight="bold")
            ax_tcurve.set_ylabel(r"Timing Coincidence Resolution $\sigma_{coinc}$ (ps)", fontweight="bold")
            ax_tcurve.set_title(f"Two-Ended Fiber Coincidence Resolution — {mod}", fontsize=11, fontweight="bold")
            ax_tcurve.set_xscale("log")
            ax_tcurve.set_xticks([25, 50, 100, 200])
            ax_tcurve.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax_tcurve.grid(True, linestyle=":", alpha=0.6)
            ax_tcurve.legend()
            fig_tcurve.tight_layout()
            fig_tcurve.savefig(two_end_dir / f"{mod}_two_end_resolution_vs_energy.png", dpi=200)
            plt.close(fig_tcurve)

    print(f"\nProcessing complete! Reports saved directly inside: {out_dir.resolve()}")

if __name__ == "__main__":
    main()