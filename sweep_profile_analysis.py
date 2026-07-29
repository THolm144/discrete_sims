"""
unified_profile_analysis.py
================================================================================
Advanced spatial, temporal, and prompt reconstruction analysis for RADiCal 
geometry variants, utilizing native Gate DoseActor (.mhd) truth extraction 
and natural timestamp/numerical directory sorting.

Optimized: Vectorized Z-layer mapping, integer PDG comparisons, and deferred Pandas grouping.
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
# CORE ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def extract_numerical_energy(label: str) -> float:
    try:
        return float(''.join(c for c in label if c.isdigit() or c == '.'))
    except ValueError:
        return 0.0

def _chunk_raw(mask, values, ev, run_tag):
    """Deferred grouping: Gather numpy arrays instead of building Pandas objects in the loop."""
    if not np.any(mask):
        return None
    n = int(mask.sum())
    return (np.full(n, run_tag, dtype=object), ev[mask].astype(np.int64), values[mask])

def _grouped_raw(chunks, how):
    """Concatenate raw chunks and build the Pandas grouping exactly once."""
    if not chunks: return {}
    
    run_tags = np.concatenate([c[0] for c in chunks])
    evs = np.concatenate([c[1] for c in chunks])
    vals = np.concatenate([c[2] for c in chunks])
    
    idx = pd.MultiIndex.from_arrays([run_tags, evs])
    s = pd.Series(vals, index=idx)
    g = s.groupby(level=[0, 1])
    
    if how == "min":
        res = g.min()
    elif how == "count":
        res = g.count()
    else:
        res = g.quantile(how)
        
    return {(k[0], int(k[1])): (int(v) if how == "count" else float(v)) for k, v in res.items()}


def get_bar_colors(ekey, idx):
    # Cohesive color families (Darker for target hits, lighter for bounced)
    energy_colors = {
        "10GeV":  {"target": "#f30808", "bounced": "#f08370"}, 
        "25GeV":  {"target": "#e66814", "bounced": "#d69f60"}, 
        "30GeV":  {"target": "#f3e306", "bounced": "#d1bb5a"}, 
        "50GeV":  {"target": "#36ff0e", "bounced": "#385735"}, 
        "70GeV":  {"target": "#1f8ab4", "bounced": "#8cd8d2"}, 
        "90GeV":  {"target": "#0f12db", "bounced": "#917ab6"}, 
        "100GeV": {"target": "#7f14e2", "bounced": "#696070"}, 
        "150GeV": {"target": "#db12ba", "bounced": "#d38fe7"} 
    }
    if ekey in energy_colors:
        return energy_colors[ekey]["target"], energy_colors[ekey]["bounced"]

    import matplotlib.colors as mcolors
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    base_col = base_colors[idx % len(base_colors)]
    rgb = mcolors.to_rgb(base_col)
    light_col = tuple(0.4 * c + 0.6 for c in rgb) 
    return base_col, light_col

def get_layer_idx_from_z(z_vals, lyso_bounds):
    """Vectorized layer mapping using binary search. Bypasses Python for-loop entirely."""
    if len(z_vals) == 0: 
        return np.array([], dtype=int)
        
    # Flatten into 1D array: [lo0-0.5, hi0+0.5, lo1-0.5, hi1+0.5, ...]
    flat_bounds = np.array([[lo - 0.5, hi + 0.5] for lo, hi in lyso_bounds]).flatten()
    
    idx = np.searchsorted(flat_bounds, z_vals)
    layer_idx = np.full(len(z_vals), -1, dtype=int)
    
    # Hits falling inside a layer bound generate an odd index
    in_layer_mask = (idx % 2 == 1)
    layer_idx[in_layer_mask] = idx[in_layer_mask] // 2
    
    return layer_idx

def analyze_profile_batch(batch_dir: Path, is_hex: bool, module_name: str, verbose_label: str = ""):
    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"), key=get_natural_sort_key)
    if not hit_files:
        return None

    # --- Detect active SiPM position ---
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

    # --- Geometry & bounds ---
    lyso_thick = _KNOWN_MODULE_LYSO_THICK[module_name]
    v_eff = v_eff_for_module(module_name)

    gap_thick_mm = lyso_thick + 2 * _TYVEK_THICK_MM
    calor_thick_mm = (_N_LYSO * gap_thick_mm) + (_N_W * _W_THICK_MM)
    lyso_bounds = get_lyso_layer_bounds(lyso_thick, calor_thick_mm)
    
    cap_xy_map = HEX_CAP_XY if is_hex else SQUARE_CAP_XY
    t_indices = list({1, 3, 5} if is_hex else {0, 1})
    e_indices = list({0, 2, 4} if is_hex else {2, 3})

    up_q_chunks, dw_q_chunks = [], []
    run_dirs = set(fpath.parent for fpath in hit_files)

    # --- DoseActor truth profiles (.mhd/.raw) ---
    truth_profiles = []
    for rdir in run_dirs:
        mhd_files = list(rdir.glob("run_Dose_edep.mhd")) or list(rdir.glob("*Dose_edep.mhd"))
        if mhd_files:
            fine_profile = load_mhd_z_profile(mhd_files[0])
            if fine_profile is not None:
                truth_profiles.append(rebin_fine_profile_to_layers(fine_profile, lyso_bounds, calor_thick_mm))

    # --- Timing histograms ---
    gt_bins = np.linspace(0.0, 100.0, 501)
    lt_bins = np.linspace(0.0, 25.0, 501)
    gt_counts = np.zeros(500)
    lt_counts = np.zeros(500)

    # --- Accumulators ---
    prompt_counts_up = np.zeros(_N_LYSO)
    prompt_counts_dw = np.zeros(_N_LYSO)
    total_events_processed = 0

    for fpath in hit_files:
        run_tag = fpath.parent.name

        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                tree = f[tk]
                if tree.num_entries == 0: continue
                
                branch_keys = set(tree.keys())
                
                # OPTIMIZATION: Check for PDGEncoding to do blazing-fast integer comparisons
                has_pdg = "PDGEncoding" in branch_keys
                branches_to_load = ["Position_X", "Position_Y", "Position_Z", "GlobalTime", "LocalTime", "EventID"]
                branches_to_load.append("PDGEncoding" if has_pdg else "ParticleName")
                
                arrs = tree.arrays(branches_to_load, library="np")
        except Exception:
            continue

        x, y, z = arrs["Position_X"], arrs["Position_Y"], arrs["Position_Z"]
        gt_raw, lt, ev = arrs["GlobalTime"], arrs["LocalTime"], arrs["EventID"]

        total_events_processed += len(np.unique(ev))

        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(dx**2 + dy**2, axis=1)  # Dropped expensive sqrt

        z_min_val, z_max_val = np.min(z), np.max(z)
        near_up = np.abs(z - z_min_val) < 5.0
        near_dw = np.abs(z - z_max_val) < 5.0

        # FAST OPTICAL CHECK
        if has_pdg:
            # Standard Geant4: Optical Photons have PDG == 0
            is_optical = (arrs["PDGEncoding"] == 0)
        else:
            pn = arrs["ParticleName"]
            if len(pn) > 0:
                target_str = b"opticalphoton" if isinstance(pn[0], bytes) else "opticalphoton"
                is_optical = (pn == target_str)
            else:
                is_optical = np.zeros(0, dtype=bool)

        is_t = np.isin(channels, t_indices)
        is_e = np.isin(channels, e_indices)
        
        m_t_up = is_e & is_optical & near_up
        m_t_dw = is_e & is_optical & near_dw

        # Use defered chunk builder
        c_up = _chunk_raw(m_t_up, lt * 1000.0, ev, run_tag)
        if c_up is not None: up_q_chunks.append(c_up)
        
        c_dw = _chunk_raw(m_t_dw, lt * 1000.0, ev, run_tag)
        if c_dw is not None: dw_q_chunks.append(c_dw)

        m_dw_opt = near_dw & is_optical & is_e

        gt_downstream_opt = gt_raw[m_dw_opt]
        lt_downstream_opt = lt[m_dw_opt]

        hist_gt, _ = np.histogram(gt_downstream_opt, bins=gt_bins)
        gt_counts += hist_gt
        hist_lt, _ = np.histogram(lt_downstream_opt, bins=lt_bins)
        lt_counts += hist_lt

        if len(lt_downstream_opt) == 0:
            continue

        # ─────────────────────────────────────────────────────────────────────
        # SINGLE-ENDED RECONSTRUCTION (Pure LocalTime, Raw Optical Mapping)
        # ─────────────────────────────────────────────────────────────────────
        lt_up_arr = lt[m_t_up]
        lt_dw_arr = lt[m_dw_opt]

        # 1. Upstream Mapping: Sensor is at z_min.
        z_recon_up_all = z_min_val + (lt_up_arr * v_eff)

        # 2. Downstream Mapping: Sensor is at z_max.
        z_recon_dw_all = z_max_val - (lt_dw_arr * v_eff)

        # Uses the blazing-fast searchsorted algorithm internally now
        layer_idx_dw = get_layer_idx_from_z(z_recon_dw_all, lyso_bounds)
        layer_idx_up = get_layer_idx_from_z(z_recon_up_all, lyso_bounds)

        valid_dw = (layer_idx_dw != -1)
        valid_up = (layer_idx_up != -1)

        if np.any(valid_dw):
            np.add.at(prompt_counts_dw, layer_idx_dw[valid_dw], 1.0)
        if np.any(valid_up):
            np.add.at(prompt_counts_up, layer_idx_up[valid_up], 1.0)

    # --- Two-ended timing using deferred grouping ---
    up_q = _grouped_raw(up_q_chunks, ARRIVAL_QUANTILE)
    dw_q = _grouped_raw(dw_q_chunks, ARRIVAL_QUANTILE)
    common_t_evs = set(up_q) & set(dw_q)

    # --- Normalization ---
    events_denom = max(1, total_events_processed)
    if truth_profiles:
        events_per_run = max(1, total_events_processed / len(run_dirs))
        mean_truth_profile = np.mean(truth_profiles, axis=0) / events_per_run
        active_edep_list = [np.sum(p) for p in truth_profiles]
    else:
        mean_truth_profile = np.zeros(_N_LYSO)
        active_edep_list = []

    if verbose_label:
        print(f"    [{verbose_label}] {len(run_dirs)} runs, {len(common_t_evs)} double-coincidences, "
              f"DoseActor Truth Mean: {np.mean(active_edep_list) if active_edep_list else 0.0:.2f} MeV/run")

    return {
        "truth_layer_profile": mean_truth_profile,
        "gt_counts": gt_counts,
        "gt_bins": gt_bins,
        "lt_counts": lt_counts,
        "lt_bins": lt_bins,
        "run_dirs": sorted(run_dirs),
        "prompt_profile_dw": prompt_counts_dw / events_denom,
        "prompt_profile_up": prompt_counts_up / events_denom,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).resolve().parent
    modules = [
        #"radi_cal_energy", "radi_cal_triple", "rc_hex", "rc_hex_triple",
        #"dsb1_radi_cal_energy", 
        "dsb1_radi_cal_triple", #"dsb1_rc_hex", 
        "dsb1_rc_hex_triple",
        #"luagce_radi_cal_energy", 
        "luagce_radi_cal_triple", #"luagce_rc_hex", 
        "luagce_rc_hex_triple"
    ]

    out_dir = base_dir / "profile_analysis"
    global_dir = out_dir / "globaltime"
    local_dir = out_dir / "localtime"
    prompt_dir = out_dir / "prompt_photon_reconstruction"

    for d in [global_dir, local_dir, prompt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    master_summary = {mod: {} for mod in modules}

    print("Master profile processing engine spawned...")
    for mod in modules:
        mod_path = base_dir / mod / "runs" / mod
        if not mod_path.exists():
            mod_path = base_dir / mod 
            if not mod_path.exists(): continue

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

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH GENERATION PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    for mod in modules:
        if mod not in master_summary or not master_summary[mod]: continue
        energy_keys = sorted(master_summary[mod].keys(), key=extract_numerical_energy)
        if not energy_keys: continue

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

            bin_centers = 0.5 * (lt_bins[:-1] + lt_bins[1:])
            peaks, _ = find_peaks(lt_counts, prominence=np.max(lt_counts) * 0.1)

            if len(peaks) == 0:
                primary_peak_idx = np.argmax(lt_counts)
            else:
                primary_peak_idx = peaks[np.argsort(lt_counts[peaks])[-1]]

            peak_time = bin_centers[primary_peak_idx]
            peak_intensity = lt_counts[primary_peak_idx]

            line, = ax_lt.plot(bin_centers, lt_counts, label=f"{ekey} (Peak: {peak_time:.3f} ns)", alpha=0.85)
            color = line.get_color()

            ax_lt.scatter(peak_time, peak_intensity, marker="*", color=color, s=120, edgecolor="black", zorder=5)

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
        ax_lt.set_yscale("linear") 
        ax_lt.grid(True, linestyle=":", alpha=0.5)
        ax_lt.legend(title="Beam Components", loc="upper right")

        fig_lt.tight_layout()
        fig_lt.savefig(local_dir / f"{mod}_localtime_spectra.png", dpi=200)
        plt.close(fig_lt)

        # ── GRAPH 6: SINGLE-ENDED RECONSTRUCTION VS TRUTH ───────────────────────
        fig_se, axs_se = plt.subplots(1, 3, figsize=(18, 5.5))
        ax_dw, ax_up, ax_truth_se = axs_se

        layers_x = np.arange(1, _N_LYSO + 1)

        for idx, ekey in enumerate(energy_keys):
            col, _ = get_bar_colors(ekey, idx)

            prof_dw = master_summary[mod][ekey].get("prompt_profile_dw", np.zeros(_N_LYSO))
            prof_up = master_summary[mod][ekey].get("prompt_profile_up", np.zeros(_N_LYSO))
            truth_prof = master_summary[mod][ekey]["truth_layer_profile"]

            # DIAGNOSTIC: peak-offset check (upstream + downstream)
            if np.any(truth_prof):
                true_peak_layer = layers_x[np.argmax(truth_prof)]

                if np.any(prof_up):
                    upstream_peak_layer = layers_x[np.argmax(prof_up)]
                    up_offset = upstream_peak_layer - true_peak_layer
                else:
                    upstream_peak_layer, up_offset = None, None

                if np.any(prof_dw):
                    downstream_peak_layer = layers_x[np.argmax(prof_dw)]
                    dw_offset = downstream_peak_layer - true_peak_layer
                else:
                    downstream_peak_layer, dw_offset = None, None

                print(f"    [{mod}:{ekey}] true={true_peak_layer}  "
                    f"upstream={upstream_peak_layer} (offset={up_offset})  "
                    f"downstream={downstream_peak_layer} (offset={dw_offset})")

            # Subplot 1: Downstream Single-Ended
            ax_dw.plot(layers_x, prof_dw, marker="o", linestyle="None", color=col, 
                       markersize=6, alpha=0.8, label=ekey)

            # Subplot 2: Upstream Single-Ended
            ax_up.plot(layers_x, prof_up, marker="o", linestyle="None", color=col, 
                       markersize=6, alpha=0.8, label=ekey)

            # Subplot 3: DoseActor Truth
            ax_truth_se.plot(layers_x, truth_prof, marker="s", linestyle="None", color=col, 
                             markersize=6, alpha=0.8, label=ekey)

        titles = [
            "Downstream Single-Ended Recon", 
            "Upstream Single-Ended Recon", 
            "DoseActor Truth Profile"
        ]
        y_labels = [
            "Reconstructed Photon Strikes", 
            "Reconstructed Photon Strikes", 
            "Mean Active Energy (MeV)"
        ]

        for ax, title, ylab in zip(axs_se, titles, y_labels):
            ax.set_xlabel("LYSO Layer Number", fontweight="bold")
            ax.set_ylabel(ylab, fontweight="bold")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlim(0, _N_LYSO + 1)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(title="Beam Energy", fontsize=9)

        fig_se.suptitle(f"Timing Reconstruction vs Truth — {mod}", fontsize=14, fontweight="bold")
        fig_se.tight_layout()

        # Save to the prompt photon reconstruction directory
        fig_se.savefig(prompt_dir / f"{mod}_timing_reconstruction.png", dpi=200)
        plt.close(fig_se)

    print(f"\nProcessing complete! Reports saved directly inside: {out_dir.resolve()}")

if __name__ == "__main__":
    main()