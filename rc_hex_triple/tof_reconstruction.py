import argparse
from pathlib import Path
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

try:
    import analysis_utils as utils
except ImportError:
    print("WARNING: Could not import 'utils'. Ensure this script is run from the OpenGATE sim directory.")
    utils = None

import warnings

from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY CONSTANTS (UPDATED TO BE COMPATIBLE WITH VARIABLE LYSO THICKNESS)
# ─────────────────────────────────────────────────────────────────────────────
_TYVEK_THICK_MM  = 0.2032
_W_THICK_MM      = 2.5
_N_LYSO          = 29
_N_W             = 28

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL KINEMATICS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS    = 299.792
REFRACTIVE_INDEX = 1.60                          # BCF-92 core index
V_LIGHT_MM_NS    = C_LIGHT_MM_NS / REFRACTIVE_INDEX
BOUNCE_FACTOR    = 0.92                          # account for TIR zig-zag
V_EFF_MM_NS      = V_LIGHT_MM_NS * BOUNCE_FACTOR

_GT_LO_NS = 0.0 
_GT_HI_NS = 50.0    

# ─────────────────────────────────────────────────────────────────────────────
# CAPILLARY XY POSITIONS  (indices 2, 3 are E-type)
# ─────────────────────────────────────────────────────────────────────────────
_HOLE_OFFSET_MM  = 3.7032
CAP_XY_MM = np.array([
    [ _HOLE_OFFSET_MM,  _HOLE_OFFSET_MM],   # 0 — T-type
    [-_HOLE_OFFSET_MM, -_HOLE_OFFSET_MM],   # 1 — T-type
    [-_HOLE_OFFSET_MM,  _HOLE_OFFSET_MM],   # 2 — E-type  ← used here
    [ _HOLE_OFFSET_MM, -_HOLE_OFFSET_MM],   # 3 — E-type  ← used here
])
_E_TYPE_INDICES  = {2, 3}
_T_TYPE_INDICES  = {0, 1}

# TIMING CALCULATION PARAMETERS
ARRIVAL_QUANTILE     = 0.10
MIN_PHOTONS_PER_FACE = 1

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (Geometry & Truth)
# ─────────────────────────────────────────────────────────────────────────────
def get_lyso_layer_bounds(lyso_thick, calor_thick):
    """Calculates active crystal bounds dynamically given current geometry configuration."""
    gap_thick = lyso_thick + 2 * _TYVEK_THICK_MM
    bounds = []
    current_z = -calor_thick / 2
    for idx in range(_N_LYSO):
        z_start = current_z + _TYVEK_THICK_MM
        z_end   = z_start + lyso_thick
        bounds.append((z_start, z_end))
        current_z += gap_thick + (_W_THICK_MM if idx < _N_W else 0)
    return bounds

def assign_channel(x_mm, y_mm):
    dists = np.hypot(CAP_XY_MM[:, 0, None] - x_mm, CAP_XY_MM[:, 1, None] - y_mm)
    return np.argmin(dists, axis=0)

def load_truth_dose_from_mhd(run_dirs: list, lyso_thick, calor_thick):
    if not utils: return None
    try:
        long_arr, _ = utils.load_calorimeter_mhd(
            run_dirs, long_glob="run_Dose_edep.mhd", trans_glob="transverse_shower_max_edep.mhd"
        )
        if long_arr is None: return None
        dz_mm, avg = 0.1, long_arr / max(len(run_dirs), 1)
        layer_edeps = []
        for (z_start, z_end) in get_lyso_layer_bounds(lyso_thick, calor_thick):
            z_offset_start = z_start - (-calor_thick / 2)
            z_offset_end   = z_end   - (-calor_thick / 2)
            i0 = max(0, min(int(round(z_offset_start / dz_mm)), len(avg)))
            i1 = max(0, min(int(round(z_offset_end   / dz_mm)), len(avg)))
            layer_edeps.append(float(np.sum(avg[i0:i1])))
        return np.array(layer_edeps)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (Timing Fit)
# ─────────────────────────────────────────────────────────────────────────────
def standard_gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian_to_peak(data, n_bins=50):
    if len(data) < 8: return 0.0, float(np.median(data)), float(np.std(data))
    q75, q25  = np.percentile(data, [75, 25])
    iqr_sigma = max((q75 - q25) / 1.349, 1.0)
    center    = np.median(data)
    lo, hi    = center - 3.0 * iqr_sigma, center + 3.0 * iqr_sigma

    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    mids = 0.5 * (edges[:-1] + edges[1:])
    smoothed  = gaussian_filter1d(counts.astype(float), sigma=2.0)
    peak_idx  = int(np.argmax(smoothed))
    mu0, A0   = float(mids[peak_idx]), float(smoothed[peak_idx])

    fit_mask = (counts > A0 * 0.15) & (np.abs(mids - mu0) < 1.5 * iqr_sigma)
    if fit_mask.sum() < 4:
        fit_mask = np.abs(mids - mu0) < 1.2 * iqr_sigma
        if fit_mask.sum() < 4: return A0, mu0, iqr_sigma

    try:
        popt, _ = curve_fit(
            standard_gaussian, mids[fit_mask], counts[fit_mask],
            p0=[A0, mu0, iqr_sigma * 0.7],
            bounds=([0.5, mu0 - iqr_sigma, 1.0], [A0 * 3.0, mu0 + iqr_sigma, iqr_sigma * 2.0]),
            method='trf', loss='soft_l1', maxfev=10000
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return A0, mu0, iqr_sigma

def clean_around_mode(arr, window_ps=60.0):
    if len(arr) == 0: return arr
    counts, edges = np.histogram(arr, bins=40)
    peak_bin      = np.argmax(gaussian_filter1d(counts.astype(float), sigma=2.0))
    mode_center   = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
    return arr[np.abs(arr - mode_center) < window_ps]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True, type=str)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    print(f"\n{'─'*60}")
    print(f"  Longitudinal Profile Reconstruction + Timing Resolution")
    print(f"{'─'*60}")

    run_dirs  = sorted([d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    hit_files = [p for d in run_dirs for p in sorted(d.glob("detector_hits_*.root"))]

    if not hit_files:
        print("  WARNING: No hit files found.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # DYNAMIC GEOMETRY SCAN: Extract parameters from first populated file
    # ─────────────────────────────────────────────────────────────────────────
    detected_z_sensor = None
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tree_key = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tree_key: continue
                z_arr = f[tree_key]["Position_Z"].array(library="np")
                if len(z_arr) > 0:
                    abs_z = np.abs(z_arr)
                    detected_z_sensor = float(np.median(abs_z[abs_z > (np.max(abs_z) - 5.0)]))
                    break
        except Exception:
            continue

    if detected_z_sensor is None:
        print("  ERROR: Could not establish physical SiPM plane location from data.")
        return

    # Back-calculate LYSO thickness depending on geometry version
    if abs(detected_z_sensor - 91.65) < 1.5:
        lyso_thick = 1.5
    elif abs(detected_z_sensor - 135.15) < 1.5:
        lyso_thick = 4.5
    else:
        # Fallback linear estimation based on cap dimension variations
        lyso_thick = 1.5 + (detected_z_sensor - 91.65) / _N_LYSO
        
    gap_thick_mm   = lyso_thick + 2 * _TYVEK_THICK_MM
    calor_thick_mm = (_N_LYSO * gap_thick_mm) + (_N_W * _W_THICK_MM)
    calor_half_mm  = calor_thick_mm / 2
    layer_pitch_mm = gap_thick_mm + _W_THICK_MM

    lyso_bounds = get_lyso_layer_bounds(lyso_thick, calor_thick_mm)

    print(f"  Detected SiPM Plane Z : ±{detected_z_sensor:.2f} mm")
    print(f"  Deduced LYSO Thickness: {lyso_thick:.2f} mm")
    print(f"  Calorimeter Thickness : {calor_thick_mm:.2f} mm")

    # ─────────────────────────────────────────────────────────────────────────
    # TRACKING DATA PASS
    # ─────────────────────────────────────────────────────────────────────────
    up_first, down_first = {}, {}
    up_times_by_ev, dw_times_by_ev = {}, {}
    t_type_best_minus_ps = []
    sipm_z_tol_mm = 2.5

    for fpath in hit_files:
        run_tag = fpath.parent.name
        try:
            with uproot.open(fpath) as f:
                tree_key = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tree_key: continue
                tree = f[tree_key]
                if tree.num_entries == 0: continue

                x  = tree["Position_X"].array(library="np")
                y  = tree["Position_Y"].array(library="np")
                z  = tree["Position_Z"].array(library="np")
                gt = tree["GlobalTime"].array(library="np")
                lt = tree["LocalTime"].array(library="np")
                ev = tree["EventID"].array(library="np")
                pn = tree["ParticleName"].array(library="np")
        except Exception as exc:
            print(f"  WARN: could not read {fpath.name}: {exc}")
            continue

        channels = assign_channel(x, y)
        near_up  = np.abs(z + detected_z_sensor) < sipm_z_tol_mm
        near_dw  = np.abs(z - detected_z_sensor) < sipm_z_tol_mm

        # ── 1. E-Type Spatial Reconstruction (GlobalTime) ──
        is_e_type = np.isin(channels, list(_E_TYPE_INDICES))
        is_prompt = (gt >= _GT_LO_NS) & (gt <= _GT_HI_NS)

        mask_e_up = is_e_type & is_prompt & near_up
        mask_e_dw = is_e_type & is_prompt & near_dw

        for eid, ti in zip(ev[mask_e_up], gt[mask_e_up]):
            key = (run_tag, int(eid))
            if key not in up_first or ti < up_first[key]:
                up_first[key] = float(ti)

        for eid, ti in zip(ev[mask_e_dw], gt[mask_e_dw]):
            key = (run_tag, int(eid))
            if key not in down_first or ti < down_first[key]:
                down_first[key] = float(ti)

        # ── 2. T-Type Timing Resolution (LocalTime) ──
        is_t_type  = np.isin(channels, list(_T_TYPE_INDICES))
        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")

        mask_t_up = is_t_type & is_optical & near_up
        mask_t_dw = is_t_type & is_optical & near_dw

        ev_t_up, lt_t_up = ev[mask_t_up], lt[mask_t_up] * 1000.0
        ev_t_dw, lt_t_dw = ev[mask_t_dw], lt[mask_t_dw] * 1000.0

        for e, t in zip(ev_t_up, lt_t_up):
            key = (run_tag, int(e))
            up_times_by_ev.setdefault(key, []).append(t)

        for e, t in zip(ev_t_dw, lt_t_dw):
            key = (run_tag, int(e))
            dw_times_by_ev.setdefault(key, []).append(t)

    # Coincidence fold
    common_t_evs = set(up_times_by_ev.keys()) & set(dw_times_by_ev.keys())
    for e in common_t_evs:
        if len(up_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE and len(dw_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE:
            t_up_q = np.quantile(up_times_by_ev[e], ARRIVAL_QUANTILE)
            t_dw_q = np.quantile(dw_times_by_ev[e], ARRIVAL_QUANTILE)
            t_type_best_minus_ps.append((t_dw_q - t_up_q) / 2.0)

    print(f"  T-Type coincident events for timing: {len(t_type_best_minus_ps):,}")
    if len(t_type_best_minus_ps) == 0:
        print("  ERROR: Zero timing coincidences detected.")
        return

    clean_bm = clean_around_mode(np.array(t_type_best_minus_ps), window_ps=100.0)
    _, _, sigma_t_ps = fit_gaussian_to_peak(clean_bm)
    
    sigma_z_mm = V_EFF_MM_NS * (sigma_t_ps / 1000.0)
    sigma_layer = sigma_z_mm / layer_pitch_mm

    print(f"  Calculated T-Type σ_t:      {sigma_t_ps:.1f} ps")
    print(f"  Equivalent spatial σ_z:     {sigma_z_mm:.2f} mm")
    print(f"  Equivalent layer error:   ± {sigma_layer:.2f} layers")

    # ── Process E-Type Spatial Reconstruction ──
    common_keys = set(up_first) & set(down_first)
    if not common_keys:
        print("  ERROR: No coincident events found for reconstruction.")
        return

    z_emit_list = [V_EFF_MM_NS * (down_first[k] - up_first[k]) / 2.0 for k in common_keys]
    z_emit_coin = np.array(z_emit_list)
    valid_coin  = z_emit_coin[(z_emit_coin >= -calor_half_mm - 15.0) & (z_emit_coin <= calor_half_mm + 15.0)]

    def kde_profile(valid_z, bounds=lyso_bounds):
        profile = np.zeros(_N_LYSO)
        if len(valid_z) < 5:
            for i, (z_min, z_max) in enumerate(bounds):
                profile[i] = np.sum((valid_z >= z_min) & (valid_z <= z_max))
        else:
            kde = gaussian_kde(valid_z, bw_method=0.15)
            for i, (z_min, z_max) in enumerate(bounds):
                profile[i] = kde.evaluate((z_min + z_max) / 2.0)[0]
        return profile

    profile_coin = kde_profile(valid_coin)[::-1]
    truth_curve = load_truth_dose_from_mhd(run_dirs, lyso_thick, calor_thick_mm)
    
    if truth_curve is not None and np.sum(profile_coin) > 0:
        profile_coin *= (np.sum(truth_curve) / np.sum(profile_coin))

    # ── Plotting ──
    layers = np.arange(1, _N_LYSO + 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    if truth_curve is not None:
        ax.bar(layers, truth_curve, color="#00bcd4", alpha=0.45,
               edgecolor="#00838f", linewidth=1.0, width=0.8, label="Simulation Truth (DoseActor)")

    ax.errorbar(layers, profile_coin, xerr=sigma_layer, color="#6a1b9a", 
                linewidth=2.5, marker="o", markersize=5, capsize=3, capthick=1.5,
                label=f"ΔT Coincidence (σ_t = {sigma_t_ps:.1f} ps → ±{sigma_layer:.2f} layers)")

    ax.set_xlabel("LYSO Layer Number")
    ax.set_ylabel("Energy / Scaled Hits (MeV)")
    ax.set_title(f"E-type SiPM ToF Reconstruction ({lyso_thick:.1f}mm LYSO Geometry)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out_path = batch_dir / "tof_reconstruction_with_errors.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved plot → {out_path}")

if __name__ == "__main__":
    main()