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

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL KINEMATICS CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS = 299.792
REFRACTIVE_INDEX = 1.60
V_LIGHT_MM_NS = C_LIGHT_MM_NS / REFRACTIVE_INDEX
BOUNCE_FACTOR = 0.92
V_EFF_MM_NS = V_LIGHT_MM_NS * BOUNCE_FACTOR

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

        # Process E-type (ToF + Distance Profiles)
        is_e = np.isin(channels, list(e_indices))
        m_e_up, m_e_dw = is_e & is_prompt & near_up, is_e & is_prompt & near_dw

        for eid, ti in zip(ev[m_e_up], gt[m_e_up]):
            key = (run_tag, int(eid))
            if key not in up_first or ti < up_first[key]: up_first[key] = float(ti)
        for eid, ti in zip(ev[m_e_dw], gt[m_e_dw]):
            key = (run_tag, int(eid))
            if key not in down_first or ti < down_first[key]: down_first[key] = float(ti)

        # Process T-type (Needed for X-error bars on ToF Plot via overall timing resolution)
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

    # 2. Timing Resolution Calculation (for ToF X-Error bars)
    common_t_evs = set(up_times_by_ev.keys()) & set(dw_times_by_ev.keys())
    all_bm_raw_ps = []
    for e in common_t_evs:
        if len(up_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE and len(dw_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE:
            t_up_q = np.quantile(up_times_by_ev[e], ARRIVAL_QUANTILE)
            t_dw_q = np.quantile(dw_times_by_ev[e], ARRIVAL_QUANTILE)
            all_bm_raw_ps.append((t_dw_q - t_up_q) / 2.0)

    clean_bm = clean_around_mode(np.array(all_bm_raw_ps), window_ps=500.0)
    _, _, sigma_t_ps = fit_gaussian_to_peak(clean_bm)

    return {
        "tof_profile": profile_counts,
        "n_e_coincidences": len(common_e_keys),
        "sigma_t_ps": sigma_t_ps,
        "pitch_mm": gap_thick_mm + _W_THICK_MM,
        "lyso_thick": lyso_thick,
        "dw_first_times": np.array(list(down_first.values()))
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

        for idx, edir in enumerate(energy_dirs):
            ekey = edir.name
            print(f"    Extracting {ekey}")
            res = extract_profile_data(edir, is_hex)
            if res is None: continue

            # ─────────────────────────────────────────────────────────────────────
            # 1. TOF PROFILE PANELS
            # ─────────────────────────────────────────────────────────────────────
            ax_t = axs_tof[idx]
            prof, s_t, pitch = res["tof_profile"], res["sigma_t_ps"], res["pitch_mm"]
            lyso_thick = res["lyso_thick"]

            sigma_z = V_EFF_MM_NS * (s_t / 1000.0)
            sigma_layer = sigma_z / pitch

            total_hits = np.sum(prof)
            norm_prof = prof / total_hits if total_hits > 0 else prof

            run_dirs = sorted(list(set(fp.parent for fp in edir.rglob("detector_hits_*.root"))))

            truth_curve = None
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

            # ─────────────────────────────────────────────────────────────────────
            # 2. DOWNSTREAM PROMPT STRIKES VS DISTANCE PANELS
            # ─────────────────────────────────────────────────────────────────────
            ax_d = axs_dist[idx]
            first_times = res["dw_first_times"]
            
            if len(first_times) > 0:
                distances_mm = first_times * V_EFF_MM_NS
                view_min, view_max = 120.0, 350.0 
                
                counts, edges, _ = ax_d.hist(distances_mm, bins=80, range=(view_min, view_max), color="#ff9800", 
                                           alpha=0.6, edgecolor="black", linewidth=0.5, 
                                           label=f"Prompt Strikes (N={len(first_times)})")
                
                if len(counts) > 0 and np.max(counts) > 0:
                    peak_idx = np.argmax(counts)
                    peak_dist = edges[peak_idx] + (edges[1] - edges[0]) / 2.0
                    ax_d.axvline(peak_dist, color="red", linestyle="--", linewidth=1.5, 
                               label=f"Peak: {peak_dist:.1f} mm")

                if utils and run_dirs:
                    try:
                        if long_arr is not None:
                            z_sensor = 110.0 
                            z_source_mm = -130.15 
                            max_hist_height = np.max(counts) if len(counts) > 0 else 1.0
                            
                            layer_dists = []
                            for (z_start, z_end) in bounds:
                                z_center = (z_start + z_end) / 2.0
                                t_expected = ((z_center - z_source_mm) / C_LIGHT_MM_NS) + (abs(z_sensor - z_center) / V_EFF_MM_NS)
                                layer_dists.append(t_expected * V_EFF_MM_NS)
                            
                            if np.sum(truth_curve) > 0:
                                norm_edeps = (truth_curve / np.max(truth_curve)) * (max_hist_height * 0.9)
                                bar_width = 2.0 
                                
                                for i, (d_val, edep_val) in enumerate(zip(layer_dists, norm_edeps)):
                                    if view_min <= d_val <= view_max: 
                                        bar_color = "#e91e63" if i in WLS_LAYERS else "#00bcd4"
                                        bar_label = "WLS Region (Sim Truth)" if (i in WLS_LAYERS and i == min(WLS_LAYERS)) else \
                                                    "Standard LYSO (Sim Truth)" if (i not in WLS_LAYERS and i == 0) else None
                                        
                                        ax_d.bar(d_val, edep_val, width=bar_width, color=bar_color, alpha=0.5, 
                                               edgecolor="black", linewidth=0.5, label=bar_label)
                    except Exception as e: pass

                ax_d.set_title(f"Prompt Strikes vs Distance: {ekey}", fontsize=11, fontweight="bold")
                ax_d.set_xlabel(f"Kinematic Pseudo-Distance (mm)", fontsize=9)
                ax_d.set_ylabel("Events", fontsize=9)
                ax_d.set_xlim(view_min, view_max)
                
                handles, labels = ax_d.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_d.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, frameon=True)
            else:
                ax_d.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax_d.grid(True, linestyle=":", alpha=0.5)

        for idx in range(n_energies, len(axs_tof)):
            fig_tof.delaxes(axs_tof[idx])
            fig_dist.delaxes(axs_dist[idx])

        fig_tof.suptitle(f"Continuous E-Type ToF Reconstructions — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_tof.tight_layout()
        fig_tof.savefig(analysis_out / f"{mod}_tof_panels.png", dpi=200)
        plt.close(fig_tof)

        fig_dist.suptitle(f"Downstream Prompt Strikes vs Distance (Cropped) — {mod}", fontsize=14, fontweight="bold", y=0.98)
        fig_dist.tight_layout()
        fig_dist.savefig(analysis_out / f"{mod}_dw_prompt_distance_cropped.png", dpi=200)
        plt.close(fig_dist)

if __name__ == "__main__":
    main()