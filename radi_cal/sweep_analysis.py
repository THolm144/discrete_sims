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
    "New Reference": {
        "c": 15.92, 
        "s": 0.0, 
        "n": 122.8, 
        "color": "darkorange", 
        "ls": "-."
    },
}

def resolution_fit_func(E, c, s, n):
    """ Energy resolution parametrization: c (+) s/sqrt(E) (+) n/E in % """
    return np.sqrt(c**2 + (s / np.sqrt(E))**2 + (n / E)**2)

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

    # Memory cleanup: Disown objects before explicit C++ deletion to prevent PyROOT double-free segfaults
    ROOT.SetOwnership(h, False)
    ROOT.SetOwnership(g, False)
    h.Delete()
    g.Delete()

    return res_tuple

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
    t_indices = list(range(len(cap_xy_map)))  # All 4 channels treat as optical sensors

    up_t_hit_chunks, dw_t_hit_chunks = [], []
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

        # --- APPLY FLAT SiPM PDE ---
        # 0.40 (40%) is a standard PDE for Hamamatsu SiPMs at ~490-530nm.
        # Adjust this number to match your specific SiPM datasheet!
        FLAT_PDE = 0.40 
        is_detected = np.random.rand(len(gt)) < FLAT_PDE

        is_t = np.isin(channels, t_indices)
        
        # Add `is_detected` to the logical masks
        m_t_up_prompt = is_t & is_optical & near_up & is_prompt & is_detected
        m_t_dw_prompt = is_t & is_optical & near_dw & is_prompt & is_detected

        c = _chunk_series(m_t_up_prompt, gt, ev, run_tag)
        if c is not None: up_t_hit_chunks.append(c)
        c = _chunk_series(m_t_dw_prompt, gt, ev, run_tag)
        if c is not None: dw_t_hit_chunks.append(c)

    up_t_hits_per_ev = _grouped(up_t_hit_chunks, "count")
    dw_t_hits_per_ev = _grouped(dw_t_hit_chunks, "count")

    all_events = sorted(list(set(up_t_hits_per_ev.keys()) | set(dw_t_hits_per_ev.keys())))
    dw_t_total = np.array([dw_t_hits_per_ev.get(k, 0) + up_t_hits_per_ev.get(k, 0) for k in all_events])

    return {"dw_t_total": dw_t_total}


def analyze_showermax_edep_batch(batch_dir: Path):
    """
    Per-event RAW dE/dx (no optical transport) summed over the shower-max
    LYSO layers — the sim analog of Walker's H1[28]. Reads the per-layer
    'showermax_edep_{layer}.root' PhaseSpaceActor outputs (EventID,
    TotalEnergyDeposit per step, opticalphoton already filtered out at the
    actor level), sums across layers per event, and returns one
    total-energy-deposited value (MeV) per event.

    Requires the 'showermax_edep_{i}' PhaseSpaceActors added in
    simulator.py — older run directories generated before those actors
    existed will simply have no matching files and this returns None.
    """
    edep_files = sorted(batch_dir.rglob("showermax_edep_*.root"))
    if not edep_files:
        return None

    edep_sum_per_ev = {}  # (run_tag, EventID) -> summed TotalEnergyDeposit across all layers
    for fpath in edep_files:
        run_tag = fpath.parent.name
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if k.split(";")[0].startswith("showermax_edep")), None)
                if not tk:
                    continue
                tree = f[tk]
                if tree.num_entries == 0:
                    continue
                arrs = tree.arrays(["EventID", "TotalEnergyDeposit"], library="np")
        except Exception:
            continue

        for e_id, dep in zip(arrs["EventID"], arrs["TotalEnergyDeposit"]):
            key = (run_tag, int(e_id))
            edep_sum_per_ev[key] = edep_sum_per_ev.get(key, 0.0) + float(dep)

    if not edep_sum_per_ev:
        return None

    return np.array(list(edep_sum_per_ev.values()))  # MeV per event

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

    # Automatically resolve path: runs/<module>/sweep_* or runs/sweep_*
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
    photon_counts_by_energy = []  # keep raw per-event arrays for the calibrated-energy curve
    edep_res_percent, edep_res_err_percent = [], []  # raw dE/dx (Walker's H1[28] analog)

    for edir in energy_dirs:
        e_val = extract_numerical_energy(edir.name)
        print(f" [+] Processing [{edir.name}]...")

        res_dict = analyze_energy_batch(edir, module_name=args.module)
        if res_dict is None or len(res_dict["dw_t_total"]) == 0:
            print(f"     [!] Skipped — no data found")
            continue

        photon_counts = res_dict["dw_t_total"]
        mean_N = np.mean(photon_counts)
        res, err = robust_resolution(photon_counts)

        energies_gev.append(e_val)
        mean_yields.append(mean_N)
        res_percent.append(res)
        res_err_percent.append(err)
        photon_counts_by_energy.append(photon_counts)

        # Raw dE/dx in the shower-max LYSO band, independent of photon yield/optics
        edep_MeV = analyze_showermax_edep_batch(edir)
        if edep_MeV is not None and len(edep_MeV) >= 2:
            res_e, err_e = robust_resolution(edep_MeV)
            edep_res_percent.append(res_e)
            edep_res_err_percent.append(err_e)
            print(f"     -> [dE/dx] Events: {len(edep_MeV)} | Mean Edep: {np.mean(edep_MeV):.2f} MeV "
                  f"| Resolution: {res_e:.2f}% ± {err_e:.2f}%")
        else:
            edep_res_percent.append(np.nan)
            edep_res_err_percent.append(np.nan)
            print(f"     -> [dE/dx] No showermax_edep_*.root found for this energy point (older run?)")

        print(f"     -> Events: {len(photon_counts)} | Mean Photons: {mean_N:.1f} | Resolution: {res:.2f}% ± {err:.2f}%")

    if not energies_gev:
        print("[-] No valid data to plot.")
        return

    energies_gev = np.array(energies_gev)
    res_percent = np.array(res_percent)
    edep_res_percent = np.array(edep_res_percent)
    edep_res_err_percent = np.array(edep_res_err_percent)
    edep_mask = np.isfinite(edep_res_percent)
    res_err_percent = np.array(res_err_percent)

    # --- Fit simulation points with physically bounded curve_fit ---
    popt_sim = [15.92, 0.0, 122.8]  # Default fallback reference
    try:
        popt, _ = curve_fit(
            resolution_fit_func, 
            energies_gev, 
            res_percent, 
            sigma=res_err_percent, 
            p0=[12.0, 50.0, 30.0],                          # Physical initial seed (c, s, n)
            bounds=([0.0, 0.0, 0.0], [30.0, 150.0, 100.0])  # Cap n <= 100 to prevent fit degeneracy
        )
        popt_sim = popt
    except Exception as e:
        print(f" [+] Fit Warning: curve_fit failed ({e}). Using default reference curves.")

    # ─────────────────────────────────────────────────────────────────────────
    # CALIBRATED-ENERGY CURVE
    # ─────────────────────────────────────────────────────────────────────────
    # Raw photon-count resolution (pink) mixes real shot noise with whatever
    # offset/pedestal sits in N_photons(E). The paper instead resolutions the
    # RECONSTRUCTED ENERGY: calibrate <N_photons> vs E_beam with a straight
    # line (N = a*E + b), invert it per event (E_reco = (N-b)/a), then take
    # sigma/mean of E_reco at each beam energy. This does not change the
    # underlying photostatistics, it just measures resolution on the same
    # quantity (energy) that the paper's 9.31/52.04/31.62 fit is measuring,
    # so the two become directly comparable.
    mean_yields_arr = np.array(mean_yields)
    calib_slope, calib_intercept = np.polyfit(energies_gev, mean_yields_arr, 1)
    print(f" [+] Calibration: N_photons = {calib_slope:.3f} * E_GeV + {calib_intercept:.3f}")

    res_calib_percent, res_calib_err_percent = [], []
    for e_val, photon_counts in zip(energies_gev, photon_counts_by_energy):
        e_reco = (np.asarray(photon_counts, dtype=float) - calib_intercept) / calib_slope
        # guard against non-physical negative reconstructed energies from the
        # pedestal subtraction at low photon count / low beam energy
        e_reco = e_reco[e_reco > 0]
        if len(e_reco) < 2:
            res_calib_percent.append(np.nan)
            res_calib_err_percent.append(np.nan)
            continue
        res_c, err_c = robust_resolution(e_reco)
        res_calib_percent.append(res_c)
        res_calib_err_percent.append(err_c)
        print(f"     -> E={e_val:.1f} GeV | Calibrated E_reco mean={np.mean(e_reco):.2f} GeV "
              f"| Resolution: {res_c:.2f}% ± {err_c:.2f}%")

    res_calib_percent = np.array(res_calib_percent)
    res_calib_err_percent = np.array(res_calib_err_percent)
    calib_mask = np.isfinite(res_calib_percent)

    popt_calib = None
    if calib_mask.sum() >= 3:
        try:
            popt_calib, _ = curve_fit(
                resolution_fit_func,
                energies_gev[calib_mask],
                res_calib_percent[calib_mask],
                sigma=res_calib_err_percent[calib_mask],
                p0=[10.0, 50.0, 30.0],
                bounds=([0.0, 0.0, 0.0], [30.0, 150.0, 100.0])
            )
        except Exception as e:
            print(f" [+] Calibrated fit warning: curve_fit failed ({e}). Not plotting a calibrated fit curve.")
            popt_calib = None

    # ─────────────────────────────────────────────────────────────────────────
    # RAW dE/dx CURVE (Walker's H1[28] analog — no optical transport at all)
    # ─────────────────────────────────────────────────────────────────────────
    popt_edep = None
    if edep_mask.sum() >= 3:
        try:
            popt_edep, _ = curve_fit(
                resolution_fit_func,
                energies_gev[edep_mask],
                edep_res_percent[edep_mask],
                sigma=edep_res_err_percent[edep_mask],
                p0=[10.0, 50.0, 30.0],
                bounds=([0.0, 0.0, 0.0], [30.0, 150.0, 100.0])
            )
        except Exception as e:
            print(f" [+] dE/dx fit warning: curve_fit failed ({e}). Not plotting a dE/dx fit curve.")
            popt_edep = None

    # Save summary dataframe
    df_summary = pd.DataFrame({
        "Energy_GeV": energies_gev,
        "Mean_Photons": mean_yields,
        "Energy_Resolution_Percent": res_percent,
        "Energy_Resolution_Err_Percent": res_err_percent,
        "Calibrated_Energy_Resolution_Percent": res_calib_percent,
        "Calibrated_Energy_Resolution_Err_Percent": res_calib_err_percent,
        "ShowerMax_dEdx_Resolution_Percent": edep_res_percent,
        "ShowerMax_dEdx_Resolution_Err_Percent": edep_res_err_percent,
    })
    df_summary.to_csv(out_dir / "sweep_4T_summary.csv", index=False)

    # ── ROOT-Style Plot Generation ───────────────────────────────────────────
    plt.figure(figsize=(9, 6.5))

    # 1. Sim Data Points (Magenta Squares)
    plt.errorbar(
        energies_gev, res_percent, yerr=res_err_percent,
        fmt='s', color='m', ecolor='m', capsize=3, elinewidth=1.2,
        label=f'sim (photon count): {popt_sim[0]:.2f}% $\\oplus$ {popt_sim[1]:.1f}%/$\\sqrt{{E}}$ $\\oplus$ {popt_sim[2]:.1f}%/E'
    )

    # 2. Fitted Sim Curve (Magenta Dashed Line)
    e_smooth = np.linspace(max(0.5, min(energies_gev) * 0.8), max(energies_gev) * 1.1, 200)
    sim_curve = resolution_fit_func(e_smooth, *popt_sim)
    plt.plot(e_smooth, sim_curve, 'm--', lw=1.8)

    
    # 2c. Raw dE/dx Data Points + Fit (Green Triangles) — Walker's H1[28]
    #     analog: energy deposited directly in the shower-max LYSO layers,
    #     with NO optical transport / photon yield involved at all. This is
    #     the cleanest test of whether the large stochastic term is real
    #     shower-sampling physics (this curve is also steep) or an artifact
    #     of the optical/light-collection chain (this curve would be flat-ish
    #     while pink/cyan stay steep).
    if edep_mask.sum() >= 1:
        if popt_edep is not None:
            edep_label = (f'sim (raw dE/dx): {popt_edep[0]:.2f}% $\\oplus$ '
                           f'{popt_edep[1]:.1f}%/$\\sqrt{{E}}$ $\\oplus$ {popt_edep[2]:.1f}%/E')
        else:
            edep_label = 'sim (raw dE/dx)'
        plt.errorbar(
            energies_gev[edep_mask], edep_res_percent[edep_mask], yerr=edep_res_err_percent[edep_mask],
            fmt='^', color='g', ecolor='g', capsize=3, elinewidth=1.2,
            label=edep_label
        )
        if popt_edep is not None:
            edep_curve = resolution_fit_func(e_smooth, *popt_edep)
            plt.plot(e_smooth, edep_curve, 'g--', lw=1.8)

    # 3. Continuous Reference Curves
    for label, params in ENERGY_REF_CURVES.items():
        ref_curve = resolution_fit_func(e_smooth, params["c"], params["s"], params["n"])
        plt.plot(
            e_smooth, 
            ref_curve, 
            ls=params.get("ls", "--"), 
            lw=1.8, 
            color=params.get("color", "gray"),
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
    plot_path = out_dir / "showermax_energy_resolution_4T.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n [✓] Results saved to {out_dir.resolve()}")
    print(f" [✓] Summary Plot: {plot_path.resolve()}")

if __name__ == "__main__":
    main()