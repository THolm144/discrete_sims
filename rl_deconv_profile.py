#!/usr/bin/env python3
"""
unfold_profile_analysis.py
===========================
Alternative longitudinal shower-profile reconstruction using Richardson-Lucy
deconvolution instead of a 2-parameter Gamma fit.

Motivation
----------
The original profile_analysis pipeline (see extract_profile_data / gamma fit)
tries to recover the shower shape from only two scalar constraints
(Z_cg asymmetry + one hardware depth-fraction window). That's essentially a
fully-determined 2-parameter fit with zero redundancy, so noise in either
input pushes the recovered shape around with no goodness-of-fit to check
against.

This script instead treats the problem as classic unfolding:

    reco_profile(z) = R * true_profile(z)

where R is a Gaussian blur matrix built directly from the *measured* ToF
resolution (sigma_t_ps -> sigma_z -> sigma_layer) that the existing pipeline
already computes via the BestMinus timing method. Since that resolution is
measured per energy point from real per-event data, R does not require any
extra truth-level tracking info -- it's derived from the same observables
you already have.

We then invert R via Richardson-Lucy deconvolution (positivity-preserving,
standard in HEP/astro unfolding) to recover the true longitudinal profile
from the raw (unsmoothed) binned z_emit histogram. Uncertainty bands come
from a Poisson bootstrap over the raw per-event z_emit list.

Pulls data from the exact same TARGET_SWEEPS / detector_hits_*.root files as
profile_analysis.py. Output goes to a new folder so nothing overwrites the
original pipeline's results.
"""

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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (identical to profile_analysis.py, so extraction stays consistent)
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

TARGET_SWEEPS = {
    "radi_cal_energy": Path("/home/uakgun/env/THOMAS/discrete_sims/radi_cal_energy/runs/radi_cal_energy/sweep_20260707_170533"),
    "radi_cal_triple": Path("/home/uakgun/env/THOMAS/discrete_sims/radi_cal_triple/runs/radi_cal_triple/sweep_20260706_171040"),
    "rc_hex": Path("/home/uakgun/env/THOMAS/discrete_sims/rc_hex/runs/rc_hex/sweep_20260706_171127"),
    "rc_hex_triple": Path("/home/uakgun/env/THOMAS/discrete_sims/rc_hex_triple/runs/rc_hex_triple/sweep_20260706_171101"),
}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS (unchanged logic from profile_analysis.py)
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
# UNFOLDING MACHINERY (new)
# ─────────────────────────────────────────────────────────────────────────────
def gaussian_response_matrix(n_bins, sigma_bins):
    """
    Build an n_bins x n_bins Gaussian blur (point-spread) matrix.
    R[i, j] = probability that a true photon originating in true-bin j is
    reconstructed in reco-bin i, given a Gaussian smearing kernel of width
    sigma_bins (in units of bin/layer index).

    Columns are normalized to sum to 1 (flux conservation: every true photon
    ends up somewhere in the reco histogram).
    """
    sigma_bins = max(float(sigma_bins), 1e-3)  # avoid singular (delta-function) kernel
    idx = np.arange(n_bins)
    # R[i, j] = kernel centered at j, evaluated at i
    diff = idx[:, None] - idx[None, :]
    R = np.exp(-0.5 * (diff / sigma_bins) ** 2)
    col_sums = R.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    R = R / col_sums
    return R

def richardson_lucy_deconvolve(observed, R, iterations=150, eps=1e-12):
    """
    Classic Richardson-Lucy update for y = R x, x,y >= 0.
    Positivity-preserving, standard unfolding method in HEP/astro.

        x_{k+1} = x_k * ( R^T ( y / (R x_k + eps) ) )

    (No extra normalization needed since R's columns already sum to 1,
    so R^T @ ones = ones.)
    """
    n = R.shape[1]
    total = np.sum(observed)
    if total <= 0:
        return np.zeros(n)
    x = np.full(n, total / n, dtype=float)  # flat prior
    Rt = R.T
    for _ in range(iterations):
        denom = R @ x + eps
        x = x * (Rt @ (observed / denom))
    return x

def bootstrap_unfold(raw_z_emits, lyso_bounds, sigma_layer, n_boot=40, iterations=150, seed=0):
    """
    Poisson-bootstrap the raw per-event z_emit list, rebin + deconvolve each
    replicate, and return (mean_unfolded, std_unfolded, mean_raw_binned).
    """
    n_bins = len(lyso_bounds)
    edges = np.array([b[0] for b in lyso_bounds] + [lyso_bounds[-1][1]])
    raw_z_emits = np.asarray(raw_z_emits)
    R = gaussian_response_matrix(n_bins, sigma_layer)

    rng = np.random.default_rng(seed)
    n_events = len(raw_z_emits)
    unfolded_reps = []
    raw_reps = []

    if n_events == 0:
        return np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins), R

    for _ in range(n_boot):
        sample_idx = rng.integers(0, n_events, size=n_events)  # resample w/ replacement
        sample = raw_z_emits[sample_idx]
        counts, _ = np.histogram(sample, bins=edges)
        raw_reps.append(counts.astype(float))
        unfolded_reps.append(richardson_lucy_deconvolve(counts.astype(float), R, iterations=iterations))

    unfolded_reps = np.array(unfolded_reps)
    raw_reps = np.array(raw_reps)
    return unfolded_reps.mean(axis=0), unfolded_reps.std(axis=0), raw_reps.mean(axis=0), R

# ─────────────────────────────────────────────────────────────────────────────
# DATA EXTRACTION (same source files/branches as profile_analysis.py, but
# returns raw per-event z_emit values instead of a pre-smoothed KDE profile)
# ─────────────────────────────────────────────────────────────────────────────
def extract_profile_data_unfold(batch_dir: Path, is_hex: bool):
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

        is_e = np.isin(channels, list(e_indices))
        m_e_up, m_e_dw = is_e & is_prompt & near_up, is_e & is_prompt & near_dw

        for eid, ti in zip(ev[m_e_up], gt[m_e_up]):
            key = (run_tag, int(eid))
            if key not in up_first or ti < up_first[key]: up_first[key] = float(ti)
        for eid, ti in zip(ev[m_e_dw], gt[m_e_dw]):
            key = (run_tag, int(eid))
            if key not in down_first or ti < down_first[key]: down_first[key] = float(ti)

        is_t = np.isin(channels, list(t_indices))
        m_t_up, m_t_dw = is_t & is_optical & near_up, is_t & is_optical & near_dw

        for e, t in zip(ev[m_t_up], lt[m_t_up] * 1000.0):
            up_times_by_ev.setdefault((run_tag, int(e)), []).append(t)
        for e, t in zip(ev[m_t_dw], lt[m_t_dw] * 1000.0):
            dw_times_by_ev.setdefault((run_tag, int(e)), []).append(t)

    # Raw per-event ToF depth estimator (no KDE smoothing -- deconvolution needs
    # actual counts, and pre-smoothing here would double-blur the profile)
    common_e_keys = set(up_first) & set(down_first)
    raw_z_emits = []
    for k in common_e_keys:
        z_est = V_EFF_MM_NS * (down_first[k] - up_first[k]) / 2.0
        if -calor_thick_mm / 2 - 15.0 <= z_est <= calor_thick_mm / 2 + 15.0:
            raw_z_emits.append(z_est)
    raw_z_emits = np.array(raw_z_emits)

    # Timing resolution -> this is the measured PSF width for the response matrix
    common_t_evs = set(up_times_by_ev.keys()) & set(dw_times_by_ev.keys())
    all_bm_raw_ps = []
    for e in common_t_evs:
        if len(up_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE and len(dw_times_by_ev[e]) >= MIN_PHOTONS_PER_FACE:
            t_up_q = np.quantile(up_times_by_ev[e], ARRIVAL_QUANTILE)
            t_dw_q = np.quantile(dw_times_by_ev[e], ARRIVAL_QUANTILE)
            all_bm_raw_ps.append((t_dw_q - t_up_q) / 2.0)

    clean_bm = clean_around_mode(np.array(all_bm_raw_ps), window_ps=500.0)
    _, _, sigma_t_ps = fit_gaussian_to_peak(clean_bm)

    pitch_mm = gap_thick_mm + _W_THICK_MM
    sigma_z_mm = V_EFF_MM_NS * (sigma_t_ps / 1000.0)
    sigma_layer = sigma_z_mm / pitch_mm if pitch_mm > 0 else 1.0

    return {
        "raw_z_emits": raw_z_emits,
        "n_e_coincidences": len(common_e_keys),
        "sigma_t_ps": sigma_t_ps,
        "sigma_layer": sigma_layer,
        "pitch_mm": pitch_mm,
        "lyso_thick": lyso_thick,
        "calor_thick": calor_thick_mm,
        "lyso_bounds": lyso_bounds,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).resolve().parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_out = base_dir / "unfolded_profile_analysis" / f"unfolded_summary_{timestamp}"
    analysis_out.mkdir(parents=True, exist_ok=True)

    print("Spawning Unfolded Longitudinal Profile Extractor...")
    print(f"Targeting outputs to: {analysis_out.relative_to(base_dir)}\n")

    try:
        import analysis_utils as utils
    except ImportError:
        utils = None
        print("[Warning] analysis_utils not found. Sim truth overlays will be disabled.")

    mod_colors = {"radi_cal_energy": "#1976d2", "radi_cal_triple": "#388e3c", "rc_hex": "#d32f2f", "rc_hex_triple": "#7b1fa2"}
    layers = np.arange(1, _N_LYSO + 1)

    for mod, target_sweep in TARGET_SWEEPS.items():
        if not target_sweep.exists():
            print(f"  Skipping '{mod}' - Target sweep not found: {target_sweep}")
            continue

        print(f"Processing Unfolded Profile Data for '{mod}'...")
        is_hex = "hex" in mod
        energy_dirs = sorted([d for d in target_sweep.iterdir() if d.is_dir() and "GeV" in d.name],
                             key=lambda p: extract_numerical_energy(p.name))
        if not energy_dirs:
            continue

        n_energies = len(energy_dirs)
        ncols = 2 if n_energies >= 2 else 1
        nrows = int(np.ceil(n_energies / ncols))

        fig, axs = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows), squeeze=False)
        axs = axs.flatten()

        for idx, edir in enumerate(energy_dirs):
            ekey = edir.name
            print(f"    Extracting + unfolding {ekey}")
            res = extract_profile_data_unfold(edir, is_hex)
            ax = axs[idx]
            if res is None or len(res["raw_z_emits"]) < 5:
                ax.text(0.5, 0.5, "Insufficient Data", ha="center", va="center")
                ax.set_title(ekey, fontsize=11, fontweight="bold")
                continue

            lyso_bounds = res["lyso_bounds"]
            sigma_layer = res["sigma_layer"]
            # NOTE: iterations kept low (~20) deliberately. Richardson-Lucy converges
            # toward an unregularized MLE solution as iterations -> large, which means
            # it starts fitting per-bin Poisson noise rather than the underlying shape
            # once run too long -- that shows up as a spiky curve riding on top of the
            # raw profile instead of a genuinely deblurred one. If the unfolded curve
            # still tracks the raw curve too closely (or is too jagged) after this fix,
            # try iterations in the 5-15 range, or switch to Tikhonov-regularized
            # inversion instead of RL.
            unfolded_mean, unfolded_std, raw_mean, R = bootstrap_unfold(
                res["raw_z_emits"], lyso_bounds, sigma_layer, n_boot=40, iterations=20
            )

            # Normalize for display (shape comparison, not absolute yield)
            def safe_norm(v):
                s = np.sum(v)
                return v / s if s > 0 else v

            raw_norm = safe_norm(raw_mean)
            unf_norm = safe_norm(unfolded_mean)
            unf_err_norm = unfolded_std / np.sum(unfolded_mean) if np.sum(unfolded_mean) > 0 else unfolded_std

            # Optional truth overlay (same source as original pipeline)
            truth_curve = None
            calor_thick = res["calor_thick"]
            lyso_thick = res["lyso_thick"]
            run_dirs = sorted(list(set(fp.parent for fp in edir.rglob("detector_hits_*.root"))))
            if utils and run_dirs:
                try:
                    long_arr, _ = utils.load_calorimeter_mhd(run_dirs, long_glob="run_Dose_edep.mhd", trans_glob="transverse_shower_max_edep.mhd")
                    if long_arr is not None:
                        dz_mm, avg = 0.1, long_arr / max(len(run_dirs), 1)
                        layer_edeps = []
                        for (z_start, z_end) in lyso_bounds:
                            z_offset_start = z_start - (-calor_thick / 2)
                            z_offset_end = z_end - (-calor_thick / 2)
                            i0 = max(0, min(int(round(z_offset_start / dz_mm)), len(avg)))
                            i1 = max(0, min(int(round(z_offset_end / dz_mm)), len(avg)))
                            layer_edeps.append(float(np.sum(avg[i0:i1])))
                        truth_curve = np.array(layer_edeps)
                except Exception:
                    truth_curve = None

            # Layer axis: reverse to match original pipeline's display convention
            raw_norm_disp = raw_norm[::-1]
            unf_norm_disp = unf_norm[::-1]
            unf_err_disp = unf_err_norm[::-1]

            if truth_curve is not None and np.sum(truth_curve) > 0:
                # NOTE: truth_curve is built directly from `lyso_bounds` (upstream-first
                # order) and is the correct physical reference -- it is NOT reversed.
                # Only the ΔT-derived reco arrays get flipped (see raw_norm_disp /
                # unf_norm_disp below), matching the convention used in
                # unified_sweep_analysis_optimized.py, where the reco profile is
                # reversed once at extraction and truth is plotted as-is.
                truth_norm_disp = truth_curve / np.sum(truth_curve)
                ax.bar(layers, truth_norm_disp, color="#00bcd4", alpha=0.30, edgecolor="#00838f",
                       linewidth=0.8, width=0.8, label="Sim Truth (DoseActor)")

            ax.plot(layers, raw_norm_disp, color="gray", linewidth=1.4, linestyle=":",
                    marker=".", markersize=4, alpha=0.8, label="Raw ΔT Profile (blurred)")
            ax.errorbar(layers, unf_norm_disp, yerr=unf_err_disp, color=mod_colors[mod],
                        linewidth=2.2, marker="o", markersize=4.5, capsize=3, capthick=1.0,
                        label=f"RL-Unfolded (σ_layer={sigma_layer:.2f})")

            ax.set_title(f"{ekey}  (N={res['n_e_coincidences']})", fontsize=11, fontweight="bold")
            ax.set_xlabel("LYSO Layer Number", fontsize=9)
            ax.set_ylabel("Normalized Density Fraction", fontsize=9)
            ax.set_xlim(0, _N_LYSO + 1)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize=7.5)

        for idx in range(n_energies, len(axs)):
            fig.delaxes(axs[idx])

        fig.suptitle(f"Richardson-Lucy Unfolded Longitudinal Profile — {mod}", fontsize=14, fontweight="bold", y=0.995)
        fig.tight_layout()
        out_path = analysis_out / f"{mod}_unfolded_profile.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"    Saved: {out_path.name}")

if __name__ == "__main__":
    main()