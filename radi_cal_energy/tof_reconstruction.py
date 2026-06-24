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

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_LYSO_THICK_MM   = 1.5
_TYVEK_THICK_MM  = 0.2032
_W_THICK_MM      = 2.5
_N_LYSO          = 29
_N_W             = 28
_GAP_THICK_MM    = _LYSO_THICK_MM + 2 * _TYVEK_THICK_MM
_CALOR_THICK_MM  = 125.2856

_CAP_LENGTH_MM   = 183.0
_SIPM_THICK_MM   = 0.3
_Z_SENSOR_MM     = _CAP_LENGTH_MM / 2 + _SIPM_THICK_MM / 2  # ~91.65 mm

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL KINEMATICS & CORRECTIONS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS    = 299.792
REFRACTIVE_INDEX = 1.60
V_LIGHT_MM_NS    = C_LIGHT_MM_NS / REFRACTIVE_INDEX

# Bouncing correction for Total Internal Reflection zigzag
BOUNCE_FACTOR    = 1.0 
V_EFF_MM_NS      = V_LIGHT_MM_NS * BOUNCE_FACTOR

_HOLE_OFFSET_MM  = 3.7032
CAP_XY_MM = np.array([
    [ _HOLE_OFFSET_MM,  _HOLE_OFFSET_MM],   
    [-_HOLE_OFFSET_MM, -_HOLE_OFFSET_MM],   
    [-_HOLE_OFFSET_MM,  _HOLE_OFFSET_MM],   # E-type
    [ _HOLE_OFFSET_MM, -_HOLE_OFFSET_MM],   # E-type
])

def get_lyso_layer_bounds():
    bounds = []
    current_z = -_CALOR_THICK_MM / 2
    for idx in range(_N_LYSO):
        z_start = current_z + _TYVEK_THICK_MM
        z_end   = z_start + _LYSO_THICK_MM
        bounds.append((z_start, z_end))
        current_z += _GAP_THICK_MM + (_W_THICK_MM if idx < _N_W else 0)
    return bounds

def assign_channel(x_mm, y_mm):
    dists = np.hypot(CAP_XY_MM[:, 0] - x_mm, CAP_XY_MM[:, 1] - y_mm)
    return int(np.argmin(dists))

def load_truth_dose_from_mhd(run_dirs: list):
    if not utils: return None
    try:
        long_arr, _ = utils.load_calorimeter_mhd(run_dirs, long_glob="run_Dose_edep.mhd", trans_glob="transverse_shower_max_edep.mhd")
        if long_arr is None: return None
        dz_mm = 0.1
        avg = long_arr / max(len(run_dirs), 1)
        layer_edeps = []
        for (z_start, z_end) in get_lyso_layer_bounds():
            z_offset_start = z_start - (-_CALOR_THICK_MM / 2)
            z_offset_end   = z_end   - (-_CALOR_THICK_MM / 2)
            i0 = max(0, min(int(round(z_offset_start / dz_mm)), len(avg)))
            i1 = max(0, min(int(round(z_offset_end / dz_mm)), len(avg)))
            layer_edeps.append(float(np.sum(avg[i0:i1])))
        return np.array(layer_edeps)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True, type=str)
    args = parser.parse_args()
    batch_dir = Path(args.batch_dir)

    print(f"\n{'─'*60}")
    print(f"  Time-of-Flight Kinematic Profile Reconstruction")
    print(f"  Effective Z-Speed (with {BOUNCE_FACTOR}x factor): {V_EFF_MM_NS:.2f} mm/ns")
    print(f"{'─'*60}")

    run_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    hit_files = [p for d in run_dirs for p in sorted(d.glob("detector_hits_*.root"))]

    if not hit_files:
        print("  WARNING: No hit files found.")
        return

    e_type_times_ns = []
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tree = f[f.keys()[0]]
                x, y, z = tree["Position_X"].array(library="np"), tree["Position_Y"].array(library="np"), tree["Position_Z"].array(library="np")
                t = tree["LocalTime"].array(library="np")
                channels = np.array([assign_channel(xi, yi) for xi, yi in zip(x, y)])
                is_e_type_downstream = ((channels == 2) | (channels == 3)) & (z > 0)
                e_type_times_ns.append(t[is_e_type_downstream])
        except Exception:
            pass

    e_type_times_ns = np.concatenate(e_type_times_ns)
    print(f"  Extracted {len(e_type_times_ns):,} total optical hits.")

    # 1. Kinematic Mapping
    z_emit_mm = _Z_SENSOR_MM - (e_type_times_ns * V_EFF_MM_NS)

    # 2. Filter valid physical space (rejects the unphysical 40ns scintillation tail)
    calor_limit = _CALOR_THICK_MM / 2 + 15.0 # Give it a slight 15mm margin
    valid_z = z_emit_mm[(z_emit_mm >= -calor_limit) & (z_emit_mm <= calor_limit)]
    print(f"  Retained {len(valid_z):,} prompt photons mapping within array bounds.")

    # 3. Apply KDE Smoothing to prevent discrete binning quantization
    lyso_bounds = get_lyso_layer_bounds()
    tof_hits_per_layer = np.zeros(_N_LYSO)
    
    if len(valid_z) > 5:
        # Generate a continuous curve from the low statistics
        kde = gaussian_kde(valid_z, bw_method=0.15)
        for i, (z_min, z_max) in enumerate(lyso_bounds):
            z_mid = (z_min + z_max) / 2.0
            tof_hits_per_layer[i] = kde.evaluate(z_mid)[0]
    else:
        # Fallback if extremely low stats
        for i, (z_min, z_max) in enumerate(lyso_bounds):
            tof_hits_per_layer[i] = np.sum((valid_z >= z_min) & (valid_z <= z_max))

    # 4. Mirror the array to align with physical shower progression
    tof_hits_per_layer = tof_hits_per_layer[::-1]

    # 5. Normalization
    truth_curve = load_truth_dose_from_mhd(run_dirs)
    layers = np.arange(1, _N_LYSO + 1)
    if truth_curve is not None and np.sum(tof_hits_per_layer) > 0:
        tof_curve_scaled = tof_hits_per_layer * (np.sum(truth_curve) / np.sum(tof_hits_per_layer))
    else:
        tof_curve_scaled = tof_hits_per_layer

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    if truth_curve is not None:
        ax.bar(layers, truth_curve, color="#00bcd4", alpha=0.6, edgecolor="#00838f", linewidth=1.2, width=0.8, label="Simulation Truth (DoseActor)")

    ax.plot(layers, tof_curve_scaled, color="#d32f2f", linewidth=2.5, marker="o", markersize=5, label="ToF Kinematic Reconstruction (KDE Smoothed)")

    ax.set_title("Longitudinal Profile: Time-of-Flight Reconstruction vs Truth")
    ax.set_xlabel("LYSO Layer Number")
    ax.set_ylabel("Energy / Scaled Hits (MeV)")
    ax.set_xticks(layers[::2])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    
    out_path = batch_dir / "tof_reconstruction_vs_truth.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved smoothed plot to: {out_path}")

if __name__ == "__main__":
    main()