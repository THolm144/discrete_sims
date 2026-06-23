import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import analysis_utils as utils
except ImportError:
    print("WARNING: Could not import 'utils'. Ensure this script is run from the OpenGATE sim directory.")
    utils = None

def get_channel_hits(sipm_hits: dict, indices: list) -> int:
    """Sums the front and back hits for the given capillary indices."""
    total = 0
    for idx in indices:
        total += sipm_hits.get(f"sipm_front_{idx}", 0)
        total += sipm_hits.get(f"sipm_back_{idx}", 0)
    return total

def reconstruct_shower_profile(e_energy: float, t_energy: float):
    """
    Fits a theoretical electromagnetic shower Gamma distribution 
    to the E-type (total) and T-type (shower-max) energies.
    """
    layers = np.arange(1, 30)
    
    if e_energy <= 0:
        return layers, np.zeros(29)

    target_ratio = t_energy / e_energy if e_energy > 0 else 0
    target_ratio = min(max(target_ratio, 0.01), 0.99)

    best_b = 0.5
    best_a = 5.0
    min_diff = float('inf')
    best_curve = np.zeros(29)

    for peak in np.linspace(6.0, 15.0, 30):
        for b in np.linspace(0.1, 1.5, 50):
            a = peak * b
            curve = (layers**a) * np.exp(-b * layers)
            curve = curve * (e_energy / np.sum(curve))
            
            t_region_sum = np.sum(curve[8:12])
            ratio = t_region_sum / e_energy
            
            diff = abs(ratio - target_ratio)
            if diff < min_diff:
                min_diff = diff
                best_b = b
                best_a = a
                best_curve = curve

    return layers, best_curve

def load_truth_dose_from_mhd(run_dirs: list):
    """
    Bypasses the text files and loads truth data directly from raw MHD files
    using the exact geometric binning logic from radi_cal_energy.py.
    """
    if not utils:
        return None

    try:
        # Pull directly from the raw simulation output files
        long_arr, _ = utils.load_calorimeter_mhd(
            run_dirs,
            long_glob="run_Dose_edep.mhd",
            trans_glob="transverse_shower_max_edep.mhd",
        )
        
        if long_arr is None:
            print("  [Warning]: No MHD dose files found in run directories.")
            return None

        # Exact physical geometry constants from radi_cal_energy.py
        _LYSO_THICK_MM   = 1.5
        _TYVEK_THICK_MM  = 0.2032
        _W_THICK_MM      = 2.5
        _GAP_THICK_MM    = _LYSO_THICK_MM + 2 * _TYVEK_THICK_MM # 1.9064 mm
        _N_LYSO          = 29
        _N_W             = 28
        dz_mm            = 0.1 # Standard voxel resolution

        avg = long_arr / max(len(run_dirs), 1)
        layer_edeps = []
        current_z = 0.0 

        # Map the continuous mesh coordinates back into the 29 distinct crystals
        for idx in range(_N_LYSO):
            z_start = current_z + _TYVEK_THICK_MM
            z_end   = z_start   + _LYSO_THICK_MM
            i0 = max(0, min(int(round(z_start / dz_mm)), len(avg)))
            i1 = max(0, min(int(round(z_end   / dz_mm)), len(avg)))
            
            layer_edeps.append(float(np.sum(avg[i0:i1])))
            current_z += _GAP_THICK_MM + (_W_THICK_MM if idx < _N_W else 0)

        return np.array(layer_edeps)
            
    except Exception as e:
        print(f"  [MHD Truth Parsing Error]: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True, type=str)
    parser.add_argument("--ly", type=float, default=12000.0, help="Light yield (photons/MeV)")
    parser.add_argument("--pde", type=float, default=0.25, help="SiPM PDE")
    parser.add_argument("--eff", type=float, default=0.005, help="Optical collection efficiency")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    
    if not utils:
        return

    # 1. Gather Simulation Run Directories
    run_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    hits_files = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    
    if not hits_files:
        print("  WARNING: No hit files found. Cannot reconstruct profile.")
        return

    hits = utils.analyse_hits(hits_files)
    
    # 2. Extract and Calibrate Energy
    t_hits = get_channel_hits(hits, [0, 1]) 
    e_hits = get_channel_hits(hits, [2, 3]) 

    calib_factor = args.ly * args.pde * args.eff
    e_energy_mev = e_hits / calib_factor
    t_energy_mev = t_hits / calib_factor

    print(f"  Total E-type energy: {e_energy_mev:.2f} MeV")
    print(f"  Shower-max T-type energy: {t_energy_mev:.2f} MeV")

    # 3. Mathematically Reconstruct Profile & Load Truth from MHD source
    layers, recon_curve = reconstruct_shower_profile(e_energy_mev, t_energy_mev)
    truth_curve = load_truth_dose_from_mhd(run_dirs)

    # 4. Plot the Comparison (Matches radi_cal_energy.py style)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    # Plot Truth Data as the official cyan bars
    if truth_curve is not None:
        ax.bar(layers, truth_curve, 
               color="#00bcd4", alpha=0.7, edgecolor="#00838f", 
               linewidth=1.2, width=0.8, label="Simulation Truth (DoseActor)")
    else:
        print("  WARNING: True profile could not be rendered (MHD data missing).")

    # Plot Reconstructed Data as a sharp contrasting orange line
    ax.plot(layers, recon_curve, color="#e65100", linewidth=2.5, marker="o", markersize=4, label="Reconstructed via SiPM Optics")
    
    # Highlight T-type sensor depth configuration (Layers 9-12)
    ax.axvspan(8.5, 12.5, color="#ffeb3b", alpha=0.25, label="T-Type Localized Window (Layers 9-12)")

    ax.set_title("Longitudinal Energy Deposition: Reconstructed vs. Truth")
    ax.set_xlabel("LYSO Layer Number")
    ax.set_ylabel("Energy Deposition (MeV)")
    ax.set_xticks(layers[::2])
    ax.legend(loc="upper right")
    
    out_path = batch_dir / "reconstructed_vs_truth.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Comparison plot saved to: {out_path.name}")

if __name__ == "__main__":
    main()