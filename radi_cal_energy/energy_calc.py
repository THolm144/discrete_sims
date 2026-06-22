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
    
    # If no energy, return an empty flat profile
    if e_energy <= 0:
        return layers, np.zeros(29)

    # Calculate the fraction of total energy that sits in the T-type layers
    target_ratio = t_energy / e_energy if e_energy > 0 else 0
    target_ratio = min(max(target_ratio, 0.01), 0.99) # Keep in bounds

    best_b = 0.5
    best_a = 5.0
    min_diff = float('inf')
    best_curve = np.zeros(29)

    # 2D Grid Search: sweep the peak location and the curve width (b parameter)
    # Peak of x^a * e^-bx is at x = a/b, so a = peak * b
    for peak in np.linspace(6.0, 15.0, 30):
        for b in np.linspace(0.1, 1.5, 50):
            a = peak * b
            curve = (layers**a) * np.exp(-b * layers)
            
            # Normalize curve so the sum of all 29 layers equals E-type total energy
            curve = curve * (e_energy / np.sum(curve))
            
            # Check energy in the T-type region (indices 8, 9, 10, 11 -> layers 9, 10, 11, 12)
            t_region_sum = np.sum(curve[8:12])
            ratio = t_region_sum / e_energy
            
            diff = abs(ratio - target_ratio)
            if diff < min_diff:
                min_diff = diff
                best_b = b
                best_a = a
                best_curve = curve

    return layers, best_curve

def load_truth_dose(batch_dir: Path):
    """Loads the pre-binned DoseActor truth data for comparison."""
    dose_file = batch_dir / "analyzed_longitudinal.txt"
    if not dose_file.exists():
        return None

    try:
        data = np.loadtxt(dose_file)
        if data.size == 0:
            return None
        
        # Extract the 1D array depending on how analyze.py formatted it
        if data.ndim == 2:
            avg_dose = data[:, 1] if data.shape[1] > 1 else data[:, 0]
        else:
            avg_dose = data
            
        # Since radi_cal_energy already bins this into 29 layers, return it directly!
        if len(avg_dose) == 29:
            return avg_dose
        else:
            print(f"  [Warning]: Expected 29 layers, but found {len(avg_dose)} in truth data.")
            return avg_dose
            
    except Exception as e:
        print(f"  [Dose Parsing Error]: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True, type=str)
    parser.add_argument("--ly", type=float, default=12000.0, help="Light yield (photons/MeV)")
    parser.add_argument("--pde", type=float, default=0.25, help="SiPM PDE")
    parser.add_argument("--eff", type=float, default=0.005, help="Optical collection efficiency (e.g., 0.005 for 0.5%)")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    
    if not utils:
        return

    # 1. Parse Simulation Hits
    run_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    hits_files = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    
    if not hits_files:
        print("  WARNING: No hit files found. Cannot reconstruct profile.")
        return

    hits = utils.analyse_hits(hits_files)
    
    # 2. Extract E-Type and T-Type Data (Fixed Indices)
    t_hits = get_channel_hits(hits, [0, 1]) # T-type (shower-max localized)
    e_hits = get_channel_hits(hits, [2, 3]) # E-type (full length)

    # Convert SiPM hits to Deposited Energy (Fixed Math)
    # Energy = Hits / (Light Yield * PDE * Collection Efficiency)
    calib_factor = args.ly * args.pde * args.eff
    e_energy_mev = e_hits / calib_factor
    t_energy_mev = t_hits / calib_factor

    print(f"  Total E-type energy: {e_energy_mev:.2f} MeV")
    print(f"  Shower-max T-type energy: {t_energy_mev:.2f} MeV")

    # 3. Mathematically Reconstruct Profile & Load Truth
    layers, recon_curve = reconstruct_shower_profile(e_energy_mev, t_energy_mev)
    truth_curve = load_truth_dose(batch_dir)

    # 4. Plot the Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot Truth Data as ghosted gray bars
    if truth_curve is not None:
        ax.bar(layers, truth_curve, color="gray", alpha=0.3, width=1.0, label="Simulation Truth (DoseActor)")

    # Plot Reconstructed Data as a sharp line
    ax.plot(layers, recon_curve, color="#2196F3", linewidth=2.5, marker="o", markersize=4, label="Reconstructed via SiPM Optics")
    
    # Highlight the T-type window (Layers 9, 10, 11, 12)
    ax.axvspan(8.5, 12.5, color="#ffeb3b", alpha=0.3, label="T-Type Localized Window (Layers 9-12)")

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