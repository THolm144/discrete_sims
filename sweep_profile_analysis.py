"""
unified_profile_analysis.py
Optimized analysis and plotting script for calorimeter profile, timing,
and longitudinal prompt photon reconstruction.
"""

import os
import glob
import re
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. CONFIGURATION & PHYSICS CONSTANTS
# ==========================================
C_LIGHT_MM_NS = 299.792458  # Speed of light in vacuum (mm/ns)

# Parent output directory
OUTPUT_DIR = "profile_analysis"
SUBFOLDERS = {
    "energy": "energy_performance",
    "globaltime": "globaltime",
    "localtime": "localtime",
    "prompt": "prompt_photon_reconstruction",
    "two_end": "two_end_timing"
}

# Module Optical Properties
# Adjust these based on your specific module geometries and materials
MODULE_CONFIGS = {
    "default": {
        "refractive_index": 1.82,     # LYSO Refractive Index
        "bounce_factor": 0.82,        # Effective velocity reduction factor due to helical bouncing paths
        "z_sensor_down": 150.0,       # Downstream SiPM z-coordinate (mm)
        "z_sensor_up": -150.0,        # Upstream SiPM z-coordinate (mm)
        "num_layers": 29,             # Number of LYSO active layers
        "lyso_z_start": -100.0,       # Start z of first active layer (mm)
        "lyso_z_end": 100.0,          # End z of last active layer (mm)
        "prompt_window_ns": 0.15,     # Half-width of prompt-photon time selection window (ns)
    }
}

# Ensure all output directories exist
for folder in SUBFOLDERS.values():
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

# Set elegant plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14, 'axes.labelsize': 12})


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def parse_metadata(filepath):
    """
    Parses beam energy (GeV) and module name/tag from the path.
    Modify regex if your naming convention differs.
    """
    filename = os.path.basename(filepath)
    energy_match = re.search(r'(\d+)\s*GeV', filepath, re.IGNORECASE)
    energy = float(energy_match.group(1)) if energy_match else 10.0
    
    module = "default"
    for m in ["module_a", "module_b", "module_c"]:
        if m in filepath.lower():
            module = m
            break
    return energy, module


def get_effective_velocity(config):
    """Calculates the signal propagation velocity along the filament (mm/ns)."""
    return (C_LIGHT_MM_NS / config["refractive_index"]) * config["bounce_factor"]


def fit_resolution(E, a, b):
    """Standard electromagnetic calorimeter resolution curve."""
    return np.sqrt((a / np.sqrt(E))**2 + b**2)


# ==========================================
# 3. DATA PROCESSING ENGINE
# ==========================================

def run_analysis(data_dir):
    # Find all root files in the target sweep/run directories
    file_pattern = os.path.join(data_dir, "**", "*.root")
    root_files = glob.glob(file_pattern, recursive=True)
    
    if not root_files:
        print(f"[-] No ROOT files found in: {data_dir}")
        return

    print(f"[+] Found {len(root_files)} data files. Starting processing pipeline...")

    # Data structures to aggregate across runs
    run_summary = {}  # Key: (module, energy) -> Dict of metrics

    for idx, filepath in enumerate(root_files):
        energy, module = parse_metadata(filepath)
        print(f"    [{idx+1}/{len(root_files)}] Processing {energy} GeV | {module}...")
        
        # Load configuration properties
        cfg = MODULE_CONFIGS.get(module, MODULE_CONFIGS["default"])
        v_eff = get_effective_velocity(cfg)
        
        # Generate layer centers
        layer_centers = np.linspace(cfg["lyso_z_start"], cfg["lyso_z_end"], cfg["num_layers"])
        
        try:
            with uproot.open(filepath) as f:
                # Check tree naming convention (adjust 'Hits' or 'Events' if needed)
                tree_name = "Hits" if "Hits" in f else f.keys()[0].split(';')[0]
                tree = f[tree_name]
                
                # Read necessary branches into memory
                # (Standard GATE branch naming convention assumed)
                data = tree.arrays([
                    "energyDeposit", "LocalTime", "GlobalTime", 
                    "PDGCode", "positionX", "positionY", "positionZ"
                ], library="np")
                
                # Check event ID groupings (if eventID branch is available)
                event_ids = tree["eventID"].array(library="np") if "eventID" in tree else np.zeros(len(data["energyDeposit"]))
                
                # --- PROCESS EVENT ENERGY DEPOSITS ---
                unique_evs, ev_counts = np.unique(event_ids, return_counts=True)
                total_energy_per_event = np.bincount(event_ids.astype(int), weights=data["energyDeposit"])
                # Filter out zero-energy events
                valid_energies = total_energy_per_event[total_energy_per_event > 0.0]
                
                # --- FILTER OPTICAL PHOTONS REGISTRATION ---
                # Optical Photons have PDGCode == 0 (or custom ID depending on your Geant4 setup)
                # Filter hits that strike near the downstream sensor bounds
                is_optical = (data["PDGCode"] == 0)
                z_hits = data["positionZ"]
                
                # Downstream Hits Selector
                is_downstream = is_optical & (np.abs(z_hits - cfg["z_sensor_down"]) < 2.5)
                # Upstream Hits Selector (For 2-ended timing)
                is_upstream = is_optical & (np.abs(z_hits - cfg["z_sensor_up"]) < 2.5)
                
                # Time vectors (converted to nanoseconds)
                global_time_dw = data["GlobalTime"][is_downstream]
                local_time_dw = data["LocalTime"][is_downstream]
                
                # --- GRAPH 4: PROMPT PHOTON RECONSTRUCTION ---
                # Pre-calculate target flight times for each LYSO layer
                target_times = np.abs(cfg["z_sensor_down"] - layer_centers) / v_eff
                prompt_counts_per_layer = np.zeros(cfg["num_layers"])
                
                for layer_idx, t_target in enumerate(target_times):
                    t_min = t_target - cfg["prompt_window_ns"]
                    t_max = t_target + cfg["prompt_window_ns"]
                    # Count downstream photons within the prompt flight time window
                    in_window = (local_time_dw >= t_min) & (local_time_dw <= t_max)
                    prompt_counts_per_layer[layer_idx] = np.sum(in_window)
                
                # Normalize prompt profiles by total number of events
                prompt_counts_per_layer /= len(unique_evs)

                # --- GRAPH 5: TWO-ENDED FIBER TIMING ---
                # For each event, retrieve the 10th-percentile (leading edge trigger) hit time
                t_up_events = []
                t_down_events = []
                
                # Efficient vector slice tracking per event
                split_indices = np.cumsum(ev_counts)[:-1]
                ev_local_times = np.split(data["LocalTime"], split_indices)
                ev_z_hits = np.split(data["positionZ"], split_indices)
                ev_pdg = np.split(data["PDGCode"], split_indices)
                
                for ev_idx in range(len(unique_evs)):
                    ev_opt = (ev_pdg[ev_idx] == 0)
                    if not np.any(ev_opt):
                        continue
                        
                    ev_z = ev_z_hits[ev_idx][ev_opt]
                    ev_lt = ev_local_times[ev_idx][ev_opt]
                    
                    # Track coordinates near sensors
                    up_mask = np.abs(ev_z - cfg["z_sensor_up"]) < 2.5
                    dw_mask = np.abs(ev_z - cfg["z_sensor_down"]) < 2.5
                    
                    if np.any(up_mask) and np.any(dw_mask):
                        # Emulate physical leading-edge trigger using 10th percentile
                        t_up_events.append(np.percentile(ev_lt[up_mask], 10))
                        t_down_events.append(np.percentile(ev_lt[dw_mask], 10))
                
                t_up_events = np.array(t_up_events)
                t_down_events = np.array(t_down_events)
                t_two_end = (t_up_events + t_down_events) / 2.0  # Average cancels out longitudinal coordinate
                
                # Populate aggregated storage map
                key = (module, energy)
                if key not in run_summary:
                    run_summary[key] = {}
                    
                run_summary[key] = {
                    "valid_energies": valid_energies,
                    "global_time_dw": global_time_dw,
                    "local_time_dw": local_time_dw,
                    "prompt_counts": prompt_counts_per_layer,
                    "t_two_end": t_two_end,
                }
                
        except Exception as e:
            print(f"[-] Error processing file {filepath}: {str(e)}")
            continue

    # ==========================================
    # 4. PLOTTING SUB-ENGINES
    # ==========================================
    unique_modules = list(set([k[0] for k in run_summary.keys()]))
    
    for mod in unique_modules:
        mod_keys = sorted([k for k in run_summary.keys() if k[0] == mod], key=lambda x: x[1])
        energies = [k[1] for k in mod_keys]
        
        # --- GRAPH 1: ENERGY PERFORMANCE ---
        print(f"[+] Generating Plot 1: Energy Performance for {mod}...")
        mean_energies = [np.mean(run_summary[k]["valid_energies"]) for k in mod_keys]
        std_energies = [np.std(run_summary[k]["valid_energies"]) for k in mod_keys]
        resolution = [std / mean for std, mean in zip(std_energies, mean_energies)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        # Linearity Plot
        ax1.errorbar(energies, mean_energies, yerr=std_energies, fmt='o', color='teal', capsize=4, label='Data')
        slope, intercept = np.polyfit(energies, mean_energies, 1)
        ax1.plot(energies, slope*np.array(energies) + intercept, '--', color='orange', label=f'Fit: {slope:.3f}*E + {intercept:.3f}')
        ax1.set_xlabel("True Beam Energy (GeV)")
        ax1.set_ylabel("Reconstructed Energy (MeV)")
        ax1.set_title("Energy Linearity")
        ax1.legend()
        ax1.grid(True, linestyle=':')
        
        # Resolution Plot
        ax2.scatter(energies, resolution, color='crimson', zorder=3, label='Simulation')
        try:
            popt, _ = curve_fit(fit_resolution, energies, resolution, p0=[0.15, 0.02], bounds=(0, [1.0, 0.2]))
            e_fit = np.linspace(min(energies), max(energies), 100)
            ax2.plot(e_fit, fit_resolution(e_fit, *popt), color='navy', label=f'Fit: {popt[0]*100:.2f}%/$\\sqrt{{E}}$ $\\oplus$ {popt[1]*100:.2f}%')
        except Exception:
            pass
        ax2.set_xlabel("True Beam Energy (GeV)")
        ax2.set_ylabel("Energy Resolution ($\\sigma_E / E$)")
        ax2.set_title("Energy Resolution Curve")
        ax2.legend()
        ax2.grid(True, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, SUBFOLDERS["energy"], f"{mod}_energy_performance.png"), dpi=300)
        plt.close()

        # --- GRAPH 2: GLOBAL TIME VS STRIP STRIKES ---
        print(f"[+] Generating Plot 2: GlobalTime vs Downstream Strikes for {mod}...")
        plt.figure(figsize=(8, 5))
        for k in mod_keys:
            times = run_summary[k]["global_time_dw"]
            if len(times) == 0: continue
            plt.hist(times, bins=150, histtype='step', lw=1.5, label=f"{k[1]} GeV")
        plt.xlabel("GlobalTime (ns)")
        plt.ylabel("Optical Photon Hits (Downstream)")
        plt.title(f"GlobalTime Distribution on Downstream SiPM - {mod}")
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, SUBFOLDERS["globaltime"], f"{mod}_globaltime_vs_strikes.png"), dpi=300)
        plt.close()

        # --- GRAPH 3: LOCAL TIME VS STRIP STRIKES ---
        print(f"[+] Generating Plot 3: LocalTime vs Downstream Strikes for {mod}...")
        plt.figure(figsize=(8, 5))
        for k in mod_keys:
            times = run_summary[k]["local_time_dw"]
            if len(times) == 0: continue
            plt.hist(times, bins=150, histtype='step', lw=1.5, label=f"{k[1]} GeV")
        plt.xlabel("LocalTime (ns)")
        plt.ylabel("Optical Photon Hits (Downstream)")
        plt.title(f"LocalTime Distribution on Downstream SiPM - {mod}")
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, SUBFOLDERS["localtime"], f"{mod}_localtime_vs_strikes.png"), dpi=300)
        plt.close()

        # --- GRAPH 4: PROMPT PHOTON RECONSTRUCTION ---
        print(f"[+] Generating Plot 4: Prompt Photon Reconstruction Profiles for {mod}...")
        plt.figure(figsize=(9, 5.5))
        for k in mod_keys:
            profile = run_summary[k]["prompt_counts"]
            layers = np.arange(1, len(profile) + 1)
            plt.plot(layers, profile, '-o', ms=4, label=f"{k[1]} GeV")
        plt.xlabel("LYSO Layer Number (Upstream -> Downstream)")
        plt.ylabel("Mean Reconstructed Photons / Event")
        plt.title(f"Prompt Photon Longitudinal Shower Profile - {mod}")
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, SUBFOLDERS["prompt"], f"{mod}_prompt_reconstruction_profile.png"), dpi=300)
        plt.close()

        # --- GRAPH 5: TWO-ENDED FIBER TIMING ---
        print(f"[+] Generating Plot 5: Standard Two-Ended Timing Performance for {mod}...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        two_end_resolutions = []
        
        # Plot Histograms for each Energy
        for k in mod_keys:
            t_avg = run_summary[k]["t_two_end"]
            if len(t_avg) < 10:
                two_end_resolutions.append(0.0)
                continue
            
            # Histogram for time average
            counts, bins, _ = ax1.hist(t_avg, bins=100, histtype='step', lw=1.2, label=f"{k[1]} GeV")
            
            # Extract timing resolution (Standard Deviation or Gaussian fit sigma)
            # Standard deviation converted to picoseconds for high-precision plotting
            std_ps = np.std(t_avg) * 1000.0  
            two_end_resolutions.append(std_ps)
            
        ax1.set_xlabel("$T_{coinc} = (t_{up} + t_{down})/2$ (ns)")
        ax1.set_ylabel("Events")
        ax1.set_title("Coincidence Time Spectra")
        ax1.legend()
        ax1.grid(True, linestyle=':')
        
        # Plot Timing Resolution vs Energy
        ax2.plot(energies, two_end_resolutions, 's--', color='darkviolet', ms=6, label='Dual-End Raw STD')
        ax2.set_xlabel("True Beam Energy (GeV)")
        ax2.set_ylabel("Coincidence Timing Resolution $\\sigma_t$ (ps)")
        ax2.set_title("Two-Ended Timing Resolution vs. Energy")
        ax2.legend()
        ax2.grid(True, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, SUBFOLDERS["two_end"], f"{mod}_two_ended_timing.png"), dpi=300)
        plt.close()

    print("[+] Done! All metrics analyzed and plots successfully generated.")

# Run execution block
if __name__ == "__main__":
    # Point this path to the folder containing your run_* directories or sweeps
    target_data_directory = "./" 
    run_analysis(target_data_directory)