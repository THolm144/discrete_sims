import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze detector hits across varying shield thicknesses.")
    parser.add_argument("--master-dir", required=True, help="Path to the master batch directory containing iron_Xcm subfolders.")
    return parser.parse_args()

def count_hits_in_root(file_path):
    """Safely count the number of entries (hits) in an OpenGATE PhaseSpace root file."""
    try:
        # OpenGATE PhaseSpace actors usually write to a tree named 'PhaseSpace'
        with uproot.open(file_path) as f:
            if "PhaseSpace;1" in f:
                return f["PhaseSpace;1"].num_entries
            elif "OpticalData;1" in f: # Fallback for some optical configs
                return f["OpticalData;1"].num_entries
            else:
                # If tree name is different, grab the first available tree
                keys = f.keys()
                if keys:
                    return f[keys[0]].num_entries
                return 0
    except Exception as e:
        print(f"  [Warning] Could not read {file_path}: {e}")
        return 0

def main():
    args = parse_args()
    master_dir = Path(args.master_dir)
    
    if not master_dir.exists():
        print(f"[Error] Master directory does not exist: {master_dir}")
        return

    # Store data as {thickness_cm: [hits_run0, hits_run1, ...]}
    thickness_data = {}
    
    # Regex to extract the thickness integer/float from the folder name (e.g., "iron_5cm")
    folder_pattern = re.compile(r"iron_([\d\.]+)cm")

    print(f"Scanning master directory: {master_dir}")
    
    for thickness_dir in master_dir.iterdir():
        if not thickness_dir.is_dir():
            continue
            
        match = folder_pattern.search(thickness_dir.name)
        if not match:
            continue
            
        thickness = float(match.group(1))
        hits_per_run = []
        
        # Look into each run_X folder
        for run_dir in thickness_dir.glob("run_*"):
            if not run_dir.is_dir():
                continue
                
            run_total_hits = 0
            # Find all detector hit root files in this run
            hit_files = run_dir.glob("detector_hits_*.root")
            
            for hf in hit_files:
                run_total_hits += count_hits_in_root(hf)
                
            hits_per_run.append(run_total_hits)
            
        if hits_per_run:
            thickness_data[thickness] = hits_per_run

    if not thickness_data:
        print("[Warning] No data found to analyze.")
        return

    # Process data for plotting
    thicknesses = sorted(thickness_data.keys())
    avg_hits = []
    std_hits = []

    print("\n--- Summary of Detector Strikes ---")
    for t in thicknesses:
        runs = thickness_data[t]
        avg = np.mean(runs)
        std = np.std(runs)
        avg_hits.append(avg)
        std_hits.append(std)
        print(f"Iron Thickness: {t:5.1f} cm | Avg Hits: {avg:.1f} ± {std:.1f} (across {len(runs)} runs)")

    # ── Plotting ─────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    
    # Use a bar plot (histogram-style) with error bars
    plt.bar(thicknesses, avg_hits, width=0.8, yerr=std_hits, capsize=5, 
            color='#3498db', edgecolor='black', alpha=0.8)
    
    # Alternatively, an overlaid line plot helps show the trend clearly
    plt.plot(thicknesses, avg_hits, color='#e74c3c', marker='o', linestyle='dashed', linewidth=2)

    plt.title('Average Detector Strikes vs. Iron Shield Thickness', fontsize=14, fontweight='bold')
    plt.xlabel('Iron Shield Thickness (cm)', fontsize=12)
    plt.ylabel('Average Number of Detector Hits', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot in the master batch directory
    plot_path = master_dir / "thickness_hits_histogram.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Histogram saved to: {plot_path}")

if __name__ == "__main__":
    main()