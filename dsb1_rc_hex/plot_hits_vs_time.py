import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import uproot
from collections import defaultdict

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================
TIME_MODE = "GlobalTime"
DEFAULT_BINS = 500
# =============================================================================

def scan_and_extract_all_channels(run_dirs: list, time_branch: str) -> dict:
    """
    Scans directories using a glob pattern exactly like the 3D plotter.
    Discovers all channel files dynamically and extracts hit times.
    Returns a dictionary mapping channel_id (str) -> np.ndarray of times.
    """
    channel_times = defaultdict(list)
    
    for run_dir in run_dirs:
        # Grab every single detector hit file present, exactly like the plotter
        for fpath in run_dir.glob("detector_hits*.root"):
            # Extract the unique channel suffix from the filename
            # e.g., "detector_hits_5.root" -> "5"
            # e.g., "detector_hits_sipm_back_0.root" -> "sipm_back_0"
            stem = fpath.stem
            channel_id = stem.replace("detector_hits_", "")
            if not channel_id or channel_id == stem:
                channel_id = "unknown"
                
            try:
                with uproot.open(fpath) as root_file:
                    # Find the TTree key inside the file dynamically
                    for key in root_file.keys():
                        clean_key = key.split(";")[0]  # Strip root cycle numbers
                        if "detector_hits" in clean_key:
                            tree = root_file[key]
                            
                            if time_branch not in tree.keys():
                                continue
                                
                            times = tree[time_branch].array(library="np")
                            if len(times) > 0:
                                channel_times[channel_id].append(times)
                            break # Found the hit tree, move to next file
            except Exception as e:
                print(f"  [Error reading {fpath.name}]: {e}")
                continue

    # Concatenate lists of arrays into single flat numpy arrays
    flat_channel_data = {
        cid: np.concatenate(arrays) 
        for cid, arrays in channel_times.items() 
        if len(arrays) > 0
    }
    return flat_channel_data

def generate_histogram(times: np.ndarray, bins: int, color: str, edge_color: str,
                       title: str, out_path: Path, time_label: str, vline_val: float = None, vline_label: str = None):
    if len(times) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Add log=True at end to switch the y-axis to a logarithmic scale
    ax.hist(times, bins=bins, color=color, alpha=0.75, edgecolor=edge_color, linewidth=0.5)
    
    if vline_val is not None:
        ax.axvline(vline_val, color="#d32f2f", linestyle="--", linewidth=2,
                   label=f"{vline_label}: {vline_val:.2f} ns")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    ax.set_title(title)
    ax.set_xlabel(f"{time_label} (ns)")
    
    # Updated label to explicitly state it is a log scale
    ax.set_ylabel("Hit Count (Log Scale)")
    
    # Grid lines look better on log scales if they follow major ticks
    ax.grid(True, which="both", linestyle=":", alpha=0.4, color="#cccccc")
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Generated profile -> {out_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Plot E-type and T-type SiPM hit structures dynamically.")
    parser.add_argument("--batch-dir", required=True, type=str, help="Path to the batch output directory")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of histogram bins")
    args = parser.parse_args()

    if TIME_MODE not in ["GlobalTime", "LocalTime"]:
        print(f"Error: TIME_MODE must be 'GlobalTime' or 'LocalTime'. Current: {TIME_MODE}")
        return

    batch_dir = Path(args.batch_dir)
    run_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

    if not run_dirs:
        print(f"Error: No run subdirectories found in {batch_dir}")
        return

    print(f"Processing all discovered channels using tracking: '{TIME_MODE}' with {args.bins} bins...")

    # Extract all hit data across all runs using the dynamic plotter methodology
    discovered_channels = scan_and_extract_all_channels(run_dirs, TIME_MODE)

    # Buckets to hold our target classifications
    up_t_times = []
    dn_t_times = []
    dn_e_times = []
    
    print("\n--- Discovered Channel Breakdown ---")
    for cid, times in sorted(discovered_channels.items()):
        hit_count = len(times)
        avg_time = np.mean(times) if hit_count > 0 else 0.0
        print(f"  Channel ID '{cid}': {hit_count:10,} total hits found | Avg: {avg_time:.3f} ns")

        # Flexible bucketing logic supporting integer IDs or descriptive string names
        if cid in ["0", "1"] or "front_0" in cid or "front_1" in cid:
            up_t_times.append(times)
        elif cid in ["4", "5"] or "back_0" in cid or "back_1" in cid:
            dn_t_times.append(times)
        elif cid in ["6", "7"] or "back_2" in cid or "back_3" in cid:
            dn_e_times.append(times)
        else:
            # Check if OpenGate reassigned your E-type channels to indices 2 and 3!
            if cid in ["2", "3"]:
                print(f"  [Notice]: Channel {cid} caught. If these are your back E-types, verify your classification logic below.")
                dn_e_times.append(times)

    # Recombine individual channel arrays into consolidated groups
    up_t_flat = np.concatenate(up_t_times) if up_t_times else np.array([])
    dn_t_flat = np.concatenate(dn_t_times) if dn_t_times else np.array([])
    dn_e_flat = np.concatenate(dn_e_times) if dn_e_times else np.array([])

    avg_up_t = np.mean(up_t_flat) if len(up_t_flat) > 0 else None
    avg_dn_t = np.mean(dn_t_flat) if len(dn_t_flat) > 0 else None

    print("\n--- Consolidated Statistics Summary ---")
    print(f"  Upstream T-type Group:   {len(up_t_flat):10,} hits | Avg {TIME_MODE}: {f'{avg_up_t:.3f} ns' if avg_up_t else 'N/A'}")
    print(f"  Downstream T-type Group: {len(dn_t_flat):10,} hits | Avg {TIME_MODE}: {f'{avg_dn_t:.3f} ns' if avg_dn_t else 'N/A'}")
    print(f"  Downstream E-type Group: {len(dn_e_flat):10,} hits")

    print("\nRendering plots...")

    if len(up_t_flat) > 0:
        generate_histogram(up_t_flat, args.bins, "#1565c0", "#0d47a1",
                           f"Upstream T-Type Group vs. {TIME_MODE}",
                           batch_dir / "upstream_ttype_time_profile.png", TIME_MODE)

    if len(dn_t_flat) > 0:
        generate_histogram(dn_t_flat, args.bins, "#26a69a", "#00695c",
                           f"Downstream T-Type Group vs. {TIME_MODE}",
                           batch_dir / "downstream_ttype_time_profile.png", TIME_MODE)

    if len(dn_e_flat) > 0:
        generate_histogram(dn_e_flat, args.bins, "#00bcd4", "#00838f",
                           f"Downstream E-Type Group vs. {TIME_MODE}",
                           batch_dir / "downstream_etype_time_profile.png", TIME_MODE,
                           vline_val=avg_dn_t, vline_label="Avg Downstream T-Type Time")

if __name__ == "__main__":
    main()