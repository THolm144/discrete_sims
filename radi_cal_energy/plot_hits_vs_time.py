import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import uproot

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================
TIME_MODE = "LocalTime"
DEFAULT_BINS = 500
# =============================================================================

def extract_times_from_channels(run_dirs: list, channel_indices: list, time_branch: str) -> np.ndarray:
    all_times = []
    
    for run_dir in run_dirs:
        for idx in channel_indices:
            fpath = run_dir / f"detector_hits_{idx}.root"
            if not fpath.exists():
                continue
                
            try:
                with uproot.open(fpath) as root_file:
                    tree_key = f"detector_hits_{idx}"
                    if tree_key in root_file:
                        tree = root_file[tree_key]
                        
                        if time_branch not in tree.keys():
                            print(f"  [Warning]: Branch '{time_branch}' not found in {fpath.name}. Skipping.")
                            continue
                            
                        times = tree[time_branch].array(library="np")
                        if len(times) > 0:
                            all_times.append(times)
            except Exception as e:
                print(f"  [Error reading {fpath.name}]: {e}")
                continue

    if all_times:
        return np.concatenate(all_times)
    return np.array([])

def generate_histogram(times: np.ndarray, bins: int, color: str, edge_color: str,
                       title: str, out_path: Path, time_label: str, vline_val: float = None, vline_label: str = None):
    if len(times) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(times, bins=bins, color=color, alpha=0.75, edgecolor=edge_color, linewidth=0.5)
    
    if vline_val is not None:
        ax.axvline(vline_val, color="#d32f2f", linestyle="--", linewidth=2,
                   label=f"{vline_label}: {vline_val:.2f} ns")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    ax.set_title(title)
    ax.set_xlabel(f"{time_label} (ns)")
    ax.set_ylabel("Hit Count")
    ax.grid(True, linestyle=":", alpha=0.6, color="#cccccc")
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Generated profile -> {out_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Plot E-type and T-type SiPM hit structures.")
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

    print(f"Processing split channels using clock tracking: '{TIME_MODE}' with {args.bins} bins...")

    # Channels present:
    #   Upstream   T-type: front_0, front_1  → indices 0, 1
    #   Downstream T-type: back_0,  back_1   → indices 4, 5
    #   Downstream E-type: back_2,  back_3   → indices 6, 7
    # Upstream E-type (front_2, front_3) not used — no readout on that end.
    up_t_times = extract_times_from_channels(run_dirs, [0, 1], TIME_MODE)
    dn_t_times = extract_times_from_channels(run_dirs, [4, 5], TIME_MODE)
    dn_e_times = extract_times_from_channels(run_dirs, [6, 7], TIME_MODE)

    avg_up_t = np.mean(up_t_times) if len(up_t_times) > 0 else None
    avg_dn_t = np.mean(dn_t_times) if len(dn_t_times) > 0 else None

    print("\n--- Statistics Summary ---")
    print(f"  Upstream T-type:   {len(up_t_times):10,} hits | Avg {TIME_MODE}: {f'{avg_up_t:.3f} ns' if avg_up_t else 'N/A'}")
    print(f"  Downstream T-type: {len(dn_t_times):10,} hits | Avg {TIME_MODE}: {f'{avg_dn_t:.3f} ns' if avg_dn_t else 'N/A'}")
    print(f"  Downstream E-type: {len(dn_e_times):10,} hits")

    print("\nRendering plots...")

    generate_histogram(up_t_times, args.bins, "#1565c0", "#0d47a1",
                       f"Upstream T-Type SiPM Hits vs. {TIME_MODE} (Channels 0 & 1)",
                       batch_dir / "upstream_ttype_time_profile.png", TIME_MODE)

    generate_histogram(dn_t_times, args.bins, "#26a69a", "#00695c",
                       f"Downstream T-Type SiPM Hits vs. {TIME_MODE} (Channels 4 & 5)",
                       batch_dir / "downstream_ttype_time_profile.png", TIME_MODE)

    generate_histogram(dn_e_times, args.bins, "#00bcd4", "#00838f",
                       f"Downstream E-Type SiPM Hits vs. {TIME_MODE} (Channels 6 & 7)",
                       batch_dir / "downstream_etype_time_profile.png", TIME_MODE,
                       vline_val=avg_dn_t, vline_label="Avg Downstream T-Type Time")

if __name__ == "__main__":
    main()