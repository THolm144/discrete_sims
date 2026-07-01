"""
analyze.py
==========
Parallelized unified batch post-processor. Reads world type from sim_metadata.json,
imports the world module, and distributes individual run directory processing across
available CPU cores using a ProcessPoolExecutor.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import uproot
import numpy as np
import analysis_utils as utils

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Parallel batch analysis dispatcher")
    p.add_argument("--batch-dir", default=None)
    p.add_argument("--world",     default=None,
                   help="World name (scopes auto-discovery; auto-detected if omitted)")
    p.add_argument("--workers",   type=int, default=None,
                   help="Number of parallel worker processes (defaults to run count or available cores)")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# WORLD LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_world(world_name: str, script_dir: Path):
    sys.path.insert(0, str(script_dir / "worlds"))
    try:
        mod = importlib.import_module(world_name)
        print(f"  World module loaded  : {world_name}")
        return mod
    except ModuleNotFoundError:
        print(f"  WARNING: world module '{world_name}' not found — skipping world hooks.")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WORK UNIT FOR ROOT FILES
# ─────────────────────────────────────────────────────────────────────────────

def _process_single_root_file(file_path_str):
    """Isolated, picklable work unit to process a single ROOT file on an independent core."""
    file_path = Path(file_path_str)
    photon_energies = []
    timing_resolutions = []
    
    try:
        with uproot.open(file_path) as file:
            if "Hits" in file:
                tree = file["Hits"]
                if "edep" in tree.keys():
                    edeps = tree["edep"].array(library="np")
                    # Filter for optical photons (< 1e-5 MeV) and convert to eV
                    optical_edeps = edeps[(edeps > 0) & (edeps < 1e-5)] * 1e6
                    photon_energies.extend(optical_edeps.tolist())

                if "time" in tree.keys():
                    times = tree["time"].array(library="np")
                    if len(times) > 1:
                        timing_resolutions.append(float(np.std(times)))
    except Exception as e:
        # Silently log to avoid multi-threaded stdout corruption
        pass

    return photon_energies, timing_resolutions

# ─────────────────────────────────────────────────────────────────────────────
# DATA EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_optical_metrics_parallel(hits_files, max_workers):
    """Distributes fallback parsing of heavy ROOT files across the process pool."""
    all_photon_energies = []
    timing_resolutions = []

    # Convert paths to strings for clean process serialization across complex environments
    file_strs = [str(f) for f in hits_files]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_single_root_file, f_str) for f_str in file_strs]
        for future in as_completed(futures):
            energies, timings = future.result()
            all_photon_energies.extend(energies)
            timing_resolutions.extend(timings)

    avg_energy = np.mean(all_photon_energies) if all_photon_energies else 0.0
    avg_timing = np.mean(timing_resolutions) if timing_resolutions else 0.0

    return avg_timing, avg_energy

# ─────────────────────────────────────────────────────────────────────────────
# REPORT ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_report(batch_dir: Path, meta: dict, results: dict) -> str:
    lines = utils.report_header(meta, batch_dir)

    hits  = results.get("hits",  {})
    exits = results.get("exits", {})
    total_primaries = meta["total_primaries"]
    total_optical   = meta["total_optical"]

    dose_edep  = results.get("dose_edep")
    total_edep = float(dose_edep.sum()) if dose_edep is not None else 0.0

    caps = meta.get("capabilities", {})

    if caps.get("optical", False) or sum(hits.values()) > 0:
        lines += utils.report_optical_section(
            hits, exits, total_optical, total_primaries, total_edep
        )

        timing_res = results.get("timing_res_ps", 0.0)
        avg_energy = results.get("avg_photon_energy_ev", 0.0)
        
        lines += ["", "─" * utils.W, "  CALIBRATION CONSTANTS", "─" * utils.W]
        c_exp = hits.get("Cerenkov", 0) / total_primaries if total_primaries else 0
        scint_lce = (hits.get("Scintillation", 0) / total_optical if total_optical > 0 else 0.0)
        edep_per_prim = total_edep / total_primaries if total_primaries else 0.0
        
        lines += [
            f"  E_dep / primary      : {edep_per_prim:.4f} MeV",
            f"  C_exp (Cer hits/prim): {c_exp:.4f}",
            f"  e_LCE (Scint hits)   : {scint_lce:.6f}",
            f"  Timing resolution    : " + (f"{timing_res:.2f} ps" if timing_res > 0 else "N/A"),
            f"  Avg Photon Energy    : " + (f"{avg_energy:.2f} eV" if avg_energy > 0 else "N/A"),
        ]

    if caps.get("dose", True):
        lines += utils.report_dose_section(results.get("dose_centers"), dose_edep, total_primaries)

    extra = results.get("extra_lines", [])
    if extra:
        lines += ["", "─" * utils.W, "  WORLD-SPECIFIC RESULTS", "─" * utils.W]
        lines += extra

    plots = results.get("plots_saved", [])
    if plots:
        lines += ["", "─" * utils.W, "  PLOTS SAVED", "─" * utils.W]
        lines += [f"  → {p}" for p in plots]

    lines += utils.report_footer()
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    batch_dir = utils.find_batch_dir(script_dir, args.world, args.batch_dir)
    run_dirs  = utils.find_runs(batch_dir)

    num_runs = len(run_dirs)
    print(f"  Batch dir   : {batch_dir}")
    print(f"  Run count   : {num_runs}")

    # Determine processing footprint relative to your 512-core ceiling
    if args.workers:
        max_workers = args.workers
    else:
        # Cap workers at the actual number of files to prevent thread spin-up waste
        max_workers = min(num_runs, os.cpu_count() or 1)
    
    print(f"  Parallel Processing Active: Using {max_workers} worker cores.")

    meta       = utils.load_batch_metadata(run_dirs, args.world)
    world_name = meta["world"]
    world      = load_world(world_name, script_dir)

    # ── Dispatch to world analyze() hook ─────────────────────────────────
    if world and hasattr(world, "analyze"):
        print(f"  Dispatching to parallelized {world_name}.analyze() …")
        # Pass max_workers downstream if your world module accepts it
        try:
            results = world.analyze(batch_dir, run_dirs, meta, utils, max_workers=max_workers)
        except TypeError:
            # Fallback if world script does not accept the max_workers keyword argument yet
            results = world.analyze(batch_dir, run_dirs, meta, utils)
    else:
        print("  No world analyze() hook — running parallel fallback analysis.")
        results = _generic_analyze_parallel(batch_dir, run_dirs, meta, max_workers)

    # ── Standard depth-dose plot ──────────────────────────────────────────
    centers = results.get("dose_centers")
    edep    = results.get("dose_edep")
    if centers is not None and edep is not None:
        plot_path = utils.plot_dose_profile(
            centers, edep, meta["total_primaries"],
            meta["phantom_cm"], world_name, batch_dir,
        )
        results.setdefault("plots_saved", []).append(plot_path.name)

    # ── Report Generation ─────────────────────────────────────────────────
    report = build_report(batch_dir, meta, results)
    print("\n" + report)
    (batch_dir / "batch_analysis.txt").write_text(report)
    print(f"\n  Report → {batch_dir / 'batch_analysis.txt'}")


def _generic_analyze_parallel(batch_dir: Path, run_dirs: list, meta: dict, max_workers: int) -> dict:
    """Fallback parallel tracking parser using Multi-Processing pools."""
    hits_files  = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    exits_files = [d / "optical_exited.root" for d in run_dirs]

    # Check if analysis_utils has internal parallel loops, otherwise execute natively
    hits        = utils.analyse_hits(hits_files)
    exits       = utils.analyse_exits(exits_files)
    centers, edep = utils.load_dose_mhd(run_dirs, meta["phantom_cm"])
    
    timing_res, avg_energy = _extract_optical_metrics_parallel(hits_files, max_workers)

    return {
        "hits":                 hits,
        "exits":                exits,
        "dose_centers":         centers,
        "dose_edep":            edep,
        "timing_res_ps":        timing_res,
        "avg_photon_energy_ev": avg_energy,
    }

if __name__ == "__main__":
    main()