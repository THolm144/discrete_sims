"""
analyze.py
==========
Unified batch post-processor.  Reads world type from sim_metadata.json,
imports the world module, and calls world.analyze() to get structured results.
analyze.py owns formatting; worlds own data extraction logic.

World analyze() contract
------------------------
    def analyze(batch_dir, run_dirs, meta, utils) -> dict

    Parameters
    ----------
    batch_dir   Path to the <energy>keV_<timestamp> directory
    run_dirs    List of Path objects for each run_N subdirectory
    meta        Dict returned by utils.load_batch_metadata()
    utils       The analysis_utils module (passed to avoid circular imports)

    Returns
    -------
    A dict with any subset of these keys:
        hits                 dict  {process: count}  — photons reaching detectors
        exits                dict  {process: count}  — photons leaving target volume
        dose_centers         np.ndarray | None       — depth bin centers in cm
        dose_edep            np.ndarray | None       — deposited energy per bin (MeV)
        timing_res_ps        float                   — timing resolution in ps (0 = N/A)
        avg_photon_energy_ev float                   — average detected optical photon energy (0 = N/A)
        extra_lines          list[str]               — world-specific report lines
        plots_saved          list[str]               — filenames of any extra plots saved

Usage:
    python3 analyze.py
    python3 analyze.py --world radi_cal
    python3 analyze.py --batch-dir runs/radi_cal/50000000keV_20250101_120000
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import uproot
import numpy as np
import analysis_utils as utils

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Unified batch analysis dispatcher")
    p.add_argument("--batch-dir", default=None)
    p.add_argument("--world",     default=None,
                   help="World name (scopes auto-discovery; auto-detected if omitted)")
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
# DATA EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_optical_metrics(hits_files):
    """Fallback helper to extract timing and energy directly from ROOT files."""
    all_photon_energies = []
    timing_resolutions = []

    for file_path in hits_files:
        try:
            with uproot.open(file_path) as file:
                if "Hits" not in file:
                    continue
                tree = file["Hits"]
                
                # Extract energy
                if "edep" in tree.keys():
                    edeps = tree["edep"].array()
                    # Filter for optical photons (< 1e-5 MeV) and convert to eV
                    optical_edeps = [e * 1e6 for e in edeps if 0 < e < 1e-5] 
                    all_photon_energies.extend(optical_edeps)

                # Extract timing
                if "time" in tree.keys():
                    times = tree["time"].array()
                    if len(times) > 1:
                        timing_resolutions.append(np.std(times))
        except Exception as e:
            print(f"  WARNING: Could not parse ROOT file {file_path}: {e}")

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

    # ── Optical section ───────────────────────────────────────────────────
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

    # ── Dose section ──────────────────────────────────────────────────────
    if caps.get("dose", True):
        lines += utils.report_dose_section(results.get("dose_centers"), dose_edep, total_primaries)

    # ── World-specific extra lines ────────────────────────────────────────
    extra = results.get("extra_lines", [])
    if extra:
        lines += ["", "─" * utils.W, "  WORLD-SPECIFIC RESULTS", "─" * utils.W]
        lines += extra

    # ── Extra plots saved ─────────────────────────────────────────────────
    plots = results.get("plots_saved", [])
    if plots:
        lines += ["", "─" * utils.W, "  PLOTS SAVED", "─" * utils.W]
        lines += [f"  → {p}" for p in plots]

    lines += utils.report_footer()
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    batch_dir = utils.find_batch_dir(script_dir, args.world, args.batch_dir)
    run_dirs  = utils.find_runs(batch_dir)

    print(f"  Batch dir   : {batch_dir}")
    print(f"  Run count   : {len(run_dirs)}")

    meta       = utils.load_batch_metadata(run_dirs, args.world)
    world_name = meta["world"]
    world      = load_world(world_name, script_dir)

    # ── Dispatch to world analyze() hook ─────────────────────────────────
    if world and hasattr(world, "analyze"):
        print(f"  Dispatching to {world_name}.analyze() …")
        results = world.analyze(batch_dir, run_dirs, meta, utils)
    else:
        print("  No world analyze() hook — running generic analysis.")
        results = _generic_analyze(batch_dir, run_dirs, meta)

    # ── Standard depth-dose plot ──────────────────────────────────────────
    centers = results.get("dose_centers")
    edep    = results.get("dose_edep")
    if centers is not None and edep is not None:
        plot_path = utils.plot_dose_profile(
            centers, edep, meta["total_primaries"],
            meta["phantom_cm"], world_name, batch_dir,
        )
        results.setdefault("plots_saved", []).append(plot_path.name)

    # ── Report ────────────────────────────────────────────────────────────
    report = build_report(batch_dir, meta, results)
    print("\n" + report)
    (batch_dir / "batch_analysis.txt").write_text(report)
    print(f"\n  Report → {batch_dir / 'batch_analysis.txt'}")

def _generic_analyze(batch_dir: Path, run_dirs: list, meta: dict) -> dict:
    """Fallback: optical hits + exits + dose + extracted uproot metrics."""
    hits_files  = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    exits_files = [d / "optical_exited.root" for d in run_dirs]

    hits        = utils.analyse_hits(hits_files)
    exits       = utils.analyse_exits(exits_files)
    centers, edep = utils.load_dose_mhd(run_dirs, meta["phantom_cm"])
    
    timing_res, avg_energy = _extract_optical_metrics(hits_files)

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