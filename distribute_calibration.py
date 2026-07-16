#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# The 12 world folders in your discrete_sims directory
WORLD_FOLDERS = [
    "dsb1_radi_cal_energy", "dsb1_radi_cal_triple", "dsb1_rc_hex", "dsb1_rc_hex_triple",
    "luagce_radi_cal_energy", "luagce_radi_cal_triple", "luagce_rc_hex", "luagce_rc_hex_triple",
    "radi_cal_energy", "radi_cal_triple", "rc_hex", "rc_hex_triple"
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. SMART PARSER TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
# This script is smart: when copied, it reads its current directory name
# to determine the correct refractive index (n) for the material.
PARSER_TEMPLATE = """import os
import uproot
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

# 1. Auto-detect world properties based on current directory name
dir_name = Path(os.getcwd()).name.lower()

if "dsb1" in dir_name:
    n_index = 1.75  # Refractive index of DSB1
elif "luagce" in dir_name:
    n_index = 1.84  # Refractive index of LuAG:Ce
else:
    n_index = 1.60  # Default BCF-92 fiber cladding/core average

V_EFF = 299.792 / n_index     # Effective speed in medium (mm/ns)
TIMING_CUT_NS = 0.50          # Your chosen ToF prompt window (ns)
SENSOR_Z_CM = 15.0            # Physical Z location of downstream SiPM (cm)

calib_dir = Path("./calib_runs")
distances_mm = []
prompt_photon_counts = []

print(f"Analyzing calibration data using refractive index n = {n_index:.2f} (v_eff = {V_EFF:.2f} mm/ns)...")

if not calib_dir.exists():
    print("ERROR: './calib_runs' directory not found. Did you run run_calib.sh first?")
    exit(1)

for run_path in sorted(calib_dir.glob("offset_*"), key=lambda p: float(p.name.split("_")[1])):
    offset_val = float(run_path.name.split("_")[1])
    dist_cm = SENSOR_Z_CM + offset_val  
    dist_mm = dist_cm * 10.0
    
    hit_files = list(run_path.glob("detector_hits_*.root"))
    if not hit_files:
        continue
        
    total_prompt_hits = 0
    expected_time_ns = dist_mm / V_EFF
    
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k), None)
                if not tk: continue
                lt = f[tk]["LocalTime"].array(library="np")
                time_diffs = np.abs(lt - expected_time_ns)
                total_prompt_hits += np.sum(time_diffs <= TIMING_CUT_NS)
        except Exception as e:
            print(f"  Warning: Failed to parse {fpath.name} ({e})")
            
    distances_mm.append(dist_mm)
    prompt_photon_counts.append(total_prompt_hits)

distances_mm = np.array(distances_mm)
prompt_photon_counts = np.array(prompt_photon_counts)

if len(distances_mm) == 0:
    print("No hit data found inside calib_runs/!")
    exit(1)

def exp_decay(d, N0, lambda_eff):
    return N0 * np.exp(-d / lambda_eff)

try:
    popt, _ = curve_fit(exp_decay, distances_mm, prompt_photon_counts, p0=[50000, 250.0])
    N0_fit, lambda_eff = popt
    print("\\n" + "="*55)
    print(f"  CALIBRATION ANALYSIS COMPLETE FOR: {dir_name}")
    print(f"  Effective Prompt Attenuation Length (L_eff): {lambda_eff:.2f} mm")
    print("="*55)
except Exception as e:
    print(f"Fit failed. Raw data array: {prompt_photon_counts}")
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. RUN SCRIPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
RUN_TEMPLATE = """#!/bin/bash
# Parallel simulation sweeping the Z-axis
OFFSETS=(-14.0 -12.5 -11.0 -9.5 -8.0 -6.5 -5.0 -3.5 -2.0 -0.5 1.0 2.5 4.0 5.5 7.0 8.5 10.0 11.5 13.0 14.5)

echo "Launching 20 parallel OpenGATE calibration runs for world: {WORLD_NAME}"

for offset in "${{OFFSETS[@]}}"; do
    OUT_DIR="./calib_runs/offset_${{offset}}"
    mkdir -p "$OUT_DIR"
    
    python simulator.py \\
        --world {WORLD_NAME} \\
        --particle opticalphoton \\
        --energy-kev 0.000003 \\
        --n 50000 \\
        --threads 1 \\
        --beam-radius 0.0 \\
        --beam-offset "$offset" \\
        --optical on \\
        --hits-optical-only on \\
        --output-dir "$OUT_DIR" > "${{OUT_DIR}}/sim.log" 2>&1 &
done

wait
echo "All calibration runs complete!"
"""

# ─────────────────────────────────────────────────────────────────────────────
# 3. AUTOMATED DEPLOYMENT & CODE PATCHING
# ─────────────────────────────────────────────────────────────────────────────
for folder in WORLD_FOLDERS:
    dest_dir = Path(folder)
    if not dest_dir.exists():
        print(f"Skipping {folder} (directory not found)")
        continue
        
    print(f"Processing {folder}...")

    # A. Patch simulator.py to add `--beam-offset`
    sim_path = dest_dir / "simulator.py"
    if sim_path.exists():
        with open(sim_path, "r") as f:
            content = f.read()
        
        # Inject the command line argument if it doesn't already exist
        if "--beam-offset" not in content:
            # Inject into parse_args()
            arg_target = 'return p.parse_args()'
            arg_replacement = 'p.add_argument("--beam-offset", type=float, default=None)\n    return p.parse_args()'
            content = content.replace(arg_target, arg_replacement)
            
            # Inject into resolve_beam_config()
            cfg_target = 'cfg = {**DEFAULT_BEAM_CONFIG, **getattr(world, "BEAM_CONFIG", {})}'
            cfg_replacement = 'cfg = {**DEFAULT_BEAM_CONFIG, **getattr(world, "BEAM_CONFIG", {})}\n    if args.beam_offset is not None:\n        cfg["offset_cm"] = args.beam_offset'
            content = content.replace(cfg_target, cfg_replacement)
            
            # Update function call signature to pass args
            def_target = 'def resolve_beam_config(world) -> dict:'
            def_replacement = 'def resolve_beam_config(world, args) -> dict:'
            content = content.replace(def_target, def_replacement)
            
            call_target = 'beam_cfg           = resolve_beam_config(world)'
            call_replacement = 'beam_cfg           = resolve_beam_config(world, args)'
            content = content.replace(call_target, call_replacement)
            
            with open(sim_path, "w") as f:
                f.write(content)
            print("  [PATCHED] simulator.py")
        else:
            print("  [SKIPPED] simulator.py (already patched)")
    else:
        print(f"  [ERROR] simulator.py not found in {folder}!")

    # B. Deploy customized run_calib.sh
    run_path = dest_dir / "run_calib.sh"
    with open(run_path, "w") as f:
        f.write(RUN_TEMPLATE.format(WORLD_NAME=folder))
    os.chmod(run_path, 0o755)
    print("  [DEPLOYED] run_calib.sh")

    # C. Deploy smart extract_prompt_attenuation.py
    parser_path = dest_dir / "extract_prompt_attenuation.py"
    with open(parser_path, "w") as f:
        f.write(PARSER_TEMPLATE)
    print("  [DEPLOYED] extract_prompt_attenuation.py")

print("\nDeployment complete! All 12 worlds are fully equipped for calibration.")