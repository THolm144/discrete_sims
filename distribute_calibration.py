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
# 1. SMART PARSER TEMPLATE (Self-correcting coordinate system)
# ─────────────────────────────────────────────────────────────────────────────
PARSER_TEMPLATE = """import os
import uproot
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

dir_name = Path(os.getcwd()).name.lower()

if "dsb1" in dir_name:
    n_index = 1.75  
elif "luagce" in dir_name:
    n_index = 1.84  
else:
    n_index = 1.60  

V_EFF = 299.792 / n_index     
TIMING_CUT_NS = 0.50          
SENSOR_Z_CM = 15.0            # Will be dynamically replaced

calib_dir = Path("./calib_runs")
offsets = []
prompt_photon_counts = []

print(f"Analyzing calibration data using refractive index n = {n_index:.2f} (v_eff = {V_EFF:.2f} mm/ns)...")

if not calib_dir.exists():
    print("ERROR: './calib_runs' directory not found. Did you run run_calib.sh first?")
    exit(1)

for run_path in sorted(calib_dir.glob("offset_*"), key=lambda p: float(p.name.split("_")[1])):
    offset_val = float(run_path.name.split("_")[1])
    
    hit_files = list(run_path.glob("detector_hits_*.root"))
    if not hit_files:
        continue
        
    total_prompt_hits = 0
    # Temporary expected time calculation for initial window filter
    expected_time_ns = (SENSOR_Z_CM * 10.0) / V_EFF 
    
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
            
    offsets.append(offset_val)
    prompt_photon_counts.append(total_prompt_hits)

offsets = np.array(offsets)
prompt_photon_counts = np.array(prompt_photon_counts)

if len(offsets) == 0:
    print("No hit data found inside calib_runs/!")
    exit(1)

if np.all(prompt_photon_counts == 0):
    print("ERROR: All prompt photon counts are zero! The beam likely missed the capillary.")
    exit(1)

# ── SELF-CORRECTING COORDINATE FLIP DETECTION ──
# Physical attenuation dictates that prompt counts must DECREASE as distance INCREASES.
# We test both upstream (+) and downstream (-) distance formulas to find the physical correlation.
d_plus = SENSOR_Z_CM + offsets
d_minus = SENSOR_Z_CM - offsets

corr_plus = np.corrcoef(d_plus, prompt_photon_counts)[0, 1] if len(offsets) > 1 else 0
corr_minus = np.corrcoef(d_minus, prompt_photon_counts)[0, 1] if len(offsets) > 1 else 0

if corr_minus < corr_plus:
    distances_mm = d_minus * 10.0
    direction_str = "Downstream Sensor Detected"
else:
    distances_mm = d_plus * 10.0
    direction_str = "Upstream Sensor Detected"

def exp_decay(d, N0, lambda_eff):
    return N0 * np.exp(-d / lambda_eff)

try:
    popt, _ = curve_fit(exp_decay, distances_mm, prompt_photon_counts, p0=[50000, 250.0])
    N0_fit, lambda_eff = popt
    print("\\n" + "="*55)
    print(f"  CALIBRATION ANALYSIS COMPLETE FOR: {dir_name}")
    print(f"  Direction: {direction_str}")
    print(f"  Effective Prompt Attenuation Length (L_eff): {lambda_eff:.2f} mm")
    print("="*55)
except Exception as e:
    print(f"Fit failed. Raw data array: {prompt_photon_counts}")
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. RUN SCRIPT TEMPLATE (Accepts X & Y targeting offsets)
# ─────────────────────────────────────────────────────────────────────────────
RUN_TEMPLATE = """#!/bin/bash
# Parallel simulation sweeping the Z-axis safely within active boundaries
OFFSETS=({OFFSETS_LIST})

echo "Launching {NUM_RUNS} parallel OpenGATE calibration runs for world: {WORLD_NAME}"
echo "Targeting Capillary coordinate: X = {BEAM_X} cm, Y = {BEAM_Y} cm"

for offset in "${{OFFSETS[@]}}"; do
    OUT_DIR="./calib_runs/offset_${{offset}}"
    mkdir -p "$OUT_DIR"
    
    python3 simulator.py \\
        --world {WORLD_NAME} \\
        --particle opticalphoton \\
        --energy-kev 0.003 \\
        --n 50000 \\
        --threads 1 \\
        --beam-radius 0.0 \\
        --beam-offset "$offset" \\
        --beam-x {BEAM_X} \\
        --beam-y {BEAM_Y} \\
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

    # A. Patch simulator.py to add `--beam-offset`, `--beam-x`, and `--beam-y`
    sim_path = dest_dir / "simulator.py"
    if sim_path.exists():
        with open(sim_path, "r") as f:
            content = f.read()
        
        if "--beam-offset" not in content:
            arg_target = 'return p.parse_args()'
            arg_replacement = (
                'p.add_argument("--beam-offset", type=float, default=None)\n'
                '    p.add_argument("--beam-x", type=float, default=None)\n'
                '    p.add_argument("--beam-y", type=float, default=None)\n'
                '    return p.parse_args()'
            )
            content = content.replace(arg_target, arg_replacement)
            
            cfg_target = 'cfg = {**DEFAULT_BEAM_CONFIG, **getattr(world, "BEAM_CONFIG", {})}'
            cfg_replacement = (
                'cfg = {**DEFAULT_BEAM_CONFIG, **getattr(world, "BEAM_CONFIG", {})}\n'
                '    if args.beam_offset is not None:\n'
                '        cfg["offset_cm"] = args.beam_offset\n'
                '    if args.beam_x is not None or args.beam_y is not None:\n'
                '        tx = args.beam_x if args.beam_x is not None else cfg.get("target_cm", [0,0,0])[0]\n'
                '        ty = args.beam_y if args.beam_y is not None else cfg.get("target_cm", [0,0,0])[1]\n'
                '        tz = cfg.get("target_cm", [0,0,0])[2]\n'
                '        cfg["target_cm"] = [tx, ty, tz]'
            )
            content = content.replace(cfg_target, cfg_replacement)
            
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

    # B. Determine correct physical bounds & beam coordinates
    if "hex" in folder:
        # Hexagonal geometry: target Capillary 0 (E-type) at R = 3.5 mm (0.35 cm)
        beam_x = 0.0
        beam_y = 0.35
    else:
        # Square/standard geometry: target center capillary
        beam_x = 0.0
        beam_y = 0.0

    if "triple" in folder:
        offsets_list = "-10.0 -8.5 -7.0 -5.5 -4.0 -2.5 -1.0 0.5 2.0 3.5 5.0 6.5 8.0 9.5"
        num_runs = 14
        sensor_z = 13.515
    else:
        offsets_list = "-6.0 -5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0"
        num_runs = 13
        sensor_z = 9.165

    # C. Deploy customized run_calib.sh
    run_path = dest_dir / "run_calib.sh"
    with open(run_path, "w") as f:
        f.write(RUN_TEMPLATE.format(
            WORLD_NAME=folder,
            OFFSETS_LIST=offsets_list,
            NUM_RUNS=num_runs,
            BEAM_X=beam_x,
            BEAM_Y=beam_y
        ))
    os.chmod(run_path, 0o755)
    print(f"  [DEPLOYED] run_calib.sh targeting ({beam_x}, {beam_y}) with {num_runs} points")

    # D. Deploy smart extract_prompt_attenuation.py
    parser_path = dest_dir / "extract_prompt_attenuation.py"
    parser_content = PARSER_TEMPLATE.replace("SENSOR_Z_CM = 15.0", f"SENSOR_Z_CM = {sensor_z}")
    with open(parser_path, "w") as f:
        f.write(parser_content)
    print(f"  [DEPLOYED] extract_prompt_attenuation.py (SENSOR_Z_CM set to {sensor_z:.3f} cm)")

print("\nDeployment complete! All 12 worlds (including hex configurations) are safe for running and fitting.")