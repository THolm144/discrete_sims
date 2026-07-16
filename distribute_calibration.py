#!/usr/bin/env python3
"""
distribute_calibration.py
=========================
Generates an optimized parallel calibration matrix queue for Discovery.
Configured for exactly 20 concurrent execution cores.
"""

import sys
import importlib.util
from pathlib import Path
import numpy as np

SWEEP_STEPS = 11            # Z-sweep resolution points
CALIB_ENERGY_KEV = 50000000  # Default calibration beam energy (50 GeV)


def discover_worlds():
    current_dir = Path(".")
    return sorted([
        p for p in current_dir.glob("**/worlds/*.py")
        if "__pycache__" not in p.parts
    ])


def load_world_metadata(filepath):
    path = Path(filepath)
    parent_dir = str(path.parent)
    sys.path.insert(0, parent_dir)
    try:
        spec = importlib.util.spec_from_file_location("temp_world_module", path)
        world = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(world)
        
        calor_thick_mm = getattr(world, "_CALOR_THICK_MM")
        calor_thick_cm = calor_thick_mm / 10.0
        half_length_cm = calor_thick_cm / 2.0
        sweep_limit_cm = round(half_length_cm * 0.8 * 2) / 2.0
        
        sipm_z_mm = getattr(world, "_SIPM_Z_MM")
        sensor_z_cm = sipm_z_mm / 10.0

        cap_positions = getattr(world, "_CAP_POSITIONS_MM")
        active_idx = 0 if len(cap_positions) == 6 else 2
        
        raw_x, raw_y = cap_positions[active_idx]
        active_x_cm = raw_x / 10.0
        active_y_cm = raw_y / 10.0

        return {
            "path_str": str(path),
            "world_dir": str(path.parent.parent),
            "name": path.stem,
            "x": active_x_cm,
            "y": active_y_cm,
            "sweep_limit": sweep_limit_cm,
            "sensor_z_cm": sensor_z_cm,
        }
    except Exception as e:
        return None
    finally:
        sys.path.pop(0)


def generate_slurm_runner(worlds):
    bash_content = []
    bash_content.append(f"""#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_calibration_discovery.sh — Parallel Multi-World Calibration (20 Cores)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=gate_calib_20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --partition=express
#SBATCH --output=calib_pool_%j.log

ROOT_DIR=$(pwd)
JOB_FILE="${{ROOT_DIR}}/calib_job_list.txt"
> "$JOB_FILE"

MAX_CONCURRENT_SIMS=20
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================================================"
echo " Preparing Execution Matrix for Multi-World Calibration Queue"
echo "========================================================================"
""")

    for w in worlds:
        z_vals = np.linspace(-w['sweep_limit'], w['sweep_limit'], SWEEP_STEPS)
        z_vals_str = " ".join([f"{z:.2f}" for z in z_vals])
        
        bash_content.append(f"""# --- Queue setup for {w['name']} ---
WORLD_DIR="${{ROOT_DIR}}/{w['world_dir']}"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${{WORLD_DIR}}/runs/calib_${{TIMESTAMP}}"
    mkdir -p "${{MASTER_BATCH_DIR}}/logs"

    Z_SWEEP=({z_vals_str})
    for Z_VAL in "${{Z_SWEEP[@]}}"; do
        RUN_OUT_DIR="${{MASTER_BATCH_DIR}}/z_${{Z_VAL}}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${{MASTER_BATCH_DIR}}/logs/z_${{Z_VAL}}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${{WORLD_DIR}} {w['path_str']} {w['x']:.5f} {w['y']:.5f} ${{Z_VAL}} ${{RUN_OUT_DIR}} ${{LOG_FILE}} {w['sensor_z_cm']:.5f} {w['name']}" >> "$JOB_FILE"
    done
fi
""")

    bash_content.append("""
TOTAL_JOBS=$(wc -l < "$JOB_FILE")
echo " [✓] Generated ${TOTAL_JOBS} calibration runs inside calib_job_list.txt."
echo "------------------------------------------------------------------------"
echo " [+] Handing execution queue to xargs pool (Concurrency = 20 Cores)..."

(
    sleep 2 
    while [ -f "$JOB_FILE" ] || [ $(pgrep -f "simulator.py" | wc -l) -gt 0 ]; do
        COMPLETED_JOBS=$(find ${ROOT_DIR}/*/runs/calib_${TIMESTAMP}/logs -name "*.log" 2>/dev/null | wc -l)
        ACTIVE_CORES=$(pgrep -f "simulator.py" | wc -l)
        printf "\\r     -> Queue Progress: %4d / %d completed | [%2d Cores Occupied]" "$COMPLETED_JOBS" "$TOTAL_JOBS" "$ACTIVE_CORES"
        sleep 1
    done
) &
TRACKER_PID=$!

# Execute tasks. We pull exactly 9 arguments per row.
xargs -P "$MAX_CONCURRENT_SIMS" -n 9 bash -c '
    WORLD_DIR="$0"
    WORLD_PATH="$1"
    BEAM_X="$2"
    BEAM_Y="$3"
    BEAM_Z="$4"
    OUT_DIR="$5"
    LOG_FILE="$6"
    SENSOR_Z="$7"
    WORLD_NAME="$8"

    # 1. Execute OpenGATE simulation targeting the specific capillary fiber path
    python3 "${WORLD_DIR}/simulator.py" \\
        --world "$WORLD_PATH" \\
        --particle "e-" \\
        --energy-kev """ + str(CALIB_ENERGY_KEV) + """ \\
        --n 24 \\
        --threads 1 \\
        --beam-radius 0.01 \\
        --optical "on" \\
        --cherenkov "off" \\
        --hits-optical-only on \\
        --physics-list "QGSP_BERT_EMV" \\
        --beam_x "$BEAM_X" \\
        --beam_y "$BEAM_Y" \\
        --beam_z "$BEAM_Z" \\
        --run-id 0 \\
        --output-dir "$OUT_DIR" > "$LOG_FILE" 2>&1

    # 2. Immediately execute analysis within the isolated thread environment
    if [ $? -eq 0 ]; then
        python3 extract_prompt_attenuation.py \\
            --run_dir "$OUT_DIR" \\
            --z_offset "$BEAM_Z" \\
            --sensor_z "$SENSOR_Z" >> "$LOG_FILE" 2>&1
    else
        echo "[-] Run failed for ${WORLD_NAME} at Z=${BEAM_Z}" >> "$LOG_FILE"
    fi
' < "$JOB_FILE"

rm -f "$JOB_FILE"
wait $TRACKER_PID 2>/dev/null

echo ""
echo "========================================================================"
echo " [✓] ALL CALIBRATION SIMULATIONS AND ANALYSIS PILES COMPLETE."
echo "========================================================================"
""")

    output_path = Path("run_calibration_discovery.sh")
    output_path.write_text("\n".join(bash_content))
    output_path.chmod(0o755)
    print(f"[+] Multi-world calibration runner script created: '{output_path}'")


if __name__ == "__main__":
    worlds = discover_worlds()
    meta_list = []
    for w in worlds:
        m = load_world_metadata(w)
        if m:
            meta_list.append(m)
    generate_slurm_runner(meta_list)