#!/usr/bin/env python3
"""
distribute_calibration.py
=========================
Master Multi-World Orchestrator.
Discovers all nested world designs, extracts their unique geometries,
and generates a dynamic master shell script to sweep and calibrate them all.
"""

import os
import sys
import importlib.util
from pathlib import Path

# --- Configuration ---
SWEEP_STEPS = 11  # Number of calibration points per world sweep


def discover_worlds():
    """Scans all subdirectories for .py files inside folders named 'worlds'."""
    current_dir = Path(".")
    world_paths = sorted([
        p for p in current_dir.glob("**/worlds/*.py")
        if "__pycache__" not in p.parts
    ])
    return world_paths


def load_world_metadata(filepath):
    """Dynamically imports a world file to extract design parameters with fallbacks."""
    path = Path(filepath)
    
    # Safely inject path context to resolve local imports within the world's directory
    parent_dir = str(path.parent)
    sys.path.insert(0, parent_dir)
    
    try:
        spec = importlib.util.spec_from_file_location("temp_world_module", path)
        world = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(world)
    except Exception as e:
        print(f"[-] Warning: Failed to import {path}. Skipping. Error: {e}")
        return None
    finally:
        sys.path.pop(0)

    try:
        # 1. Extract physical boundaries
        calor_thick_mm = getattr(world, "_CALOR_THICK_MM")
        calor_thick_cm = calor_thick_mm / 10.0
        half_length_cm = calor_thick_cm / 2.0
        
        # Scale sweep limit safely to 80% of active half-length
        sweep_limit_cm = round(half_length_cm * 0.8 * 2) / 2.0
        
        # 2. Extract Sensor position
        sipm_z_mm = getattr(world, "_SIPM_Z_MM")
        sensor_z_cm = sipm_z_mm / 10.0

        # 3. Resolve active calibration coordinates (First E-type capillary)
        cap_positions = getattr(world, "_CAP_POSITIONS_MM")
        
        if hasattr(world, "_E_TYPE_INDICES"):
            e_type_indices = getattr(world, "_E_TYPE_INDICES")
            active_idx = sorted(list(e_type_indices))[0]
        else:
            # Fallback based on capillary count: Hex structures default to index 0, Square to index 2
            active_idx = 0 if len(cap_positions) == 6 else 2
        
        raw_x, raw_y = cap_positions[active_idx]
        active_x_cm = raw_x / 10.0
        active_y_cm = raw_y / 10.0

        is_hex = len(cap_positions) == 6
        geometry_type = "Hexagonal" if is_hex else "Square"

        return {
            "path_str": str(path),
            "name": path.stem,
            "type": geometry_type,
            "x": active_x_cm,
            "y": active_y_cm,
            "calor_thick_cm": calor_thick_cm,
            "sweep_limit": sweep_limit_cm,
            "sensor_z_cm": sensor_z_cm,
            "active_index": active_idx
        }
    except AttributeError as e:
        print(f"[-] Warning: Skipping {path.name} due to missing standard parameters: {e}")
        return None


def generate_config_string(meta):
    """Generates the content of a python configuration block for a specific world."""
    return f"""# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "{meta['path_str']}"
WORLD_NAME     = "{meta['name']}"
GEOMETRY_TYPE  = "{meta['type']}"
ACTIVE_INDEX   = {meta['active_index']}

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = {meta['x']:.5f}
BEAM_Y_CM      = {meta['y']:.5f}

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = {meta['calor_thick_cm']:.5f}
SENSOR_Z_CM    = {meta['sensor_z_cm']:.5f}
"""


def write_master_bash_runner(worlds_meta):
    """Generates run_all_calibs.sh with individualized configuration steps for every world."""
    import numpy as np

    bash_content = []
    bash_content.append("#!/bin/bash")
    bash_content.append("# =====================================================================")
    bash_content.append("# AUTO-GENERATED MASTER MULTI-WORLD CALIBRATION RUNNER")
    bash_content.append("# Runs sequential Z-sweeps with dynamically scaled boundaries")
    bash_content.append("# =====================================================================")
    bash_content.append("\nset -e\n")

    for meta in worlds_meta:
        # Generate sweep points for this specific world length
        z_values = np.linspace(-meta['sweep_limit'], meta['sweep_limit'], SWEEP_STEPS)
        z_values_str = " ".join([f"{z:.2f}" for z in z_values])
        config_data = generate_config_string(meta)

        # Write clean section headers
        bash_content.append(f"# {'='*76}")
        bash_content.append(f"# PIPELINE FOR WORLD: {meta['name']} ({meta['type']})")
        bash_content.append(f"# {'='*76}")
        
        # Inject the python config file dynamic rewrite
        bash_content.append("cat << 'EOF' > calibration_config.py")
        bash_content.append(config_data.strip())
        bash_content.append("EOF")
        bash_content.append("echo '[+] Applied configuration for world: " + meta['name'] + "'\n")

        # Define sweep loops
        bash_content.append(f"Z_SWEEP_VALUES=({z_values_str})")
        bash_content.append(f"for z_val in \"${{Z_SWEEP_VALUES[@]}}\"; do")
        bash_content.append("    echo \"[*] Sweeping " + meta['name'] + " at Z = ${z_val} cm...\"")
        
        # Prepare output dir
        out_dir = f"output/{meta['name']}/calib_z_${{z_val}}"
        bash_content.append(f"    mkdir -p \"{out_dir}\"")
        
        # Run simulation & analysis command
        # NOTE: Adapt these to point to your actual executable pipeline scripts!
        bash_content.append(f"    # python3 run_simulation.py \\")
        bash_content.append(f"    #     --world \"{meta['path_str']}\" \\")
        bash_content.append(f"    #     --beam_x \"{meta['x']:.5f}\" \\")
        bash_content.append(f"    #     --beam_y \"{meta['y']:.5f}\" \\")
        bash_content.append(f"    #     --beam_z \"${{z_val}}\" \\")
        bash_content.append(f"    #     --output_dir \"{out_dir}\"")
        bash_content.append("    ")
        bash_content.append(f"    # python3 extract_prompt_attenuation.py --run_dir \"{out_dir}\" --z_offset \"${{z_val}}\"")
        bash_content.append("done\n")

    output_path = Path("run_all_calibs.sh")
    output_path.write_text("\n".join(bash_content))
    output_path.chmod(0o755)
    print(f"[+] Master script successfully generated: '{output_path}'")


if __name__ == "__main__":
    print("[*] Scanning workspace subdirectories for world modules...")
    world_paths = discover_worlds()
    
    if not world_paths:
        print("[-] Error: No world modules found in any nested 'worlds/' folders!")
        sys.exit(1)
        
    print(f"[+] Found {len(world_paths)} world candidate files.")
    
    validated_meta = []
    for path in world_paths:
        meta = load_world_metadata(path)
        if meta:
            validated_meta.append(meta)
            
    if not validated_meta:
        print("[-] Error: No valid configurations could be extracted.")
        sys.exit(1)

    print("\n=====================================================================")
    print(" 📋 Multi-World Calibration Sequence Map")
    print("=====================================================================")
    print(f"{'World Name':<25} | {'Type':<10} | {'Target (X, Y) cm':<20} | {'Sweep Limits':<15}")
    print("-" * 77)
    for m in validated_meta:
        coord_str = f"({m['x']:.4f}, {m['y']:.4f})"
        sweep_str = f"±{m['sweep_limit']:.2f} cm"
        print(f"{m['name']:<25} | {m['type']:<10} | {coord_str:<20} | {sweep_str:<15}")
    print("=====================================================================\n")

    write_master_bash_runner(validated_meta)
    print("[+] Setup complete! Execute with: ./run_all_calibs.sh")