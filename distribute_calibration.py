#!/usr/bin/env python3
"""
distribute_calibration.py
=========================
Robustly parses active capillary coordinates from each world module and updates
all local shell execution scripts, supporting multi-line commands and echo lines.
"""

import os
import re
import sys
import importlib.util
from pathlib import Path
from types import ModuleType

# ─────────────────────────────────────────────────────────────────────────────
# OPENGATE MOCK SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_opengate_mocks():
    """
    Creates dummy modules for OpenGate and its submodules.
    This prevents ImportErrors when running this script on systems (like DAQ)
    that do not have Geant4 or OpenGate packages installed.
    """
    mock_opengate = ModuleType("opengate")
    mock_geometry = ModuleType("opengate.geometry")
    mock_volumes = ModuleType("opengate.geometry.volumes")
    
    mock_opengate.geometry = mock_geometry
    mock_geometry.volumes = mock_volumes
    
    sys.modules["opengate"] = mock_opengate
    sys.modules["opengate.geometry"] = mock_geometry
    sys.modules["opengate.geometry.volumes"] = mock_volumes
    
    class DummyClass:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return DummyClass
        
    mock_volumes.TubsVolume = DummyClass
    mock_volumes.BoxVolume = DummyClass
    mock_volumes.subtract_volumes = lambda *args, **kwargs: DummyClass()

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC WORLD PARSING
# ─────────────────────────────────────────────────────────────────────────────
def extract_beam_coordinates(world_path: Path):
    """
    Loads the world python file dynamically and extracts the first active capillary coordinate (in cm).
    """
    module_name = world_path.stem
    spec = importlib.util.spec_from_file_location(module_name, world_path)
    module = importlib.util.module_from_spec(spec)
    
    # Add directory to sys.path to allow internal imports
    worlds_dir = str(world_path.parent)
    if worlds_dir not in sys.path:
        sys.path.insert(0, worlds_dir)
        
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error executing world module: {e}")

    # 1. Look for capillary positions
    positions = None
    for attr in ['_CAP_POSITIONS_MM', 'CAP_POSITIONS_MM', '_CAP_POSITIONS', 'CAP_POSITIONS']:
        if hasattr(module, attr):
            positions = getattr(module, attr)
            break
            
    if not positions:
        raise ValueError("Could not find capillary positions variable (e.g. _CAP_POSITIONS_MM).")
        
    # 2. Look for active indices (E-type or Active)
    active_indices = None
    for attr in ['_E_TYPE_INDICES', 'E_TYPE_INDICES', '_ACTIVE_INDICES', 'ACTIVE_INDICES', '_ACTIVE_CAPS']:
        if hasattr(module, attr):
            active_indices = getattr(module, attr)
            break
            
    if not active_indices:
        # Scan internal variables for any sets containing 'ACTIVE' or 'E_TYPE'
        for key, val in vars(module).items():
            if isinstance(val, (set, list, tuple)) and any(x in key.upper() for x in ['ACTIVE', 'E_TYPE', 'E_IND']):
                active_indices = val
                break
                
    # Fallback to index 0 if no active markers are set
    if not active_indices:
        active_indices = {0}
        
    active_idx = list(active_indices)[0]
    pos_mm = positions[active_idx]
    
    # Convert millimeters to centimeters (standard for Beam Targets)
    x_cm = pos_mm[0] / 10.0
    y_cm = pos_mm[1] / 10.0
    
    return x_cm, y_cm, active_idx

# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT PATCHING UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def patch_shell_script(script_path: Path, x_cm: float, y_cm: float) -> bool:
    """
    Finds commands calling 'simulator.py' and injects/updates '--beam-x' and '--beam-y'.
    Handles multi-line backslash formats and hardcoded echo outputs robustly.
    """
    with open(script_path, "r") as f:
        content = f.read()
        
    # 1. Update hardcoded target coordinate echo lines
    # Match strings like: echo "Targeting Capillary coordinate: X = 0.0 cm, Y = 0.0 cm"
    orig_content = content
    content = re.sub(
        r'echo "Targeting Capillary coordinate: X\s*=\s*[-\d\.]+\s*cm,\s*Y\s*=\s*[-\d\.]+\s*cm"',
        f'echo "Targeting Capillary coordinate: X = {x_cm:.5f} cm, Y = {y_cm:.5f} cm"',
        content
    )
    
    # 2. Clean and update simulator commands
    lines = content.splitlines()
    new_lines = []
    updated_run_line = False
    
    for line in lines:
        # If it's a standalone parameter line containing only --beam-x or --beam-y (common in multiline setups)
        is_pure_beam_arg = re.match(r'^\s*--beam-[xy](?:\s+|=\s*)[-\d\.]+\s*(\\?)\s*$', line)
        if is_pure_beam_arg:
            # Drop the standalone line entirely. Backslash line breaks will safely carry forward.
            updated_run_line = True
            continue
            
        # If it's the main simulator execution call
        if "simulator.py" in line and not line.strip().startswith("#"):
            # Strip out any existing inline --beam-x or --beam-y parameters
            line = re.sub(r'--beam-x(?:\s+|=\s*)[-\d\.]+', '', line)
            line = re.sub(r'--beam-y(?:\s+|=\s*)[-\d\.]+', '', line)
            
            # Re-inject the correct, fresh coordinates right after simulator.py
            line = line.replace("simulator.py", f"simulator.py --beam-x {x_cm:.5f} --beam-y {y_cm:.5f}")
            updated_run_line = True
            
        new_lines.append(line)
        
    final_content = "\n".join(new_lines) + "\n"
    
    # Save only if we actually modified something
    if final_content != orig_content or updated_run_line:
        with open(script_path, "w") as f:
            f.write(final_content)
        return True
        
    return False

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    setup_opengate_mocks()
    base_dir = Path(__file__).resolve().parent
    
    print("=" * 80)
    print(" RADiCAL Calibration Coordination & Deployer Script (Multi-line Patched)")
    print("=" * 80)
    
    results = []
    
    # Scan subdirectories containing simulator.py
    for sim_dir in base_dir.iterdir():
        if not sim_dir.is_dir() or not (sim_dir / "simulator.py").exists():
            continue
            
        worlds_dir = sim_dir / "worlds"
        if not worlds_dir.exists():
            results.append((sim_dir.name, "No worlds directory", "-", "-", "Skipped"))
            continue
            
        # Get world python files
        world_files = [f for f in worlds_dir.glob("*.py") if f.name != "__init__.py"]
        if not world_files:
            results.append((sim_dir.name, "No world .py modules found", "-", "-", "Skipped"))
            continue
            
        # Prioritize non-utility files
        world_file = next((f for f in world_files if "test" not in f.name and "helper" not in f.name), world_files[0])
        
        try:
            x_cm, y_cm, active_idx = extract_beam_coordinates(world_file)
            coords_str = f"({x_cm:+.5f}, {y_cm:+.5f})"
            
            # Look for all .sh scripts inside the module directory to patch
            sh_scripts = list(sim_dir.glob("*.sh"))
            patched_scripts = []
            
            for script in sh_scripts:
                if patch_shell_script(script, x_cm, y_cm):
                    patched_scripts.append(script.name)
                    
            if patched_scripts:
                status = f"Patched: {', '.join(patched_scripts)}"
            else:
                status = "No execution scripts found/modified"
                
            results.append((sim_dir.name, world_file.name, f"Index {active_idx}", coords_str, status))
            
        except Exception as e:
            results.append((sim_dir.name, world_file.name, "Error", "-", f"Failed: {str(e)[:40]}..."))

    # Print summary report
    print("\n### Deployment Summary Table\n")
    print(f"| {'Directory':<24} | {'World Module':<26} | {'Active Fiber':<12} | {'Target Coords (cm)':<22} | {'Deployment Status':<35} |")
    print(f"|{'-'*26}|{'-'*28}|{'-'*14}|{'-'*24}|{'-'*37}|")
    for r in results:
        print(f"| {r[0]:<24} | {r[1]:<26} | {r[2]:<12} | {r[3]:<22} | {r[4]:<35} |")
    print()

if __name__ == "__main__":
    main()