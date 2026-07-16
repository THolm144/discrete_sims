#!/usr/bin/env python3
"""
patch_all_calibs.py
===================
Automatically searches all subdirectories for 'extract_prompt_attenuation.py',
creates a safe backup, and patches them to support dynamic command-line 
parameters without breaking existing argument parsing.
"""

import os
import re
from pathlib import Path

# The interceptor code consumes our custom CLI overrides before argparse can complain
INTERCEPTOR_CODE = """# --- AUTO-PATCHED FOR DYNAMIC TIMING ---
import sys as _sys
_sensor_z_override = None
_z_offset_override = None
try:
    if '--sensor_z' in _sys.argv:
        _idx = _sys.argv.index('--sensor_z')
        if _idx + 1 < len(_sys.argv):
            _sensor_z_override = float(_sys.argv[_idx+1])
            _sys.argv.pop(_idx+1)
            _sys.argv.pop(_idx)
    if '--z_offset' in _sys.argv:
        _idx = _sys.argv.index('--z_offset')
        if _idx + 1 < len(_sys.argv):
            _z_offset_override = float(_sys.argv[_idx+1])
            _sys.argv.pop(_idx+1)
            _sys.argv.pop(_idx)
except Exception as _e:
    print(f"[-] Auto-patcher interceptor error: {_e}")
# ----------------------------------------
"""

# Supported timing variables across your various scripts
TARGET_VARS = [
    "expected_time_ns", "expected_time", "expected_t", 
    "t_expected", "t_expected_ns", "transit_time", "arrival_time"
]

def patch_content(content, filepath):
    if "AUTO-PATCHED FOR DYNAMIC TIMING" in content:
        print(f"  [~] Already patched: {filepath}")
        return None

    lines = content.splitlines()
    if not lines:
        return None

    # Inject the interceptor right at the top
    if lines[0].startswith("#!"):
        lines.insert(1, INTERCEPTOR_CODE)
    else:
        lines.insert(0, INTERCEPTOR_CODE)

    new_content = "\n".join(lines)
    patched_any = False

    for var in TARGET_VARS:
        pattern = rf"^(\s*{var}\s*=\s*)([^#\n]+)"
        
        # This checks local and global namespaces for any variation of v_eff, defaulting to 16.5
        replacement = (
            r"\1((_sensor_z_override - _z_offset_override) / "
            r"locals().get('v_eff', globals().get('v_eff', locals().get('V_EFF', globals().get('V_EFF', 16.5)))) "
            r"if (_sensor_z_override is not None and _z_offset_override is not None) else (\2))"
        )
        
        new_content, count = re.subn(pattern, replacement, new_content, flags=re.MULTILINE)
        if count > 0:
            print(f"  [+] Patched '{var}' inside: {filepath} ({count} matches)")
            patched_any = True

    if not patched_any:
        print(f"  [!] Warning: Couldn't detect typical timing variable names in {filepath}.")
        print("      Applying the command-line interceptor anyway to prevent argument parser crashes.")
    
    return new_content

def main():
    print("========================================================================")
    print(" Running Auto-Patcher for Multi-World Calibration Scripts")
    print("========================================================================")
    
    current_dir = Path(".")
    target_files = sorted(current_dir.glob("**/extract_prompt_attenuation.py"))
    
    if not target_files:
        print("[-] No 'extract_prompt_attenuation.py' scripts found in subdirectories!")
        return

    for path in target_files:
        print(f"[*] Processing: {path}")
        content = path.read_text()
        
        # Create a backup (.bak) if it doesn't already exist
        backup_path = path.with_suffix(".py.bak")
        if not backup_path.exists():
            backup_path.write_text(content)
            print(f"  [✓] Created backup: {backup_path}")
            
        new_content = patch_content(content, path)
        if new_content is not None:
            path.write_text(new_content)
            print(f"  [✓] Successfully updated: {path}")
            
    print("\n========================================================================")
    print(" [✓] AUTO-PATCH COMPLETE! All calibration analysis files prepared.")
    print("========================================================================")

if __name__ == "__main__":
    main()