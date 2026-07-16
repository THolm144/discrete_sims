# --- AUTO-PATCHED FOR DYNAMIC TIMING ---
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

import os
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
SENSOR_Z_CM = 13.515            # Will be dynamically replaced

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
    expected_time_ns = ((_sensor_z_override - _z_offset_override) / locals().get('v_eff', globals().get('v_eff', locals().get('V_EFF', globals().get('V_EFF', 16.5)))) if (_sensor_z_override is not None and _z_offset_override is not None) else ((SENSOR_Z_CM * 10.0) / V_EFF ))
    
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
    print("\n" + "="*55)
    print(f"  CALIBRATION ANALYSIS COMPLETE FOR: {dir_name}")
    print(f"  Direction: {direction_str}")
    print(f"  Effective Prompt Attenuation Length (L_eff): {lambda_eff:.2f} mm")
    print("="*55)
except Exception as e:
    print(f"Fit failed. Raw data array: {prompt_photon_counts}")