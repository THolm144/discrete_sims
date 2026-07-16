import os
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
    print("\n" + "="*55)
    print(f"  CALIBRATION ANALYSIS COMPLETE FOR: {dir_name}")
    print(f"  Effective Prompt Attenuation Length (L_eff): {lambda_eff:.2f} mm")
    print("="*55)
except Exception as e:
    print(f"Fit failed. Raw data array: {prompt_photon_counts}")
