import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================================================================
# 1. METHODOLOGY: SETUP AND CALIBRATION
# ==============================================================================
def equalize_channels(data_array):
    """
    Normalizes each channel's response based on its centroid to 
    correct for individual SiPM gain variations.
    """
    normalized_data = data_array.copy()
    for i in range(normalized_data.shape[1]):
        centroid = np.mean(normalized_data[:, i])
        normalized_data[:, i] /= centroid
    return normalized_data

# ==============================================================================
# 2. METHODOLOGY: ENERGY RECONSTRUCTION
# ==============================================================================
def calculate_total_energy(low_gain_data):
    """E_meas = Sum(Amplitudes of all 8 low-gain SiPM signals)"""
    return np.sum(low_gain_data, axis=1)

# ==============================================================================
# 3. METHODOLOGY: WAVEFORM GENERATION (Bridging Geant4 to Analysis)
# ==============================================================================
def simulate_sipm_waveform(photon_times_ns, time_step_ps=10, max_time_ps=5000):
    """
    Converts discrete Geant4 photon arrival times into a digitized waveform array.
    
    Parameters:
    - photon_times_ns: List of photon arrival times from Geant4 in nanoseconds.
    - time_step_ps: The size of each time bin (e.g., 10 ps per sample).
    - max_time_ps: Total length of the waveform to simulate in ps.
    """
    # Convert Geant4 nanoseconds to picoseconds
    times_ps = np.array(photon_times_ns) * 1000 
    
    # Create time bins (like digitizer samples)
    bins = np.arange(0, max_time_ps + time_step_ps, time_step_ps)
    
    # Histogram the hits. This gives us the number of photons per time bin
    pulse_shape, _ = np.histogram(times_ps, bins=bins)
    
    # Simulate the rising edge of the voltage waveform (integrating charge)
    waveform = np.cumsum(pulse_shape) 
    
    return waveform

# ==============================================================================
# 4. METHODOLOGY: TIMING RECONSTRUCTION
# ==============================================================================
def get_timing_resolution(high_gain_waveforms, threshold, ref_times_mcp, time_step_ps=10):
    """
    Calculates time relative to MCP reference for each event.
    """
    delta_times = []
    
    for i, waveform in enumerate(high_gain_waveforms):
        # Find where the waveform crosses the threshold
        crossings = np.where(waveform > threshold)[0]
        
        if len(crossings) == 0:
            continue # Skip events that never reach the threshold
            
        idx = crossings[0]
        
        if idx == 0:
            t_rise_bin = 0 # Reached threshold immediately
        else:
            # Linear interpolation between samples for sub-bin precision
            fraction = (threshold - waveform[idx-1]) / (waveform[idx] - waveform[idx-1])
            t_rise_bin = (idx - 1) + fraction
            
        # Convert bin index back to actual time in ps
        t_rise_ps = t_rise_bin * time_step_ps
        
        # Calculate difference from the MCP reference for this event
        delta_t = t_rise_ps - ref_times_mcp[i]
        delta_times.append(delta_t)
        
    return np.std(delta_times)

# ==============================================================================
# 5. METHODOLOGY: RESOLUTION ANALYSIS
# ==============================================================================
def timing_res_func(E, a, b):
    """Energy dependence follows: sigma_t = a/sqrt(E) + b"""
    return a / np.sqrt(E) + b


import argparse
from pathlib import Path
import uproot # Make sure to pip install uproot if you haven't!

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=str, required=True, help="Directory with simulation output")
    args = parser.parse_args()
    
    batch_path = Path(args.batch_dir)
    print(f"--- Calculating Timing Resolution for {batch_path.name} ---")
    
    threshold_pe = 5 # 5 photoelectrons threshold
    mock_mcp_refs = []
    real_waveforms = []
    
    # 1. Find all the detector hit ROOT files in this run directory
    hit_files = list(batch_path.glob("detector_hits_*.root"))
    
    if not hit_files:
        print("No hit files found! Did the simulation save 'detector_hits_X.root'?")
    else:
        # 2. Loop through the files to extract photon hit times
        for root_file in hit_files:
            try:
                # Open the ROOT file and grab the PhaseSpace tree
                with uproot.open(root_file) as f:
                    tree = f["PhaseSpace"] # Or whatever your Geant4 tree is named
                    
                    # Extract GlobalTime of all photon hits in nanoseconds
                    # (Note: Geant4 outputs in ns by default)
                    photon_times_ns = tree["GlobalTime"].array(library="np")
                    
                    if len(photon_times_ns) > 0:
                        # Convert to waveform
                        waveform = simulate_sipm_waveform(photon_times_ns, time_step_ps=10)
                        real_waveforms.append(waveform)
                        
                        # Mock the MCP reference time for this specific event
                        # We assume the beam hits at ~1.0 ns with 15ps MCP jitter
                        true_start_ns = 1.0 
                        mcp_time_ps = (true_start_ns * 1000) + np.random.normal(0, 15)
                        mock_mcp_refs.append(mcp_time_ps)
                        
            except Exception as e:
                print(f"Error reading {root_file.name}: {e}")
        
        # 3. Calculate and print the actual timing resolution
        if len(real_waveforms) > 0:
            sigma_t = get_timing_resolution(real_waveforms, threshold=threshold_pe, ref_times_mcp=mock_mcp_refs)
            print(f"Calculated Timing Resolution (sigma_t): {sigma_t:.2f} ps")
        else:
            print("No valid waveforms generated to calculate timing resolution.")