import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================================================================
# 1. METHODOLOGY: SETUP AND CALIBRATION
# ==============================================================================
def equalize_channels(data_array):
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
def simulate_sipm_waveform(photon_times_ns, time_step_ps=10, max_time_ps=5000,
                           wls_filter_ns=None):
    """
    Converts discrete Geant4 photon arrival times into a digitized waveform array.

    Parameters:
    - photon_times_ns : List of photon arrival times from Geant4 in nanoseconds.
    - time_step_ps    : The size of each time bin (e.g., 10 ps per sample).
    - max_time_ps     : Total length of the waveform to simulate in ps.
    - wls_filter_ns   : If set, photons arriving AFTER this time (in ns) are
                        discarded before building the waveform.

                        Rationale: BCF-92 has a WLS re-emission time constant of
                        2.7 ns (WLSTIMECONSTANT in Materials.xml). When a UV
                        scintillation photon is absorbed by the WLS fiber, the
                        excited fluorescent molecule re-emits a visible photon
                        after a random exponential delay with mean 2.7 ns. This
                        produces the broad tail seen in E-type LocalTime spectra
                        (photons arriving 0.5-8+ ns after the shower).

                        For TIMING reconstruction we only want the prompt
                        Cherenkov signal (arriving ~0.1 ns, coincident with the
                        T-type reference). Setting wls_filter_ns to e.g. 0.3-0.5
                        isolates this prompt peak and rejects WLS-delayed photons
                        that would smear the timing resolution.

                        For ENERGY reconstruction, leave wls_filter_ns=None to
                        integrate the full photon yield including WLS-converted
                        scintillation light.
    """
    times_ps = np.array(photon_times_ns) * 1000

    # ── WLS prompt-photon filter ──────────────────────────────────────────────
    # Reject photons that arrived too late to be prompt Cherenkov signal.
    # These late arrivals are dominated by WLS re-emission (BCF-92 tau = 2.7 ns)
    # and would broaden the timing peak if included in the threshold crossing.
    if wls_filter_ns is not None:
        wls_filter_ps = wls_filter_ns * 1000
        times_ps = times_ps[times_ps <= wls_filter_ps]
    # ─────────────────────────────────────────────────────────────────────────

    bins = np.arange(0, max_time_ps + time_step_ps, time_step_ps)
    pulse_shape, _ = np.histogram(times_ps, bins=bins)
    waveform = np.cumsum(pulse_shape)
    return waveform

# ==============================================================================
# 4. METHODOLOGY: TIMING RECONSTRUCTION
# ==============================================================================
def get_timing_resolution(high_gain_waveforms, threshold, ref_times_mcp, time_step_ps=10):
    """Calculates time relative to MCP reference for each event."""
    delta_times = []

    for i, waveform in enumerate(high_gain_waveforms):
        crossings = np.where(waveform > threshold)[0]

        if len(crossings) == 0:
            continue

        idx = crossings[0]

        if idx == 0:
            t_rise_bin = 0
        else:
            fraction = (threshold - waveform[idx-1]) / (waveform[idx] - waveform[idx-1])
            t_rise_bin = (idx - 1) + fraction

        t_rise_ps = t_rise_bin * time_step_ps
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
import uproot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=str, required=True)
    parser.add_argument("--wls-filter-ns", type=float, default=0.4,
                        help="Reject photons arriving after this time (ns) to "
                             "suppress WLS-delayed hits. Default 0.4 ns sits "
                             "between the prompt Cherenkov peak (~0.1 ns) and "
                             "the BCF-92 WLS tail (tau=2.7 ns). "
                             "Set to 0 to disable filtering.")
    args = parser.parse_args()

    batch_path    = Path(args.batch_dir)
    wls_filter_ns = args.wls_filter_ns if args.wls_filter_ns > 0 else None

    print(f"--- Calculating Timing Resolution for {batch_path.name} ---")
    if wls_filter_ns is not None:
        print(f"    WLS filter: rejecting photons after {wls_filter_ns} ns "
              f"(BCF-92 tau = 2.7 ns)")
    else:
        print(f"    WLS filter: DISABLED (full photon yield)")

    threshold_pe   = 5
    mock_mcp_refs  = []
    real_waveforms = []

    hit_files = list(batch_path.glob("run_*/detector_hits_*.root"))

    if not hit_files:
        print("No hit files found!")
    else:
        for root_file in hit_files:
            try:
                with uproot.open(root_file) as f:
                    # OpenGATE names the tree after the file stem, not "PhaseSpace"
                    tree_key = [k.split(";")[0] for k in f.keys()][0]
                    tree = f[tree_key]
                    photon_times_ns = tree["GlobalTime"].array(library="np")

                    if len(photon_times_ns) > 0:
                        waveform = simulate_sipm_waveform(
                            photon_times_ns,
                            time_step_ps=10,
                            wls_filter_ns=wls_filter_ns,
                        )
                        real_waveforms.append(waveform)

                        true_start_ns = 1.0
                        mcp_time_ps   = (true_start_ns * 1000) + np.random.normal(0, 15)
                        mock_mcp_refs.append(mcp_time_ps)

            except Exception as e:
                print(f"Error reading {root_file.name}: {e}")

        if len(real_waveforms) > 0:
            sigma_t = get_timing_resolution(
                real_waveforms,
                threshold=threshold_pe,
                ref_times_mcp=mock_mcp_refs,
            )
            print(f"Calculated Timing Resolution (sigma_t): {sigma_t:.2f} ps")
        else:
            print("No valid waveforms generated.")