#!/usr/bin/env python3
"""
calibration_config.py
=======================
Single shared source of the per-module, per-energy timing calibration
(T0 offset and bounce factor) used to compute raw_z_emits from BestMinus
ToF. Both build_empirical_response_matrix.py and unfold_profile_analysis.py
import from here so the response kernel is always fit against the SAME
raw_z_emits it will later be used to deconvolve. (Previously the builder
script called extract_profile_data_unfold with no calibration at all,
which would have fit the kernel against uncalibrated/default-offset data
while the consumer used the real per-energy calibration -- a silent
mismatch worse than the missing-energy-term bug.)
"""

DEFAULT_T0_OFFSET_NS = -0.32

CALIBRATION_T0_OFFSETS = {
    "radi_cal_energy": {25.0: -0.320, 50.0: -0.300, 100.0: -0.330, 200.0: -0.320},
    "radi_cal_triple": {25.0: -0.370, 50.0: -0.270, 100.0: -0.250, 200.0: -0.190},
    "rc_hex":          {25.0: -0.290, 50.0: -0.310, 100.0: -0.180, 200.0: -0.085},
    "rc_hex_triple":   {25.0: -0.370, 50.0: -0.300, 100.0: -0.220, 200.0: -0.050},
}

CALIBRATION_BOUNCE_FACTORS = {
    "radi_cal_energy": {25.0: 1.17, 50.0: 1.20, 100.0: 1.23, 200.0: 1.27},
    "radi_cal_triple": {25.0: 1.45, 50.0: 1.50, 100.0: 1.55, 200.0: 1.65},
    "rc_hex":          {25.0: 1.02, 50.0: 1.02, 100.0: 1.01, 200.0: 1.01},
    "rc_hex_triple":   {25.0: 1.19, 50.0: 1.23, 100.0: 1.33, 200.0: 1.73},
}


def get_calibration(mod, energy_val):
    """Returns (t0_offset, bounce_factor) for a given module/energy, falling
    back to DEFAULT_T0_OFFSET_NS / 1.0 if the module or energy isn't in the
    tables (matches the .get(...) fallback pattern used elsewhere)."""
    t0_offset = CALIBRATION_T0_OFFSETS.get(mod, {}).get(energy_val, DEFAULT_T0_OFFSET_NS)
    bounce_factor = CALIBRATION_BOUNCE_FACTORS.get(mod, {}).get(energy_val, 1.0)
    return t0_offset, bounce_factor