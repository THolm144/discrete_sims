"""
timing_res.py  —  RADiCAL timing resolution via BestMinus (paper Method 4)
==========================================================================
Digital twin of the real CERN beam-test electronics chain:

  Geant4 photon hit times
       │
       ▼
  Assign to 1-of-8 SiPM channels by geometry (XY → capillary, Z → up/down)
       │
       ▼
  Build digitized high-gain waveform per channel per event
  (mimics CAEN DT5742 @ 5 GS/s, 200 ps/sample)
       │
       ▼
  Fixed-threshold leading-edge crossing with linear interpolation
       │
       ▼
  Δt_DW = mean(t_SiPM_downstream - t_MCP) over 4 downstream channels
  Δt_UP = mean(t_SiPM_upstream   - t_MCP) over 4 upstream channels
  BestMinus = (Δt_DW - Δt_UP) / 2   [MCP-independent]
       │
       ▼
  Select events in energy bins 6-8 (upper third of photon-count peak)
       │
       ▼
  Gaussian fit to BestMinus distribution → σ_t

Reference: Perez-Lara et al., NIM A 1068 (2024) 169737, Section 5.2 / Method 4
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import uproot
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY CONSTANTS  (must match radi_cal.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

_TYVEK_MM      = 0.008 * 25.4            # 0.2032 mm
_CAP_LENGTH_MM = 183.0
_SIPM_THICK_MM = 0.3
_SIPM_Z_MM     = _CAP_LENGTH_MM / 2 + _SIPM_THICK_MM / 2   # 91.65 mm

_CALOR_XY_MM   = 14.0 + 2 * _TYVEK_MM   # 14.4064 mm
_HOLE_INSET_MM = 3.5
_HOLE_OFFSET   = _CALOR_XY_MM / 2 - _HOLE_INSET_MM         # 3.7032 mm

# (x, y) centre of each capillary in mm — index matches sipm_front/back_N
CAP_XY_MM = np.array([
    [ _HOLE_OFFSET,  _HOLE_OFFSET],   # cap 0 / sipm_*_0
    [ _HOLE_OFFSET, -_HOLE_OFFSET],   # cap 1 / sipm_*_1
    [-_HOLE_OFFSET,  _HOLE_OFFSET],   # cap 2 / sipm_*_2
    [-_HOLE_OFFSET, -_HOLE_OFFSET],   # cap 3 / sipm_*_3
])

# ─────────────────────────────────────────────────────────────────────────────
# DIGITIZER PARAMETERS  (mimic CAEN DT5742)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_PS       = 200       # 5 GS/s  → 200 ps per sample
WINDOW_PS       = 50_000    # 50 ns acquisition window
SPTR_SIGMA_PS   = 150       # SiPM single-photon time resolution (Gaussian jitter)
THRESHOLD_PE    = 3         # fixed threshold in photoelectrons on leading edge
MCP_SIGMA_PS    = 15        # MCP reference timing jitter (σ)
MCP_TRUE_PS     = 0.0       # beam crossing reference (arbitrary offset)

# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL ASSIGNMENT  (geometry only — no simulation truth)
# ─────────────────────────────────────────────────────────────────────────────

def assign_channel(x_mm, y_mm, z_mm):
    """
    Map a photon hit position to one of 8 SiPM channel indices.

    Channels 0-3 : downstream (sipm_back,  z > 0)
    Channels 4-7 : upstream   (sipm_front, z < 0)

    Within each end, the capillary index (0-3) is the nearest XY neighbour.
    Returns -1 if the hit is not on a SiPM face (i.e. z not near ±SIPM_Z).
    """
    # Accept only hits within ±1 mm of a SiPM face in Z
    if abs(abs(z_mm) - _SIPM_Z_MM) > 1.0:
        return -1

    # Nearest capillary by XY distance
    dists = np.hypot(CAP_XY_MM[:, 0] - x_mm, CAP_XY_MM[:, 1] - y_mm)
    cap_idx = int(np.argmin(dists))

    # Downstream (back, z > 0) → channels 0-3
    # Upstream   (front, z < 0) → channels 4-7
    if z_mm > 0:
        return cap_idx          # 0-3  downstream
    else:
        return cap_idx + 4      # 4-7  upstream


# ─────────────────────────────────────────────────────────────────────────────
# WAVEFORM BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_waveform(hit_times_ns, rng):
    """
    Convert a list of photon arrival times (ns) into a digitized waveform
    mimicking the CAEN DT5742 high-gain channel output.

      1. Convert ns → ps
      2. Add Gaussian SPTR jitter per photon
      3. Histogram into 200 ps bins (5 GS/s)
      4. The histogram IS the pulse-height waveform (not cumsum)
         — threshold crossing on this gives the leading-edge time

    Returns (waveform array, bin_edges_ps).
    """
    if len(hit_times_ns) == 0:
        bins = np.arange(0, WINDOW_PS + SAMPLE_PS, SAMPLE_PS)
        return np.zeros(len(bins) - 1, dtype=float), bins

    times_ps = np.asarray(hit_times_ns, dtype=float) * 1000.0
    times_ps = times_ps + rng.normal(0.0, SPTR_SIGMA_PS, size=len(times_ps))

    bins = np.arange(0, WINDOW_PS + SAMPLE_PS, SAMPLE_PS)
    waveform, _ = np.histogram(times_ps, bins=bins)
    return waveform.astype(float), bins


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD CROSSING  (leading edge, linear interpolation)
# ─────────────────────────────────────────────────────────────────────────────

def threshold_crossing_ps(waveform, threshold):
    """
    Find the first time the waveform rises above `threshold` PE,
    using linear interpolation between adjacent samples.

    Returns crossing time in ps, or None if threshold is never reached.
    """
    crossings = np.where(waveform >= threshold)[0]
    if len(crossings) == 0:
        return None

    idx = crossings[0]
    if idx == 0:
        return 0.0

    # Linear interpolation: fraction of sample width
    y0, y1 = waveform[idx - 1], waveform[idx]
    if y1 == y0:
        frac = 0.0
    else:
        frac = (threshold - y0) / (y1 - y0)

    return (idx - 1 + frac) * SAMPLE_PS


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN FIT
# ─────────────────────────────────────────────────────────────────────────────

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian(data, n_bins=60):
    """
    Fit a Gaussian to `data` and return (mu, sigma, sigma_err).
    Falls back to (mean, std, nan) if fit fails.
    """
    if len(data) < 10:
        return np.mean(data), np.std(data), np.nan

    mu0, s0 = np.mean(data), np.std(data)
    lo, hi  = mu0 - 3 * s0, mu0 + 3 * s0
    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])

    try:
        p0 = [counts.max(), mu0, s0]
        popt, pcov = curve_fit(gaussian, centres, counts, p0=p0, maxfev=5000)
        sigma_err  = np.sqrt(pcov[2, 2]) if pcov[2, 2] >= 0 else np.nan
        return popt[1], abs(popt[2]), sigma_err
    except Exception:
        return mu0, s0, np.nan


# ─────────────────────────────────────────────────────────────────────────────
# ENERGY BIN SELECTION  (bins 6-8 of 9, i.e. upper third of peak)
# ─────────────────────────────────────────────────────────────────────────────

def select_high_energy_events(photon_counts_per_event, fraction_lo=5/9, fraction_hi=8/9):
    """
    Mimic the paper's bin 6-8 selection:
    keep events whose total photon count falls in the upper-middle portion
    of the distribution (where E_meas ≈ E_beam, minimal shower leakage).

    `fraction_lo` and `fraction_hi` are quantile bounds on the distribution.
    """
    lo = np.quantile(photon_counts_per_event, fraction_lo)
    hi = np.quantile(photon_counts_per_event, fraction_hi)
    mask = (photon_counts_per_event >= lo) & (photon_counts_per_event <= hi)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(batch_dir: Path):
    rng = np.random.default_rng(seed=42)

    # ── 1. Collect all ROOT files ─────────────────────────────────────────
    hit_files = sorted(batch_dir.rglob("detector_hits_*.root"))
    if not hit_files:
        print("  No detector_hits_*.root files found.")
        return None

    # ── 2. Load all hits across files ─────────────────────────────────────
    all_event_id  = []
    all_x, all_y, all_z = [], [], []
    all_time_ns   = []
    all_particle  = []

    file_offset = 0   # shift EventID to make it globally unique across files
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tree = f[f.keys()[0]]
                ev   = tree["EventID"].array(library="np").astype(int)
                x    = tree["Position_X"].array(library="np")
                y    = tree["Position_Y"].array(library="np")
                z    = tree["Position_Z"].array(library="np")
                t    = tree["GlobalTime"].array(library="np")
                pn   = tree["ParticleName"].array(library="np")

                all_event_id.append(ev + file_offset)
                all_x.append(x);  all_y.append(y);  all_z.append(z)
                all_time_ns.append(t)
                all_particle.append(pn)
                file_offset += int(ev.max()) + 1
        except Exception as e:
            print(f"  Warning: could not read {fpath.name}: {e}")

    if not all_event_id:
        print("  No valid data loaded.")
        return None

    event_id  = np.concatenate(all_event_id)
    x_mm      = np.concatenate(all_x)
    y_mm      = np.concatenate(all_y)
    z_mm      = np.concatenate(all_z)
    time_ns   = np.concatenate(all_time_ns)
    particle  = np.concatenate(all_particle)

    # ── 3. Filter: optical photons only ───────────────────────────────────
    is_optical = (particle == b"opticalphoton") | (particle == "opticalphoton")
    event_id   = event_id[is_optical]
    x_mm       = x_mm[is_optical]
    y_mm       = y_mm[is_optical]
    z_mm       = z_mm[is_optical]
    time_ns    = time_ns[is_optical]

    if len(event_id) == 0:
        print("  No optical photon hits found.")
        return None

    # ── 4. Assign each hit to a SiPM channel (geometry only) ──────────────
    channels = np.array([assign_channel(x, y, z)
                         for x, y, z in zip(x_mm, y_mm, z_mm)])

    # Keep only hits that landed on a SiPM face
    on_sipm   = channels >= 0
    event_id  = event_id[on_sipm]
    time_ns   = time_ns[on_sipm]
    channels  = channels[on_sipm]

    if len(event_id) == 0:
        print("  No hits on SiPM faces found. Check geometry Z tolerance.")
        return None

    print(f"  Total SiPM hits loaded : {len(event_id):,}")

    # ── 5. Per-event processing ────────────────────────────────────────────
    unique_events = np.unique(event_id)
    print(f"  Events with SiPM hits  : {len(unique_events):,}")

    best_minus_ps     = []   # (Δt_DW - Δt_UP) / 2  per event
    photons_per_event = []   # total SiPM photon count per event (proxy for E_meas)

    # Mock MCP: beam crosses at t=0 + Gaussian jitter
    mcp_times = rng.normal(MCP_TRUE_PS, MCP_SIGMA_PS, size=len(unique_events))

    for ev_i, (ev_id, mcp_t) in enumerate(zip(unique_events, mcp_times)):
        mask = event_id == ev_id
        ev_times    = time_ns[mask]
        ev_channels = channels[mask]

        photons_per_event.append(int(mask.sum()))

        # Build waveform and get threshold-crossing time for each of 8 channels
        t_channel = {}   # channel_idx → crossing time in ps (or None)
        for ch in range(8):
            ch_times = ev_times[ev_channels == ch]
            waveform, _ = build_waveform(ch_times, rng)
            t_cross = threshold_crossing_ps(waveform, THRESHOLD_PE)
            t_channel[ch] = t_cross

        # Downstream average: channels 0-3
        dw_crossings = [t_channel[ch] for ch in range(4) if t_channel[ch] is not None]
        # Upstream average: channels 4-7
        up_crossings = [t_channel[ch] for ch in range(4, 8) if t_channel[ch] is not None]

        # Need at least 1 crossing on each end
        if not dw_crossings or not up_crossings:
            continue

        delta_dw = np.mean(dw_crossings) - mcp_t   # Δt_DW
        delta_up = np.mean(up_crossings) - mcp_t   # Δt_UP

        bm = (delta_dw - delta_up) / 2.0            # BestMinus
        best_minus_ps.append(bm)

    if len(best_minus_ps) < 10:
        print(f"  Too few valid events ({len(best_minus_ps)}) to fit.")
        return None

    best_minus_ps     = np.array(best_minus_ps)
    photons_per_event = np.array(photons_per_event[:len(best_minus_ps)])

    # ── 6. Energy bin selection: bins 6-8 of 9 (paper Method 4) ───────────
    energy_mask = select_high_energy_events(photons_per_event)
    selected    = best_minus_ps[energy_mask]

    print(f"  Events after bin 6-8 selection: {energy_mask.sum():,} "
          f"/ {len(best_minus_ps):,}")

    if len(selected) < 10:
        print("  Too few events in selected energy bins — using all events.")
        selected = best_minus_ps

    # ── 7. Gaussian fit → σ_t ─────────────────────────────────────────────
    mu, sigma, sigma_err = fit_gaussian(selected)

    return {
        "sigma_t_ps":      sigma,
        "sigma_t_err_ps":  sigma_err,
        "mu_ps":           mu,
        "n_events_total":  len(best_minus_ps),
        "n_events_selected": int(energy_mask.sum()),
        "best_minus_all":  best_minus_ps,
        "best_minus_sel":  selected,
        "photons_per_event": photons_per_event,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=str, required=True)
    args   = parser.parse_args()

    batch_path = Path(args.batch_dir)
    energy_label = batch_path.name   # e.g. "25000000keV"
    print(f"\n{'─'*60}")
    print(f"  Timing Resolution — BestMinus (Method 4)")
    print(f"  Batch : {energy_label}")
    print(f"{'─'*60}")

    result = run(batch_path)

    if result is None:
        print("  Could not compute timing resolution.")
    else:
        s  = result["sigma_t_ps"]
        se = result["sigma_t_err_ps"]
        n  = result["n_events_selected"]

        err_str = f" ± {se:.2f}" if not np.isnan(se) else ""
        print(f"\n  BestMinus σ_t  =  {s:.2f}{err_str} ps")
        print(f"  (Gaussian fit over {n} events in energy bins 6-8)")

        # Save result
        out_txt = batch_path / "timing_resolution.txt"
        with open(out_txt, "w") as f:
            f.write(f"Method          : BestMinus (paper Method 4)\n")
            f.write(f"sigma_t_ps      : {s:.4f}\n")
            f.write(f"sigma_t_err_ps  : {se:.4f}\n")
            f.write(f"mu_ps           : {result['mu_ps']:.4f}\n")
            f.write(f"n_events_total  : {result['n_events_total']}\n")
            f.write(f"n_events_sel    : {result['n_events_selected']}\n")
        print(f"  Saved → {out_txt}")

        # Paper comparison
        try:
            energy_kev = float(energy_label.replace("keV", ""))
            energy_gev = energy_kev / 1e6
            a, b = 256.0, 17.5   # paper constants
            sigma_paper = np.sqrt((a / np.sqrt(energy_gev))**2 + b**2)
            print(f"\n  Paper formula σ_t = {sigma_paper:.2f} ps  "
                  f"(a/√E ⊕ b, E={energy_gev:.0f} GeV)")
            print(f"  Ratio sim/paper   = {s/sigma_paper:.2f}")
        except Exception:
            pass