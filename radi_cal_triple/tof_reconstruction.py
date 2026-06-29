import argparse
from pathlib import Path
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

try:
    import analysis_utils as utils
except ImportError:
    print("WARNING: Could not import 'utils'. Ensure this script is run from the OpenGATE sim directory.")
    utils = None

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY CONSTANTS  (Updated for radi_cal_triple.py)
# ─────────────────────────────────────────────────────────────────────────────
_LYSO_THICK_MM   = 4.5                      # 1.5 mm * 3 (Triple Variant)
_TYVEK_THICK_MM  = 0.2032
_W_THICK_MM      = 2.5
_N_LYSO          = 29
_N_W             = 28
_GAP_THICK_MM    = _LYSO_THICK_MM + 2 * _TYVEK_THICK_MM   # 4.9064 mm
_CALOR_THICK_MM  = 212.2856                 # Expanded Active Volume Stack

_SIPM_THICK_MM   = 0.3
# Dynamically calculate long overhang matching world construction exactly
_CAP_LENGTH_MM   = _CALOR_THICK_MM + 57.7144 
_Z_SENSOR_MM     = _CAP_LENGTH_MM / 2 + _SIPM_THICK_MM / 2   # ≈ 135.15 mm

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL KINEMATICS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS    = 299.792
REFRACTIVE_INDEX = 1.60                          # BCF-92 core index
V_LIGHT_MM_NS    = C_LIGHT_MM_NS / REFRACTIVE_INDEX
BOUNCE_FACTOR    = 0.92                           # set < 1 to account for TIR zig-zag
V_EFF_MM_NS      = V_LIGHT_MM_NS * BOUNCE_FACTOR

_GT_LO_NS = 0.25   # reject Cherenkov from beam halo arriving too early
_GT_HI_NS = 1.5    # reject BCF-92 WLS delayed tail (τ ≈ 2.7 ns)
# ─────────────────────────────────────────────────────────────────────────────
# CAPILLARY XY POSITIONS  (indices 2, 3 are E-type)
# ─────────────────────────────────────────────────────────────────────────────
_HOLE_OFFSET_MM  = 3.7032
CAP_XY_MM = np.array([
    [ _HOLE_OFFSET_MM,  _HOLE_OFFSET_MM],   # 0 — T-type
    [-_HOLE_OFFSET_MM, -_HOLE_OFFSET_MM],   # 1 — T-type
    [-_HOLE_OFFSET_MM,  _HOLE_OFFSET_MM],   # 2 — E-type  ← used here
    [ _HOLE_OFFSET_MM, -_HOLE_OFFSET_MM],   # 3 — E-type  ← used here
])
_E_TYPE_INDICES  = {2, 3}

# Tolerance for SiPM z-position matching (must be > _SIPM_THICK_MM/2 = 0.15 mm)
_SIPM_Z_TOL_MM   = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_lyso_layer_bounds():
    """Return (z_start, z_end) in calorimeter-centred coordinates for each LYSO layer."""
    bounds = []
    current_z = -_CALOR_THICK_MM / 2
    for idx in range(_N_LYSO):
        z_start = current_z + _TYVEK_THICK_MM
        z_end   = z_start + _LYSO_THICK_MM
        bounds.append((z_start, z_end))
        current_z += _GAP_THICK_MM + (_W_THICK_MM if idx < _N_W else 0)
    return bounds


def assign_channel(x_mm, y_mm):
    dists = np.hypot(CAP_XY_MM[:, 0] - x_mm, CAP_XY_MM[:, 1] - y_mm)
    return int(np.argmin(dists))


def load_truth_dose_from_mhd(run_dirs: list):
    if not utils:
        return None
    try:
        long_arr, _ = utils.load_calorimeter_mhd(
            run_dirs,
            long_glob="run_Dose_edep.mhd",
            trans_glob="transverse_shower_max_edep.mhd",
        )
        if long_arr is None:
            return None
        dz_mm = 0.1
        avg   = long_arr / max(len(run_dirs), 1)
        layer_edeps = []
        for (z_start, z_end) in get_lyso_layer_bounds():
            z_offset_start = z_start - (-_CALOR_THICK_MM / 2)
            z_offset_end   = z_end   - (-_CALOR_THICK_MM / 2)
            i0 = max(0, min(int(round(z_offset_start / dz_mm)), len(avg)))
            i1 = max(0, min(int(round(z_offset_end   / dz_mm)), len(avg)))
            layer_edeps.append(float(np.sum(avg[i0:i1])))
        return np.array(layer_edeps)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ToF longitudinal reconstruction using E-type SiPM arrival times."
    )
    parser.add_argument("--batch-dir", required=True, type=str)
    args = parser.parse_args()

    batch_dir     = Path(args.batch_dir)
    lyso_bounds   = get_lyso_layer_bounds()
    calor_half_mm = _CALOR_THICK_MM / 2

    print(f"\n{'─'*60}")
    print(f"  Time-of-Flight Kinematic Profile Reconstruction")
    print(f"  Using: E-type SiPM coincidence ΔT (front vs back face)")
    print(f"  v_eff = {V_EFF_MM_NS:.2f} mm/ns  (n={REFRACTIVE_INDEX}, bounce={BOUNCE_FACTOR})")
    print(f"  GlobalTime window: [{_GT_LO_NS}, {_GT_HI_NS}] ns")
    print(f"{'─'*60}")

    run_dirs  = sorted([d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    hit_files = [p for d in run_dirs for p in sorted(d.glob("detector_hits_*.root"))]

    if not hit_files:
        print("  WARNING: No hit files found.")
        return

    # Per-event earliest GlobalTime on each face.
    # Key: (run_dir_name, event_id) to avoid collisions across runs.
    up_first   = {}   # -> min GlobalTime on upstream face
    down_first = {}   # -> min GlobalTime on downstream face

    n_files_read = 0
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                all_keys = f.keys()
                if not all_keys:
                    continue  # Empty file — no hits this run, skip silently
                tree_key = next(
                    (k for k in all_keys if "detector_hits" in k.split(";")[0]),
                    None
                )
                if tree_key is None:
                    print(f"  WARN: no detector_hits tree in {fpath.name}")
                    continue
                tree = f[tree_key]
                if tree.num_entries == 0:
                    continue
                if "GlobalTime" not in tree.keys():
                    print(f"  WARN: GlobalTime branch missing in {fpath.name}")
                    continue
                x        = tree["Position_X"].array(library="np")
                y        = tree["Position_Y"].array(library="np")
                z        = tree["Position_Z"].array(library="np")
                t        = tree["GlobalTime"].array(library="np")
                event_id = tree["EventID"].array(library="np")
        except Exception as exc:
            print(f"  WARN: could not read {fpath.name}: {exc}")
            continue
        n_files_read += 1

        run_tag = fpath.parent.name   # e.g. "run_3" — scopes EventID per run

        # ── Channel assignment ────────────────────────────────────────────────
        channels  = np.array([assign_channel(xi, yi) for xi, yi in zip(x, y)])
        is_e_type = np.isin(channels, list(_E_TYPE_INDICES))

        # ── GlobalTime prompt window ──────────────────────────────────────────
        is_prompt = (t >= _GT_LO_NS) & (t <= _GT_HI_NS)

        # ── Face identification by z-sign ─────────────────────────────────────
        near_upstream   = np.abs(z + _Z_SENSOR_MM) < _SIPM_Z_TOL_MM
        near_downstream = np.abs(z - _Z_SENSOR_MM) < _SIPM_Z_TOL_MM

        mask_up   = is_e_type & is_prompt & near_upstream
        mask_down = is_e_type & is_prompt & near_downstream

        # ── Per-event earliest arrival time on each face ──────────────────────
        for eid, ti in zip(event_id[mask_up], t[mask_up]):
            key = (run_tag, int(eid))
            if key not in up_first or ti < up_first[key]:
                up_first[key] = float(ti)

        for eid, ti in zip(event_id[mask_down], t[mask_down]):
            key = (run_tag, int(eid))
            if key not in down_first or ti < down_first[key]:
                down_first[key] = float(ti)

    print(f"  Read {n_files_read} ROOT files from {len(run_dirs)} run directories.")
    print(f"  Events with upstream   hit: {len(up_first):,}")
    print(f"  Events with downstream hit: {len(down_first):,}")

    # ── Coincidence: only events with hits on BOTH faces ─────────────────────
    common_keys = set(up_first) & set(down_first)
    print(f"  Coincident events (both faces):  {len(common_keys):,}")

    if len(common_keys) == 0:
        print("  ERROR: No coincident events found. Check geometry constants and GlobalTime window.")
        return

    z_emit_list = []
    for key in common_keys:
        t_up   = up_first[key]    # upstream SiPM   = front face (z = −Z_sensor)
        t_down = down_first[key]  # downstream SiPM = back face  (z = +Z_sensor)
        # Positive ΔT means downstream photon arrives later → emission point
        # is closer to the front (negative z / upstream side).
        # z_emit = v_eff * (t_down − t_up) / 2   (centred at calorimeter midplane)
        delta_t = t_down - t_up
        z_emit  = V_EFF_MM_NS * delta_t / 2.0
        z_emit_list.append(z_emit)

    z_emit_coin = np.array(z_emit_list)
    print(f"  z_emit range: {z_emit_coin.min():.1f} to {z_emit_coin.max():.1f} mm")
    print(f"  z_emit mean:  {z_emit_coin.mean():.1f} mm  (calorimeter centre = 0, front = −{calor_half_mm:.1f} mm)")

    # ── Physical-space filter ─────────────────────────────────────────────────
    margin_mm  = 15.0
    z_lo, z_hi = -calor_half_mm - margin_mm, calor_half_mm + margin_mm
    valid_coin  = z_emit_coin[(z_emit_coin >= z_lo) & (z_emit_coin <= z_hi)]
    print(f"  After physical-bounds filter: {len(valid_coin):,} events retained")

    # ── KDE → layer profile ───────────────────────────────────────────────────
    def kde_profile(valid_z, n_layers=_N_LYSO, bounds=lyso_bounds):
        profile = np.zeros(n_layers)
        if len(valid_z) < 5:
            for i, (z_min, z_max) in enumerate(bounds):
                profile[i] = np.sum((valid_z >= z_min) & (valid_z <= z_max))
        else:
            kde = gaussian_kde(valid_z, bw_method=0.15)
            for i, (z_min, z_max) in enumerate(bounds):
                z_mid      = (z_min + z_max) / 2.0
                profile[i] = kde.evaluate(z_mid)[0]
        return profile

    profile_coin = kde_profile(valid_coin)

    # Mirror so layer 1 = beam-entry face (−z) and layer 29 = back face (+z)
    profile_coin = profile_coin[::-1]

    # ── Scale to truth if available ───────────────────────────────────────────
    truth_curve = load_truth_dose_from_mhd(run_dirs)

    def scale_to_truth(profile, truth):
        if truth is not None and np.sum(profile) > 0:
            return profile * (np.sum(truth) / np.sum(profile))
        return profile

    profile_coin_sc = scale_to_truth(profile_coin, truth_curve)

    # ── Plot ──────────────────────────────────────────────────────────────────
    layers = np.arange(1, _N_LYSO + 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    if truth_curve is not None:
        ax.bar(layers, truth_curve, color="#00bcd4", alpha=0.45,
               edgecolor="#00838f", linewidth=1.0, width=0.8,
               label="Simulation Truth (DoseActor)")

    ax.plot(layers, profile_coin_sc, color="#6a1b9a", linewidth=2.5,
            marker="o", markersize=5,
            label=f"ΔT Coincidence Reconstruction  (n={len(valid_coin)} events)")

    ax.set_xlabel("LYSO Layer Number")
    ax.set_ylabel("Energy / Scaled Hits (MeV)")
    ax.set_title("E-type SiPM ToF Reconstruction — ΔT Coincidence vs Truth")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"ToF Longitudinal Profile  |  v_eff = {V_EFF_MM_NS:.1f} mm/ns"
        f"  |  GlobalTime window [{_GT_LO_NS}, {_GT_HI_NS}] ns",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    out_path = batch_dir / "tof_reconstruction_etype_sipm.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved plot → {out_path}")

    # ── Sanity diagnostics ────────────────────────────────────────────────────
    peak_coin = int(np.argmax(profile_coin)) + 1
    print(f"\n  Peak layer (ΔT coincidence): {peak_coin}")
    if truth_curve is not None:
        peak_truth = int(np.argmax(truth_curve)) + 1
        print(f"  Peak layer (truth):          {peak_truth}")
    print()


if __name__ == "__main__":
    main()