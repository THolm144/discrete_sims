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
# GEOMETRY CONSTANTS  (must match radi_cal_energy.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
_LYSO_THICK_MM   = 1.5
_TYVEK_THICK_MM  = 0.2032
_W_THICK_MM      = 2.5
_N_LYSO          = 29
_N_W             = 28
_GAP_THICK_MM    = _LYSO_THICK_MM + 2 * _TYVEK_THICK_MM
_CALOR_THICK_MM  = 125.2856

_CAP_LENGTH_MM   = 183.0
_SIPM_THICK_MM   = 0.3
# z-coordinate of the SiPM face centres (±, symmetric about origin)
_Z_SENSOR_MM     = _CAP_LENGTH_MM / 2 + _SIPM_THICK_MM / 2   # ≈ 91.65 mm

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL KINEMATICS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MM_NS    = 299.792
REFRACTIVE_INDEX = 1.60                          # BCF-92 core index
V_LIGHT_MM_NS    = C_LIGHT_MM_NS / REFRACTIVE_INDEX
BOUNCE_FACTOR    = 1.0                           # set < 1 to account for TIR zig-zag
V_EFF_MM_NS      = V_LIGHT_MM_NS * BOUNCE_FACTOR

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
    parser.add_argument(
        "--prompt-cut-ns", type=float, default=0.4,
        help="Max LocalTime (ns) to accept as a prompt photon (default: 0.4 ns)."
    )
    args = parser.parse_args()

    batch_dir      = Path(args.batch_dir)
    prompt_cut_ns  = args.prompt_cut_ns
    lyso_bounds    = get_lyso_layer_bounds()
    calor_half_mm  = _CALOR_THICK_MM / 2

    print(f"\n{'─'*60}")
    print(f"  Time-of-Flight Kinematic Profile Reconstruction")
    print(f"  Using: E-type SiPM face arrival times")
    print(f"  v_eff = {V_EFF_MM_NS:.2f} mm/ns  (n={REFRACTIVE_INDEX}, bounce={BOUNCE_FACTOR})")
    print(f"  Prompt-photon cut: LocalTime < {prompt_cut_ns} ns")
    print(f"{'─'*60}")

    run_dirs  = sorted([d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    hit_files = [p for d in run_dirs for p in sorted(d.glob("detector_hits_*.root"))]

    if not hit_files:
        print("  WARNING: No hit files found.")
        return

    # Accumulate reconstructed emission-z values from both SiPM faces separately
    # so we can cross-check and overlay them in the final plot.
    z_emit_upstream   = []   # from sipm_front_2/3  (z < 0 face)
    z_emit_downstream = []   # from sipm_back_2/3   (z > 0 face)

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

        # ── Channel assignment ────────────────────────────────────────────────
        channels = np.array([assign_channel(xi, yi) for xi, yi in zip(x, y)])
        is_e_type = np.isin(channels, list(_E_TYPE_INDICES))

        # ── Prompt-photon cut (remove BCF-92 WLS delayed tail ~2.7 ns) ───────
        #is_prompt = t < prompt_cut_ns

        # ── Face identification by z-sign ─────────────────────────────────────
        # Upstream SiPMs are at z ≈ −_Z_SENSOR_MM
        # Downstream SiPMs are at z ≈ +_Z_SENSOR_MM
        near_upstream   = np.abs(z + _Z_SENSOR_MM) < _SIPM_Z_TOL_MM
        near_downstream = np.abs(z - _Z_SENSOR_MM) < _SIPM_Z_TOL_MM

        mask_up   = is_e_type  & near_upstream
        mask_down = is_e_type  & near_downstream
        if np.any(mask_up):
            t_up = t[mask_up]
            print(f"  Upstream   GlobalTime: min={t_up.min():.3f}  max={t_up.max():.3f}  mean={t_up.mean():.3f} ns  n={len(t_up)}")

        if np.any(mask_down):
            t_dn = t[mask_down]
            print(f"  Downstream GlobalTime: min={t_dn.min():.3f}  max={t_dn.max():.3f}  mean={t_dn.mean():.3f} ns  n={len(t_dn)}")

        # ── Kinematic back-projection ─────────────────────────────────────────
        # Upstream face (z = −Z_sensor): photon travelled in −z direction from
        # emission point, so  z_emit = −Z_sensor + t·v_eff
        if np.any(mask_up):
            z_emit_upstream.append(-_Z_SENSOR_MM + t[mask_up] * V_EFF_MM_NS)

        # Downstream face (z = +Z_sensor): photon travelled in +z direction,
        # so  z_emit = +Z_sensor − t·v_eff
        if np.any(mask_down):
            z_emit_downstream.append(_Z_SENSOR_MM - t[mask_down] * V_EFF_MM_NS)

    print(f"  Read {n_files_read} ROOT files from {len(run_dirs)} run directories.")

    z_emit_up   = np.concatenate(z_emit_upstream)   if z_emit_upstream   else np.array([])
    z_emit_down = np.concatenate(z_emit_downstream) if z_emit_downstream else np.array([])

    print(f"  Upstream   E-type SiPM hits (prompt): {len(z_emit_up):,}")
    print(f"  Downstream E-type SiPM hits (prompt): {len(z_emit_down):,}")

    # ── Physical-space filter ─────────────────────────────────────────────────
    margin_mm  = 15.0
    z_lo, z_hi = -calor_half_mm - margin_mm, calor_half_mm + margin_mm

    valid_up   = z_emit_up  [(z_emit_up   >= z_lo) & (z_emit_up   <= z_hi)]
    valid_down = z_emit_down[(z_emit_down >= z_lo) & (z_emit_down <= z_hi)]

    print(f"  After physical-bounds filter:")
    print(f"    Upstream   retained: {len(valid_up):,}")
    print(f"    Downstream retained: {len(valid_down):,}")

    # ── KDE → layer histogram for each face ───────────────────────────────────
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

    profile_up   = kde_profile(valid_up)
    profile_down = kde_profile(valid_down)

    # Combine: simple sum of both faces (doubles statistics, consistent direction)
    profile_combined = profile_up + profile_down

    # Mirror so layer 1 = beam-entry face (−z) and layer 29 = back face (+z)
    profile_up       = profile_up      [::-1]
    profile_down     = profile_down    [::-1]
    profile_combined = profile_combined[::-1]

    # ── Scale to truth if available ───────────────────────────────────────────
    truth_curve = load_truth_dose_from_mhd(run_dirs)

    def scale_to_truth(profile, truth):
        if truth is not None and np.sum(profile) > 0:
            return profile * (np.sum(truth) / np.sum(profile))
        return profile

    profile_up_sc   = scale_to_truth(profile_up,       truth_curve)
    profile_down_sc = scale_to_truth(profile_down,     truth_curve)
    profile_comb_sc = scale_to_truth(profile_combined, truth_curve)

    # ── Plot ──────────────────────────────────────────────────────────────────
    layers = np.arange(1, _N_LYSO + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # ── Top panel: individual faces ───────────────────────────────────────────
    ax = axes[0]
    if truth_curve is not None:
        ax.bar(layers, truth_curve, color="#00bcd4", alpha=0.45,
               edgecolor="#00838f", linewidth=1.0, width=0.8,
               label="Simulation Truth (DoseActor)")
    ax.plot(layers, profile_up_sc,   color="#1565c0", linewidth=2.0,
            marker="^", markersize=5, label="Upstream SiPM (sipm_front_2/3) — ToF")
    ax.plot(layers, profile_down_sc, color="#b71c1c", linewidth=2.0,
            marker="v", markersize=5, label="Downstream SiPM (sipm_back_2/3) — ToF")
    ax.set_ylabel("Energy / Scaled Hits (MeV)")
    ax.set_title("E-type SiPM ToF Reconstruction — Individual Faces vs Truth")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    # ── Bottom panel: combined ────────────────────────────────────────────────
    ax = axes[1]
    if truth_curve is not None:
        ax.bar(layers, truth_curve, color="#00bcd4", alpha=0.45,
               edgecolor="#00838f", linewidth=1.0, width=0.8,
               label="Simulation Truth (DoseActor)")
    ax.plot(layers, profile_comb_sc, color="#6a1b9a", linewidth=2.5,
            marker="o", markersize=5,
            label="Combined (upstream + downstream, KDE smoothed)")
    ax.set_xlabel("LYSO Layer Number")
    ax.set_ylabel("Energy / Scaled Hits (MeV)")
    ax.set_title("E-type SiPM ToF Reconstruction — Combined vs Truth")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"ToF Longitudinal Profile  |  v_eff = {V_EFF_MM_NS:.1f} mm/ns"
        f"  |  prompt cut = {prompt_cut_ns} ns",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    out_path = batch_dir / "tof_reconstruction_etype_sipm.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved plot → {out_path}")

    # ── Sanity diagnostics ────────────────────────────────────────────────────
    peak_up   = int(np.argmax(profile_up))   + 1
    peak_down = int(np.argmax(profile_down)) + 1
    peak_comb = int(np.argmax(profile_combined)) + 1
    print(f"\n  Peak layer (upstream face):   {peak_up}")
    print(f"  Peak layer (downstream face): {peak_down}")
    print(f"  Peak layer (combined):        {peak_comb}")
    if truth_curve is not None:
        peak_truth = int(np.argmax(truth_curve)) + 1
        print(f"  Peak layer (truth):           {peak_truth}")
    print()


if __name__ == "__main__":
    main()