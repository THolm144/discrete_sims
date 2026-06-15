"""
worlds/radi_cal.py
==================
RADiCAL Shashlik calorimeter — 29-layer LYSO/W sampling calorimeter
with embedded quartz capillaries and a DSB1 wavelength-shifting centre capillary.

New-contract exports
--------------------
CAPABILITIES, BEAM_CONFIG, PHANTOM_CM, TARGET_VOLUME_NAME,
DETECTOR_VOLUME_NAMES, analyze(), get_geometry_primitives()
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

CAPABILITIES = {
    "optical":          True,
    "dose":             True,
    "sipm_hits":        True,
    "optical_exits":    True,
    "calorimeter_mode": True,
}

# Beam enters the -Z face, traveling in +Z direction
BEAM_CONFIG = {
    "direction": [0, 0, 1],
    "target_cm": [0, 0, 0],
    "offset_cm": 2.0,
}

TARGET_VOLUME_NAME = "target_volume"

PHANTOM_CM = [1.4, 1.4, 18.3]

EXPECTED_DEDX = 1.0

ACTIVATE_CALORIMETER_SETTINGS = True
CALORIMETER_Z_RES_MM = 0.1

ACTIVE_Z_RANGES_MM = [
    [0.0, PHANTOM_CM[2] * 10.0]
]

# Minimum photons per event per channel to fire the timing trigger.
# With 10 SiPM channels and ~5.7 hits/primary on average, threshold=1
# ensures we capture timing data. Raise this as statistics improve.
TIMING_TRIGGER_THRESHOLD = 1

DETECTOR_VOLUME_NAMES = [
    "sipm_up_1", "sipm_up_2", "sipm_up_3", "sipm_up_4", "sipm_up_c",
    "sipm_dn_1", "sipm_dn_2", "sipm_dn_3", "sipm_dn_4", "sipm_dn_c",
]

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL GEOMETRY CONSTANTS (used by build_world + get_geometry_primitives)
# ─────────────────────────────────────────────────────────────────────────────

_LYSO_THICK_MM   = 1.5
_W_THICK_MM      = 2.5
_TYVEK_THICK_MM  = 0.2
_CAP_OUTER_MM    = 0.575
_CAP_INNER_MM    = 0.475
_DSB_LENGTH_MM   = 15.0
_DSB_CENTER_Z_MM = -17.6
_CAP_TOTAL_MM    = 183.0
_SIPM_THICK_MM   = 1.0
_SIPM_XY_MM      = 1.5

_CAP_POSITIONS_MM = [
    [0,    0   ],
    [-3.5, -3.5],
    [-3.5,  3.5],
    [ 3.5, -3.5],
    [ 3.5,  3.5],
]


# ─────────────────────────────────────────────────────────────────────────────
# WORLD CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_world(sim, units):
    world = sim.world
    world.size     = [50 * units.cm, 50 * units.cm, 50 * units.cm]
    world.material = "Air"

    lyso_thick  = _LYSO_THICK_MM  * units.mm
    w_thick     = _W_THICK_MM     * units.mm
    tyvek_thick = _TYVEK_THICK_MM * units.mm

    stack_thick = (
        29 * lyso_thick
        + 28 * w_thick
        + 56 * tyvek_thick
    )

    cap_total = _CAP_TOTAL_MM * units.mm
    dsb_z_min = (_DSB_CENTER_Z_MM - _DSB_LENGTH_MM / 2) * units.mm
    dsb_z_max = (_DSB_CENTER_Z_MM + _DSB_LENGTH_MM / 2) * units.mm

    # Target volume container (Acts as the overarching experimental envelope)
    target = sim.add_volume("Box", TARGET_VOLUME_NAME)
    target.mother   = "world"
    target.size     = [14 * units.mm, 14 * units.mm, cap_total]
    target.material = "Air"

    cap_positions = [
        [x * units.mm, y * units.mm]
        for x, y in _CAP_POSITIONS_MM
    ]

    # 1. BUILD CONTINUOUS CAPILLARIES DIRECTLY IN THE TARGET ENVELOPE
    _build_continuous_capillaries(sim, cap_positions, units, cap_total, dsb_z_min, dsb_z_max)

    # 2. BUILD SOLID LAYERS (Geant4 handles the nested volumes automatically)
    front_gap_thick = (cap_total - stack_thick) / 2.0
    current_z = -(stack_thick / 2.0)

    # Front air extension
    _build_solid_plate(sim, "front_gap", front_gap_thick, "Air", 
                       current_z - front_gap_thick / 2.0, units)

    # Main structural stacking
    for layer_idx in range(29):
        current_z += lyso_thick / 2.0
        _build_solid_plate(sim, f"lyso_{layer_idx}", lyso_thick, "LYSO", 
                           current_z, units)
        current_z += lyso_thick / 2.0

        if layer_idx < 28:
            for label, thick, mat in [("tyvek_f", tyvek_thick, "Tyvek"),
                                      ("w",       w_thick,     "Tungsten"),
                                      ("tyvek_b", tyvek_thick, "Tyvek")]:
                current_z += thick / 2.0
                _build_solid_plate(sim, f"{label}_{layer_idx}", thick, mat, 
                                     current_z, units)
                current_z += thick / 2.0

    # Back air extension
    _build_solid_plate(sim, "back_gap", front_gap_thick, "Air", 
                       current_z + front_gap_thick / 2.0, units)

    # 3. SIPMs
    sipm_thick = _SIPM_THICK_MM * units.mm
    sipm_xy    = _SIPM_XY_MM   * units.mm
    z_up = -(cap_total / 2.0) - sipm_thick / 2.0
    z_dn =  (cap_total / 2.0) + sipm_thick / 2.0

    up_names = ["sipm_up_c", "sipm_up_1", "sipm_up_2", "sipm_up_3", "sipm_up_4"]
    dn_names = ["sipm_dn_c", "sipm_dn_1", "sipm_dn_2", "sipm_dn_3", "sipm_dn_4"]

    for i, (cx, cy) in enumerate(cap_positions):
        for name, z_pos in [(up_names[i], z_up), (dn_names[i], z_dn)]:
            sv = sim.add_volume("Box", name)
            sv.mother      = "world"
            sv.size        = [sipm_xy, sipm_xy, sipm_thick]
            sv.translation = [cx, cy, z_pos]
            sv.material    = "G4_Si"

    return sim


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_continuous_capillaries(sim, cap_positions, units, cap_total, dsb_z_min, dsb_z_max):
    """Creates unbroken capillary tubes spanning the full target length."""
    dsb_z_center = _DSB_CENTER_Z_MM * units.mm
    dsb_length   = _DSB_LENGTH_MM * units.mm

    for cap_idx, (cx, cy) in enumerate(cap_positions):
        if cap_idx == 0:
            hole = sim.add_volume("Tubs", "center_air_channel")
            hole.mother      = TARGET_VOLUME_NAME
            hole.rmin        = 0.0
            hole.rmax        = _CAP_OUTER_MM * units.mm
            hole.dz          = cap_total / 2.0
            hole.translation = [cx, cy, 0]
            hole.material    = "Air"
            continue

        # Continuous Quartz Wall outer sheath
        wall = sim.add_volume("Tubs", f"cap_{cap_idx}_wall")
        wall.mother      = TARGET_VOLUME_NAME
        wall.rmin        = _CAP_INNER_MM * units.mm
        wall.rmax        = _CAP_OUTER_MM * units.mm
        wall.dz          = cap_total / 2.0
        wall.translation = [cx, cy, 0]
        wall.material    = "Quartz"

        # Inner fluid/core split into 3 segments along Z
        up_len = (dsb_z_min - (-cap_total / 2.0))
        if up_len > 0:
            up_core = sim.add_volume("Tubs", f"cap_{cap_idx}_core_up")
            up_core.mother      = TARGET_VOLUME_NAME
            up_core.rmin        = 0.0
            up_core.rmax        = _CAP_INNER_MM * units.mm
            up_core.dz          = up_len / 2.0
            up_core.translation = [cx, cy, -cap_total/2.0 + up_len/2.0]
            up_core.material    = "Quartz"

        dsb_core = sim.add_volume("Tubs", f"cap_{cap_idx}_core_dsb1")
        dsb_core.mother      = TARGET_VOLUME_NAME
        dsb_core.rmin        = 0.0
        dsb_core.rmax        = _CAP_INNER_MM * units.mm
        dsb_core.dz          = dsb_length / 2.0
        dsb_core.translation = [cx, cy, dsb_z_center]
        dsb_core.material    = "DSB1"

        dn_len = ((cap_total / 2.0) - dsb_z_max)
        if dn_len > 0:
            dn_core = sim.add_volume("Tubs", f"cap_{cap_idx}_core_dn")
            dn_core.mother      = TARGET_VOLUME_NAME
            dn_core.rmin        = 0.0
            dn_core.rmax        = _CAP_INNER_MM * units.mm
            dn_core.dz          = dn_len / 2.0
            dn_core.translation = [cx, cy, cap_total/2.0 - dn_len/2.0]
            dn_core.material    = "Quartz"


def _build_solid_plate(sim, name, thickness, material, z_pos, units):
    """Creates a clean solid slab block without programmatic errors."""
    plate = sim.add_volume("Box", name)
    plate.mother      = TARGET_VOLUME_NAME
    plate.size        = [14 * units.mm, 14 * units.mm, thickness]
    plate.translation = [0, 0, z_pos]
    plate.material    = material


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HOOK
# ─────────────────────────────────────────────────────────────────────────────

def analyze(batch_dir, run_dirs, meta, utils):
    """
    RADiCAL-specific analysis:
    - Longitudinal shower profile aggregated across runs
    - Transverse shower core at shower maximum
    - SiPM hits split by upstream / downstream
    - Calorimeter batch aggregation → analyzed_longitudinal_profile.txt
    """
    import matplotlib.pyplot as plt

    hits_files  = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    exits_files = [d / "optical_exited.root" for d in run_dirs]

    hits       = utils.analyse_hits(hits_files)
    exits      = utils.analyse_exits(exits_files)
    timing_res = (utils.extract_timing_resolution(hits_files,
                      threshold_photon=TIMING_TRIGGER_THRESHOLD)
                  if hits_files else 0.0)

    # Longitudinal + transverse .mhd accumulation
    long_arr, trans_arr = utils.load_calorimeter_mhd(
        run_dirs,
        long_glob  = "run_Dose_edep.mhd",
        trans_glob = "transverse_shower_max_edep.mhd",
    )

    # Aggregate batch .txt / .npy files → analyzed_longitudinal_profile.txt
    _aggregate_batch(batch_dir, run_dirs, meta, utils)

    extra_lines = []
    plots_saved = []

    # Longitudinal profile plot
    if long_arr is not None:
        dz_mm = meta.get("dose_spacing_mm", 0.1)
        bins  = np.arange(len(long_arr)) * dz_mm
        avg   = long_arr / max(len(run_dirs), 1)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(bins, avg, color="#d32f2f", lw=2, label="Avg edep per bin")
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Energy deposition (MeV)")
        ax.set_title(f"RADiCAL longitudinal profile — {len(run_dirs)} run(s)")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        out = batch_dir / "radical_longitudinal_profile.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        plots_saved.append(out.name)
        extra_lines.append(f"  Peak edep bin: {bins[np.argmax(avg)]:.1f} mm")
        extra_lines.append(f"  Peak edep val: {avg.max():.4f} MeV")

    # Transverse heatmap
    if trans_arr is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ext = [-7, 7, -7, 7]
        im  = ax.imshow(trans_arr / max(len(run_dirs), 1),
                        extent=ext, cmap="inferno", origin="lower")
        fig.colorbar(im, ax=ax, label="Energy density")
        ax.set_title("Transverse shower core at shower maximum")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        fig.tight_layout()
        out = batch_dir / "radical_transverse_core.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        plots_saved.append(out.name)

    # Upstream vs downstream SiPM hit split
    up_hits = sum(
        hits.get(k, 0)
        for k in hits
        if any(n in k for n in ["sipm_up", "sipm_up_c"])
    )
    dn_hits = sum(hits.values()) - up_hits
    extra_lines += [
        f"  Upstream SiPM hits   : {up_hits:,}",
        f"  Downstream SiPM hits : {dn_hits:,}",
    ]

    return {
        "hits":          hits,
        "exits":         exits,
        "dose_centers":  None,   # calorimeter mode — no standard dose profile
        "dose_edep":     None,
        "timing_res_ps": timing_res,
        "extra_lines":   extra_lines,
        "plots_saved":   plots_saved,
    }


def _aggregate_batch(batch_dir, run_dirs, meta, utils):
    """Write analyzed_longitudinal_profile.txt from per-run _Dose.txt files."""
    dz_mm         = meta.get("dose_spacing_mm", 0.1)
    active_ranges = meta.get("active_z_ranges", None)
    long_acc      = None
    n             = 0

    for run_dir in run_dirs:
        run_id   = int(run_dir.name.split("_")[-1])
        dose_txt = batch_dir / f"run_{run_id}_Dose.txt"
        if dose_txt.exists():
            try:
                _, energy = np.loadtxt(dose_txt, unpack=True, usecols=(0, 1))
                long_acc   = energy.astype(float) if long_acc is None else long_acc + energy
                n         += 1
            except Exception:
                pass

    if n == 0 or long_acc is None:
        return

    avg = long_acc / n
    out = batch_dir / "analyzed_longitudinal_profile.txt"

    if active_ranges:
        layer_energies = []
        for z_start, z_end in active_ranges:
            i0 = int(round(z_start / dz_mm))
            i1 = int(round(z_end   / dz_mm))
            layer_energies.append(np.sum(avg[i0:i1]))
        np.savetxt(str(out),
                   np.c_[np.arange(len(layer_energies)) + 1, layer_energies],
                   header="Layer_ID Deposited_Energy", fmt="%d %.6e")
    else:
        np.savetxt(str(out),
                   np.c_[np.arange(len(avg)), avg],
                   header="Bin_ID Normalized_Energy", fmt="%d %.6e")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT_3D HOOK
# ─────────────────────────────────────────────────────────────────────────────

def get_geometry_primitives() -> list[dict]:
    prims = []
    cap_total_cm = _CAP_TOTAL_MM / 10.0

    # Target bounding box
    prims.append({
        "type":      "box",
        "center":    [0.0, 0.0, 0.0],
        "half":      [0.7, 0.7, cap_total_cm / 2],
        "color":     "#00ffcc",
        "label":     "Calorimeter stack",
        "alpha":     0.2,
        "linewidth": 0.8,
    })

    # Capillaries (outer envelope only, one per corner + center)
    for x_mm, y_mm in _CAP_POSITIONS_MM:
        prims.append({
            "type":   "tube",
            "center": [x_mm / 10.0, y_mm / 10.0, 0.0],
            "rmax":   _CAP_OUTER_MM / 10.0,
            "height": cap_total_cm,
            "color":  "#00cfff",
            "label":  "Capillary" if (x_mm, y_mm) == _CAP_POSITIONS_MM[1] else "",
            "alpha":  0.5,
            "linewidth": 0.6,
        })

    # DSB1 filament
    dsb_cm = _DSB_LENGTH_MM / 10.0
    prims.append({
        "type":   "tube",
        "center": [0.0, 0.0, _DSB_CENTER_Z_MM / 10.0],
        "rmax":   _CAP_INNER_MM / 10.0,
        "height": dsb_cm,
        "color":  "#ff9800",
        "label":  "DSB1 filament",
        "alpha":  0.9,
        "linewidth": 1.2,
    })

    # Upstream and downstream SiPMs
    z_up_cm = -(cap_total_cm / 2.0) - _SIPM_THICK_MM / 10.0 / 2.0
    z_dn_cm =  (cap_total_cm / 2.0) + _SIPM_THICK_MM / 10.0 / 2.0
    for x_mm, y_mm in _CAP_POSITIONS_MM:
        for z_cm, lbl in [(z_up_cm, "SiPM upstream" if (x_mm, y_mm) == _CAP_POSITIONS_MM[0] else ""),
                          (z_dn_cm, "SiPM downstream" if (x_mm, y_mm) == _CAP_POSITIONS_MM[0] else "")]:
            prims.append({
                "type":      "box",
                "center":    [x_mm / 10.0, y_mm / 10.0, z_cm],
                "half":      [_SIPM_XY_MM / 20.0, _SIPM_XY_MM / 20.0, _SIPM_THICK_MM / 20.0],
                "color":     "#f1c40f",
                "label":     lbl,
                "alpha":     0.9,
                "linewidth": 1.2,
            })

    return prims

#test change