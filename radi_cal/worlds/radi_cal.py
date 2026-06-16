"""
worlds/radi_cal.py
==================
RADiCAL Shashlik calorimeter — 29-layer LYSO/W sampling calorimeter
with embedded quartz capillaries and a air centre capillary.

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

TIMING_TRIGGER_THRESHOLD = 1

# Screen volume names match old SiPM names so analyze.py/metadata are unchanged
# Sensitive detector volumes: capillary tubes in the gap layers.
# front_gap = upstream face, back_gap = downstream face.
# cap_0_air = centre air bore; cap_N_core = quartz core of outer capillaries.
DETECTOR_VOLUME_NAMES = [
    "front_gap_cap_0_air",
    "front_gap_cap_1_core", "front_gap_cap_2_core",
    "front_gap_cap_3_core", "front_gap_cap_4_core",
    "back_gap_cap_0_air",
    "back_gap_cap_1_core", "back_gap_cap_2_core",
    "back_gap_cap_3_core", "back_gap_cap_4_core",
]

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL GEOMETRY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_LYSO_THICK_MM   = 1.5
_W_THICK_MM      = 2.5
_TYVEK_THICK_MM  = 0.2
_CAP_OUTER_MM    = 0.575
_CAP_INNER_MM    = 0.475
_DSB_LENGTH_MM   = 15.0
_DSB_CENTER_Z_MM = -17.6
_CAP_TOTAL_MM    = 183.0
_SCREEN_THICK_MM = 0.1   # thin air screen — transparent to optical photons
_SCREEN_XY_MM    = 1.5   # slightly larger than capillary outer diameter

_CAP_POSITIONS_MM = [
    [0,    0   ],
    [-3.5, -3.5],
    [-3.5,  3.5],
    [ 3.5, -3.5],
    [ 3.5,  3.5],
]


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_layer_with_capillaries(sim, name, thickness, material, z_pos, units):
    """
    Creates a solid layer plate and populates it with local capillary segments.
    The daughter segments automatically displace the plate's base material.
    """
    plate = sim.add_volume("Box", name)
    plate.mother = TARGET_VOLUME_NAME
    plate.size = [14 * units.mm, 14 * units.mm, thickness]
    plate.material = material
    plate.translation = [0, 0, z_pos]

    z_min = z_pos - thickness / 2.0
    z_max = z_pos + thickness / 2.0
    dsb_z_min = (_DSB_CENTER_Z_MM - _DSB_LENGTH_MM / 2.0) * units.mm
    dsb_z_max = (_DSB_CENTER_Z_MM + _DSB_LENGTH_MM / 2.0) * units.mm

    cap_dz = (thickness / 2.0)

    for cap_idx, (cx, cy) in enumerate(_CAP_POSITIONS_MM):
        cx_g4 = cx * units.mm
        cy_g4 = cy * units.mm

        if cap_idx == 0:
            air_seg = sim.add_volume("Tubs", f"{name}_cap_0_air")
            air_seg.mother = name
            air_seg.rmin = 0.0
            air_seg.rmax = _CAP_OUTER_MM * units.mm
            air_seg.dz = cap_dz
            air_seg.translation = [cx_g4, cy_g4, 0]
            air_seg.material = "Air"
        else:
            wall_seg = sim.add_volume("Tubs", f"{name}_cap_{cap_idx}_wall")
            wall_seg.mother = name
            wall_seg.rmin = _CAP_INNER_MM * units.mm
            wall_seg.rmax = _CAP_OUTER_MM * units.mm
            wall_seg.dz = cap_dz
            wall_seg.translation = [cx_g4, cy_g4, 0]
            wall_seg.material = "Quartz"

            if z_min >= dsb_z_min and z_max <= dsb_z_max:
                core_material = "DSB1"
            else:
                core_material = "Quartz"

            core_seg = sim.add_volume("Tubs", f"{name}_cap_{cap_idx}_core")
            core_seg.mother = name
            core_seg.rmin = 0.0
            core_seg.rmax = _CAP_INNER_MM * units.mm
            core_seg.dz = cap_dz
            core_seg.translation = [cx_g4, cy_g4, 0]
            core_seg.material = core_material

    return plate




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
    cap_total   = _CAP_TOTAL_MM   * units.mm

    stack_thick = (29 * lyso_thick) + (28 * w_thick) + (56 * tyvek_thick)

    front_gap_thick = (cap_total - stack_thick) / 2.0
    back_gap_thick  = front_gap_thick

    # Target volume container
    target = sim.add_volume("Box", TARGET_VOLUME_NAME)
    target.mother   = "world"
    target.size     = [14 * units.mm, 14 * units.mm, cap_total]
    target.material = "Air"

    # Start layout from the far upstream (-Z) boundary of the stack
    current_z = -(stack_thick / 2.0)

    # 1. Front gap
    _build_layer_with_capillaries(sim, "front_gap", front_gap_thick, "Air",
                                  current_z - front_gap_thick / 2.0, units)

    # 2. Main absorber/scintillator stack
    for layer_idx in range(29):
        current_z += lyso_thick / 2.0
        _build_layer_with_capillaries(sim, f"lyso_{layer_idx}", lyso_thick, "LYSO", current_z, units)
        current_z += lyso_thick / 2.0

        if layer_idx < 28:
            for label, thick, mat in [("tyvek_f", tyvek_thick, "Tyvek"),
                                      ("w",       w_thick,     "Tungsten"),
                                      ("tyvek_b", tyvek_thick, "Tyvek")]:
                current_z += thick / 2.0
                _build_layer_with_capillaries(sim, f"{label}_{layer_idx}", thick, mat, current_z, units)
                current_z += thick / 2.0

    # 3. Back gap
    _build_layer_with_capillaries(sim, "back_gap", back_gap_thick, "Air",
                                  current_z + back_gap_thick / 2.0, units)


    return sim


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HOOK
# ─────────────────────────────────────────────────────────────────────────────

def analyze(batch_dir, run_dirs, meta, utils):
    import matplotlib.pyplot as plt

    hits_files  = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    exits_files = [d / "optical_exited.root" for d in run_dirs]

    hits       = utils.analyse_hits(hits_files)
    exits      = utils.analyse_exits(exits_files)
    timing_res = (utils.extract_timing_resolution(hits_files,
                      threshold_photon=TIMING_TRIGGER_THRESHOLD)
                  if hits_files else 0.0)

    long_arr, trans_arr = utils.load_calorimeter_mhd(
        run_dirs,
        long_glob  = "run_Dose_edep.mhd",
        trans_glob = "transverse_shower_max_edep.mhd",
    )

    _aggregate_batch(batch_dir, run_dirs, meta, utils)

    extra_lines = []
    plots_saved = []

    if long_arr is not None:
        dz_mm = meta.get("dose_spacing_mm", 0.1)
        avg   = long_arr / max(len(run_dirs), 1)

        lyso_thick  = _LYSO_THICK_MM
        w_thick     = _W_THICK_MM
        tyvek_thick = _TYVEK_THICK_MM
        cap_total   = _CAP_TOTAL_MM

        stack_thick = (29 * lyso_thick) + (28 * w_thick) + (56 * tyvek_thick)
        front_gap   = (cap_total - stack_thick) / 2.0

        layer_numbers = list(range(1, 30))
        layer_edeps   = []

        current_z = front_gap

        for layer_idx in range(29):
            z_start = current_z
            z_end   = current_z + lyso_thick

            idx_start = int(round(z_start / dz_mm))
            idx_end   = int(round(z_end   / dz_mm))

            idx_start = max(0, min(idx_start, len(avg)))
            idx_end   = max(0, min(idx_end,   len(avg)))

            layer_energy = float(np.sum(avg[idx_start:idx_end]))
            layer_edeps.append(layer_energy)

            current_z += lyso_thick
            if layer_idx < 28:
                current_z += (tyvek_thick * 2) + w_thick

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(layer_numbers, layer_edeps, color="#00bcd4", alpha=0.7,
               edgecolor="#00838f", linewidth=1.2, width=0.8, label="LYSO Layer Total")
        ax.plot(layer_numbers, layer_edeps, color="#00838f", linestyle="--",
                alpha=0.5, marker="o", ms=4)
        ax.set_xlabel("LYSO Layer Number")
        ax.set_ylabel("Integrated Energy Deposition (MeV)")
        ax.set_title(f"RADiCAL Longitudinal Profile by Active Layer — {len(run_dirs)} run(s)")
        ax.set_xticks(layer_numbers)
        ax.set_xticklabels([str(n) for n in layer_numbers], fontsize=8)
        ax.set_xlim(0, 30)
        ax.grid(True, alpha=0.2, linestyle="--", axis="y")
        ax.legend(loc="upper right")
        fig.tight_layout()
        out = batch_dir / "radical_longitudinal_profile.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        plots_saved.append(out.name)

        if layer_edeps:
            max_layer_idx = np.argmax(layer_edeps)
            extra_lines.append(f"  Peak LYSO Layer: {layer_numbers[max_layer_idx]}")
            extra_lines.append(f"  Peak Layer edep: {layer_edeps[max_layer_idx]:.4f} MeV")

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

    up_hits = sum(
        hits.get(k, 0)
        for k in hits
        if any(n in k for n in ["sipm_up", "sipm_up_c"])
    )
    dn_hits = sum(hits.values()) - up_hits
    extra_lines += [
        f"  Upstream screen hits   : {up_hits:,}",
        f"  Downstream screen hits : {dn_hits:,}",
    ]

    return {
        "hits":          hits,
        "exits":         exits,
        "dose_centers":  None,
        "dose_edep":     None,
        "timing_res_ps": timing_res,
        "extra_lines":   extra_lines,
        "plots_saved":   plots_saved,
    }


def _aggregate_batch(batch_dir, run_dirs, meta, utils):
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

    prims.append({
        "type":      "box",
        "center":    [0.0, 0.0, 0.0],
        "half":      [0.7, 0.7, cap_total_cm / 2],
        "color":     "#00ffcc",
        "label":     "Calorimeter stack",
        "alpha":     0.2,
        "linewidth": 0.8,
    })

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

    dsb_cm = _DSB_LENGTH_MM / 10.0
    for idx, (x_mm, y_mm) in enumerate(_CAP_POSITIONS_MM):
        if idx == 0:
            continue
        prims.append({
            "type":   "tube",
            "center": [x_mm / 10.0, y_mm / 10.0, _DSB_CENTER_Z_MM / 10.0],
            "rmax":   _CAP_INNER_MM / 10.0,
            "height": dsb_cm,
            "color":  "#ff9800",
            "label":  "DSB1 filament" if idx == 1 else "",
            "alpha":  0.9,
            "linewidth": 1.2,
        })

    # Draw screens instead of SiPMs
    screen_thick_cm = _SCREEN_THICK_MM / 10.0
    screen_xy_cm    = _SCREEN_XY_MM    / 10.0
    z_up_cm = -(cap_total_cm / 2.0) + screen_thick_cm / 2.0
    z_dn_cm =  (cap_total_cm / 2.0) - screen_thick_cm / 2.0

    for i, (x_mm, y_mm) in enumerate(_CAP_POSITIONS_MM):
        for z_cm, lbl_up, lbl_dn in [
            (z_up_cm,
             "Screen upstream"   if i == 0 else "",
             ""),
            (z_dn_cm,
             "",
             "Screen downstream" if i == 0 else ""),
        ]:
            label = lbl_up or lbl_dn
            prims.append({
                "type":      "box",
                "center":    [x_mm / 10.0, y_mm / 10.0, z_cm],
                "half":      [screen_xy_cm / 2.0, screen_xy_cm / 2.0, screen_thick_cm / 2.0],
                "color":     "#f1c40f",
                "label":     label,
                "alpha":     0.6,
                "linewidth": 1.0,
            })

    return prims


def add_optical_surfaces(sim, units):
    detector_volumes = getattr(sim.volume_manager, "volumes", {})

    for vol_name in detector_volumes:
        if "lyso" in vol_name.lower() and "_cap_" not in vol_name.lower():
            sim.physics_manager.add_optical_surface(
                TARGET_VOLUME_NAME, vol_name, "Tyvek")
            sim.physics_manager.add_optical_surface(
                vol_name, TARGET_VOLUME_NAME, "Tyvek")
        # Screens live inside target_volume — no optical surface needed;
        # they record crossings without perturbing photon transport.