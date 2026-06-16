"""
worlds/radi_cal.py
==================
RADiCAL Shashlik calorimeter — 29-layer LYSO/W sampling calorimeter
with localized shower-max DSB1 inserts and physical SiPM detectors.
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

BEAM_CONFIG = {
    "direction": [0, 0, 1],
    "target_cm": [0.2, 0.2, 0],  # Shifted 2mm off-center to avoid the central air capillary
    "offset_cm": 12.0,           # Pushed back to spawn outside the -91.5mm front face
}

TARGET_VOLUME_NAME = "target_volume"

PHANTOM_CM = [1.4, 1.4, 18.3]
EXPECTED_DEDX = 1.0
ACTIVATE_CALORIMETER_SETTINGS = True
CALORIMETER_Z_RES_MM = 0.1
ACTIVE_Z_RANGES_MM = [[0.0, PHANTOM_CM[2] * 10.0]]
TIMING_TRIGGER_THRESHOLD = 1

DETECTOR_VOLUME_NAMES = [
    "sipm_front_1", "sipm_front_2", "sipm_front_3", "sipm_front_4",
    "sipm_back_1",  "sipm_back_2",  "sipm_back_3",  "sipm_back_4",
]

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_MODULE_XY_MM     = 14.0
_LYSO_THICK_MM    = 1.5
_W_THICK_MM       = 2.5
_CAP_OUTER_MM     = 0.575
_CAP_INNER_MM     = 0.475
_CAP_TOTAL_MM     = 183.0

_SIPM_XY_MM       = 1.0
_SIPM_THICK_MM    = 0.1

_SHOWER_MAX_RANGE = range(9, 12)

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

def _build_layer(sim, name, thickness, material, z_pos, units, is_shower_max=False):
    plate = sim.add_volume("Box", name)
    plate.mother = TARGET_VOLUME_NAME
    plate.size = [_MODULE_XY_MM * units.mm, _MODULE_XY_MM * units.mm, thickness * units.mm]
    plate.material = material
    plate.translation = [0, 0, z_pos * units.mm]

    cap_dz = (thickness / 2.0) * units.mm

    for cap_idx, (cx, cy) in enumerate(_CAP_POSITIONS_MM):
        if cap_idx == 0:
            air = sim.add_volume("Tubs", f"{name}_cap_0_air")
            air.mother = name
            air.rmin, air.rmax = 0.0, _CAP_OUTER_MM * units.mm
            air.dz = cap_dz
            air.translation = [cx * units.mm, cy * units.mm, 0]
            air.material = "Air"
            continue

        wall = sim.add_volume("Tubs", f"{name}_cap_{cap_idx}_wall")
        wall.mother = name
        wall.rmin, wall.rmax = _CAP_INNER_MM * units.mm, _CAP_OUTER_MM * units.mm
        wall.dz = cap_dz
        wall.translation = [cx * units.mm, cy * units.mm, 0]
        wall.material = "Quartz"

        core_mat = "DSB1" if is_shower_max else "Quartz"
        core = sim.add_volume("Tubs", f"{name}_cap_{cap_idx}_core")
        core.mother = name
        core.rmin, core.rmax = 0.0, _CAP_INNER_MM * units.mm
        core.dz = cap_dz
        core.translation = [cx * units.mm, cy * units.mm, 0]
        core.material = core_mat

    return plate

def _build_sipms(sim, units):
    z_front = -(_CAP_TOTAL_MM / 2.0 + _SIPM_THICK_MM / 2.0)
    z_back  =  (_CAP_TOTAL_MM / 2.0 + _SIPM_THICK_MM / 2.0)

    for cap_idx, (cx, cy) in enumerate(_CAP_POSITIONS_MM):
        if cap_idx == 0:
            continue

        front = sim.add_volume("Box", f"sipm_front_{cap_idx}")
        front.mother = "world"
        front.size = [_SIPM_XY_MM * units.mm, _SIPM_XY_MM * units.mm, _SIPM_THICK_MM * units.mm]
        front.material = "G4_Si"
        front.translation = [cx * units.mm, cy * units.mm, z_front * units.mm]

        back = sim.add_volume("Box", f"sipm_back_{cap_idx}")
        back.mother = "world"
        back.size = [_SIPM_XY_MM * units.mm, _SIPM_XY_MM * units.mm, _SIPM_THICK_MM * units.mm]
        back.material = "G4_Si"
        back.translation = [cx * units.mm, cy * units.mm, z_back * units.mm]

# ─────────────────────────────────────────────────────────────────────────────
# WORLD CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_world(sim, units):
    world = sim.world
    world.size     = [50 * units.cm, 50 * units.cm, 50 * units.cm]
    world.material = "Air"

    target = sim.add_volume("Box", TARGET_VOLUME_NAME)
    target.mother   = "world"
    target.size     = [_MODULE_XY_MM * units.mm, _MODULE_XY_MM * units.mm, _CAP_TOTAL_MM * units.mm]
    target.material = "Air"

    _build_sipms(sim, units)

    stack_thick = (29 * _LYSO_THICK_MM) + (28 * _W_THICK_MM)
    gap_thick   = (_CAP_TOTAL_MM - stack_thick) / 2.0
    current_z   = -(_CAP_TOTAL_MM / 2.0)

    current_z += gap_thick / 2.0
    _build_layer(sim, "front_gap", gap_thick, "Air", current_z, units)
    current_z += gap_thick / 2.0

    for layer_idx in range(29):
        current_z += _LYSO_THICK_MM / 2.0
        _build_layer(sim, f"lyso_{layer_idx}", _LYSO_THICK_MM, "LYSO", current_z, units, layer_idx in _SHOWER_MAX_RANGE)
        current_z += _LYSO_THICK_MM / 2.0

        if layer_idx < 28:
            current_z += _W_THICK_MM / 2.0
            _build_layer(sim, f"w_{layer_idx}", _W_THICK_MM, "Tungsten", current_z, units)
            current_z += _W_THICK_MM / 2.0

    current_z += gap_thick / 2.0
    _build_layer(sim, "back_gap", gap_thick, "Air", current_z, units)

    return sim

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HOOKS
# ─────────────────────────────────────────────────────────────────────────────

def analyze(batch_dir, run_dirs, meta, utils):
    import matplotlib.pyplot as plt

    hits_files  = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    exits_files = [d / "optical_exited.root" for d in run_dirs]

    hits       = utils.analyse_hits(hits_files)
    exits      = utils.analyse_exits(exits_files)
    timing_res = (utils.extract_timing_resolution(hits_files, threshold_photon=TIMING_TRIGGER_THRESHOLD) if hits_files else 0.0)

    long_arr, trans_arr = utils.load_calorimeter_mhd(run_dirs, long_glob="run_Dose_edep.mhd", trans_glob="transverse_shower_max_edep.mhd")
    _aggregate_batch(batch_dir, run_dirs, meta, utils)

    extra_lines = []
    plots_saved = []

    if long_arr is not None:
        dz_mm = meta.get("dose_spacing_mm", 0.1)
        avg   = long_arr / max(len(run_dirs), 1)

        stack_thick = (29 * _LYSO_THICK_MM) + (28 * _W_THICK_MM)
        front_gap   = (_CAP_TOTAL_MM - stack_thick) / 2.0
        layer_numbers, layer_edeps = list(range(1, 30)), []
        current_z = front_gap

        for layer_idx in range(29):
            z_start, z_end = current_z, current_z + _LYSO_THICK_MM
            idx_start = max(0, min(int(round(z_start / dz_mm)), len(avg)))
            idx_end   = max(0, min(int(round(z_end / dz_mm)), len(avg)))
            layer_edeps.append(float(np.sum(avg[idx_start:idx_end])))
            current_z += _LYSO_THICK_MM + (_W_THICK_MM if layer_idx < 28 else 0)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(layer_numbers, layer_edeps, color="#00bcd4", alpha=0.7, edgecolor="#00838f", linewidth=1.2, width=0.8)
        ax.set_xlabel("LYSO Layer Number")
        ax.set_ylabel("Energy Deposition (MeV)")
        fig.tight_layout()
        out = batch_dir / "radical_longitudinal.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        plots_saved.append(out.name)

    up_hits = sum(hits.get(k, 0) for k in hits if "sipm_front" in k)
    dn_hits = sum(hits.get(k, 0) for k in hits if "sipm_back" in k)
    extra_lines += [f"  Upstream SiPM hits: {up_hits:,}", f"  Downstream SiPM hits: {dn_hits:,}"]

    return {"hits": hits, "exits": exits, "timing_res_ps": timing_res, "extra_lines": extra_lines, "plots_saved": plots_saved}

def _aggregate_batch(batch_dir, run_dirs, meta, utils):
    dz_mm = meta.get("dose_spacing_mm", 0.1)
    active_ranges = meta.get("active_z_ranges", None)
    long_acc, n = None, 0

    for run_dir in run_dirs:
        dose_txt = batch_dir / f"run_{run_dir.name.split('_')[-1]}_Dose.txt"
        if dose_txt.exists():
            try:
                _, energy = np.loadtxt(dose_txt, unpack=True, usecols=(0, 1))
                long_acc = energy.astype(float) if long_acc is None else long_acc + energy
                n += 1
            except Exception: pass

    if n > 0 and long_acc is not None:
        avg = long_acc / n
        out = batch_dir / "analyzed_longitudinal.txt"
        if active_ranges:
            energies = [np.sum(avg[int(round(zs/dz_mm)):int(round(ze/dz_mm))]) for zs, ze in active_ranges]
            np.savetxt(str(out), np.c_[np.arange(len(energies)) + 1, energies], fmt="%d %.6e")
        else:
            np.savetxt(str(out), np.c_[np.arange(len(avg)), avg], fmt="%d %.6e")

def get_geometry_primitives() -> list[dict]:
    prims = [{"type": "box", "center": [0.0, 0.0, 0.0], "half": [0.7, 0.7, _CAP_TOTAL_MM / 20.0], "color": "#00ffcc", "alpha": 0.15}]
    for idx, (x_mm, y_mm) in enumerate(_CAP_POSITIONS_MM):
        prims.append({"type": "tube", "center": [x_mm/10, y_mm/10, 0.0], "rmax": _CAP_OUTER_MM/10, "height": _CAP_TOTAL_MM/10, "color": "#00cfff", "alpha": 0.3})
    return prims

def add_optical_surfaces(sim, units):
    detector_volumes = getattr(sim.volume_manager, "volumes", {})
    for vol_name in detector_volumes:
        if vol_name.startswith("lyso_") and "_cap_" not in vol_name:
            sim.physics_manager.add_optical_surface(TARGET_VOLUME_NAME, vol_name, "Tyvek")
            sim.physics_manager.add_optical_surface(vol_name, TARGET_VOLUME_NAME, "Tyvek")