"""
worlds/radi_cal.py
==================
RADiCAL Shashlik calorimeter — 29-layer LYSO/W sampling calorimeter
with localized shower-max DSB1 inserts and physical SiPM detectors.
"""

import numpy as np
import matplotlib.pyplot as plt
import uproot

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
ACTIVE_Z_RANGES_MM = None
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

def get_active_ranges():
    """Generates the Z-bounds for the 29 LYSO layers to pass into meta."""
    ranges = []
    stack_thick = (29 * _LYSO_THICK_MM) + (28 * _W_THICK_MM)
    gap_thick = (_CAP_TOTAL_MM - stack_thick) / 2.0
    current_z = -(_CAP_TOTAL_MM / 2.0) + gap_thick
    
    for i in range(29):
        ranges.append((current_z, current_z + _LYSO_THICK_MM))
        current_z += (_LYSO_THICK_MM + _W_THICK_MM)
    return ranges

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
    hits_files  = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    exits_files = [d / "optical_exited.root" for d in run_dirs]

    hits       = utils.analyse_hits(hits_files)
    exits      = utils.analyse_exits(exits_files)
    
    extra_lines = []
    plots_saved = []

    # ── Extract Timing and Photon Energy ─────────────────────────────────────
    all_energies, times = [], []
    for f in hits_files:
        try:
            with uproot.open(f) as file:
                tree_names = [k for k in file.keys() if ";" in k]
                if not tree_names: continue
                
                tree = file[tree_names[0]]
                keys = tree.keys()
                key_map = {k.lower(): k for k in keys}
                
                if not times: 
                    extra_lines.append(f"  [DEBUG] ROOT Tree    : {tree_names[0]}")
                    extra_lines.append(f"  [DEBUG] ROOT Branches: {', '.join(keys[:10])}")

                for e_name in ["energy", "kineticenergy", "edep"]:
                    if e_name in key_map:
                        arr = tree[key_map[e_name]].array(library="np") * 1e6
                        all_energies.extend([val for val in arr if val > 1e-5])
                        break
                        
                for t_name in ["time", "globaltime", "localtime"]:
                    if t_name in key_map:
                        times.extend(tree[key_map[t_name]].array(library="np"))
                        break
        except Exception as e: 
            extra_lines.append(f"  [DEBUG] Uproot Error on {f.name}: {e}")

    timing_res = float(np.std(times)) if len(times) > 1 else 0.0
    avg_energy = float(np.mean(all_energies)) if len(all_energies) > 0 else 0.0

    # ── Aggregate Dose Data ──────────────────────────────────────────────────
    _aggregate_batch(batch_dir, run_dirs, meta, utils)
    centers, dose_edep = utils.load_dose_mhd(run_dirs, meta["phantom_cm"])

    # ── Modular Plotting Hook ────────────────────────────────────────────────
    _generate_bar_plot(batch_dir, meta, plots_saved, extra_lines)

    return {
        "hits": hits, 
        "exits": exits, 
        "timing_res_ps": timing_res, 
        "avg_photon_energy_ev": avg_energy,
        "dose_centers": centers,
        "dose_edep": dose_edep,
        "extra_lines": extra_lines, 
        "plots_saved": plots_saved
    }

def _generate_bar_plot(batch_dir, meta, plots_saved, extra_lines):
    """Generates the absolute MeV bar chart matching the requested style."""
    analyzed_txt = batch_dir / "analyzed_longitudinal.txt"
    if not analyzed_txt.exists():
        return
        
    try:
        layers, energies_mev = np.loadtxt(analyzed_txt, unpack=True)
        
        # Ensure array shapes handle single-layer edge cases
        if layers.ndim == 0:
            layers = np.array([layers])
            energies_mev = np.array([energies_mev])
            
        plt.figure(figsize=(10, 5))
        plt.bar(layers, energies_mev, color='#48cae4', edgecolor='#0077b6', width=0.8)
        
        plt.xlabel("LYSO Layer Number", fontsize=12)
        plt.ylabel("Energy Deposition (MeV)", fontsize=12)
        plt.title("RADiCAL Longitudinal Shower Profile", fontsize=14)
        plt.xlim(0, 30)
        plt.grid(False)
        
        plot_file = batch_dir / "radical_longitudinal.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        plots_saved.append(plot_file.name)
        
        shower_max_layer = layers[np.argmax(energies_mev)]
        extra_lines.append(f"  Shower Max Layer     : {int(shower_max_layer)}")
        
    except Exception as e:
        extra_lines.append(f"  Warning: Could not plot longitudinal data: {e}")

def _aggregate_batch(batch_dir, run_dirs, meta, utils):
    dz_mm = meta.get("dose_spacing_mm", 0.1)
    # Default to the generator if none provided by the runner
    active_ranges = meta.get("active_z_ranges", get_active_ranges())
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