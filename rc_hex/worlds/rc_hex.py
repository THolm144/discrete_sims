"""
worlds/rc_hex.py
=========================
RADiCAL Shashlik calorimeter — hexagonal energy-measurement variant.

Stack:  29 Tyvek-wrapped LYSO plates interleaved with 28 tungsten absorbers.
        LYSO | W | LYSO | W | … | W | LYSO   (125.29 mm total)

Capillaries (6, alternating):
    Regular Hexagon (7mm side). Capillaries placed on the apothem of each face, 
    at a radial distance of 3.5mm from the center.
    
    E-type (indices 0, 2, 4) — Top face is index 0:
        Quartz sleeve (Ø1.15 mm outer) with a continuous BCF-92 WLS 
        filament (Ø0.90 mm inner) running the full length.

    T-type (indices 1, 3, 5):
        Quartz rod (Ø1.15 mm, 183 mm) with a short bore at shower-max
        (LYSO layers 8–11) holding a BCF-92 WLS filament (Ø0.90 mm, ~15 mm).

SiPMs:  12 active silicon tiles (6 front, 6 back) + 2 hexagonal FR4 readout cards.
"""

import numpy as np
import opengate.geometry.volumes as vol_module

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

TARGET_VOLUME_NAME = "calorimeter"

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_HEX_SIDE_MM     = 7.0
_LYSO_THICK_MM   = 1.5
_TYVEK_THICK_MM  = 0.008 * 25.4           # 0.2032 mm
_W_THICK_MM      = 2.5
_N_LYSO          = 29
_N_W             = 28

_GAP_THICK_MM    = _LYSO_THICK_MM + 2 * _TYVEK_THICK_MM   # 1.9064 mm
_CALOR_THICK_MM  = _N_LYSO * _GAP_THICK_MM + _N_W * _W_THICK_MM  # 125.2856 mm

# Hexagon Math
_APOTHEM_MM      = _HEX_SIDE_MM * np.sqrt(3) / 2  # ~6.062 mm
_CAP_R_MM        = 3.5                            # Safe radial distance from center

_CAP_OUTER_MM    = 1.150 / 2              # 0.575 mm  — quartz rod outer radius
_CAP_INNER_MM    = 0.950 / 2              # 0.475 mm  — inner bore radius
_CAP_LENGTH_MM   = 183.0

_FILAMENT_R_MM   = 0.900 / 2              # 0.45 mm   — BCF-92 filament radius

# ── Capillary Positions (6 faces) ─────────────────────────────────────────────
_CAP_POSITIONS_MM = []
for i in range(6):
    angle = np.pi / 2 + i * (np.pi / 3)
    cx = _CAP_R_MM * np.cos(angle)
    cy = _CAP_R_MM * np.sin(angle)
    _CAP_POSITIONS_MM.append([cx, cy])

_E_TYPE_INDICES  = {0, 2, 4} # 0 is Top
_T_TYPE_INDICES  = {1, 3, 5}

# ── Shower-max band (T-type bore region) ──────────────────────────────────────
_SHOWER_FIRST    = 8                      
_SHOWER_LAST     = 11                     
_LAYER_PITCH_MM  = _GAP_THICK_MM + _W_THICK_MM
_FIRST_CTR_MM    = _GAP_THICK_MM/2 + _SHOWER_FIRST * _LAYER_PITCH_MM
_LAST_CTR_MM     = _GAP_THICK_MM/2 + _SHOWER_LAST  * _LAYER_PITCH_MM
_BAND_FRONT_MM   = _FIRST_CTR_MM - _GAP_THICK_MM/2
_BAND_BACK_MM    = _LAST_CTR_MM  + _GAP_THICK_MM/2
_FILAMENT_LEN_MM = _BAND_BACK_MM - _BAND_FRONT_MM             
_FILAMENT_Z_MM   = -_CALOR_THICK_MM/2 + 0.5*(_BAND_FRONT_MM + _BAND_BACK_MM)

# ── SiPM / card geometry ─────────────────────────────────────────────────────
_SIPM_XY_MM      = 1.2
_SIPM_THICK_MM   = 0.3
_CARD_THICK_MM   = 1.6
_CARD_HOLE_R_MM  = 2.0
_SIPM_Z_MM       = _CAP_LENGTH_MM/2 + _SIPM_THICK_MM/2
_CARD_Z_MM       = _CAP_LENGTH_MM/2 + _SIPM_THICK_MM + 0.1 + _CARD_THICK_MM/2

_WORLD_XY_MM     = 3.0 * _HEX_SIDE_MM
_WORLD_Z_MM      = 1.5 * max(_CAP_LENGTH_MM, _CALOR_THICK_MM)

# ── Simulator metadata ────────────────────────────────────────────────────────
PHANTOM_CM       = [_HEX_SIDE_MM/10, _HEX_SIDE_MM/10, _CALOR_THICK_MM/10]
EXPECTED_DEDX    = 1.0
ACTIVATE_CALORIMETER_SETTINGS = True
CALORIMETER_Z_RES_MM  = 0.1
ACTIVE_Z_RANGES_MM    = [[0.0, _CALOR_THICK_MM]]
TIMING_TRIGGER_THRESHOLD = 1

DETECTOR_VOLUME_NAMES = [
    f"sipm_front_{i}" for i in range(6)
] + [
    f"sipm_back_{i}" for i in range(6)
]

BEAM_CONFIG = {
    "direction": [0, 0, 1],
    "target_cm": [0, 0, 0],
    "offset_cm": _SIPM_Z_MM/10 + 2.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _drill_holes(base_vol, name, half_dz_mm, mm, clearance=0.010):
    bore_dz = (half_dz_mm + 0.1) * mm
    result  = base_vol
    for i, (cx, cy) in enumerate(_CAP_POSITIONS_MM):
        bore      = vol_module.TubsVolume(name=f"{name}_bore_{i}")
        bore.rmin = 0.0
        # Stagger the drill clearance so nested daughter volumes don't share identical boundaries
        bore.rmax = (_CAP_OUTER_MM + clearance) * mm  
        bore.dz   = bore_dz
        result    = vol_module.subtract_volumes(
            result, bore,
            translation=[cx * mm, cy * mm, 0],
            new_name=f"{name}_sub{i}",
        )
    return result

def _make_gap(name, mm):
    base = vol_module.HexagonVolume(name=f"{name}_hex")
    base.radius = (_HEX_SIDE_MM + _TYVEK_THICK_MM) * mm
    base.height = _GAP_THICK_MM * mm  
    return _drill_holes(base, name, _GAP_THICK_MM/2, mm, clearance=0.012)

def _make_lyso(name, mm):
    base = vol_module.HexagonVolume(name=f"{name}_hex")
    base.radius = _HEX_SIDE_MM * mm  
    base.height = _LYSO_THICK_MM * mm
    return _drill_holes(base, name, _LYSO_THICK_MM/2, mm, clearance=0.014)

def _make_abso(name, mm):
    base = vol_module.HexagonVolume(name=f"{name}_hex")
    base.radius = (_HEX_SIDE_MM + _TYVEK_THICK_MM) * mm
    base.height = _W_THICK_MM * mm
    return _drill_holes(base, name, _W_THICK_MM/2, mm, clearance=0.012)

def _build_capillaries(sim, mm):
    half_cap   = _CAP_LENGTH_MM / 2 * mm
    half_calor = _CALOR_THICK_MM / 2 * mm

    for i, (cx, cy) in enumerate(_CAP_POSITIONS_MM):
        if i in _E_TYPE_INDICES:
            # Active Quartz Cladding Sleeve (Mothered to world to prevent overlap with drilled calorimeter)
            sleeve = sim.add_volume("Tubs", f"cap_{i}_active_sleeve")
            sleeve.mother      = "world"
            sleeve.rmin        = _FILAMENT_R_MM * mm
            sleeve.rmax        = _CAP_OUTER_MM * mm     
            sleeve.dz          = half_calor             
            sleeve.translation = [cx * mm, cy * mm, 0]
            sleeve.material    = "G4_SILICON_DIOXIDE"

            # Continuous active core filament
            core = sim.add_volume("Tubs", f"cap_{i}_active_core")
            core.mother        = "world"
            core.rmin          = 0.0
            core.rmax          = _FILAMENT_R_MM * mm
            core.dz            = half_calor
            core.translation   = [cx * mm, cy * mm, 0]
            core.material      = "BCF92"

            # Upstream/Downstream Passive Quartz Extensions
            tail_len_z       = (half_cap - half_calor)
            z_pos_front      = -(half_calor + tail_len_z / 2)
            tail_f             = sim.add_volume("Tubs", f"cap_{i}_tail_front")
            tail_f.mother      = "world"
            tail_f.rmin        = 0.0
            tail_f.rmax        = _CAP_OUTER_MM * mm
            tail_f.dz          = tail_len_z / 2
            tail_f.translation = [cx * mm, cy * mm, z_pos_front]
            tail_f.material    = "G4_SILICON_DIOXIDE"

            z_pos_back       = (half_calor + tail_len_z / 2)
            tail_b             = sim.add_volume("Tubs", f"cap_{i}_tail_back")
            tail_b.mother      = "world"
            tail_b.rmin        = 0.0
            tail_b.rmax        = _CAP_OUTER_MM * mm
            tail_b.dz          = tail_len_z / 2
            tail_b.translation = [cx * mm, cy * mm, z_pos_back]
            tail_b.material    = "G4_SILICON_DIOXIDE"

        else:
            # ── T-TYPE ──
            rod_base      = vol_module.TubsVolume(name=f"cap_{i}_rod")
            rod_base.rmin = 0.0
            rod_base.rmax = _CAP_OUTER_MM * mm
            rod_base.dz   = half_cap

            bore          = vol_module.TubsVolume(name=f"cap_{i}_bore")
            bore.rmin     = 0.0
            bore.rmax     = (_CAP_INNER_MM + 0.05) * mm
            bore.dz       = (_FILAMENT_LEN_MM / 2 + 0.01) * mm

            quartz_vol    = vol_module.subtract_volumes(
                rod_base, bore,
                translation=[0, 0, _FILAMENT_Z_MM * mm],
                new_name=f"cap_{i}",
            )
            quartz_vol.name        = f"cap_{i}"
            quartz_vol.mother      = "world"
            quartz_vol.material    = "G4_SILICON_DIOXIDE"
            quartz_vol.translation = [cx * mm, cy * mm, 0]
            sim.add_volume(quartz_vol)

            filament             = sim.add_volume("Tubs", f"cap_{i}_filament")
            filament.mother      = "world"
            filament.rmin        = 0.0
            filament.rmax        = _FILAMENT_R_MM * mm
            filament.dz          = (_FILAMENT_LEN_MM / 2) * mm
            filament.translation = [cx * mm, cy * mm, _FILAMENT_Z_MM * mm]
            filament.material    = "BCF92"

def _build_sipms(sim, mm):
    for end_name, sgn in [("front", -1), ("back", +1)]:
        z_sipm = sgn * _SIPM_Z_MM * mm
        z_card = sgn * _CARD_Z_MM * mm

        card_base      = vol_module.HexagonVolume(name=f"card_{end_name}_base")
        card_base.radius = _HEX_SIDE_MM * mm
        card_base.height   = (_CARD_THICK_MM / 2) * mm
        
        card_hole      = vol_module.TubsVolume(name=f"card_{end_name}_hole")
        card_hole.rmin = 0.0
        card_hole.rmax = _CARD_HOLE_R_MM * mm
        card_hole.dz   = (_CARD_THICK_MM + 0.1) * mm
        
        card_vol        = vol_module.subtract_volumes(
            card_base, card_hole, new_name=f"card_{end_name}"
        )
        card_vol.name        = f"card_{end_name}"
        card_vol.mother      = "world"
        card_vol.material    = "G4_SILICON_DIOXIDE"
        card_vol.translation = [0, 0, z_card]
        sim.add_volume(card_vol)

        for cap_idx, (cx, cy) in enumerate(_CAP_POSITIONS_MM):
            sipm             = sim.add_volume("Box", f"sipm_{end_name}_{cap_idx}")
            sipm.mother      = "world"
            sipm.size        = [_SIPM_XY_MM * mm, _SIPM_XY_MM * mm, _SIPM_THICK_MM * mm]
            sipm.material    = "G4_Si"
            sipm.translation = [cx * mm, cy * mm, z_sipm]

# ─────────────────────────────────────────────────────────────────────────────
# WORLD CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_world(sim, units):
    mm = units.mm

    world          = sim.world
    world.size     = [_WORLD_XY_MM * mm, _WORLD_XY_MM * mm, _WORLD_Z_MM * mm]
    world.material = "G4_AIR"

    calor_base = vol_module.HexagonVolume(name="calorimeter_base")
    calor_base.radius = (_HEX_SIDE_MM + _TYVEK_THICK_MM + 0.010) * mm
    calor_base.height = (_CALOR_THICK_MM + 0.020) * mm
    
    # Change calor_base.dz/mm to (calor_base.height / 2) / mm
    calor_vol = _drill_holes(calor_base, "calorimeter", (calor_base.height / 2) / mm, mm, clearance=0.010)
    calor_vol.name        = TARGET_VOLUME_NAME
    calor_vol.mother      = "world"
    calor_vol.material    = "G4_AIR"
    calor_vol.translation = [0, 0, 0]
    sim.add_volume(calor_vol)

    _build_capillaries(sim, mm)
    _build_sipms(sim, mm)

    z_pos = -_CALOR_THICK_MM / 2

    for i in range(_N_LYSO):
        z_pos   += _GAP_THICK_MM / 2
        gap_vol  = _make_gap(f"gap_{i}", mm)
        gap_vol.name        = f"gap_{i}"
        gap_vol.mother      = TARGET_VOLUME_NAME
        gap_vol.material    = "Tyvek"
        gap_vol.translation = [0, 0, z_pos * mm]
        sim.add_volume(gap_vol)

        lyso_vol             = _make_lyso(f"lyso_{i}", mm)
        lyso_vol.name        = f"lyso_{i}"
        lyso_vol.mother      = f"gap_{i}"
        lyso_vol.material    = "LYSO"
        lyso_vol.translation = [0, 0, 0]
        sim.add_volume(lyso_vol)

        z_pos += _GAP_THICK_MM / 2

        if i < _N_W:
            z_pos   += _W_THICK_MM / 2
            abso_vol = _make_abso(f"abso_{i}", mm)
            abso_vol.name        = f"abso_{i}"
            abso_vol.mother      = TARGET_VOLUME_NAME
            abso_vol.material    = "Tungsten"
            abso_vol.translation = [0, 0, z_pos * mm]
            sim.add_volume(abso_vol)
            z_pos += _W_THICK_MM / 2

    return sim

# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL SURFACES
# ─────────────────────────────────────────────────────────────────────────────

def add_optical_surfaces(sim, units):
    vols = sim.volume_manager.volumes

    for i in range(_N_LYSO):
        lyso_name = f"lyso_{i}"
        gap_name  = f"gap_{i}"
        if lyso_name in vols and gap_name in vols:
            sim.physics_manager.add_optical_surface(lyso_name, gap_name, "Tyvek")
            sim.physics_manager.add_optical_surface(gap_name, lyso_name, "Tyvek")

    for cap_idx in _E_TYPE_INDICES:
        core_name   = f"cap_{cap_idx}_active_core"
        sleeve_name = f"cap_{cap_idx}_active_sleeve"
        tail_b_name = f"cap_{cap_idx}_tail_back"
        tail_f_name = f"cap_{cap_idx}_tail_front"

        if core_name in vols and sleeve_name in vols:
            sim.physics_manager.add_optical_surface(core_name, sleeve_name, "Polished")
        if core_name in vols and tail_b_name in vols:
            sim.physics_manager.add_optical_surface(core_name, tail_b_name, "Polished")
        if core_name in vols and tail_f_name in vols:
            sim.physics_manager.add_optical_surface(core_name, tail_f_name, "Polished")

    for cap_idx in _T_TYPE_INDICES:
        rod_name  = f"cap_{cap_idx}"
        plug_name = f"cap_{cap_idx}_filament"
        if rod_name in vols and plug_name in vols:
            sim.physics_manager.add_optical_surface(plug_name, rod_name, "Polished")

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HOOKS
# ─────────────────────────────────────────────────────────────────────────────

def analyze(batch_dir, run_dirs, meta, utils):
    import matplotlib.pyplot as plt

    hits_files  = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    exits_files = [d / "optical_exited.root" for d in run_dirs]

    hits       = utils.analyse_hits(hits_files) if hits_files else {}
    exits      = utils.analyse_exits(exits_files) if exits_files else {}
    timing_res = (utils.extract_timing_resolution(
                  hits_files, threshold_photon=TIMING_TRIGGER_THRESHOLD)
                  if hits_files else 0.0)

    long_arr, trans_arr = utils.load_calorimeter_mhd(
        run_dirs,
        long_glob="run_Dose_edep.mhd",
        trans_glob="transverse_shower_max_edep.mhd",
    )
    _aggregate_batch(batch_dir, run_dirs, meta, utils)

    plots_saved = []

    if long_arr is not None:
        dz_mm = meta.get("dose_spacing_mm", 0.1)
        avg   = long_arr / max(len(run_dirs), 1)

        layer_edeps = []
        current_z   = 0.0   

        for idx in range(_N_LYSO):
            z_start = current_z + _TYVEK_THICK_MM
            z_end   = z_start   + _LYSO_THICK_MM
            i0 = max(0, min(int(round(z_start / dz_mm)), len(avg)))
            i1 = max(0, min(int(round(z_end   / dz_mm)), len(avg)))
            layer_edeps.append(float(np.sum(avg[i0:i1])))
            current_z += _GAP_THICK_MM + (_W_THICK_MM if idx < _N_W else 0)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(range(1, _N_LYSO + 1), layer_edeps,
               color="#00bcd4", alpha=0.7, edgecolor="#00838f",
               linewidth=1.2, width=0.8)
        ax.set_xlabel("LYSO Layer Number")
        ax.set_ylabel("Energy Deposition (MeV)")
        ax.set_title("RADiCAL Energy — Hexagonal Longitudinal Shower Profile")
        fig.tight_layout()
        out = batch_dir / "radical_energy_longitudinal.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        plots_saved.append(out.name)

    up_hits = sum(hits.get(k, 0) for k in hits if "sipm_front" in k)
    dn_hits = sum(hits.get(k, 0) for k in hits if "sipm_back"  in k)
    extra_lines = [
        f"  Upstream SiPM hits:   {up_hits:,}",
        f"  Downstream SiPM hits: {dn_hits:,}",
    ]

    return {
        "hits":          hits,
        "exits":         exits,
        "timing_res_ps": timing_res,
        "extra_lines":   extra_lines,
        "plots_saved":   plots_saved,
    }

def _aggregate_batch(batch_dir, run_dirs, meta, utils):
    dz_mm         = meta.get("dose_spacing_mm", 0.1)
    active_ranges = meta.get("active_z_ranges", None)
    long_acc, n   = None, 0

    for run_dir in run_dirs:
        dose_txt = batch_dir / f"run_{run_dir.name.split('_')[-1]}_Dose.txt"
        if dose_txt.exists():
            try:
                _, energy = np.loadtxt(dose_txt, unpack=True, usecols=(0, 1))
                long_acc  = (energy.astype(float) if long_acc is None
                             else long_acc + energy)
                n += 1
            except Exception:
                pass

    if n > 0 and long_acc is not None:
        avg = long_acc / n
        out = batch_dir / "analyzed_longitudinal.txt"
        if active_ranges:
            energies = [
                np.sum(avg[int(round(zs/dz_mm)):int(round(ze/dz_mm))])
                for zs, ze in active_ranges
            ]
            np.savetxt(str(out),
                       np.c_[np.arange(len(energies))+1, energies],
                       fmt="%d %.6e")
        else:
            np.savetxt(str(out),
                       np.c_[np.arange(len(avg)), avg],
                       fmt="%d %.6e")

def get_geometry_primitives() -> list[dict]:
    prims = [{
        "type":   "hexagon",
        "center": [0.0, 0.0, 0.0],
        "side":   _HEX_SIDE_MM/10,
        "height": _CALOR_THICK_MM/10,
        "color":  "#00ffcc",
        "alpha":  0.15,
    }]
    for i, (cx, cy) in enumerate(_CAP_POSITIONS_MM):
        color = "#ff9900" if i in _E_TYPE_INDICES else "#00cfff"
        prims.append({
            "type":   "tube",
            "center": [cx/10, cy/10, 0.0],
            "rmax":   _CAP_OUTER_MM/10,
            "height": _CAP_LENGTH_MM/10,
            "color":  color,   
            "alpha":  0.35,
        })

    return prims