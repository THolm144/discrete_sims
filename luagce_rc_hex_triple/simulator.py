"""
simulator.py
============
CLI-driven OpenGATE simulation.  World modules declare their capabilities
via a CAPABILITIES dict; the simulator wires actors accordingly.
CLI flags can override world defaults (e.g. --optical off).
"""

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from opengate.actors.filters import GateFilterBuilder
import numpy as np
import opengate as gate


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CAPABILITIES = {
    "optical":                False,
    "dose":                   True,
    "sipm_hits":              False,
    "optical_exits":          False,
    "calorimeter_mode":       False,
    "sipm_hits_optical_only": False,
}

DEFAULT_BEAM_CONFIG = {
    "direction": [0, 0, 1],
    "target_cm": [0, 0, 0],
    "offset_cm": 2.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OpenGATE modular simulator")
    p.add_argument("--world",        default="quartz_8x8")
    p.add_argument("--particle",     default="proton")
    p.add_argument("--energy-kev",   type=float, default=500_000)
    p.add_argument("--n",            type=int,   default=10_000)
    p.add_argument("--threads",      type=int,   default=8)
    p.add_argument("--beam-radius",  type=float, default=1.0,
                   help="Beam disc radius in cm (default: 1.0)")
    p.add_argument("--output-dir",   default=None)
    p.add_argument("--physics-list", default="G4EmStandardPhysics_option4")
    p.add_argument("--run-id",       type=int,   default=None)
    # ── capability overrides ──────────────────────────────────────────────────
    p.add_argument("--optical",      choices=["on", "off", "world"],  default="world",
                   help="Override world optical capability. 'world' = respect world manifest.")
    p.add_argument("--dose",         choices=["on", "off", "world"],  default="world")
    p.add_argument("--sipm-hits",    choices=["on", "off", "world"],  default="world")
    p.add_argument("--optical-exits", choices=["on", "off", "world"], default="off",
                   help="Toggle optical exit PhaseSpaceActor (default: off).")
    p.add_argument("--cherenkov",    choices=["on", "off"],           default="on",
                   help="Toggle Cherenkov radiation when optical physics is enabled.")
    p.add_argument("--hits-optical-only", choices=["on", "off", "world"], default="world",
                   help="Filter detector_hits actors to opticalphoton hits only.")
    p.add_argument("--beam-offset", type=float, default=None)
    p.add_argument("--beam-x", type=float, default=None)
    p.add_argument("--beam-y", type=float, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# WORLD LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_world(world_name: str, script_dir: Path):
    sys.path.insert(0, str(script_dir / "worlds"))
    try:
        return importlib.import_module(world_name)
    except ModuleNotFoundError:
        print(f"ERROR: world module '{world_name}' not found in worlds/")
        sys.exit(1)


def resolve_capabilities(world, args) -> dict:
    caps = {**DEFAULT_CAPABILITIES, **getattr(world, "CAPABILITIES", {})}

    override_map = {
        "optical":                args.optical,
        "dose":                   args.dose,
        "sipm_hits":              getattr(args, "sipm_hits", "world"),
        "optical_exits":          getattr(args, "optical_exits", "world"),
        "sipm_hits_optical_only": getattr(args, "hits_optical_only", "world"),
    }
    for key, val in override_map.items():
        if val == "on":
            caps[key] = True
        elif val == "off":
            caps[key] = False

    return caps


def resolve_beam_config(world, args) -> dict:
    cfg = {**DEFAULT_BEAM_CONFIG, **getattr(world, "BEAM_CONFIG", {})}
    if args.beam_offset is not None:
        cfg["offset_cm"] = args.beam_offset
    if args.beam_x is not None or args.beam_y is not None:
        tx = args.beam_x if args.beam_x is not None else cfg.get("target_cm", [0,0,0])[0]
        ty = args.beam_y if args.beam_y is not None else cfg.get("target_cm", [0,0,0])[1]
        tz = cfg.get("target_cm", [0,0,0])[2]
        cfg["target_cm"] = [tx, ty, tz]
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────

def resolve_output_dirs(args, script_dir: Path) -> tuple[Path, Path]:
    if args.output_dir:
        batch_dir = Path(args.output_dir)
    else:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = script_dir / "runs" / args.world / f"{int(args.energy_kev)}keV_{ts}"

    run_id  = args.run_id if args.run_id is not None else 0
    run_dir = batch_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir, run_dir


# ─────────────────────────────────────────────────────────────────────────────
# ACTOR WIRING
# ─────────────────────────────────────────────────────────────────────────────

def wire_actors(sim, world, caps: dict, run_dir: Path, units) -> dict:
    registry = {
        "optical_exited_actor": None,
        "hit_actors":           [],
        "dose_actor":           None,
        "is_calorimeter":       caps["calorimeter_mode"],
    }

    target_vol       = getattr(world, "TARGET_VOLUME_NAME", "calorimeter")
    detector_volumes = getattr(world, "DETECTOR_VOLUME_NAMES", [])

    # ── Optical exits ─────────────────────────────────────────────────────
    if caps["optical"] and caps.get("optical_exits", False):
        valid_vols = sim.volume_manager.volumes
        if target_vol not in valid_vols or target_vol in ["world", "calorimeter"]:
            print(f"[ACTOR] SAFELY SKIPPING 'optical_exited' for volume '{target_vol}' "
                  f"(prevents OpenGATE mother-volume C++ exception).")
        else:
            exited = sim.add_actor("PhaseSpaceActor", "optical_exited")
            exited.attached_to     = target_vol
            exited.output_filename = "optical_exited.root"
            exited.steps_to_store  = "exiting"
            exited.attributes = [
                "ParticleName", "KineticEnergy", "TrackCreatorProcess",
                "Position", "TrackID", "EventID", "GlobalTime",
            ]
            registry["optical_exited_actor"] = exited

    # ── Per-channel screen hit actors ─────────────────────────────────────
    if caps["sipm_hits"] and detector_volumes:
        print(f"[ACTOR] Registered volumes: {list(sim.volume_manager.volumes.keys())[:20]} ...")

        optical_only = caps.get("sipm_hits_optical_only", False)
        hit_filter = None
        if optical_only:
            F = GateFilterBuilder()
            hit_filter = (F.ParticleName == "opticalphoton")
            print("[ACTOR] detector_hits actors filtering to opticalphoton hits only "
                  "(sipm_hits_optical_only=True).")

        for idx, vol_name in enumerate(detector_volumes):
            if vol_name not in sim.volume_manager.volumes:
                print(f"  WARNING: screen volume '{vol_name}' not found — skipping.")
                continue
            hits = sim.add_actor("PhaseSpaceActor", f"detector_hits_{idx}")
            hits.attached_to                = vol_name
            hits.authorize_repeated_volumes = True
            hits.output_filename            = f"detector_hits_{idx}.root"
            hits.steps_to_store             = "entering"

            if hit_filter is not None:
                hits.filter = hit_filter

            hits.attributes = [
                "ParticleName", "Position", "EventID", "GlobalTime", "TrackCreatorProcess", "LocalTime",
            ]
            registry["hit_actors"].append(hits)
            if caps.get("optical", False):
                killer = sim.add_actor("KillActor", f"sensor_killer_{idx}")
                killer.attached_to = vol_name
                if hit_filter is not None:
                    killer.filter = hit_filter
                else:
                    F = GateFilterBuilder()
                    killer.filter = (F.ParticleName == "opticalphoton")

    # ── Shower-max raw dE/dx (no optical transport) ────────────────────────
    shower_first = getattr(world, "_SHOWER_FIRST", None)
    shower_last  = getattr(world, "_SHOWER_LAST", None)
    if shower_first is not None and shower_last is not None:
        found_any = False
        for i in range(shower_first, shower_last + 1):
            vol_name = f"lyso_{i}"
            if vol_name in sim.volume_manager.volumes:
                found_any = True
                edep_ps = sim.add_actor("PhaseSpaceActor", f"showermax_edep_{i}")
                
                # FIX: Attach to a single string volume name, not a list
                edep_ps.attached_to     = vol_name
                edep_ps.output_filename = f"showermax_edep_{i}.root"
                
                
                edep_ps.steps_to_store  = "all"  
                
                F_edep = GateFilterBuilder()
                edep_ps.filter    = (F_edep.ParticleName != "opticalphoton")
                edep_ps.attributes = ["EventID", "TotalEnergyDeposit"]
                
                # Add to registry
                registry[f"showermax_edep_actor_{i}"] = edep_ps
                print(f"[ACTOR] Configured 'showermax_edep' actor for volume: {vol_name}")
                
        if not found_any:
            print(f"  WARNING: no lyso_{{{shower_first}..{shower_last}}} volumes found — "
                  f"skipping showermax_edep actors.")

    # ── Dose actor ────────────────────────────────────────────────────────
    if caps["dose"]:
        phantom_cm = world.PHANTOM_CM
        dose_vol = target_vol if target_vol in sim.volume_manager.volumes else "calorimeter"
        if caps["calorimeter_mode"]:
            dose = _wire_calorimeter_dose(sim, world, dose_vol, phantom_cm, units)
        else:
            dose = _wire_standard_dose(sim, dose_vol, phantom_cm, units)

        if hasattr(world, "configure_dose_actor"):
            world.configure_dose_actor(dose, units)

        registry["dose_actor"] = dose

    return registry


def _wire_standard_dose(sim, target_vol: str, phantom_cm: list, units):
    dose = sim.add_actor("DoseActor", "dose_actor")
    dose.attached_to      = target_vol
    dose.output_filename  = "edep.mhd"
    dose.size = [
        int(round(phantom_cm[0] * 10)),
        int(round(phantom_cm[1] * 10)),
        int(round(phantom_cm[2] * 10)),
    ]
    dose.spacing                = [1.0 * units.mm] * 3
    dose.hit_type               = "random"
    dose.edep.active            = True
    dose.dose.active            = False
    dose.edep_uncertainty.active = False
    return dose


def _wire_calorimeter_dose(sim, world, target_vol: str, phantom_cm: list, units):
    res_mm = getattr(world, "CALORIMETER_Z_RES_MM", 0.1)
    nz     = int(round((phantom_cm[2] * 10.0) / res_mm))

    dose = sim.add_actor("DoseActor", "longitudinalDose")
    dose.attached_to             = target_vol
    dose.output_filename         = "run_Dose.mhd"
    dose.size                    = [1, 1, nz]
    dose.spacing                 = [
        phantom_cm[0] * 10.0 * units.mm,
        phantom_cm[1] * 10.0 * units.mm,
        res_mm * units.mm,
    ]
    dose.hit_type                = "random"
    dose.edep.active             = True
    dose.dose.active             = False
    dose.edep_uncertainty.active = False

    trans_res_mm = 0.1
    nx = int(round((phantom_cm[0] * 10.0) / trans_res_mm))
    ny = int(round((phantom_cm[1] * 10.0) / trans_res_mm))

    trans = sim.add_actor("DoseActor", "transverseShowerMax")
    trans.attached_to             = target_vol
    trans.output_filename         = "transverse_shower_max.mhd"
    trans.size                    = [nx, ny, 1]
    trans.spacing                 = [trans_res_mm * units.mm,
                                     trans_res_mm * units.mm,
                                     1.0 * units.mm]
    trans.hit_type                = "random"
    trans.edep.active             = True
    trans.dose.active             = False
    trans.edep_uncertainty.active = False

    return dose


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE
# ─────────────────────────────────────────────────────────────────────────────

def add_beam_source(sim, args, world, beam_cfg: dict, units):
    direction = np.array(beam_cfg["direction"], dtype=float)
    direction /= np.linalg.norm(direction)

    target_cm  = np.array(beam_cfg["target_cm"], dtype=float)
    offset_cm  = beam_cfg["offset_cm"]

    source_pos_cm = target_cm - direction * offset_cm

    source = sim.add_source("GenericSource", f"{args.particle}_beam")
    source.particle             = args.particle
    source.energy.mono          = args.energy_kev * units.keV
    source.position.type        = "disc"
    source.position.radius      = args.beam_radius * units.cm
    source.position.translation = [
        source_pos_cm[0] * units.cm,
        source_pos_cm[1] * units.cm,
        source_pos_cm[2] * units.cm,
    ]
    source.direction.type     = "momentum"
    source.direction.momentum = direction.tolist()

    events_per_thread = max(1, int(args.n / args.threads))
    source.n = events_per_thread
    return source


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS
# ─────────────────────────────────────────────────────────────────────────────

def configure_physics(sim, args, script_dir: Path, world,
                      caps: dict, target_vol: str, units):
    sim.physics_manager.physics_list_name = args.physics_list

    sim.g4_commands_before_init.append("/run/setCut 0.1 mm")
    sim.g4_commands_before_init.append("/process/optical/scintillation/setTrackSecondariesFirst true")
    sim.g4_commands_before_init.append("/process/optical/cerenkov/setTrackSecondariesFirst true")
    sim.g4_commands_before_init.append("/process/optical/cerenkov/setMaxPhotons 300")

    surface_file = script_dir / "SurfaceProperties.xml"
    if surface_file.exists():
        sim.physics_manager.surface_properties_file = str(surface_file)

    if caps["optical"]:
        optical_file = script_dir / "Materials.xml"
        sim.physics_manager.special_physics_constructors.G4OpticalPhysics = True
        if optical_file.exists():
            sim.physics_manager.optical_properties_file = str(optical_file)

        if getattr(args, "cherenkov", "on") == "off":
            sim.g4_commands_before_init.append("/process/optical/processActivation Cerenkov false")
            print("[SIM] Optical physics ENABLED (Scintillation ONLY, Cherenkov DISABLED).")
        else:
            sim.g4_commands_before_init.append("/process/optical/processActivation Cerenkov true")
            print("[SIM] Optical physics ENABLED (Both Scintillation & Cherenkov ACTIVE).")
    else:
        sim.physics_manager.special_physics_constructors.G4OpticalPhysics = False
        print("[SIM] Optical physics DISABLED.")


# ─────────────────────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────────────────────

def save_metadata(args, batch_dir: Path, run_dir: Path, world,
                  caps: dict, beam_cfg: dict, actor_registry: dict):
    dose = actor_registry["dose_actor"]
    metadata = {
        "world":              args.world,
        "particle":           args.particle,
        "energy_kev":         args.energy_kev,
        "n_primaries":        args.n,
        "threads":            args.threads,
        "beam_radius_cm":     args.beam_radius,
        "physics_list":       args.physics_list,
        "batch_dir":          str(batch_dir),
        "output_dir":         str(run_dir),
        "material":           getattr(world, "MATERIAL",    "unknown"),
        "phantom_cm":         getattr(world, "PHANTOM_CM",  None),
        "target_volume":      getattr(world, "TARGET_VOLUME_NAME", "calorimeter"),
        "detector_volumes":   getattr(world, "DETECTOR_VOLUME_NAMES", []),
        "expected_dedx":      getattr(world, "EXPECTED_DEDX", 1.0),
        "active_z_ranges_mm": getattr(world, "ACTIVE_Z_RANGES_MM", None),
        "dose_spacing_mm":    getattr(world, "CALORIMETER_Z_RES_MM", 0.1),
        "capabilities":       caps,
        "beam_config":        beam_cfg,
        "dose_size_vox":      dose.size if dose else None,
    }
    path = run_dir / "sim_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"  Metadata → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    batch_dir, run_dir = resolve_output_dirs(args, script_dir)
    world              = load_world(args.world, script_dir)
    caps               = resolve_capabilities(world, args)
    beam_cfg           = resolve_beam_config(world, args)

    print("=" * 60)
    print(f"  World       : {args.world}")
    print(f"  Particle    : {args.particle}")
    print(f"  Energy      : {args.energy_kev / 1000:.1f} MeV")
    print(f"  Particles   : {args.n:,}   Threads: {args.threads}")
    print(f"  Capabilities: {caps}")
    print(f"  Beam dir    : {beam_cfg['direction']}")
    print(f"  Batch dir   : {batch_dir}")
    print("=" * 60)

    sim = gate.Simulation()
    if args.run_id is not None:
        sim.random_seed = 1000 + args.run_id
    sim.output_dir = str(run_dir)

    stats = sim.add_actor("SimulationStatisticsActor", "sim_stats")
    stats.output_filename  = "stats.json"
    stats.track_types_flag = True

    units = gate.g4_units
    world.build_world(sim, units)
    if hasattr(world, "add_optical_surfaces"):
        world.add_optical_surfaces(sim, units)

    target_vol     = getattr(world, "TARGET_VOLUME_NAME", "calorimeter")
    actor_registry = wire_actors(sim, world, caps, run_dir, units)

    add_beam_source(sim, args, world, beam_cfg, units)
    configure_physics(sim, args, script_dir, world, caps, target_vol, units)

    db_path = script_dir / "GateMaterials.db"
    if db_path.exists():
        sim.volume_manager.add_material_database(str(db_path))

    sim.number_of_threads = args.threads
    sim.progress_bar      = True

    save_metadata(args, batch_dir, run_dir, world, caps, beam_cfg, actor_registry)

    # ─── OPTICAL PHOTON TIME CUT (GLOBAL FAIL-SAFE) ───────────────────────
    if caps["optical"]:
        print("[ACTOR] Attaching global optical photon lifetime tracking cut.")
        global_time_cut = sim.add_actor("KillActor", "global_optical_time_breaker")
        global_time_cut.attached_to = "world"

        from opengate.actors.filters import GateFilterBuilder
        F = GateFilterBuilder()

        global_time_cut.filter = (
            (F.ParticleName == "opticalphoton") &
            (F.GlobalTime > 20.0 * units.ns)
        )

    sim.run()


if __name__ == "__main__":
    main()