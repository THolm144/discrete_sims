"""
simulator.py (Modified for Track Graphing)
==========================================
CLI-driven OpenGATE simulation with full optical track logging capabilities.
"""

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import opengate as gate


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS & CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CAPABILITIES = {
    "optical":          False,
    "dose":             True,
    "sipm_hits":        False,
    "optical_exits":    False,
    "calorimeter_mode": False,
}

DEFAULT_BEAM_CONFIG = {
    "direction": [0, 0, 1],
    "target_cm": [0, 0, 0],
    "offset_cm": 2.0,
}

def parse_args():
    p = argparse.ArgumentParser(description="OpenGATE modular simulator")
    p.add_argument("--world",        default="iron_quartz_array")
    p.add_argument("--particle",     default="proton")
    p.add_argument("--energy-kev",   type=float, default=500_000)
    p.add_argument("--n",            type=int,   default=10_000)
    p.add_argument("--threads",      type=int,   default=64)
    p.add_argument("--beam-radius",  type=float, default=1.0)
    p.add_argument("--output-dir",   default=None)
    p.add_argument("--physics-list", default="G4EmStandardPhysics_option4")
    p.add_argument("--run-id",       type=int,   default=None)
    p.add_argument("--optical",      choices=["on", "off", "world"],  default="world")
    p.add_argument("--dose",         choices=["on", "off", "world"],  default="world")
    p.add_argument("--sipm-hits",    choices=["on", "off", "world"],  default="world")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# CODE LOGIC MODIFICATIONS
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
    override_map = {"optical": args.optical, "dose": args.dose, "sipm_hits": args.sipm_hits}
    for key, val in override_map.items():
        if val == "on": caps[key] = True
        elif val == "off": caps[key] = False
    return caps

def resolve_beam_config(world) -> dict:
    return {**DEFAULT_BEAM_CONFIG, **getattr(world, "BEAM_CONFIG", {})}

def resolve_output_dirs(args, script_dir: Path) -> tuple[Path, Path]:
    if args.output_dir:
        batch_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = script_dir / "runs" / args.world / f"{int(args.energy_kev)}keV_{ts}"
    run_id = args.run_id if args.run_id is not None else 0
    run_dir = batch_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir, run_dir


def wire_actors(sim, world, caps: dict, run_dir: Path, units) -> dict:
    registry = {
        "optical_exited_actor": None,
        "hit_actors":           [],
        "dose_actor":           None,
        "track_actor":          None,
        "is_calorimeter":       caps["calorimeter_mode"],
    }

    target_vol       = getattr(world, "TARGET_VOLUME_NAME",    "target")
    detector_volumes = getattr(world, "DETECTOR_VOLUME_NAMES", [])

    # ── NEW: OPTICAL TRACK LOGGING ACTOR ─────────────────────────────────────
    if caps["optical"]:
        # Stores step-by-step positions of photons anywhere inside the mother target volume
        tracks = sim.add_actor("PhaseSpaceActor", "optical_tracks")
        tracks.attached_to     = target_vol
        tracks.output_filename = "optical_tracks.root"
        tracks.steps_to_store  = "all"  # "all" saves every spatial step coordinate
        tracks.attributes = [
            "Position", "TrackID", "EventID", "ParticleName", "KineticEnergy"
        ]
        registry["track_actor"] = tracks
        print("[SIM] Attached Step-by-Step Track Logging Actor.")

    # ── Optical exits ─────────────────────────────────────────────────────
    if caps["optical"] and caps.get("optical_exits", False):
        exited = sim.add_actor("PhaseSpaceActor", "optical_exited")
        exited.attached_to     = target_vol
        exited.output_filename = "optical_exited.root"
        exited.steps_to_store  = "exiting"
        exited.attributes = ["ParticleName", "KineticEnergy", "TrackCreatorProcess", "Position", "TrackID", "EventID", "GlobalTime"]
        registry["optical_exited_actor"] = exited

    # ── Per-detector hit actors ───────────────────────────────────────────
    if caps["sipm_hits"] and detector_volumes:
        for idx, vol_name in enumerate(detector_volumes):
            if vol_name not in sim.volume_manager.volumes: continue
            hits = sim.add_actor("PhaseSpaceActor", f"detector_hits_{idx}")
            hits.attached_to     = vol_name
            hits.authorize_repeated_volumes = True
            hits.output_filename = f"detector_hits_{idx}.root"
            hits.steps_to_store  = "entering"
            hits.attributes = ["ParticleName", "KineticEnergy", "Position", "TrackCreatorProcess", "TrackID", "EventID"]
            registry["hit_actors"].append(hits)

    # ── Dose actor ────────────────────────────────────────────────────────
    if caps["dose"]:
        phantom_cm = world.PHANTOM_CM
        dose = _wire_standard_dose(sim, target_vol, phantom_cm, units)
        registry["dose_actor"] = dose

    return registry


def _wire_standard_dose(sim, target_vol: str, phantom_cm: list, units):
    dose = sim.add_actor("DoseActor", "dose_actor")
    dose.attached_to      = target_vol
    dose.output_filename  = "edep.mhd"
    dose.size             = [int(round(phantom_cm[0] * 10)), int(round(phantom_cm[1] * 10)), int(round(phantom_cm[2] * 10))]
    dose.spacing          = [1.0 * units.mm] * 3
    dose.hit_type         = "random"
    dose.edep.active      = True
    return dose

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
    source.position.translation = (source_pos_cm * units.cm).tolist()
    source.direction.type     = "momentum"
    source.direction.momentum = direction.tolist()
    source.n = max(1, int(args.n / args.threads))
    return source

def configure_physics(sim, args, script_dir: Path, world, caps: dict, target_vol: str, units):
    sim.physics_manager.physics_list_name = args.physics_list
    surface_file = script_dir / "SurfaceProperties.xml"
    if surface_file.exists():
        sim.physics_manager.surface_properties_file = str(surface_file)

    if caps["optical"]:
        sim.physics_manager.special_physics_constructors.G4OpticalPhysics = True
        optical_file = script_dir / "Materials.xml"
        if optical_file.exists():
            sim.physics_manager.optical_properties_file = str(optical_file)
    else:
        sim.physics_manager.special_physics_constructors.G4OpticalPhysics = False

def save_metadata(args, batch_dir: Path, run_dir: Path, world, caps: dict, beam_cfg: dict, actor_registry: dict):
    metadata = {
        "world": args.world, "particle": args.particle, "energy_kev": args.energy_kev,
        "n_primaries": args.n, "threads": args.threads, "capabilities": caps, "phantom_cm": getattr(world, "PHANTOM_CM",  None)
    }
    with open(run_dir / "sim_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


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
    beam_cfg           = resolve_beam_config(world)

    sim = gate.Simulation()
    if args.run_id is not None:
        sim.random_seed = 1000 + args.run_id
    sim.output_dir = str(run_dir)

    units = gate.g4_units
    world.build_world(sim, units)

    target_vol = getattr(world, "TARGET_VOLUME_NAME", "target")
    actor_registry = wire_actors(sim, world, caps, run_dir, units)

    add_beam_source(sim, args, world, beam_cfg, units)
    configure_physics(sim, args, script_dir, world, caps, target_vol, units)

    db_path = script_dir / "GateMaterials.db"
    if db_path.exists():
        sim.volume_manager.add_material_database(str(db_path))

    sim.number_of_threads = args.threads
    save_metadata(args, batch_dir, run_dir, world, caps, beam_cfg, actor_registry)
    
    sim.run()
    print(f"\nExecution Complete. Track files located in: {run_dir}")


if __name__ == "__main__":
    main()