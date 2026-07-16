#!/usr/bin/env python3
"""
run_integrated_attenuation.py
================================================================================
A completely self-contained, high-fidelity optical attenuation simulation 
for the RADiCAL capillary structure.

Includes:
- Dynamic XML generation (using your provided Materials & Surface configurations)
- Procedural single-capillary geometry construction
- Optical photon direct injection sweeps
- Clean-up automation for intermediate ROOT files
- Exponential curve fitting with R² quality scoring
- Subprocess pattern execution to circumvent the SimulationEngine singleton limitation.
================================================================================
"""

import os
import sys
import json
import shutil
import uproot
import numpy as np
from pathlib import Path
import opengate as gate
from opengate.actors.filters import GateFilterBuilder

# ─────────────────────────────────────────────────────────────────────────────
# 1. MATERIALS & SURFACES CONFIGURATIONS (YOUR XML DATA)
# ─────────────────────────────────────────────────────────────────────────────

MATERIALS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<materials>
  <material name="G4_SILICON_DIOXIDE">
    <propertiestable>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="1.455"/>
        <ve energy="2.07" value="1.458"/>
        <ve energy="2.48" value="1.462"/>
        <ve energy="3.10" value="1.470"/>
        <ve energy="4.13" value="1.485"/>
        <ve energy="6.20" value="1.540"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="8.0"/>
        <ve energy="6.20" value="8.0"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="LuAg_Ce_WLS">
    <propertiestable>
      <property name="WLSTIMECONSTANT" value="70" unit="ns"/>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="1.84"/>
        <ve energy="2.00" value="1.84"/>
        <ve energy="2.31" value="1.84"/>
        <ve energy="2.61" value="1.84"/>
        <ve energy="3.10" value="1.84"/>
        <ve energy="6.20" value="1.84"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="5.0"/>
        <ve energy="2.31" value="5.0"/>
        <ve energy="6.20" value="5.0"/>
      </propertyvector>
      <propertyvector name="WLSABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="1000.0"/>
        <ve energy="2.31" value="1000.0"/>
        <ve energy="2.48" value="1.000"/>
        <ve energy="2.64" value="0.005"/>
        <ve energy="2.75" value="0.002"/>
        <ve energy="2.90" value="0.005"/>
        <ve energy="3.54" value="1000.0"/>
        <ve energy="6.20" value="1000.0"/>
      </propertyvector>
      <propertyvector name="WLSCOMPONENT" energyunit="eV">
        <ve energy="1.55" value="0.00"/>
        <ve energy="1.77" value="0.15"/>
        <ve energy="2.07" value="0.60"/>
        <ve energy="2.31" value="1.00"/>
        <ve energy="2.48" value="0.45"/>
        <ve energy="2.61" value="0.00"/>
        <ve energy="6.20" value="0.00"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="LYSO">
    <propertiestable>
      <property name="SCINTILLATIONYIELD"         value="320"/>
      <property name="RESOLUTIONSCALE"            value="1.0"/>
      <property name="SCINTILLATIONTIMECONSTANT1" value="40" unit="ns"/>
      <property name="SCINTILLATIONYIELD1"        value="1.0"/>
      <propertyvector name="SCINTILLATIONCOMPONENT1" energyunit="eV">
        <ve energy="2.5" value="0.10"/>
        <ve energy="2.6" value="0.25"/>
        <ve energy="2.7" value="0.50"/>
        <ve energy="2.8" value="0.80"/>
        <ve energy="2.9" value="1.00"/>
        <ve energy="3.0" value="0.80"/>
        <ve energy="3.1" value="0.50"/>
        <ve energy="3.2" value="0.25"/>
        <ve energy="3.3" value="0.10"/>
      </propertyvector>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="2.5" value="1.82"/>
        <ve energy="3.3" value="1.82"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.771" value="0.0313"/>
        <ve energy="1.907" value="0.0313"/>
        <ve energy="2.066" value="0.0344"/>
        <ve energy="2.254" value="0.0344"/>
        <ve energy="2.480" value="0.0313"/>
        <ve energy="2.755" value="0.0313"/>
        <ve energy="2.883" value="0.0344"/>
        <ve energy="2.952" value="0.0380"/>
        <ve energy="3.024" value="0.0313"/>
        <ve energy="3.061" value="0.0265"/>
        <ve energy="3.100" value="0.0215"/>
        <ve energy="3.139" value="0.0161"/>
        <ve energy="3.179" value="0.0112"/>
        <ve energy="3.221" value="0.0078"/>
        <ve energy="3.263" value="0.0058"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="G4_AIR">
    <propertiestable>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="1.000277"/>
        <ve energy="6.20" value="1.000277"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="50"/>
        <ve energy="6.20" value="50"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="G4_WATER">
    <propertiestable>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="1.333"/>
        <ve energy="2.07" value="1.334"/>
        <ve energy="2.48" value="1.336"/>
        <ve energy="3.10" value="1.339"/>
        <ve energy="4.13" value="1.344"/>
        <ve energy="6.20" value="1.355"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="20.0"/>
        <ve energy="6.20" value="20.0"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="G4_Si">
    <propertiestable>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="3.5"/>
        <ve energy="6.20" value="3.5"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="0.001"/>
        <ve energy="6.20" value="0.001"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="ScintX">
    <propertiestable>
      <property name="SCINTILLATIONYIELD"         value="100"/>
      <property name="RESOLUTIONSCALE"            value="1.0"/>
      <property name="SCINTILLATIONTIMECONSTANT1" value="2.0" unit="ns"/>
      <property name="SCINTILLATIONYIELD1"        value="1.0"/>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="1.53"/>
        <ve energy="2.07" value="1.53"/>
        <ve energy="2.48" value="1.53"/>
        <ve energy="3.10" value="1.53"/>
        <ve energy="4.13" value="1.53"/>
        <ve energy="6.20" value="1.53"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="2.0"/>
        <ve energy="6.20" value="2.0"/>
      </propertyvector>
      <propertyvector name="SCINTILLATIONCOMPONENT1" energyunit="eV">
        <ve energy="2.2000" value="0.000"/>
        <ve energy="2.2576" value="0.086"/>
        <ve energy="2.3105" value="0.110"/>
        <ve energy="2.3674" value="0.121"/>
        <ve energy="2.4797" value="0.181"/>
        <ve energy="2.5777" value="0.292"/>
        <ve energy="2.6355" value="0.354"/>
        <ve energy="2.6802" value="0.459"/>
        <ve energy="2.7462" value="0.627"/>
        <ve energy="2.8509" value="0.799"/>
        <ve energy="2.8868" value="0.923"/>
        <ve energy="2.9307" value="1.000"/>
        <ve energy="2.9610" value="0.953"/>
        <ve energy="2.9878" value="0.841"/>
        <ve energy="2.9999" value="0.757"/>
        <ve energy="3.0260" value="0.659"/>
        <ve energy="3.0644" value="0.561"/>
        <ve energy="3.0996" value="0.561"/>
        <ve energy="3.1190" value="0.323"/>
        <ve energy="3.1446" value="0.170"/>
        <ve energy="3.2000" value="0.000"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="Quartz">
    <propertiestable>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="1.455"/>
        <ve energy="2.07" value="1.458"/>
        <ve energy="2.48" value="1.462"/>
        <ve energy="3.10" value="1.470"/>
        <ve energy="4.13" value="1.485"/>
        <ve energy="6.20" value="1.540"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="500.0"/>
        <ve energy="6.20" value="500.0"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="DSB1">
    <propertiestable>
      <property name="WLSTIMECONSTANT" value="3.5" unit="ns"/>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="1.57"/>
        <ve energy="6.20" value="1.57"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="10.0"/>
        <ve energy="6.20" value="10.0"/>
      </propertyvector>
      <propertyvector name="WLSABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="1000.0"/>
        <ve energy="2.40" value="1000.0"/>
        <ve energy="2.48" value="5.000"/>
        <ve energy="2.55" value="0.100"/>
        <ve energy="2.70" value="0.001"/>
        <ve energy="2.95" value="0.0001"/>
        <ve energy="3.54" value="0.0001"/>
        <ve energy="6.20" value="0.0001"/>
      </propertyvector>
      <propertyvector name="WLSCOMPONENT" energyunit="eV">
        <ve energy="1.55" value="0.00"/>
        <ve energy="1.90" value="0.05"/>
        <ve energy="2.15" value="0.30"/>
        <ve energy="2.35" value="0.75"/>
        <ve energy="2.51" value="1.00"/>
        <ve energy="2.58" value="0.40"/>
        <ve energy="2.64" value="0.00"/>
        <ve energy="6.20" value="0.00"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="Air">
    <propertiestable>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="1.000277"/>
        <ve energy="6.20" value="1.000277"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="50.0"/>
        <ve energy="6.20" value="50.0"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="Tungsten">
    <propertiestable>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="3.4"/>
        <ve energy="6.20" value="3.4"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="1e-6"/>
        <ve energy="6.20" value="1e-6"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="Tyvek">
    <propertiestable>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.55" value="1.57"/>
        <ve energy="6.20" value="1.57"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.55" value="0.001"/>
        <ve energy="6.20" value="0.001"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="BCF92">
    <propertiestable>
      <property name="WLSTIMECONSTANT" value="2.7" unit="ns"/>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.00" value="1.60"/>
        <ve energy="2.00" value="1.60"/>
        <ve energy="2.25" value="1.60"/>
        <ve energy="2.48" value="1.60"/>
        <ve energy="2.52" value="1.60"/>
        <ve energy="2.64" value="1.60"/>
        <ve energy="2.76" value="1.60"/>
        <ve energy="2.88" value="1.60"/>
        <ve energy="3.02" value="1.60"/>
        <ve energy="3.10" value="1.60"/>
        <ve energy="3.31" value="1.60"/>
      </propertyvector>
      <propertyvector name="ABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.00" value="3.5"/>
        <ve energy="2.00" value="3.5"/>
        <ve energy="2.25" value="3.5"/>
        <ve energy="2.48" value="3.5"/>
        <ve energy="2.52" value="3.5"/>
        <ve energy="2.64" value="3.5"/>
        <ve energy="2.76" value="3.5"/>
        <ve energy="2.88" value="3.5"/>
        <ve energy="3.02" value="3.5"/>
        <ve energy="3.10" value="3.5"/>
        <ve energy="3.31" value="3.5"/>
      </propertyvector>
      <propertyvector name="WLSABSLENGTH" unit="m" energyunit="eV">
        <ve energy="1.00" value="10000.0"/>
        <ve energy="2.00" value="10000.0"/>
        <ve energy="2.25" value="10000.0"/>
        <ve energy="2.48" value="10000.0"/>
        <ve energy="2.52" value="8.0"/>
        <ve energy="2.64" value="1.0"/>
        <ve energy="2.76" value="0.050"/>
        <ve energy="2.88" value="0.003"/>
        <ve energy="3.02" value="0.0003"/>
        <ve energy="3.10" value="0.0005"/>
        <ve energy="3.31" value="0.001"/>
      </propertyvector>
      <propertyvector name="WLSCOMPONENT" energyunit="eV">
        <ve energy="1.00" value="0.00"/>
        <ve energy="2.00" value="0.00"/>
        <ve energy="2.25" value="0.30"/>
        <ve energy="2.48" value="0.90"/>
        <ve energy="2.52" value="1.00"/>
        <ve energy="2.64" value="0.55"/>
        <ve energy="2.76" value="0.10"/>
        <ve energy="2.88" value="0.00"/>
        <ve energy="3.02" value="0.00"/>
        <ve energy="3.10" value="0.00"/>
        <ve energy="3.31" value="0.00"/>
      </propertyvector>
    </propertiestable>
  </material>
  <material name="Galactic">
    <propertiestable>
      <propertyvector name="RINDEX" energyunit="eV">
        <ve energy="1.50" value="1.0"/>
        <ve energy="6.50" value="1.0"/>
      </propertyvector>
    </propertiestable>
  </material>
</materials>
"""

SURFACES_XML = """<?xml version="1.0" encoding="utf-8"?>
<surfaces>
  <surface name="Polished" model="unified" type="dielectric_dielectric" finish="polished" sigmaalpha="0.0">
    <propertiestable>
      <propertyvector name="REFLECTIVITY" energyunit="eV">
        <ve energy="1.0" value="0.0"/>
        <ve energy="4.5" value="0.0"/>
      </propertyvector>
    </propertiestable>
  </surface>
  <surface model="unified" name="SiPM_surface" type="dielectric_metal" sigmaalpha="0.0" finish="polished">
    <propertiestable>
      <propertyvector name="REFLECTIVITY" energyunit="eV">
        <ve energy="1.38" value="0.0"/>
        <ve energy="4.13" value="0.0"/>
      </propertyvector>
      <propertyvector name="EFFICIENCY" energyunit="eV">
        <ve energy="1.38" value="0.01"/>
        <ve energy="1.77" value="0.10"/>
        <ve energy="2.07" value="0.25"/>
        <ve energy="2.25" value="0.30"/>
        <ve energy="2.76" value="0.40"/>
        <ve energy="3.10" value="0.35"/>
        <ve energy="3.54" value="0.15"/>
        <ve energy="4.13" value="0.05"/>
      </propertyvector>
    </propertiestable>
  </surface>
  <surface name="Tyvek" model="unified" finish="groundfrontpainted" type="dielectric_dielectric">
      <propertiestable>
          <propertyvector name="REFLECTIVITY" energyunit="eV">
              <ve energy="1.0" value="0.98"/>
              <ve energy="4.5" value="0.98"/>
          </propertyvector>
          <propertyvector name="DIFFUSLOBCONSTANT" energyunit="eV">
              <ve energy="1.0" value="1.0"/>
              <ve energy="4.5" value="1.0"/>
          </propertyvector>
      </propertiestable>
  </surface>
</surfaces>
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. PHYSICAL PARAMETERS & CALIBRATION MAP
# ─────────────────────────────────────────────────────────────────────────────

WLS_PEAK_ENERGY_EV = {
    "DSB1": 2.51,         # 494nm
    "LuAg_Ce_WLS": 2.31,  # 535nm
    "BCF92": 2.52         # 492nm
}

N_PHOTONS_PER_RUN = 20000  # Number of optical photons injected per step
THREADS = 8                # CPU threads to allocate local runs


# ─────────────────────────────────────────────────────────────────────────────
# 3. GEOMETRY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_capillary_world(sim, length_mm, wls_material, units):
    """
    Constructs a high-fidelity isolated capillary system embedded inside
    an absorbing Tungsten matrix to prevent cladding propagation reflection.
    """
    script_dir = Path(__file__).resolve().parent
    db_path = script_dir / "GateMaterials.db"
    
    if db_path.exists():
        db_str = str(db_path)
        loaded_dbs = [str(f) for f in sim.volume_manager.material_database.filenames]
        if db_str not in loaded_dbs and "GateMaterials.db" not in [Path(f).name for f in loaded_dbs]:
            sim.volume_manager.add_material_database(db_str)
    else:
        print(f"[Warning] Could not find local GateMaterials.db at: {db_path}")

    # 1. Expand the master world coordinates to envelope the active capillary
    world = sim.world
    world.size = [30.0 * units.mm, 30.0 * units.mm, (length_mm + 20.0) * units.mm]
    world.material = "Air"
    
    # 2. Surround the capillary with dense, absorbent Tungsten (Box uses full dimensions)
    absorber = sim.add_volume("Box", "absorber")
    absorber.parent = world
    absorber.material = "Tungsten"
    absorber.size = [10.0 * units.mm, 10.0 * units.mm, length_mm * units.mm]
    absorber.translation = [0, 0, 0]
    
    # 3. Nest the Quartz capillary shell (Tubs uses half-length dz)
    quartz_sleeve = sim.add_volume("Tubs", "quartz_sleeve")
    quartz_sleeve.parent = absorber
    quartz_sleeve.material = "Quartz"
    quartz_sleeve.rmax = 0.5 * units.mm
    quartz_sleeve.rmin = 0.0 * units.mm
    quartz_sleeve.dz = (length_mm / 2.0) * units.mm  # FIXED: dz is half-length
    quartz_sleeve.translation = [0, 0, 0]
    
    # 4. Nest the Core wavelength shifting (WLS) filament inside the Quartz (Tubs uses half-length dz)
    wls_core = sim.add_volume("Tubs", "wls_core")
    wls_core.parent = quartz_sleeve
    wls_core.material = wls_material
    wls_core.rmax = 0.3 * units.mm
    wls_core.rmin = 0.0 * units.mm
    wls_core.dz = (length_mm / 2.0) * units.mm  # FIXED: dz is half-length
    wls_core.translation = [0, 0, 0]
    
    # 5. Position the Downstream Sensor (Tubs uses half-length dz)
    sipm_down = sim.add_volume("Tubs", "sipm_down")
    sipm_down.parent = world
    sipm_down.material = "G4_Si"
    sipm_down.rmax = 0.5 * units.mm
    sipm_down.rmin = 0.0 * units.mm
    sipm_down.dz = 0.1 * units.mm  # FIXED: dz is half-length (0.2mm total thickness)
    sipm_down.translation = [0, 0, (length_mm / 2.0 + 0.1) * units.mm]


# ─────────────────────────────────────────────────────────────────────────────
# 4. CORE ENGINE WORKER (EXECUTED BY SPAWNED SUBPROCESS)
# ─────────────────────────────────────────────────────────────────────────────

def execute_actual_simulation(length_mm, wls_material, z_offset_mm, peak_ev):
    """
    Constructs and starts the simulation thread. Since Geant4 utilizes an un-destroyable
    C++ singleton system in memory, this is handled within spawned worker processes.
    """
    sim = gate.Simulation()
    sim.output_dir = "temp_attenuation_run"
    sim.number_of_threads = THREADS
    sim.progress_bar = False
    
    units = gate.g4_units
    
    # Load custom materials dynamically
    if os.path.exists("GateMaterials.db"):
        sim.volume_manager.add_material_database("GateMaterials.db")
        
    build_capillary_world(sim, length_mm, wls_material, units)
    
    # Inject isotropic optical photons inside the WLS core at the dynamic Z offset
    source = sim.add_source("GenericSource", "isotropic_optical_source")
    source.particle = "opticalphoton"
    source.polarization = [1, 0, 0]  # FIXED: Suppresses ZeroPolarization warning block!
    source.energy.mono = peak_ev * units.eV
    source.position.type = "cylinder"
    source.position.radius = 0.28 * units.mm
    source.position.dz = 0.1 * units.mm
    source.position.translation = [0.0 * units.mm, 0.0 * units.mm, z_offset_mm * units.mm]
    source.direction.type = "iso"
    source.n = max(1, int(N_PHOTONS_PER_RUN / THREADS))
    
    # Assign physics list and bind custom property tables
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.special_physics_constructors.G4OpticalPhysics = True
    sim.physics_manager.optical_properties_file = "Materials_Att.xml"
    sim.physics_manager.surface_properties_file = "Surfaces_Att.xml"
    
    # Configure detectors and count output hits
    hits_down = sim.add_actor("PhaseSpaceActor", "hits_down")
    hits_down.attached_to = "sipm_down"
    hits_down.output_filename = "hits_down.root"
    hits_down.steps_to_store = "entering"
    
    F = GateFilterBuilder()
    hits_down.filter = (F.ParticleName == "opticalphoton")
    hits_down.attributes = ["EventID", "GlobalTime"]
    
    # Absorb photons striking the SiPM immediately to prevent back-reflection loops
    killer = sim.add_actor("KillActor", "sipm_kill")
    killer.attached_to = "sipm_down"
    killer.filter = (F.ParticleName == "opticalphoton")
    
    # Global optical photon lifetime safety limit (50ns) to kill trapped infinite modes
    global_cut = sim.add_actor("KillActor", "optical_time_breaker")
    global_cut.attached_to = "world"
    global_cut.filter = (F.ParticleName == "opticalphoton") & (F.GlobalTime > 50.0 * units.ns)
    
    # Run simulation
    sim.run()


# ─────────────────────────────────────────────────────────────────────────────
# 5. SUBPROCESS ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_point_simulation(length_mm, wls_material, z_offset_mm, peak_ev):
    """
    Bypasses Geant4's SimulationEngine singleton crash by spinning up
    individual clean Python processes for every iteration point.
    """
    import subprocess
    
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        str(length_mm),
        str(wls_material),
        str(z_offset_mm),
        str(peak_ev)
    ]
    
    # Synchronously blocks parent until output is generated cleanly
    subprocess.run(cmd, check=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. DATA EXTRACTION & ATTENUATION FITTING
# ─────────────────────────────────────────────────────────────────────────────

def extract_sipm_hits():
    """
    Parses active hits directly from the output ROOT tree.
    """
    root_path = Path("temp_attenuation_run/hits_down.root")
    if not root_path.exists():
        return 0
    try:
        with uproot.open(root_path) as file:
            keys = file.keys()
            if not keys:
                return 0
            tree_name = keys[0]
            tree = file[tree_name]
            return len(tree["EventID"].array(library="np"))
    except Exception as e:
        print(f"  [!] Failed to read hits: {e}")
        return 0


def calculate_attenuation_length(distances_mm, hit_counts):
    """
    Solves for effective attenuation length using a log-linear fit:
    ln(I) = ln(I_0) - d / lambda_eff
    """
    d = np.array(distances_mm, dtype=float)
    I = np.array(hit_counts, dtype=float)
    
    # Only fit parameters where hits were successfully detected
    valid = (d > 0) & (I > 0)
    if np.sum(valid) < 2:
        return np.nan, np.nan, np.nan
        
    d_fit = d[valid]
    y_fit = np.log(I[valid])
    
    slope, intercept = np.polyfit(d_fit, y_fit, 1)
    
    lambda_eff_mm = -1.0 / slope if slope != 0 else np.nan
    I0 = np.exp(intercept)
    
    # Determine R²
    y_pred = slope * d_fit + intercept
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    return lambda_eff_mm, I0, r_squared


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN CONTROLLER & SWEEP EXECUTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("========================================================================")
    print("      Executing Integrated Calibration Attenuation Sweep Core           ")
    print("========================================================================")
    
    # Write dynamic configuration profiles
    Path("Materials_Att.xml").write_text(MATERIALS_XML)
    Path("Surfaces_Att.xml").write_text(SURFACES_XML)
    print("[✓] Dynamic property configuration files initialized locally.")
    
    wls_materials = ["DSB1", "LuAg_Ce_WLS", "BCF92"]
    lengths = [120, 360]  # Normal (12cm) vs. Triple-Stack (36cm)
    
    final_results = {}
    
    for mat in wls_materials:
        final_results[mat] = {}
        peak_ev = WLS_PEAK_ENERGY_EV[mat]
        
        for L in lengths:
            print(f"\n[*] Processing configuration: Material={mat} | Length={L}mm")
            print("-" * 72)
            
            # Construct a wide spatial sweep (from deep in the core up to the downstream end)
            half_L = L / 2.0
            z_points = np.linspace(-0.8 * half_L, 0.8 * half_L, 6)
            
            distances_mm = []
            hit_counts = []
            
            for idx, z_val in enumerate(z_points):
                # Calculate the exact distance to the downstream SiPM surface
                dist_to_sensor = half_L - z_val
                distances_mm.append(dist_to_sensor)
                
                # Execute simulated run
                shutil.rmtree("temp_attenuation_run", ignore_errors=True)
                run_point_simulation(L, mat, z_val, peak_ev)
                
                # Retrieve stats
                hits = extract_sipm_hits()
                hit_counts.append(hits)
                
                print(f"  -> Run {idx+1}/6: Source Z={z_val:6.1f} mm | Distance to SiPM={dist_to_sensor:6.1f} mm | Hits Collected={hits:6d}")
                
            # Compute exponential metrics
            lambda_mm, I0, r_sq = calculate_attenuation_length(distances_mm, hit_counts)
            final_results[mat][L] = {
                "lambda_mm": lambda_mm,
                "I0": I0,
                "r_squared": r_sq
            }
            
            lambda_cm_str = f"{lambda_mm / 10.0:6.2f} cm" if not np.isnan(lambda_mm) else "Failed Fit"
            print(f"  [✓] Fit Completed: Effective Attenuation Length = {lambda_cm_str} (R² = {r_sq:.5f})")
            
    # Cleaning environment footprint
    shutil.rmtree("temp_attenuation_run", ignore_errors=True)
    for f in ["Materials_Att.xml", "Surfaces_Att.xml"]:
        if os.path.exists(f):
            os.remove(f)
            
    # ─────────────────────────────────────────────────────────────────────────────
    # OUTPUT DASHBOARD REPORT
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("                   FINAL ATTENUATION ANALYSIS REPORT")
    print("=" * 72)
    print(f"| {'WLS Material':<15} | {'Length (mm)':<12} | {'λ_eff (mm)':<12} | {'λ_eff (cm)':<12} | {'R² Fit':<8} |")
    print("-" * 72)
    
    table_rows = []
    for mat in wls_materials:
        for L in lengths:
            metrics = final_results[mat][L]
            l_mm = metrics["lambda_mm"]
            r2 = metrics["r_squared"]
            
            if np.isnan(l_mm):
                l_mm_str = "N/A"
                l_cm_str = "N/A"
                r2_str = "N/A"
            else:
                l_mm_str = f"{l_mm:8.2f}"
                l_cm_str = f"{l_mm/10.0:8.2f}"
                r2_str = f"{r2:.5f}"
                
            print(f"| {mat:<15} | {L:<12} | {l_mm_str:<12} | {l_cm_str:<12} | {r2_str:<8} |")
            table_rows.append({
                "material": mat,
                "length_mm": L,
                "lambda_mm": l_mm,
                "r_squared": r2
            })
            
    print("=" * 72)
    
    # Save a JSON log for safe record-keeping
    with open("capillary_attenuation_results.json", "w") as jf:
        json.dump(final_results, jf, indent=4)
    print("[✓] Raw fitting array structures backed up to: 'capillary_attenuation_results.json'")
    print("[✓] Process complete!\n")


if __name__ == "__main__":
    # --- SUBPROCESS WORKER INTERCEPT ---
    # Intercept spawned worker tasks cleanly before triggering master loops.
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        L_val = float(sys.argv[2])
        mat_val = sys.argv[3]
        z_val = float(sys.argv[4])
        peak_val = float(sys.argv[5])
        
        execute_actual_simulation(L_val, mat_val, z_val, peak_val)
        sys.exit(0)
        
    main()