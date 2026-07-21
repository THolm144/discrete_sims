# Discrete Sims

A collection of Monte Carlo simulation and analysis tools for studying scintillation detector response using **OpenGATE**. This repository contains utilities for detector simulation, calibration, attenuation studies, timing analysis, and profile reconstruction developed as part of research into radiation detector performance.

The codebase is organized around a typical simulation workflow:

```
Detector Geometry
        │
        ▼
 OpenGATE Simulation
        │
        ▼
  ROOT Output Files
        │
        ▼
 Python Analysis
        │
 ┌──────┼────────┐
 │      │        │
 ▼      ▼        ▼
Timing  Energy  Calibration
Analysis Analysis  & Response
        │
        ▼
   Figures & Reports
```

---

## Features

- Monte Carlo detector simulations using OpenGATE
- Detector timing and energy analysis
- Automated parameter sweep analysis
- Publication-quality plotting and diagnostics

---

## Repository Structure

```
.
├── analysis/                  # Analysis utilities
├── calibration/               # Calibration data and scripts
├── configs/                   # Simulation configuration files
├── profiles/                  # Detector profile utilities
├── worlds/                    # OpenGATE detector geometries
├── *.py                       # Simulation and analysis scripts
├── GateMaterials.db           # Material database
├── Materials*.xml             # Material definitions
└── Surfaces*.xml              # Optical surface definitions
```

The project is still under active development, so the directory structure and organization may evolve over time.

---

## Requirements

The project primarily relies on

- Python 3.10+
- OpenGATE
- NumPy
- SciPy
- Matplotlib
- Uproot
- pandas
- pathlib

Some scripts may require additional scientific Python packages depending on the analysis being performed.

---

## Installation

Clone the repository

```bash
git clone https://github.com/THolm144/discrete_sims.git
cd discrete_sims
```

Install the required Python packages

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not yet available, install the dependencies manually.

You will also need a working installation of **OpenGATE** compatible with your simulation environment.

---

## Typical Workflow

1. Configure detector geometry and materials.
2. Run an OpenGATE simulation.
3. Produce ROOT output files.
4. Analyze detector response using the supplied Python scripts.
5. Generate plots, calibration products, and summary statistics.

Depending on the study, additional calibration or reconstruction steps may be performed.

---

## Included Analyses

Current analyses include work related to


- detector timing performance
- energy calibration
- profile reconstruction
- parameter sweeps
- diagnostic plotting

Each analysis script is designed to be run independently for a specific study or dataset.

---

## Materials

The repository includes material and optical surface definitions used by OpenGATE, including

- `GateMaterials.db`
- `Materials*.xml`
- `Surfaces*.xml`

These define detector materials, optical properties, and surface interactions used during simulation.

---

## Output

Simulation and analysis scripts generate outputs such as

- ROOT files
- reconstructed profiles
- timing reports
- summary statistics
- diagnostic figures

Most generated files are intended to be excluded from version control and reproduced as needed.

---

## Development Status

This repository is an active research project rather than a finalized software package. The codebase continues to evolve as new detector studies and analysis methods are developed.

Future improvements include

- improved package organization
- expanded documentation
- automated testing
- configuration through YAML/JSON
- command-line interfaces
- example datasets and tutorials

---

## Current Work

### ScintX Simulation
Simulation of the conditions at the University of Iowa FLASH beam facility. This project models the response of the patented **ScintillatorX** material and compares the simulated response with experimental measurements to determine its light yield.

### Quartz/Iron Calorimeter Simulation
Tests the concept of extracting additional information from a CMS-style calorimeter by adding a Cherenkov-only detector consisting of an iron absorber followed by a grid of quartz cubes instrumented with SiPMs. The simulation evaluates whether the detector can reconstruct the longitudinal shower profile by measuring the number of SiPM hits as a function of increasing iron thickness.

### RADiCAL Simulations
A collection of proof-of-concept simulations exploring several proposed iterations of the RADiCAL shashlik-style calorimeter.

#### Geometries

- **Standard Square**
  - Baseline module.
  - Contains four capillaries:
    - **2 T-type** fibers with wavelength shifter only at shower maximum.
    - **2 E-type** fibers with wavelength shifter along the entire fiber.

- **Triple Square**
  - Same geometry as the Standard Square.
  - LYSO scintillator blocks are **3× thicker**.

- **Standard Hexagon**
  - Hexagonal module.
  - Contains:
    - **3 T-type** fibers.
    - **3 E-type** fibers.

- **Triple Hexagon**
  - Same geometry as the Standard Hexagon.
  - LYSO scintillator blocks are **3× thicker**.

#### Wavelength-Shifting Fibers

Each geometry is simulated with the following wavelength shifters:

- **DSB1**
- **BCF92**
- **LuAG:Ce**

## Contributing

Contributions, bug reports, and suggestions are welcome. If you encounter an issue or have an idea for improving the project, please open an issue or submit a pull request.

---

## Citation

If this software contributes to published work, please cite the associated publication (to be added).

---

## License

A license has not yet been selected for this repository. Before reusing or distributing this code, please consult the repository owner.
