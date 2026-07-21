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

## Current work
---
- ScintX simulation - simulation of the conditions at the UIowa FLASH beam facility contructed to model the response of a patented material, ScintillatorX, and discover its light yield in combination with experiment.

- quartz/iron calorimeter simulation - tests the idea of pulling more data from a CMS-style calorimeter by adding a Cherenkov-only detector that is an iron shield in front of a grid of quartz cubes with sipms behind them. This simulation specifically verifies the capability of such a device to reproduce a shower profile by measuring the number of sipm hits vs increasing thickness of shielding.

- RADiCAL simulations - these are primarily proof of concept tests for various theorized iterations of the RADiCAL shashlik-style calorimeter. They are split into 4 different geometries and 3 different types of wavelength-shifting fibre. The geometries are as follows: 
       - standard square, which is the basic module, where 2 of the capillaries therein are t-type with wavelength shifter at shower max only and the other 2 are e-type and have wls throughout.
       - triple square, which is the same as the standard squaree except the LYSO scintillating blocks are three times as thick.
       - standard hexagon, which is heaxagonally shaped and has 3 of each t-type and e-type.
       - triple hexagon, which is the same as the standard hex module except for having three times as much LYSO.

The three wavelength shifters simulated in each of the four modules are DSB1, BCF92, and LuAG:Ce.
---

## Contributing

Contributions, bug reports, and suggestions are welcome. If you encounter an issue or have an idea for improving the project, please open an issue or submit a pull request.

---

## Citation

If this software contributes to published work, please cite the associated publication (to be added).

---

## License

A license has not yet been selected for this repository. Before reusing or distributing this code, please consult the repository owner.
