The project is still under active development, so the directory structure and organization may evolve over time.

---

## Requirements

The project primarily relies on

- Python 3.10+
- OpenGATE
- NumPy
- SciPy
- Matplotlib
- Uproot (ROOT file I/O for hit/PhaseSpace trees, without requiring a full ROOT install)
- ROOT / PyROOT (used for Minuit-based Gaussian fits in the sweep analysis)
- pandas
- itk and/or SimpleITK (reading DoseActor `.mhd` volumes)
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

If a `requirements.txt` file is not yet available, install the dependencies manually, e.g.:

```bash
pip install numpy scipy matplotlib uproot pandas itk SimpleITK
```

You will also need a working installation of **OpenGATE** compatible with your simulation environment. ROOT/PyROOT is typically installed separately (e.g. via a conda-forge `root` package or a system/CVMFS stack), since it isn't pip-installable everywhere.

---

## Typical Workflow

1. Configure detector geometry and materials.
2. Run an OpenGATE simulation.
3. Produce ROOT output files (and DoseActor `.mhd` volumes).
4. Analyze detector response using the supplied Python scripts.
5. Generate plots and summary statistics.

For the full RADiCAL sweep specifically:

```bash
# Launch/collect the full multi-module RADiCAL sweep (geometries × wavelength shifters × beam energies)
./run_12_sweep.sh

# Sync raw output from a remote cluster to the local analysis machine
./sync.sh        # or rsync_job.sh, depending on target

# Analyze
python 12_sweep_analysis.py
python 12_sweep_analysis.py --from-cache master_summary_<timestamp>.pkl   # re-plot from a cached summary
```

---

## Included Analyses

Current analyses include work related to

- detector timing performance
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

## Current Work

### ScintX Simulation
Simulation of the conditions at the University of Iowa FLASH beam facility. This project models the response of the patented **ScintillatorX** material and compares the simulated response with experimental measurements to determine its light yield.

### Quartz/Iron Calorimeter Simulation
Tests the concept of extracting additional information from a CMS-style calorimeter by adding a Cherenkov-only detector consisting of an iron absorber followed by a grid of quartz cubes instrumented with SiPMs. The simulation evaluates whether the detector can reconstruct the longitudinal shower profile by measuring the number of SiPM hits as a function of increasing iron thickness.

### RADiCAL Simulations
A collection of proof-of-concept simulations exploring several proposed iterations of the RADiCAL shashlik-style calorimeter.

#### Geometries
- **Original Model**
  - See square standard, but all T-type fibres.
- **Standard Square**
  - Baseline module.
  - Contains four capillaries:
    - **2 T-type** fibers with wavelength shifter only at shower maximum.
    - **2 E-type** fibers with wavelength shifter along the entire fiber.
- **Triple Square**
  - Same geometry as the Standard Square.
  - LYSO scintillator blocks are **3× thicker** at shower max (layer 8).
- **Standard Hexagon**
  - Hexagonal module.
  - Contains:
    - **3 T-type** fibers.
    - **3 E-type** fibers.
- **Triple Hexagon**
  - Same geometry as the Standard Hexagon.
  - LYSO scintillator blocks are **3× thicker** at shower max (layer 8).

#### Wavelength-Shifting Fibers

Each geometry is simulated with the following wavelength shifters:

- **DSB1**
- **BCF92**
- **LuAG:Ce**

Crossing the 2 square/hexagon geometries (standard and triple-thickness) with the 3 wavelength shifters gives the 12 module variants processed together by `12_sweep_analysis.py`. For each variant, optical photons are classified as Scintillation or Cerenkov via `TrackCreatorProcess`, T-type and E-type fiber hits are read out separately from ROOT `detector_hits_*.root` PhaseSpace trees, and double-ended (upstream/downstream) timing is used to reconstruct time-of-flight, shower depth, and a depth-corrected (light-collection-efficiency-corrected) event energy. Results (timing resolution vs. energy, energy resolution vs. energy, longitudinal shower profiles) are plotted per module and compared across modules, including overlays against test-beam data and a reference paper (arXiv:2401.01747).

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

## Contributing

Contributions, bug reports, and suggestions are welcome. If you encounter an issue or have an idea for improving the project, please open an issue or submit a pull request.

---

## Citation

If this software contributes to published work, please cite the associated publication (to be added).

---

## License

A license has not yet been selected for this repository. Before reusing or distributing this code, please consult the repository owner.