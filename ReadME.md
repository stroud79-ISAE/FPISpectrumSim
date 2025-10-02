# Thermospheric Wind Doppler Analysis

This repository contains Python scripts for processing atmospheric model data (from **WACCM-X** or **TIEGCM**) to perform Doppler wind and temperature analysis in the thermosphere. The code calculates volume emission rates (VER) at 630 nm, performs vertical and line-of-sight (LOS) interpolations, computes Doppler spectra, and generates output NetCDF files with integrated wind and temperature data along specified LOS paths.

The pipeline supports parallel processing for efficiency and handles data transformation for TIEGCM files to match WACCM-X structure.

---

## Features
- **Model Support**: Processes data from WACCM-X (native) and TIEGCM (via wrapper).
- **VER Calculation**: Computes 630 nm red-line emission rates using physical rate coefficients.
- **Interpolation**: Vertical interpolation to uniform altitude grids and direct interpolation to scattered LOS points.
- **LOS Calculation**: Generates points along LOS paths in cardinal directions (N, E, S, W) using geodetic transformations.
- **Doppler Processing**: Computes LOS winds, Doppler spectra, and integrated wind/temperature from spectra.
- **Parallelization**: Uses Joblib for multi-core processing over time and directions.
- **Output**: Generates NetCDF files with time-series of LOS winds, temperatures, spectra, and Doppler-derived values.
- **Locations**: Pre-defined locations (e.g., Kiruna, Longyearbyen) with customisable coordinates.

---

## Project Structure
    therm-wind-doppler/
    ├── main.py                # Entry point; parses arguments and orchestrates processing
    ├── doppler.py             # LOS path generation and Doppler spectrum calculations
    ├── constants.py           # Physical constants (e.g., Boltzmann constant, atomic masses)
    ├── coord_transforms.py    # Coordinate transformations (WGS84, ECEF, ENU)
    ├── utils.py               # Utilities for location info, output paths, and dataset enhancements
    ├── tiegcm_wrapper.py      # Transforms TIEGCM data to WACCM-X structure
    ├── redline.py             # Calculates 630 nm volume emission rate (VER)
    ├── interpolation.py       # Vertical interpolation to uniform altitude grids
    ├── files_to_process.py    # Lists files in a directory for batch processing
    ├── README.md              # This file
    ├── requirements.txt       # Lists dependencies

---

## Requirements
- Python 3.8+
- Libraries:
  - `xarray`
  - `numpy`
  - `scipy`
  - `joblib`
  - `pyproj`
  - `netCDF4`

Install dependencies:
    pip install xarray numpy scipy joblib pyproj netCDF4

Optional: set environment variable `SLURM_CPUS_PER_TASK` to control parallel jobs.

---

## Installation
    git clone https://github.com/stroud79-ISAE/FPISpectrumSim.git
    cd FPISpectrumSim
    cd ForwardModel

Ensure all `.py` files are in the same directory.

Set up a virtual environment:
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt

---

## Usage

### Basic Command
    python main.py path/to/input.nc --model_type WACCMX --locations kir lyb

### Command-Line Arguments
- `input_file` (**required**): Path to input NetCDF file (WACCM-X or TIEGCM format).
- `--model_type` (default: `"WACCMX"`): `"WACCMX"` or `"TIEGCM"` (activates wrapper).
- `--locations` (default: `['kir','lyb','sod','caj','msh']`): List of location codes.
- `--output_dir` (default: `/rds/projects/r/reidb-waccm-x/generated_files/therm_wind/`): Base output directory.
- `--grid_spacing_km` (default: 2.828): Grid spacing (km).
- `--los_elevation_deg` (default: 45): LOS elevation angle (°).
- `--los_max_distance_km` (default: 600): Max LOS distance (km).
- `--alt_init` (default: 50.0): Initial observer altitude (m).

### Example
Process a TIEGCM file for Kiruna and Sodankylä:
    python main.py /path/to/tiegcm_file.nc --model_type TIEGCM --locations kir sod --output_dir /output/path --los_max_distance_km 1000

---

## Batch Processing
Generate a list of files:
    python files_to_process.py /path/to/directory files_to_process.txt

Then loop over the list in a shell script or SLURM job to process multiple files.

---

## Output Format
Outputs are NetCDF files (e.g., `Kiruna_YYYY-MM-DD_therm_winds.nc`) containing:

- `LOS_wind_{direction}`: LOS wind speed (m/s).
- `LOS_T_{direction}`: LOS temperature (K).
- `wind_doppler_{direction}`: Integrated Doppler wind (m/s).
- `temp_doppler_{direction}`: Integrated Doppler temperature (K).
- `spectrum_{direction}`: Normalised Doppler spectrum.

**Dimensions:** `time`, `distance`, `frequency_bins`.

---

## Contributing
- Fork the repo and submit pull requests.
- Report issues via GitHub Issues.
- Follow **PEP8** and include docstrings.


---

## Acknowledgement & Citation

If you use this project in your research or work, please cite it.

### Plain Text Acknowledgement

A simple acknowledgement in a `README.md` or a presentation is always appreciated:

> This project uses FPISpectrumSim/ForwardModel (https://github.com/stroud79-ISAE/FPISpectrumSim.git), developed by Joseph Stroud.

### For Academic Use (BibTeX)

For academic papers, please use the following BibTeX entry.

```bibtex
@software{STROUD_2025_FPISpectrumSim,
  author = {Mr Joseph Stroud},
  title = {FPISpectrumSim},
  month = {October},
  year = {2025},
  publisher = {GitHub},
  version = {v1.0.0},
  url = {[https://github.com/stroud79-ISAE/FPISpectrumSim.git](https://github.com/stroud79-ISAE/FPISpectrumSim.git)}
}
