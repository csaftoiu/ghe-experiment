# Greenhouse Effect Experiment

Code and data for "A Genuine Demonstration of the Radiative Greenhouse Effect" - an experimental demonstration showing that surfaces under infrared-opaque panes (borosilicate) become warmer than under infrared-transparent panes (calcium fluoride), directly demonstrating the radiative greenhouse effect.

## Repository Structure

```
ghe-experiment/
├── data/
│   ├── main_exp/           # Main experiment raw data (24-hour runs)
│   │   ├── *.csv           # Raw temperature logs from data loggers (exported w/ T&D software)
│   │   ├── condition_ranges.csv  # Time ranges for sun/cloud conditions
│   │   ├── drop_ranges.csv       # Time ranges to exclude (adjustments, etc.)
│   │   ├── sun_*.csv             # Solar noon times
│   │   └── processed/      # Processed data (generated, gitignored)
│   ├── materials/          # Optical material properties
│   │   ├── *_transmission.csv    # Transmission spectra
│   │   └── *_refrindex*.csv      # Refractive index data
│   ├── simple_variant/     # Simple variant experiment data
│   │   └── newdemo_figure1.csv
│   └── raw_tnd_data/       # Original T&D data logger exports (zipped)
├── 3d_designs/             # OpenSCAD files for apparatus
│   ├── main_apparatus_base.scad
│   ├── thermocouple_housing.scad
│   ├── modified_pyran_pyrge_housing.scad
│   └── ...
├── figures/                # Generated figures (gitignored)
├── ghe_exp/                # Python module
│   └── graph_config.py     # Plot styling configuration
├── main_exp_plots.py       # Main experiment data processing & plotting
├── pane_properties.py      # Material transmission/refractive index plots
├── simple_variant_plots.py # Simple variant figure generation
├── model.py                # Thermal model (steady-state & time-evolution)
├── generate_all_figures.sh # Generate all paper figures
└── run_all_sims.sh         # Run all model simulations
```

## Quick Start

### Generate All Figures

```bash
./generate_all_figures.sh
```

This will:
1. Process raw experiment data (if not already processed)
2. Generate transmission spectrum plot
3. Generate clear-sky temperature difference figure
4. Generate cloudy-sky temperature difference figure
5. Generate simple variant figures

Output PDFs are saved to `figures/`.

### Run All Simulations

```bash
./run_all_sims.sh
```

Runs the thermal model for:
- Peak sunlight and nighttime conditions
- Single-layer and two-layer pane configurations
- Various counterfactual scenarios (equal solar, equal IR, etc.)

## Scripts

### `pane_properties.py`

Plot material optical properties.

```bash
# Generate transmission spectrum plot
python pane_properties.py transmission
```

### `main_exp_plots.py`

Process and plot main experiment data.

### `simple_variant_plots.py`

Generate figures for the simplified variant experiment.

```bash
# Main experiment figure (Figure 5)
python simple_variant_plots.py main_exp

# Glass comparison figure (Figure 6)
python simple_variant_plots.py glass_exp
```

### `model.py`

Thermal model for steady-state analysis and time evolution.

```bash
# Steady-state analysis
python model.py run_ss peak                    # Peak sunlight conditions
python model.py run_ss night                   # Nighttime conditions
python model.py run_ss peak --two-layer-pane   # Two-layer pane model

# Counterfactual analysis
python model.py run_ss peak --counterfactuals=equal_solar      # Equal solar transmission
python model.py run_ss peak --counterfactuals=equal_ir         # Equal IR transmission
python model.py run_ss peak --counterfactuals=equal_solar,equal_ir  # Both

# Time-series fitting (requires processed data)
python model.py run_tsf data/main_exp/processed/June2021_SBC_full.csv
```

## Data

### Main Experiment (`data/main_exp/`)

24-hour temperature recordings with three apparatuses (positions A, B, C) covered by different optical panes:
- **S** = Sapphire, **B** = Borosilicate, **C** = CaF2
- Filename format: `{Date}_{ABC_order}_{condition}.csv`
- Example: `June2021_SBC_full.csv` = June 20-21, A=Sapph, B=Boro, C=CaF2

### Material Properties (`data/materials/`)

Transmission spectra and refractive index data for borosilicate, sapphire, and calcium fluoride from 0.2-80 μm wavelength range.

### Simple Variant (`data/simple_variant/`)

Short-duration swap experiment demonstrating causality of the greenhouse effect.

## 3D Designs

OpenSCAD files for all apparatus components. Open in OpenSCAD to render and export STL files for 3D printing.

## Citation

If you use this code or data, please cite.
