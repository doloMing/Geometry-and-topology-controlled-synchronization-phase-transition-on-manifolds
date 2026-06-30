# Numerical experiments

This directory contains the numerical workflow used for the manuscript figures in Geometry- and topology-controlled synchronization phase transition on manifolds.

## Directory structure

- `notebooks/`: executable notebooks, one for each manuscript data figure.
- `src/`: shared numerical and plotting modules.
- `data/`: generated CSV files.
- `figures/`: generated figure files.
- `logs/`: optional local run logs.

## Running the figure notebooks

Run a notebook from `notebooks/` to regenerate the corresponding data files in
`data/` and figure files in `figures/`.

1. `fig01_linear_threshold_curves.ipynb`: Figure 1.
2. `fig02_growth_rate_finite_size.ipynb`: Figure 2.
3. `fig03_topological_transition.ipynb`: Figure 3.
4. `fig04_nonspherical_validation.ipynb`: Figure 4.
5. `fig05_direct_dynamics.ipynb`: Figure 5.

For Figure 5, `RECOMPUTE = False` loads the existing CSV files and regenerates
the figure. Set `RECOMPUTE = True` to rerun the direct finite-\(N\) dynamical
simulations.

`make_notebooks.py` rebuilds the notebook files from the source workflow. It is
only needed after editing the notebook-generation script.

## Source modules

- `manifold_sync.py`: shared manifold synchronization routines.
- `finite_oscillator_dynamics.py`: direct finite-\(N\) oscillator simulations
  and figure assembly for the final dynamical tests.
- `representative_manifold_dynamics.py`: product manifold and non-spherical
  representative manifold simulations.
- `paths.py`: output-directory definitions.
- `plotting.py`: shared plotting style and save helpers.
- `run_tablevi_representative_dynamics.py`: command-line helper for regenerating
  the Table VI representative manifold data.

## Data files used by the manuscript figures

- Figure 1: `fig01_spectral_raw.csv`, `fig01_threshold_summary.csv`.
- Figure 2: `fig02_finite_sample_geometry.csv`,
  `fig02_finite_sample_threshold.csv`.
- Figure 3: `fig03_defect_indices.csv`, `fig03_defect_summary.csv`,
  `fig03_topological_transition_samples.csv`,
  `fig03_topological_transition_summary.csv`,
  `fig03_topological_transition_widths.csv`.
- Figure 4: `fig04_nonspherical_kappa_samples.csv`,
  `fig04_nonspherical_threshold_samples.csv`,
  `fig04_nonspherical_defect_indices.csv`,
  `fig04_nonspherical_defect_summary.csv`,
  `fig04_nonspherical_coefficient_samples.csv`,
  `fig04_nonspherical_transition_summary.csv`.
- Figure 5: `fig05_construction_conditioned_raw.csv`,
  `fig05_construction_conditioned_summary.csv`,
  `fig05_construction_conditioned_fit.csv`,
  `fig05_construction_conditioned_merged.csv`,
  `fig05_construction_conditioned_ensemble.csv`,
  `fig05_construction_conditioned_high_resolution_D3_D6.csv`,
  `fig05_product_formal_raw.csv`, `fig05_product_formal_summary.csv`,
  `fig05_tablevi_formal_raw.csv`, and
  `fig05_tablevi_formal_summary.csv`.

The directory `../data_to_upload/` contains the upload-ready copy of these CSV
files.
