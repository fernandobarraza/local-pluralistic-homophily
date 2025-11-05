
# Local Pluralistic Homophily — Reproducible Pipeline
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Reproducible](https://img.shields.io/badge/reproducible-ready-blue.svg)](#reproduce-main-results) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17204707.svg)](https://doi.org/10.5281/zenodo.17204707)

Code and data layout for the paper  
**“Local pluralistic homophily and community boundaries”** (Applied Sciences, 2025).

This repository provides a lightweight, community-aware pipeline to compute:

- Neighborhood-centered pluralistic homophily \(\tilde h_v\) (s1).
- Robustness curves based on the **normalized total inter-community weight** \(\widehat{{W}}_{{\mathrm{{inter}}}}(p)\) (Figure 1).
- Edge-ratio immunization effectiveness (**IER**) comparing \(\tilde h_v\) vs. BridgeCC, consolidated in a **forest plot** (main result).

It also ships optional utilities for descriptive tables, GCC fragmentation snapshots, AUC/first-separation over \( \widehat{{W}}_{{\mathrm{{inter}}}} \), validation, and LFR helpers.

> **Note on CI**: This repository **does not run CI automatically**. Any workflows under `.github/workflows/` are disabled or manual (`workflow_dispatch`).

---

## Repository layout

```
core/                      # Main paper pipeline (minimal deps)
  calc_hv.py               # Step 1: per-node \tilde h_v (s1)
  calc_w_inter_curves.py   # Step 2A: \widehat{{W}}_{{inter}}(p) curves (+ random band)
  plot_w_inter_curves_2x4.py
  calc_all_metrics.py      # Structural baselines (e.g., BridgeCC) needed by IER
  calc_edge_ratio.py       # IER computation & consolidation
  forest_plot.py           # Forest plot of IER (main figure)
  run_pipeline_core.py     # Orchestrates 1 → 2A → IER + forest (main)

supplement/                # Optional analyses (not required for main results)
  calc_sensitivity_curves.py
  plot_sensitivity_with_gcc.py

aux/                       # Utilities / helpers (table-ready outputs, checks, viz)
  calc_global_metrics.py   # Descriptive metrics per dataset (tables)
  calc_auc_w_inter.py      # ΔAUC and p_sep over normalized \widehat{{W}}_{{inter}}(p)
  calc_rho_gcc.py          # ρ_GCC trajectories & snapshots at p
  unify_metrics.py
  viz/plot_overlapping_communities.py
  validation/check_networks.py
  stats/calc_dataset_descriptives.py

lfr/                       # LFR helpers (Lancichinetti–Fortunato benchmarks)
  lfr_convert.py
  gen_node_communities.py
  generate_lfr_baselines.sh
  generate_lfr_8.sh

datasets/<key>/            # (not tracked) Preprocessed datasets (Zenodo)
outputs/                   # (gitignored) Results and figures
```

---

## Installation

- Python **3.9–3.11** (tested).
- Runtime deps:
```bash
pip install -r requirements.txt
```
> On macOS, if `python-igraph` complains, `brew install cairo glpk` can help.

Developer tools (optional):
```bash
pip install -r requirements-dev.txt
```

---

## Datasets

Preprocessed datasets are hosted on **Zenodo** (see `DATASETS.md` for structure).  
If `<key>_node_to_communities.csv` is missing but `<key>_communities.txt` exists, you can auto-generate it:

```bash
python lfr/gen_node_communities.py   --network-file datasets/<key>/<key>_network.txt   --communities-file datasets/<key>/<key>_communities.txt   --out-dir datasets   --dataset-key <key>
```

**Dataset DOI (concept, always latest):**  
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17204706-blue)](https://doi.org/10.5281/zenodo.17204706)


<details>
<summary>Version-specific DOI used in the paper (for exact reproducibility)</summary>

- v2: https://doi.org/10.5281/zenodo.17536223  
- v1: https://doi.org/10.5281/zenodo.17204707
</details>

---

## Reproduce main results

### Figure 1 — \(\widehat{{W}}_{{\mathrm{{inter}}}}(p)\) grid (2×4)

```bash
# Step 1: \tilde h_v
python core/calc_hv.py   --data-dir datasets   --dataset-key <key>   --results-dir outputs/proc3   --methods s1

# Step 2A: curves per dataset (repeat for each <key>)
python core/calc_w_inter_curves.py   --edges datasets/<key>/<key>_network.txt   --node2comms datasets/<key>/<key>_node_to_communities.csv   --metrics-file outputs/proc3/<key>_hv_metrics.csv   --out-dir outputs/meta_graph_eval   --no-plot

# Grid plot (8 datasets)
python core/plot_w_inter_curves_2x4.py   --datasets dblp lj youtube so deezer github amazon twitch   --in-dir outputs/meta_graph_eval   --out-file outputs/meta_graph_eval/meta_eval_grid_4x2_norm.png
```

### Main result — IER forest (\(\tilde h_v\) vs BridgeCC)

```bash
# Structural baselines required by IER
python core/calc_all_metrics.py   --data-dir datasets   --dataset-key <key>   --proc3-dir outputs/proc3   --metrics bridgecc   --force

# IER compute and consolidate
python core/calc_edge_ratio.py   --datasets dblp lj youtube so deezer github amazon twitch   --data-dir datasets   --proc3-dir outputs/proc3   --out-dir outputs/edge_ratio   --taus 0.00 0.10 0.20 0.30   --budget 0.10   --n-boot 300   --n-perm 500   --seed 42

# Forest plot (choose τ, e.g., 0.30)
python core/forest_plot.py   --in-dir outputs/edge_ratio   --datasets dblp lj youtube so deezer github amazon twitch   --tau 0.30   --out-file outputs/edge_ratio/figs/forest_meta_bw.pdf
```

> Shortcut: `core/run_pipeline_core.py` executes the main figures end-to-end.

---

## Optional utilities

- **Descriptive tables**:  
  `python aux/calc_global_metrics.py --datasets … --data-dir datasets --out-file outputs/proc2/ext_networks_metrics_addons.csv`
- **AUC & first separation** over \( \widehat{{W}}_{{\mathrm{{inter}}}} \):  
  `python aux/calc_auc_w_inter.py --datasets … --in-dir outputs/meta_graph_eval`
- **ρ_GCC snapshots** at \(p\in\{{0.05,0.10\}}\):  
  `python aux/calc_rho_gcc.py --datasets … --data-dir datasets --proc3-dir outputs/proc3`

---

## Reproducibility notes

- All scripts are **CLI-first** with deterministic defaults when applicable.
- Random baselines expose `--random-reps` (default 20). Set a global `--seed` where provided.
- Node-removal uses **static rankings** (efficiency) as per Hébert‑Dufresne et al. (2013).

---

## Citing

If you use this work, please cite **the paper**, **the dataset**, and (optionally) **the code**.

**Paper**  
> Barraza, F., Ramírez-Ovalle, C., Álvarez, A., & Fernández, A. (2025).  
> *Local Pluralistic Homophily for Boundary-Spanning Node Detection in Overlapping Community Networks*.  
> (journal info and DOI to appear)

**Dataset (Zenodo)**  
> Barraza, F. (2025). *Local Pluralistic Homophily — Preprocessed Datasets* [Data set]. Zenodo.  
> **Concept DOI (latest):** https://doi.org/10.5281/zenodo.17204706  
> *For exact replication of the manuscript:* **v2 DOI:** https://doi.org/10.5281/zenodo.17536223

**Software (GitHub)**  
> Barraza, F. (2025). *local-pluralistic-homophily — Reproducible Pipeline* [Source code]. GitHub.  
> https://github.com/fernandobarraza/local-pluralistic-homophily  
> *(Pin a specific version for reproducibility, e.g., tag `v1.0.0` or commit `abcdef1`.)*
---

## License

MIT — see [`LICENSE`](LICENSE).
