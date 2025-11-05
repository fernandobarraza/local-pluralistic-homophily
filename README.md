# Local Pluralistic Homophily — Reproducible Pipeline

Code and data layout for the paper  
**“Local pluralistic homophily and community boundaries”** (Applied Sciences, 2025).

This repository provides a lightweight, community-aware pipeline to compute:

- Neighborhood-centered pluralistic homophily \(\tilde h_v\) (s1).
- Robustness curves based on the **normalized total inter-community weight** \(\widehat{W}_{\mathrm{inter}}(p)\) (Figure 1).
- Edge-ratio immunization effectiveness (**IER**) comparing \(\tilde h_v\) vs. BridgeCC, consolidated in a **forest plot** (main result).

It also ships optional utilities for descriptive tables, GCC fragmentation snapshots, AUC/first-separation over \( \widehat{W}_{\mathrm{inter}} \), validation, and LFR helpers.

---

## Repository layout

```
core/                      # Main paper pipeline (minimal deps)
  calc_hv.py
  calc_w_inter_curves.py
  plot_w_inter_curves_2x4.py
  calc_all_metrics.py
  calc_edge_ratio.py
  forest_plot.py
  run_pipeline_core.py

supplement/                # Optional analyses
  calc_sensitivity_curves.py
  plot_sensitivity_with_gcc.py

aux/                       # Utilities / helpers
  calc_global_metrics.py
  calc_auc_w_inter.py
  calc_rho_gcc.py
  unify_metrics.py
  ...

lfr/                       # LFR benchmark helpers
datasets/<key>/            # (not tracked) data
outputs/                   # results and figures
```

---

## Installation

Requires Python **3.9–3.11**.

```bash
pip install -r requirements.txt
# optional developer tools
pip install -r requirements-dev.txt
```

---

## Datasets

Preprocessed datasets are hosted on **Zenodo** (see `DATASETS.md`).  
If `<key>_node_to_communities.csv` is missing:

```bash
python lfr/gen_node_communities.py   --network-file datasets/<key>/<key>_network.txt   --communities-file datasets/<key>/<key>_communities.txt   --out-dir datasets   --dataset-key <key>
```

---

## Reproduce main results

### Figure 1 — \(\widehat{W}_{\mathrm{inter}}(p)\) grid (2×4)

```bash
python core/calc_hv.py --data-dir datasets --dataset-key <key>   --results-dir outputs/proc3 --methods s1

python core/calc_w_inter_curves.py   --edges datasets/<key>/<key>_network.txt   --node2comms datasets/<key>/<key>_node_to_communities.csv   --metrics-file outputs/proc3/<key>_hv_metrics.csv   --out-dir outputs/meta_graph_eval --no-plot

python core/plot_w_inter_curves_2x4.py   --datasets dblp lj youtube so deezer github amazon twitch   --in-dir outputs/meta_graph_eval   --out-file outputs/meta_graph_eval/meta_eval_grid_4x2_norm.png
```

### Main result — IER forest

```bash
python core/calc_all_metrics.py --data-dir datasets --dataset-key <key>   --proc3-dir outputs/proc3 --metrics bridgecc --force

python core/calc_edge_ratio.py   --datasets dblp lj youtube so deezer github amazon twitch   --data-dir datasets --proc3-dir outputs/proc3 --out-dir outputs/edge_ratio   --taus 0.00 0.10 0.20 0.30 --budget 0.10 --n-boot 300 --n-perm 500 --seed 42

python core/forest_plot.py   --in-dir outputs/edge_ratio   --datasets dblp lj youtube so deezer github amazon twitch   --tau 0.30   --out-file outputs/edge_ratio/figs/forest_meta_bw.pdf
```

> Shortcut: `core/run_pipeline_core.py` executes both main figures end-to-end.

---

## Optional utilities

- **Descriptive tables**: `python aux/calc_global_metrics.py --datasets … --data-dir datasets`
- **AUC & first separation**: `python aux/calc_auc_w_inter.py --datasets … --in-dir outputs/meta_graph_eval`
- **ρ_GCC snapshots**: `python aux/calc_rho_gcc.py --datasets … --data-dir datasets --proc3-dir outputs/proc3`

---

## CI / Workflows

This repository **does not run CI automatically**.  
All GitHub Actions are disabled or manual (`workflow_dispatch`).

---

## Citing

Please cite both the paper and this repository.  
We recommend archiving a release on Zenodo for DOI persistence.

---

## License

MIT — see `LICENSE`.
