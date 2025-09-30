# Pluralistic Homophily — Reproducible Pipeline

Code for the paper *Local pluralistic homophily and community boundaries* (Applied Sciences, 2025).  
This repository contains a lightweight, community-aware pipeline to compute:

- Neighborhood-centered pluralistic homophily \(\tilde h_v\)
- Robustness curves based on the **normalized total inter–community weight** \(\widehat{W}_{\mathrm{inter}}(p)\)
- Sensitivity curves \(\tilde h(p)\) and **GCC** fraction \(\rho_{\mathrm{GCC}}(p)\)

It also ships auxiliary tools for validation, descriptive tables, supplemental homophily variants (edge/community level), and LFR generation.

---

## Repository layout

```
core/                    # Main paper pipeline (minimal dependencies)
  calc_hv.py             # Step 1: per-node \tilde h_v (s1)
  calc_w_inter_curves.py # Step 2A: \widehat{W}_{inter}(p) curves (+ random band)
  plot_w_inter_curves_2x2.py
  calc_sensitivity_curves.py  # Step 2B: \tilde h(p) and GCC curves
  plot_sensitivity_with_gcc.py
  run_pipeline_core.py   # Orchestrates 1→2A→2B (+ plots)

aux/                     # Optional analyses (not required for the paper)
  calc_all_metrics.py    # Structural baselines (e.g., BridgeCC, CI)
  calc_ghv.py            # Global-mean version (previous paper)
  unify_metrics.py       # Legacy (optional)
  stats/calc_dataset_descriptives.py       # Table of dataset statistics
  validation/check_networks.py             # Sanity checks for inputs
  supplement/calc_edge_homophily.py        # \tilde h_e (edge-level, SI)
  supplement/calc_community_homophily.py   # \tilde h_C (community-level, SI)
  viz/plot_overlapping_communities.py      # Toy visualization
  run_pipeline_aux.py    # Runner for ghv/structural baselines

lfr/                     # LFR helpers (authors' benchmark required)
  generate_lfr_8.sh                  # 8 LFRs (appendix)
  generate_lfr_baselines.sh          # LFR_low/mid/high (main figures)
  lfr_convert.py                     # Convert .dat → repo layout (0-based)
  gen_node_communities.py            # Build node→communities CSV from communities.txt

datasets/<key>/          # (not tracked) Preprocessed datasets (Zenodo)
outputs/                 # (gitignored) Results and figures
```

---

## Installation

- Python 3.10+
- Install runtime dependencies:
```bash
pip install -r requirements.txt
```
> Note: `python-igraph` and `networkit` may require system packages/compilers.  
> On macOS: `brew install cairo glpk` can help if `igraph` asks for it.

Dev tools (optional):
```bash
pip install -r requirements-dev.txt
```

---

## Datasets

Preprocessed datasets are hosted on **Zenodo** (see `DATASETS.md` for the folder layout).  
If a dataset folder is complete (network, communities, node→communities), the pipeline uses it as-is.

**Auto-generation fallback:**  
If `datasets/<key>/<key>_node_to_communities.csv` is missing but `*_communities.txt` exists, the runners will call:
```bash
python lfr/gen_node_communities.py   --network-file datasets/<key>/<key>_network.txt   --communities-file datasets/<key>/<key>_communities.txt   --out-dir datasets --dataset-key <key>
```

---

## Main pipeline (paper)

Compute \(\tilde h_v\), then \(\widehat{W}_{inter}(p)\), then \(\tilde h(p)\) and GCC.  
All outputs land under `outputs/`.

```bash
python core/run_pipeline_core.py   --datasets dblp lj youtube so   --data-dir datasets   --proc3-dir outputs/proc3   --meta-dir outputs/meta_graph_eval   --curves-dir outputs/node_removal_curves   --sens-dir outputs/sensitivity
```

- **Step 1**: `calc_hv.py` → `outputs/proc3/<key>_hv_metrics.csv`
- **Step 2A**: `calc_w_inter_curves.py` → `outputs/meta_graph_eval/<key>/meta_eval_<key>.csv`
- **Step 2B**: `calc_sensitivity_curves.py` → `outputs/sensitivity/summary_curves.csv` (+ markers)

Figures:
- `plot_w_inter_curves_2x2.py` → `meta_eval_grid_2x2_norm.png` (when 4 datasets are given)
- `plot_sensitivity_with_gcc.py` → `sensitivity_panel_with_gcc.png`

---

## Auxiliary comparisons (optional)

Add structural baselines or compute the global-mean homophily (ghv):
```bash
# BridgeCC only on top of hv table
python aux/run_pipeline_aux.py   --datasets so youtube   --data-dir datasets   --proc1-dir outputs/proc1   --proc3-dir outputs/proc3   --metrics bridgecc

# BridgeCC + eigenvector + CI, also compute ghv
python aux/run_pipeline_aux.py   --datasets so youtube   --data-dir datasets   --proc1-dir outputs/proc1   --proc3-dir outputs/proc3   --with-ghv   --metrics bridgecc eigenvector ci --ci-radius 2
```

---

## LFR generation (optional)

We rely on the original **LFR benchmarks** by Lancichinetti & Fortunato:  
<https://github.com/andrealancichinetti/LFRbenchmarks>

1) Compile the benchmark.
2) Generate raw LFR datasets:
```bash
bash lfr/generate_lfr_baselines.sh /path/to/benchmark ./lfr_out
bash lfr/generate_lfr_8.sh         /path/to/benchmark ./lfr_out
```
3) Convert to the repo layout (0-based, dedup, no self-loops):
```bash
python lfr/lfr_convert.py --in ./lfr_out --key LFR_low  --out datasets
python lfr/lfr_convert.py --in ./lfr_out --key LFR_mid  --out datasets
python lfr/lfr_convert.py --in ./lfr_out --key LFR_high --out datasets
```
4) Build node→communities if needed:
```bash
python lfr/gen_node_communities.py --data-dir datasets --datasets LFR_low LFR_mid LFR_high
```

---

## Makefile shortcuts

```bash
make core     # prints help for core runner
make aux      # prints help for aux runner
make lint     # ruff check
make test     # smoke tests (--help) without data
```

---

## Reproducibility

- All scripts are **CLI-first** with deterministic defaults when applicable.
- Random baselines: use `--random-reps` (default 20) and set an environment seed if needed.
- The pipeline uses **static rankings** for node removal (efficiency) as per Hébert-Dufresne (2013).

---

## Citing

Please cite the paper and this repository. A `CITATION.cff` file is included (GitHub will render a “Cite this repository” button).  
We recommend archiving a release on Zenodo to mint a DOI.

---

## License

MIT — see `LICENSE`.
