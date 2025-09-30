# Local Pluralistic Homophily

This repository accompanies the paper  
**“Local pluralistic homophily and community boundaries”**  

It provides the preprocessing pipeline, code, and standardized datasets (via Zenodo) used in the analyses.

---

## Datasets

Preprocessed edge lists and community assignments are hosted on Zenodo:  
[https://doi.org/10.5281/zenodo.17204707](https://doi.org/10.5281/zenodo.17204707)

### Core datasets (analyzed in the article)

- **DBLP** (co-authorship, venue communities)  
- **StackOverflow (SO)** (proxy tag communities)  
- **LiveJournal (LJ)** (interest-group communities)  
- **YouTube** (friendship, group subscriptions)  

### Extended datasets (not analyzed in the article, shared for completeness and reproducibility)

- **Deezer** (music genre communities)  
- **Orkut** (user-defined groups)  
- **Twitch** (streamer–viewer communities)  
- **GitHub** (repository-star communities)  

### Synthetic benchmarks

- **text** (toy illustrative example)  
- **LFR_low**, **LFR_mid**, **LFR_high** (controlled overlap benchmarks)  
- **LFR1 … LFR8** (appendix benchmarks for $h_v$ vs. $\tilde h_v$ comparison)  

> Note: LFR datasets were generated with the official LFR benchmark by Lancichinetti & Fortunato.

---

## File formats

- **Edgelist** (`*_network.txt`): space-separated pairs `u v`, undirected, no self-loops, duplicates removed, 0-based contiguous node IDs as strings.  
- **Communities** (`*_communities.txt`): each line lists node IDs in one community (TAB-separated), supports overlap.  
- **Node→communities map** (`*_node_to_communities.csv`): rows are variable-length `(node_id,comm1,comm2,…)`; empty memberships are uncommon but permitted.  

---

## Citation

If you use this repository or datasets, please cite:

Barraza, F. (2025). *Local Pluralistic Homophily* [Data set]. Zenodo.  
[https://doi.org/10.5281/zenodo.17204707](https://doi.org/10.5281/zenodo.17204707)
