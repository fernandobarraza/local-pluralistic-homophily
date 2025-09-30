#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compute dataset-level descriptive metrics (table-friendly) for multiple datasets.

Overview
--------
For each dataset key <key>, loads the network and communities, restricts to the
giant component, and reports standard descriptors plus the network-level
pluralistic homophily h_s1 (i.e., the assortativity of the per-node s1 attribute).

Metrics reported
----------------
- N (nodes), E (edges), <d> (mean degree), density
- |C| (number of communities after intersect with nodes in GCC)
- <m> (mean #memberships per node among nodes with m>=1), max_m
- Coverage and overlap: prop_nodes_in_comms, prop_overlapped (conditional), prop_overlapped_all
- r (degree assortativity), avg_clustering
- h_s1 (network-level pluralistic homophily via assortativity over s1)

Inputs
------
Either use --datasets with standard layout:
  datasets/<key>/<key>_network.txt
  datasets/<key>/<key>_communities.txt
Or pass explicit files with --network-file/--communities-file for a single dataset.

Outputs
-------
- Per-dataset CSVs under <out-dir>/<key>/
- A concatenated summary CSV: <out-dir>/summary_descriptives.csv

Examples
--------
Standard layout with several datasets:
  python calc_dataset_descriptives.py --data-dir datasets --out-dir outputs/descriptives \\
      --datasets dblp lj youtube so LFR_low LFR_mid LFR_high

Single dataset with explicit files:
  python calc_dataset_descriptives.py --network-file datasets/so/so_network.txt \\
      --communities-file datasets/so/so_communities.txt --out-dir outputs/descriptives
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from igraph import Graph

from common_functions import assign_scalar_values, load_communities

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_dataset_descriptives")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_dataset_descriptives",
        description="Compute descriptive metrics for datasets (for table reporting).",
    )
    # Standard layout
    p.add_argument("--datasets", nargs="+", help="Dataset keys to process.")
    p.add_argument("--data-dir", type=Path, help="Base path for datasets/<key>/ folders.")
    # Single explicit
    p.add_argument("--network-file", type=Path, help="Explicit NCOL undirected edge list.")
    p.add_argument("--communities-file", type=Path, help="Explicit communities file (one community per line).")
    # I/O and options
    p.add_argument("--communities-sep", default="\\t", help="Separator in communities file (default: tab).")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/descriptives"),
                   help="Output base directory. Default: outputs/descriptives")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def load_layout(data_dir: Path, key: str, sep: str) -> Tuple[Path, List[List[str]]]:
    net = data_dir / key / f"{key}_network.txt"
    comms_path = data_dir / key / f"{key}_communities.txt"
    comms = load_communities(comms_path, separator=sep)
    return net, comms


def compute_homophily_s1(graph: Graph, communities: List[List[str]]) -> float | None:
    assign_scalar_values(graph, "s1", communities)
    # try 'scalar_value' first
    try:
        return float(graph.assortativity(types1="scalar_value", directed=False))
    except Exception:
        pass
    try:
        return float(graph.assortativity(types1="s1", directed=False))
    except Exception:
        return None


def compute_descriptives_single(net_path: Path, communities: List[List[str]]) -> Dict[str, object]:
    g = Graph.Read_Ncol(str(net_path), directed=False)
    if not g.is_connected():
        g = g.components().giant()

    N = g.vcount()
    E = g.ecount()
    degs = g.degree()
    mean_d = float(np.mean(degs)) if N else 0.0
    density = float(E / (N * (N - 1) / 2)) if N > 1 else 0.0

    present = set(g.vs["name"])
    comms = [[n for n in c if n in present] for c in communities]
    comms = [c for c in comms if len(c) > 0]

    memberships: Dict[str, set] = {}
    for cid, c in enumerate(comms):
        for u in c:
            memberships.setdefault(u, set()).add(cid)
    m_vals = [len(s) for s in memberships.values()]
    m_mean = float(np.mean(m_vals)) if m_vals else 0.0
    m_max = int(max(m_vals)) if m_vals else 0

    n_in_comms = len(m_vals)
    overlapped = sum(1 for m in m_vals if m > 1)
    prop_nodes_in = float(n_in_comms / N) if N else 0.0
    prop_over_cond = float(overlapped / n_in_comms) if n_in_comms else 0.0
    prop_over_all = float(overlapped / N) if N else 0.0

    r_deg = float(g.assortativity_degree(directed=False))
    clustering = g.transitivity_local_undirected(vertices=None)
    avg_clust = float(np.nanmean(clustering)) if len(clustering) else 0.0

    h_s1 = compute_homophily_s1(g, comms)
    return {
        "N": int(N), "E": int(E), "<d>": round(mean_d, 2), "density": round(density, 5),
        "|C|": int(len(comms)), "<m>": round(m_mean, 2), "max_m": int(m_max),
        "prop_nodes_in_comms": round(prop_nodes_in, 4),
        "prop_overlapped": round(prop_over_cond, 4),
        "prop_overlapped_all": round(prop_over_all, 4),
        "r": round(r_deg, 4), "avg_clustering": round(avg_clust, 4),
        "h_s1": (round(h_s1, 4) if h_s1 is not None else None),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    out_summary = []
    out_base = args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)

    try:
        if args.network_file and args.communities_file:
            key = args.network_file.stem.replace("_network", "")
            comms = load_communities(args.communities_file, separator=args.communities_sep)
            metrics = compute_descriptives_single(args.network_file, comms)
            df = pd.DataFrame([{**metrics, "dataset": key}])
            (out_base / f"{key}_descriptives.csv").write_text(df.to_csv(index=False), encoding="utf-8")
            out_summary.append(df)
        else:
            if not args.datasets or not args.data_dir:
                raise ValueError("Provide --datasets and --data-dir, or explicit --network-file/--communities-file.")
            for key in args.datasets:
                net, comms = load_layout(args.data_dir, key, args.communities_sep)
                metrics = compute_descriptives_single(net, comms)
                df = pd.DataFrame([{**metrics, "dataset": key}])
                out_dir = out_base / key
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{key}_descriptives.csv").write_text(df.to_csv(index=False), encoding="utf-8")
                out_summary.append(df)

        if out_summary:
            summary = pd.concat(out_summary, ignore_index=True)
            (out_base / "summary_descriptives.csv").write_text(summary.to_csv(index=False), encoding="utf-8")
            log.info("Summary written: %s", out_base / "summary_descriptives.csv")

        return 0
    except Exception as e:
        log.exception("Fatal error: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
