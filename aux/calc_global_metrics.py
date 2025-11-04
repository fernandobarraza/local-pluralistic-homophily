#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
Compute descriptive network & community metrics per dataset (table-ready).

Overview
--------
For each dataset key, this script reads:
  <data-dir>/<key>/<key>_network.txt           (NCOL, undirected)
  <data-dir>/<key>/<key>_communities.txt       (one community per line)

It optionally extracts the Giant Connected Component (GCC) for consistency across
metrics, computes coverage/overlap indicators, degree assortativity r, average
local clustering, and the pluralistic homophily at network level (h_s1).

Outputs
-------
A single CSV with one row per dataset:
  Dataset, N, E, <d>, density, |C|, <m>, max_m, avg_comm_size, max_comm_size,
  prop_nodes_in_comms, prop_overlapped, prop_overlapped_all, r, avg_clustering, h_s1

Examples
--------
python aux/calc_global_metrics.py \
  --datasets dblp lj youtube so deezer github amazon twitch \
  --data-dir datasets \
  --out-file outputs/proc2/ext_networks_metrics_addons.csv

python aux/calc_global_metrics.py \
  --datasets amazon twitch \
  --data-dir datasets \
  --no-gcc \
  --h-methods s1 \
  --out-file outputs/proc2/metrics_am_tw.csv \
  --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from igraph import Graph

# Import from repo (core/common_functions.py should be importable from CWD)
from common_functions import assign_scalar_values, load_communities

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_global_metrics")


# ----------------------------- helpers ----------------------------- #
def need(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def _compute_h(graph: Graph, method: str, communities: List[List[str]]) -> Optional[float]:
    """
    Apply assign_scalar_values and compute assortativity for the appropriate attribute.
    Tries 'scalar_value' first (convention), falls back to 'method' name (e.g., 's1').
    """
    assign_scalar_values(graph, method, communities)

    # try 'scalar_value'
    try:
        h = float(graph.assortativity(types1="scalar_value", directed=False))
        return round(h, 4)
    except Exception:
        pass

    # fallback: method name
    try:
        h = float(graph.assortativity(types1=method, directed=False))
        return round(h, 4)
    except Exception:
        return None


def _plurality_stats(communities: List[List[str]], present_names: Iterable[str]):
    """
    Build per-node plurality m_v (number of communities per node) restricted to nodes present.
    Returns: (avg_plurality, max_plurality, n_in_comms, n_overlapped, m_list)
    """
    present = set(present_names)
    # intersect with present nodes and drop empties
    comms = [[n for n in c if n in present] for c in communities]
    comms = [c for c in comms if c]

    memberships: Dict[str, int] = {}
    for idx, c in enumerate(comms):
        for n in c:
            memberships[n] = memberships.get(n, 0) + 1

    m_list = list(memberships.values())
    n_in = len(m_list)
    n_ov = sum(1 for m in m_list if m > 1)

    avg_m = round(float(np.mean(m_list)), 2) if m_list else 0.0
    max_m = int(max(m_list)) if m_list else 0
    return avg_m, max_m, n_in, n_ov, comms, m_list


def _avg_clustering(graph: Graph) -> float:
    vals = graph.transitivity_local_undirected(vertices=None, mode="nan")
    # igraph returns np.nan for deg<2 with mode="nan"; ignore them
    return round(float(np.nanmean(vals)), 4) if len(vals) else 0.0


def _basic_density(graph: Graph) -> float:
    n = graph.vcount()
    e = graph.ecount()
    return round(e / (n * (n - 1) / 2.0), 5) if n > 1 else 0.0


# ----------------------------- per-dataset ----------------------------- #
def process_dataset(
    key: str,
    data_dir: Path,
    use_gcc: bool,
    h_methods: List[str],
) -> Optional[dict]:
    net_path = data_dir / key / f"{key}_network.txt"
    com_path = data_dir / key / f"{key}_communities.txt"

    need(net_path, f"network for {key}")
    need(com_path, f"communities for {key}")

    log.info("Dataset=%s | reading network: %s", key, net_path)
    g = Graph.Read_Ncol(str(net_path), directed=False)

    if use_gcc and not g.is_connected():
        log.debug("[%s] extracting Giant Connected Component", key)
        g = g.components().giant()

    N = g.vcount()
    E = g.ecount()
    deg = g.degree()
    avg_d = round(float(np.mean(deg)), 2) if N else 0.0
    dens = _basic_density(g)

    log.debug("[%s] reading communities: %s", key, com_path)
    comms_raw = load_communities(str(com_path))
    avg_m, max_m, n_in, n_ov, comms, _m_list = _plurality_stats(comms_raw, g.vs["name"])

    prop_in = round(n_in / N, 4) if N else 0.0
    prop_ov_all = round(n_ov / N, 4) if N else 0.0
    prop_ov_cond = round(n_ov / n_in, 4) if n_in else 0.0

    avg_c = _avg_clustering(g)
    r_deg = round(float(g.assortativity_degree(directed=False)), 4) if N else 0.0

    metrics = {
        "Dataset": key,
        "N": N,
        "E": E,
        "<d>": avg_d,
        "density": dens,
        "|C|": len(comms),
        "<m>": avg_m,        # mean plurality (over nodes with >=1 community)
        "max_m": max_m,
        "avg_comm_size": round(float(np.mean([len(c) for c in comms])), 2) if comms else 0.0,
        "max_comm_size": int(max((len(c) for c in comms), default=0)),
        "prop_nodes_in_comms": prop_in,
        "prop_overlapped": prop_ov_cond,     # conditional on being in >=1 community
        "prop_overlapped_all": prop_ov_all,  # over all nodes
        "r": r_deg,
        "avg_clustering": avg_c,
    }

    # Network-level pluralistic homophily (h_method → h_<method>)
    for m in h_methods:
        try:
            h_val = _compute_h(g, m, comms)
        except Exception as e:
            log.exception("[%s] error computing h_%s: %s", key, m, e)
            h_val = None
        metrics[f"h_{m}"] = h_val

    return metrics


# ----------------------------- CLI ----------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_global_metrics",
        description="Compute descriptive metrics for datasets (network + communities).",
    )
    p.add_argument("--datasets", nargs="+", required=True, help="Dataset keys to process.")
    p.add_argument("--data-dir", type=Path, required=True, help="Base datasets directory.")
    p.add_argument(
        "--out-file",
        type=Path,
        default=Path("outputs/proc2/ext_networks_metrics_addons.csv"),
        help="Output CSV path.",
    )
    p.add_argument(
        "--h-methods",
        nargs="+",
        default=["s1"],
        help="Scalar methods to compute network-level homophily (default: s1).",
    )
    p.add_argument(
        "--no-gcc",
        action="store_true",
        help="Do NOT extract GCC (by default, GCC is used if the graph is disconnected).",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for key in args.datasets:
        row = process_dataset(
            key=key,
            data_dir=args.data_dir,
            use_gcc=(not args.no_gcc),
            h_methods=list(args.h_methods),
        )
        if row is not None:
            rows.append(row)

    if not rows:
        log.error("No metrics were produced. Check inputs.")
        return 2

    df = pd.DataFrame(rows)
    df.to_csv(args.out_file, index=False)
    log.info("Written: %s", args.out_file)

    # Pretty print to console
    with pd.option_context(
        "display.max_colwidth",
        None,
        "display.precision",
        6,
        "display.float_format",
        lambda x: f"{x:.6f}",
    ):
        print("\n== Global metrics ==")
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
