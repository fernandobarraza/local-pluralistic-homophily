#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compute edge-level pluralistic homophily \u007E h_e for a graph.

Overview
--------
Given a graph and a node scalar attribute (assigned via `assign_scalar_values`),
this script computes per-edge dissimilarities and the normalized global score
using `calculate_h_e_normalized` from common_functions.

Outputs a per-edge table containing:
- endpoints (Node 1 ID, Node 2 ID),
- h_e value (signed), and
- optional copy of the node scalar values at each endpoint (for auditability).

Inputs
------
Either:
  --data-dir <dir> --dataset-key <key>
or:
  --network-file, --communities-file, --node2comms-file

Outputs
-------
- <out-dir>/<key>/<key>_he_metrics.csv

Options
-------
- --method s1        : node scalar assignment method (default s1)
- --include-nodes    : also write scalar values for both endpoints (default on)
- --log-level        : INFO/DEBUG/...

Examples
--------
Standard layout:
    python calc_edge_homophily.py --data-dir datasets --dataset-key toy6

Explicit paths:
    python calc_edge_homophily.py --network-file datasets/toy6/toy6_network.txt \
        --communities-file datasets/toy6/toy6_communities.txt \
        --node2comms-file datasets/toy6/toy6_node_to_communities.csv \
        --out-dir outputs/supplemental
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, List

import pandas as pd
from igraph import Graph

from common_functions import (
    assign_scalar_values,
    calculate_h_e_normalized,
    load_communities,
    load_node_to_communities_map,
)

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_edge_homophily")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_edge_homophily",
        description="Compute per-edge pluralistic homophily h_e and write a table.",
    )
    # Standard layout
    p.add_argument("--data-dir", type=Path, help="Base datasets directory.")
    p.add_argument("--dataset-key", type=str, help="Dataset key.")
    # Explicit inputs
    p.add_argument("--network-file", type=Path, help="NCOL undirected edge list.")
    p.add_argument("--communities-file", type=Path, help="Communities file.")
    p.add_argument("--node2comms-file", type=Path, help="Node→communities CSV mapping.")
    # Options
    p.add_argument("--method", type=str, default="s1", help="Scalar assignment method for nodes. Default: s1")
    p.add_argument("--include-nodes", action=argparse.BooleanOptionalAction, default=True,
                   help="Also include per-endpoint scalar values in the output (default: on).")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/supplemental"),
                   help="Base output directory. Default: outputs/supplemental")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def infer_paths(args: argparse.Namespace) -> Tuple[str, Path, Path, Path, Path]:
    if args.network_file and args.communities_file and args.node2comms_file:
        key = args.dataset_key or args.network_file.stem.replace("_network", "")
        out_dir = args.out_dir / key
        return key, args.network_file, args.communities_file, args.node2comms_file, out_dir

    if not args.data_dir or not args.dataset_key:
        raise ValueError("Provide either explicit paths OR --data-dir + --dataset-key.")

    key = args.dataset_key
    base = Path(args.data_dir) / key
    net = base / f"{key}_network.txt"
    comms = base / f"{key}_communities.txt"
    n2c = base / f"{key}_node_to_communities.csv"
    out_dir = args.out_dir / key
    return key, net, comms, n2c, out_dir


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    try:
        key, net_path, comms_path, n2c_path, out_dir = infer_paths(args)
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("[%s] Loading graph and communities", key)
        g = Graph.Read_Ncol(str(net_path), directed=False)
        communities = load_communities(comms_path)
        node2comms = load_node_to_communities_map(n2c_path)

        # Assign node scalars
        assign_scalar_values(g, args.method, communities, node2comms)
        scalar_values = g.vs["scalar_value"]

        # Compute per-edge values
        lam, deltas_e, h_global_e = calculate_h_e_normalized(g)
        log.info("[%s] global h_e = %.6f", key, h_global_e)

        # Build edge table
        rows: List[dict] = []
        for eid, e in enumerate(g.es):
            u = g.vs[e.source]["name"]
            v = g.vs[e.target]["name"]
            row = {"Edge ID": eid, "Node 1 ID": u, "Node 2 ID": v, "he_s1": deltas_e[eid]}
            if args.include_nodes:
                row[f"{args.method}_node1"] = scalar_values[e.source]
                row[f"{args.method}_node2"] = scalar_values[e.target]
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_out = out_dir / f"{key}_he_metrics.csv"
        df.to_csv(csv_out, index=False)
        log.info("[%s] wrote %s", key, csv_out)
        return 0

    except Exception as e:
        log.exception("Fatal error: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
