#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compute community-level pluralistic homophily h̃_C for overlapping communities.

Overview
--------
For a given graph G(V,E) and an overlapping community cover C = {C_1, …, C_K},
this script computes a per-community score h̃_C(k) measuring within-community
alignment of the node-level scalar attribute (e.g., s1). The node attribute is
assigned via `assign_scalar_values(graph, method, communities, node2comms)`.

Two outputs are produced:
1) Per-community table with h̃_C, community size, and optional global h (assortativity).
2) An optional dataset-level summary (weighted average by |C_k|).

This script implements the supplemental equations reported in the SI.

Inputs
------
Either:
  --data-dir <dir> --dataset-key <key>
or:
  --network-file, --communities-file, --node2comms-file

Outputs
-------
- <out-dir>/<key>/<key>_hc_metrics.csv  (per-community)
- (optional) <out-dir>/<key>/<key>_hc_summary.csv  (single-row summary)

Notes
-----
- The function `calculate_h_c` (from common_functions) is used for the score.
- If communities without internal edges exist, we output NaN for their h̃_C.

Examples
--------
Standard layout:
    python calc_community_homophily.py --data-dir datasets --dataset-key toy6

Explicit paths:
    python calc_community_homophily.py --network-file datasets/toy6/toy6_network.txt \
        --communities-file datasets/toy6/toy6_communities.txt \
        --node2comms-file datasets/toy6/toy6_node_to_communities.csv \
        --out-dir outputs/supplemental
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from igraph import Graph

from common_functions import (
    assign_scalar_values,
    load_communities,
    load_node_to_communities_map,
    calculate_h_c,
)

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_community_homophily")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_community_homophily",
        description="Compute per-community pluralistic homophily h̃_C.",
    )
    # Standard layout
    p.add_argument("--data-dir", type=Path, help="Base datasets directory.")
    p.add_argument("--dataset-key", type=str, help="Dataset key (e.g., toy6, so, dblp).")
    # Explicit inputs
    p.add_argument("--network-file", type=Path, help="NCOL undirected edge list.")
    p.add_argument("--communities-file", type=Path, help="Communities file (one community per line).")
    p.add_argument("--node2comms-file", type=Path, help="Node→communities CSV mapping.")
    # Options
    p.add_argument("--method", type=str, default="s1", help="Scalar assignment method for nodes. Default: s1")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/supplemental"),
                   help="Base output directory. Default: outputs/supplemental")
    p.add_argument("--write-summary", action=argparse.BooleanOptionalAction, default=True,
                   help="Write also a 1-row summary with weighted average. Default: on")
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

        # Assign node scalars (e.g., s1)
        assign_scalar_values(g, args.method, communities, node2comms)
        scalars = g.vs["scalar_value"]
        lambda_val = max(scalars) if len(scalars) else 0.0

        # Compute h_c per community using provided function
        hc_dict, h_global = calculate_h_c(g, communities, scalars, lambda_val)

        # Build output table
        df = pd.DataFrame.from_dict(hc_dict, orient="index", columns=[f"hc_{args.method}"])
        df.index.name = "community_id"
        df["num_nodes"] = [len(c) for c in communities]
        df["h_global"] = h_global

        csv_out = out_dir / f"{key}_hc_metrics.csv"
        df.to_csv(csv_out, index=False)
        log.info("[%s] per-community metrics → %s", key, csv_out)

        if args.write_summary:
            weights = np.array(df["num_nodes"], dtype=float)
            vals = pd.to_numeric(df[f"hc_{args.method}"], errors="coerce").to_numpy()
            wavg = float(np.nansum(vals * weights) / np.nansum(weights)) if weights.sum() > 0 else np.nan
            df_sum = pd.DataFrame([{
                "dataset": key,
                "hc_weighted": wavg,
                "h_global": float(h_global) if h_global is not None else np.nan,
                "n_communities": int(len(communities)),
            }])
            sum_out = out_dir / f"{key}_hc_summary.csv"
            df_sum.to_csv(sum_out, index=False)
            log.info("[%s] summary → %s", key, sum_out)

        return 0

    except Exception as e:
        log.exception("Fatal error: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
