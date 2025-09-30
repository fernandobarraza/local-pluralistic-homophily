#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compute node-level local pluralistic homophily proxies (legacy hv) for given scalar methods.

Overview
--------
This script computes, for each node, the legacy local proxy `hv_method` (based on the
deprecated "normalized" variant used in earlier experiments) together with the underlying
scalar attribute used by the method. It supports multiple scalar methods in one run.
Results are appended/merged into a per-dataset CSV.

Notes
-----
- This script preserves the historical behavior relied upon by downstream steps by using
  `calculate_h_v_normalized` from `common_functions`. For new analyses, prefer computing
  global assortativity directly via `graph.assortativity(types1="s1", directed=False)`
  and/or using `calculate_ghv` if a per-node vector is needed.
- Input files can be passed explicitly or inferred from `--data-dir` + `--dataset-key`.

Inputs
------
- Edge list in ncol format (undirected).
- Communities file (one community per line; tab-separated by default).
- Node-to-communities CSV (node_id, comm1, comm2, ...).

Outputs
-------
- CSV with columns:
  - "Node ID" (string),
  - for each method m in --methods: "hv_m" and "m" (scalar_value for method m).

Examples
--------
Explicit file paths:
    python calc_hv.py \
        --edges data/so/so_network.txt \
        --communities data/so/so_communities.txt \
        --node2comms data/so/so_node_to_communities.csv \
        --out outputs/proc1/so_hv_metrics.csv \
        --methods s1 s0

Dataset layout (data-dir + dataset-key):
    python calc_hv.py \
        --data-dir /path/to/datasets \
        --dataset-key so \
        --results-dir outputs/proc1

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from igraph import Graph

from common_functions import (
    assign_scalar_values,
    calculate_h_v_normalized,  # kept for backward-compat experiments
    load_communities,
    load_node_to_communities_map,
    DATASETS,
)

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_hv")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_hv",
        description="Compute legacy local homophily proxies hv_{method} and scalar values per node.",
    )

    # Direct file inputs (optional if using dataset layout)
    p.add_argument("--edges", type=Path, help="Path to edge list file (ncol).")
    p.add_argument("--communities", type=Path, help="Path to communities file (one line per community).")
    p.add_argument("--node2comms", type=Path, help="Path to node->communities CSV (node_id,comm1,comm2,...).")
    p.add_argument("--out", type=Path, help="Output CSV path. If omitted, inferred from results-dir + dataset-key.")

    # Dataset layout inputs (alternative)
    p.add_argument("--data-dir", type=Path, default=None, help="Base directory containing dataset folders.")
    p.add_argument("--dataset-key", type=str, default=None,
                   help=f"Dataset key in {{{', '.join(DATASETS.keys())}}}.")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Directory where the output CSV will be written (name inferred from dataset-key).")

    p.add_argument("--methods", nargs="+", default=["s1", "s0"],
                   choices=["s0", "s1", "s2", "s3", "s4"],
                   help="Scalar methods to compute. Defaults to ['s1', 's0'].")
    p.add_argument("--communities-sep", default="\\t",
                   help="Separator used within each communities line. Default: tab.")
    p.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def infer_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    # If explicit paths are given, use them
    if args.edges and args.communities and args.node2comms:
        if args.out is None:
            raise ValueError("--out must be provided when using explicit file paths.")
        return args.edges, args.communities, args.node2comms, args.out

    # Otherwise infer from data-dir + dataset-key
    if not args.data_dir or not args.dataset_key:
        raise ValueError("Either provide explicit file paths or use --data-dir + --dataset-key.")

    key = args.dataset_key
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset key '{key}'. Valid: {', '.join(DATASETS.keys())}")

    dataset_dir = Path(args.data_dir) / key
    edges = dataset_dir / f"{key}_network.txt"
    comms = dataset_dir / f"{key}_communities.txt"
    node2 = dataset_dir / f"{key}_node_to_communities.csv"

    if args.results_dir:
        out = Path(args.results_dir) / f"{key}_hv_metrics.csv"
    else:
        # Default alongside the dataset directory
        out = dataset_dir / f"{key}_hv_metrics.csv"

    return edges, comms, node2, out


def load_or_init_df(out_path: Path, node_ids: List[str]) -> pd.DataFrame:
    if out_path.exists():
        df = pd.read_csv(out_path)
        if "Node ID" in df.columns:
            df = df.set_index("Node ID")
        else:
            # If index is already node ids
            df = df.set_index(df.columns[0])
        return df
    else:
        return pd.DataFrame(index=node_ids)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    try:
        edges_path, comms_path, node2_path, out_path = infer_paths(args)
        log.info("Edges: %s", edges_path)
        log.info("Communities: %s", comms_path)
        log.info("Node->Communities: %s", node2_path)
        log.info("Output: %s", out_path)

        if out_path.exists() and not args.force:
            log.info("Output exists and --force not set. New results will be merged into existing file.")

        # Load inputs
        graph = Graph.Read_Ncol(str(edges_path), directed=False)
        if "name" not in graph.vs.attributes():
            graph.vs["name"] = [str(v.index) for v in graph.vs]
        log.info("Graph: %d nodes, %d edges", graph.vcount(), graph.ecount())

        communities = load_communities(comms_path, separator=args.communities_sep)
        log.info("Communities: %d sets loaded", len(communities))

        node_to_comms = load_node_to_communities_map(node2_path)

        # DataFrame init (merge behavior)
        node_ids = [str(x) for x in graph.vs["name"]]
        df = load_or_init_df(out_path, node_ids)

        # Compute for each scalar method
        for method in args.methods:
            log.info("Assigning scalar values for method=%s", method)
            assign_scalar_values(
                graph=graph,
                method=method,
                communities=communities,
                node_to_comms_dict=node_to_comms,
                out_attr="scalar_value",
            )
            scalar_values = graph.vs["scalar_value"]

            # Legacy hv (normalized variant)
            log.info("Computing legacy hv for method=%s", method)
            lambda_value, deltas_v, h_global = calculate_h_v_normalized(graph, scalar_name="scalar_value")
            log.debug("lambda_value=%.6f | h_global=%.6f", lambda_value, h_global)

            # Persist into DataFrame
            df[f"hv_{method}"] = deltas_v
            df[f"{method}"] = scalar_values

        # Persist to CSV
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if df.index.name == "Node ID":
            df.to_csv(out_path)
        else:
            df_out = df.reset_index().rename(columns={"index": "Node ID"})
            df_out.to_csv(out_path, index=False)

        log.info("Results written to: %s", out_path)
        return 0

    except Exception as e:
        log.exception("Fatal error: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
