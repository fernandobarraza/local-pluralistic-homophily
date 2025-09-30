#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compute per-node ghv (global-mean local homophily proxy) for one or more scalar methods.

Overview
--------
This script computes, for each node, the classical ghv vector (as in the previous paper),
based on the global-mean normalization of the scalar attribute. It also stores the
underlying scalar values used for each method. Results are written to a per-dataset CSV
with the suffix "_ghv_metrics.csv".

Notes
-----
- The computation uses `calculate_ghv` from `common_functions`, which ensures that
  the degree-weighted mean of ghv equals the global assortativity returned by igraph.
- Inputs can be provided explicitly, or inferred from `--data-dir` + `--dataset-key`.

Inputs
------
- Edge list in NCOL format (undirected).
- Communities file (one community per line; tab-separated by default).

Outputs
-------
- CSV with columns:
  - "Node ID": node identifier (string),
  - For each scalar method m in --methods: "ghv_m" and the scalar column "m"
    containing the node-level scalar used to compute ghv.

Examples
--------
Explicit file paths:
    python calc_ghv.py \
        --edges data/so/so_network.txt \
        --communities data/so/so_communities.txt \
        --out outputs/proc1/so_ghv_metrics.csv \
        --methods s2

Dataset layout (data-dir + dataset-key):
    python calc_ghv.py \
        --data-dir /path/to/datasets \
        --dataset-key so \
        --results-dir outputs/proc1 \
        --methods s2

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from igraph import Graph

from common_functions import (
    assign_scalar_values,
    calculate_ghv,
    load_communities,
    DATASETS,
)

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_ghv")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_ghv",
        description="Compute per-node ghv vector (previous-paper definition) and scalar values per node.",
    )

    # Direct file inputs (optional if using dataset layout)
    p.add_argument("--edges", type=Path, help="Path to edge list file (NCOL).")
    p.add_argument("--communities", type=Path, help="Path to communities file (one line per community).")
    p.add_argument("--out", type=Path, help="Output CSV path. If omitted, inferred from results-dir + dataset-key.")

    # Dataset layout inputs (alternative)
    p.add_argument("--data-dir", type=Path, default=None, help="Base directory containing dataset folders.")
    p.add_argument("--dataset-key", type=str, default=None,
                   help=f"Dataset key in {{{', '.join(DATASETS.keys())}}}.")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Directory where the output CSV will be written (name inferred from dataset-key).")

    p.add_argument("--methods", nargs="+", default=["s2"],
                   choices=["s0", "s1", "s2", "s3", "s4"],
                   help="Scalar methods to compute. Default: ['s2'].")
    p.add_argument("--communities-sep", default="\\t",
                   help="Separator used within each communities line. Default: tab.")

    p.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def infer_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    # If explicit paths are given, use them
    if args.edges and args.communities:
        if args.out is None:
            raise ValueError("--out must be provided when using explicit file paths.")
        return args.edges, args.communities, args.out

    # Otherwise infer from data-dir + dataset-key
    if not args.data_dir or not args.dataset_key:
        raise ValueError("Either provide explicit file paths or use --data-dir + --dataset-key.")

    key = args.dataset_key
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset key '{key}'. Valid: {', '.join(DATASETS.keys())}")

    dataset_dir = Path(args.data_dir) / key
    edges = dataset_dir / f"{key}_network.txt"
    comms = dataset_dir / f"{key}_communities.txt"

    if args.results_dir:
        out = Path(args.results_dir) / f"{key}_ghv_metrics.csv"
    else:
        # Default alongside the dataset directory
        out = dataset_dir / f"{key}_ghv_metrics.csv"

    return edges, comms, out


def load_or_init_df(out_path: Path, node_ids: list[str]) -> pd.DataFrame:
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
        edges_path, comms_path, out_path = infer_paths(args)
        log.info("Edges: %s", edges_path)
        log.info("Communities: %s", comms_path)
        log.info("Output: %s", out_path)

        if out_path.exists() and not args.force:
            log.error("Output exists: %s (use --force to overwrite or merge manually).", out_path)
            return 1

        # Load inputs
        graph = Graph.Read_Ncol(str(edges_path), directed=False)
        if "name" not in graph.vs.attributes():
            graph.vs["name"] = [str(v.index) for v in graph.vs]
        log.info("Graph: %d nodes, %d edges", graph.vcount(), graph.ecount())

        communities = load_communities(comms_path, separator=args.communities_sep)
        log.info("Communities: %d sets loaded", len(communities))

        # DataFrame init
        node_ids = [str(x) for x in graph.vs["name"]]
        df = load_or_init_df(out_path, node_ids)

        # Compute ghv for each scalar method
        for method in args.methods:
            log.info("Assigning scalar values for method=%s", method)
            assign_scalar_values(
                graph=graph,
                method=method,
                communities=communities,
                node_to_comms_dict=None,
                out_attr="scalar_value",
            )
            scalar_values = graph.vs["scalar_value"]

            # ghv vector based on global mean normalization
            log.info("Computing ghv for method=%s", method)
            ghv_vec, h_global = calculate_ghv(graph, scalar_name="scalar_value")
            log.debug("h_global (igraph assortativity) = %.6f", h_global)

            # Degree-weighted check: equals h_global
            deg = np.array(graph.degree())
            M = graph.ecount()
            h_check = float(np.sum(deg * ghv_vec) / (2 * M))
            log.info("Degree-weighted mean(ghv) = %.6f | h_global = %.6f", h_check, h_global)

            # Persist into DataFrame
            df[f"ghv_{method}"] = ghv_vec
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
