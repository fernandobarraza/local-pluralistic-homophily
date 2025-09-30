#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compute structural baseline metrics and merge into a unified per-node table (auxiliary analysis).

Overview
--------
This script augments the per-node unified table with selected *structural* metrics
(e.g., BridgeCC, Collective Influence, degree, betweenness, etc.). It is intended as an
**auxiliary comparison** against local pluralistic homophily measures (tilde-h_v and ghv),
and is **not part of the core pipeline** used in the paper (which focuses on
W_inter, h̃(p), and rho_GCC).

Inputs
------
- Unified per-node metrics CSV (typically produced by `unify_metrics.py`).
- Network file (NetworKit edgelist with 0-based IDs; tab-separated).
- Communities file (one community per line; default: tab-separated).

Outputs
-------
- `<key>_all_metrics.csv` (or a custom path via `--out`) with new columns appended.

Notes
-----
- By default, only **BridgeCC** is computed (used in the toy example).
- Additional metrics can be requested via `--metrics ...`.
- Nodes in the unified table that are not present in the graph are dropped.

Examples
--------
Default paths inferred from `--data-dir`, `--dataset-key`, and `--proc3-dir`:
    python calc_all_metrics_v3.py \
        --data-dir datasets \
        --dataset-key so \
        --proc3-dir outputs/proc3 \
        --metrics bridgecc

Multiple metrics and explicit output path:
    python calc_all_metrics_v3.py \
        --data-dir datasets \
        --dataset-key so \
        --proc3-dir outputs/proc3 \
        --metrics degree betweenness ci \
        --ci-radius 2 \
        --out outputs/proc3/so_all_metrics.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

from common_functions import (
    compute_bridgecc,
    compute_collective_influence,
    compute_approximate_betweenness,
    get_overlapping_nodes,
    load_communities,
    load_graph_nk,
)

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_all_metrics_v3")

VALID_METRICS = {"degree", "betweenness", "eigenvector", "bridgecc", "overlap_neigh", "ci"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_all_metrics_v3",
        description="Compute selected structural metrics and append them to the unified per-node table.",
    )
    # Inference from dataset layout
    p.add_argument("--data-dir", type=Path, help="Base directory containing dataset folders.")
    p.add_argument("--dataset-key", type=str, help="Dataset key (e.g., so, dblp, lj, youtube, text, LFR_*).")
    p.add_argument("--proc3-dir", type=Path, default=Path("outputs/proc3"),
                   help="Directory where unified and output tables live. Default: outputs/proc3")

    # Explicit paths (alternative to dataset layout)
    p.add_argument("--unified-file", type=Path, help="Path to <key>_unified_metrics.csv.")
    p.add_argument("--network-file", type=Path, help="Path to <key>_network.txt (NetworKit edge list).")
    p.add_argument("--communities-file", type=Path, help="Path to <key>_communities.txt (one community per line).")

    # Metrics selection and options
    p.add_argument("--metrics", nargs="+", default=["bridgecc"], choices=sorted(VALID_METRICS),
                   help="Structural metrics to compute and append. Default: bridgecc")
    p.add_argument("--ci-radius", type=int, default=2, help="Radius for Collective Influence (default: 2).")
    p.add_argument("--communities-sep", default="\\t", help="Separator in the communities file (default: tab).")

    # Optional extra columns
    p.add_argument("--add-num-communities", action="store_true",
                   help="Add per-node number of communities (derived from the communities file).")
    p.add_argument("--add-delta-hv", action="store_true",
                   help="If hv_s1 and hv_s0 exist, add delta_hv and abs_delta_hv.")

    # Output and runtime
    p.add_argument("--out", type=Path, help="Output CSV path. Default: <proc3-dir>/<key>_all_metrics.csv")
    p.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def infer_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Return (unified_csv, network_file, communities_file, out_csv)."""
    # If explicit paths provided, use them
    if args.unified_file and args.network_file and args.communities_file:
        out = args.out if args.out else args.unified_file.with_name(
            args.unified_file.name.replace("_unified_metrics.csv", "_all_metrics.csv")
        )
        return args.unified_file, args.network_file, args.communities_file, out

    # Otherwise rely on dataset layout
    if not args.data_dir or not args.dataset_key:
        raise ValueError("Provide either explicit paths or --data-dir + --dataset-key.")

    key = args.dataset_key
    proc3 = args.proc3_dir
    dataset_dir = Path(args.data_dir) / key

    unified = proc3 / f"{key}_unified_metrics.csv"
    net = dataset_dir / f"{key}_network.txt"
    comms = dataset_dir / f"{key}_communities.txt"
    out = args.out if args.out else (proc3 / f"{key}_all_metrics.csv")
    return unified, net, comms, out


def read_csv_safe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Node ID" not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: "Node ID"})
    return df


def build_node_to_comms(communities: Sequence[Sequence[str]]) -> Dict[int, Set[int]]:
    """
    Build a mapping node_id -> set of community indices from a list of communities.
    Node IDs are cast to int; communities are indexed by their position in the list.
    """
    node_to_comms: Dict[int, Set[int]] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            u = int(node)
            node_to_comms.setdefault(u, set()).add(cid)
    return node_to_comms


def append_metric_column(df: pd.DataFrame, col: str, mapping: Dict[int, float]) -> None:
    df[col] = df["Node ID"].map(mapping)
    n_nan = int(df[col].isna().sum())
    if n_nan > 0:
        log.warning("Column '%s' has %d NaN values after mapping.", col, n_nan)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    try:
        unified_path, network_path, communities_path, out_path = infer_paths(args)
        log.info("Unified metrics: %s", unified_path)
        log.info("Network: %s", network_path)
        log.info("Communities: %s", communities_path)
        log.info("Output: %s", out_path)

        if out_path.exists() and not args.force:
            log.error("Output exists: %s (use --force to overwrite).", out_path)
            return 1

        # Load base table
        df_all = read_csv_safe(unified_path)
        df_all["Node ID"] = df_all["Node ID"].astype(int)

        # Load graph and filter rows to valid nodes
        Gk = load_graph_nk(network_path, directed=False)
        valid_nodes = set(Gk.iterNodes())
        df_all = df_all[df_all["Node ID"].isin(valid_nodes)].copy()
        log.info("Nodes after filtering by graph: %d", len(df_all))

        # Optionally load communities if needed
        need_communities = args.add_num_communities or ("overlap_neigh" in args.metrics)
        node_to_comms: Optional[Dict[int, Set[int]]] = None
        if need_communities:
            communities = load_communities(communities_path, separator=args.communities_sep)
            node_to_comms = build_node_to_comms(communities)
            log.info("Communities loaded: %d sets | nodes with memberships: %d",
                     len(communities), len(node_to_comms))

        # Compute selected metrics
        metrics = set(args.metrics)

        if "degree" in metrics and "degree" not in df_all.columns:
            log.info("Computing degree …")
            deg_map = {u: Gk.degree(u) for u in Gk.iterNodes()}
            append_metric_column(df_all, "degree", deg_map)

        if "betweenness" in metrics and "betweenness" not in df_all.columns:
            log.info("Computing approximate betweenness …")
            abt = compute_approximate_betweenness(Gk)
            append_metric_column(df_all, "betweenness", abt)

        if "eigenvector" in metrics and "eigenvector" not in df_all.columns:
            log.info("Computing eigenvector centrality …")
            import networkit as nk  # local import to avoid import cost if unused
            eig = nk.centrality.EigenvectorCentrality(Gk)
            eig.run()
            eig_map = {u: eig.score(u) for u in Gk.iterNodes()}
            append_metric_column(df_all, "eigenvector", eig_map)

        if "bridgecc" in metrics and "bridgecc" not in df_all.columns:
            log.info("Computing BridgeCC …")
            bridgecc_map = compute_bridgecc(Gk)
            append_metric_column(df_all, "bridgecc", bridgecc_map)

        if "overlap_neigh" in metrics and "overlap_neigh" not in df_all.columns:
            if node_to_comms is None:
                raise RuntimeError("Communities required to compute 'overlap_neigh' but not available.")
            log.info("Computing overlap_neigh …")
            overlap_nodes = get_overlapping_nodes(node_to_comms)
            ordered = overlap_neigh_rank(Gk, overlap_nodes)
            max_rank = len(ordered)
            rank_map = {node: max_rank - idx for idx, node in enumerate(ordered)}
            append_metric_column(df_all, "overlap_neigh", rank_map)

        if "ci" in metrics and "ci" not in df_all.columns:
            log.info("Computing Collective Influence (l=%d) …", args.ci_radius)
            ci_map = compute_collective_influence(Gk, args.ci_radius)
            append_metric_column(df_all, "ci", ci_map)

        # Optional extras
        if args.add_num_communities and node_to_comms is not None and "num_communities" not in df_all.columns:
            log.info("Adding num_communities …")
            num_map = {u: len(members) for u, members in node_to_comms.items()}
            append_metric_column(df_all, "num_communities", num_map)

        if args.add_delta_hv and "hv_s1" in df_all.columns and "hv_s0" in df_all.columns:
            if "delta_hv" not in df_all.columns:
                df_all["delta_hv"] = df_all["hv_s1"] - df_all["hv_s0"]
            if "abs_delta_hv" not in df_all.columns:
                df_all["abs_delta_hv"] = df_all["delta_hv"].abs()

        # Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_path, index=False)
        log.info("Saved: %s (%d nodes, %d columns)", out_path, len(df_all), len(df_all.columns))
        return 0

    except Exception as e:
        log.exception("Fatal error: %s", e)
        return 2


def overlap_neigh_rank(Gk, overlapping_nodes: Iterable[int]) -> List[int]:
    """
    Compute the overlap-neighborhood ranking directly (NetworKit variant).
    Returns nodes sorted by degree in the 1-hop neighborhood of overlapping nodes.
    """
    khoplist: Set[int] = set()
    for u in overlapping_nodes:
        for v in Gk.iterNeighbors(u):
            khoplist.add(v)
    deg_map = {v: Gk.degree(v) for v in khoplist}
    return sorted(deg_map, key=lambda x: deg_map[x], reverse=True)


if __name__ == "__main__":
    raise SystemExit(main())
