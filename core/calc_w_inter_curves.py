#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compute W_inter(p) curves on the community graph (with random band)
under node removals, using an incremental update scheme.

Overview
--------
Given:
- a network (NCOL undirected edge list),
- a node→communities mapping (CSV: node_id, c1, c2, ...),
- and a per-node metrics table containing a targeting column (default: hv_s1),

this script simulates node removals in a prescribed order and computes the curve
W_inter(p) = sum of inter–community edge weights in the community graph H_p.
An edge (u, v) contributes weight |C_u| * |C_v| only when C_u ∩ C_v = ∅ and both
endpoints are still present. Computation is incremental in O(E) per trajectory.

Intended use
------------
This script is part of the auxiliary analysis to validate that targeting nodes with
low neighborhood-centered homophily (tilde-h_v, stored as 'hv_s1' by default) causes
a faster drop in inter–community weight than random removals. It is not part of the
core pipeline metrics (which focus on W_inter(p), \tilde h(p), and rho_GCC(p) curves),
but it produces the W_inter(p) curve used in the paper.

Inputs
------
- Edge list: <data-dir>/<key>/<key>_network.txt (NCOL, undirected).
- Node→communities CSV: <data-dir>/<key>/<key>_node_to_communities.csv.
- Per-node metrics CSV: <proc3-dir>/<key>_all_metrics.csv (must include --hv-col).

Outputs
-------
- CSV: <out-dir>/<key>/meta_eval_<key>.csv with columns [p, W_inter, strategy].
- (optional) PNG plot with low_hv vs. random band.

CLI summary
-----------
--dataset-key <key> or explicit --edges/--node2comms/--metrics-file
--hv-col hv_s1
--step-p 0.02
--random-reps 20
--edge-sample-frac 1.0
--seed 12345
--plot / --no-plot
--out-dir outputs/meta_graph_eval

Examples
--------
Infer all paths from layout:
    python calc_W_inter_comm_graph.py \
        --data-dir datasets \
        --proc3-dir outputs/proc3 \
        --dataset-key so \
        --plot

Explicit file paths:
    python calc_W_inter_comm_graph.py \
        --edges datasets/so/so_network.txt \
        --node2comms datasets/so/so_node_to_communities.csv \
        --metrics-file outputs/proc3/so_all_metrics.csv \
        --hv-col hv_s1 \
        --out-dir outputs/meta_graph_eval/so \
        --no-plot
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt

from common_functions import load_node_to_communities_map

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_W_inter_comm_graph")

# Default dataset keys used in the project
DATASETS = ["so", "dblp", "lj", "youtube", "text", "LFR_low", "LFR_mid", "LFR_high"]


# -------------------- IO helpers --------------------
def load_graph_ncol(path: Path) -> ig.Graph:
    g = ig.Graph.Read_Ncol(str(path), directed=False)
    if "name" not in g.vs.attributes():
        g.vs["name"] = [str(v.index) for v in g.vs]
    return g


def load_node2comms_safe(path_csv: Path, delimiters=(",", ";", "\t", " ")):
    """Robustly load node→communities into dict[str -> frozenset[int]]."""
    last_err = None
    for d in delimiters:
        try:
            m = load_node_to_communities_map(str(path_csv), delimiter=d)
            if not m:
                raise ValueError("Empty map; trying next delimiter")
            norm = {
                str(u): frozenset(int(c) for c in cs if str(c).strip() != "")
                for u, cs in m.items()
            }
            if not norm:
                raise ValueError("Normalization to frozensets failed")
            return norm
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read node→communities mapping: {path_csv}") from last_err


def is_inter(Cu: frozenset, Cv: frozenset) -> bool:
    """True if communities are strictly disjoint and both non-empty."""
    if not Cu or not Cv:
        return False
    return Cu.isdisjoint(Cv)


# -------------------- Precomputation --------------------
def precompute_inter_weights(
    g: ig.Graph,
    names: np.ndarray,
    node2comms: Dict[str, frozenset],
    edge_sample_frac: float = 1.0,
) -> Tuple[float, List[List[Tuple[int, int]]]]:
    """
    Precompute inter-community weights.

    Returns
    -------
    W0 : float
        Initial total inter–community weight (possibly estimated if sampled).
    inter_adj : list[list[tuple[int,int]]]
        For each node u, a list of (v, w_e) with v > u only, for edges that are inter.
        Using v>u avoids double subtraction during incremental updates.
    """
    n = g.vcount()
    C_of = [node2comms.get(names[i], frozenset()) for i in range(n)]
    inter_adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    W0 = 0.0

    chosen = None
    if edge_sample_frac < 1.0:
        E = g.ecount()
        take = int(max(1, round(edge_sample_frac * E)))
        chosen = set(np.random.choice(E, size=take, replace=False).tolist())

    for eid, e in enumerate(g.es):
        if (chosen is not None) and (eid not in chosen):
            continue
        u, v = e.tuple
        Cu, Cv = C_of[u], C_of[v]
        if not is_inter(Cu, Cv):
            continue
        w = len(Cu) * len(Cv)
        W0 += w
        if u < v:
            inter_adj[u].append((v, w))
        else:
            inter_adj[v].append((u, w))

    return W0, inter_adj


def evaluate_order_incremental(
    order_idx: np.ndarray,
    p_grid: np.ndarray,
    W0: float,
    inter_adj: List[List[Tuple[int, int]]],
    n: int,
) -> pd.DataFrame:
    """
    Incrementally update W_inter as nodes are removed following 'order_idx'.

    Notes
    -----
    At step k, node order_idx[k] is removed. For each stored (u, v) with v > u,
    the weight is subtracted only if both endpoints are alive at removal time.
    """
    alive = np.ones(n, dtype=bool)
    W = W0
    rows = []

    # Robust discretization: unique k values from p_grid
    k_grid = np.unique(np.clip(np.rint(p_grid * n).astype(int), 0, n))

    k_target_idx = 0
    for k in range(n + 1):
        # Record current W at grid points
        if k_target_idx < len(k_grid) and k == k_grid[k_target_idx]:
            p = k / n
            rows.append((p, W))
            k_target_idx += 1
            if k_target_idx >= len(k_grid):
                break

        if k == n:
            break

        u = order_idx[k]
        if not alive[u]:
            continue

        for v, w in inter_adj[u]:
            if alive[v]:
                W -= w
        alive[u] = False

    return pd.DataFrame(rows, columns=["p", "W_inter"])


# -------------------- CLI & main --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_w_inter_curves",
        description="Compute W_inter(p) on the community graph for given node removal strategies.",
    )

    # Dataset layout inference
    p.add_argument("--data-dir", type=Path, help="Base datasets directory.")
    p.add_argument("--proc3-dir", type=Path, default=Path("outputs/proc3"),
                   help="Directory where per-node metrics live. Default: outputs/proc3")
    p.add_argument("--dataset-key", type=str, help=f"Dataset key in {DATASETS}.")

    # Explicit files (alternative)
    p.add_argument("--edges", type=Path, help="Path to network edge list (NCOL, undirected).")
    p.add_argument("--node2comms", type=Path, help="Path to node->communities CSV.")
    p.add_argument("--metrics-file", type=Path, help="Path to per-node metrics CSV (must include --hv-col).")

    # Targeting / strategies
    p.add_argument("--hv-col", type=str, default="hv_s1",
                   help="Column in metrics CSV used for low-hv targeting. Default: hv_s1")
    p.add_argument("--random-reps", type=int, default=20, help="Number of random replicate trajectories.")
    p.add_argument("--seed", type=int, default=12345, help="Random seed.")

    # Discretization / performance
    p.add_argument("--step-p", type=float, default=0.02, help="Grid step for p in [0,1]. Default: 0.02")
    p.add_argument("--edge-sample-frac", type=float, default=1.0,
                   help="Fraction of edges to sample for W_inter approximation (1.0 uses all).")

    # Output / plotting / logging
    p.add_argument("--out-dir", type=Path, default=Path("outputs/meta_graph_eval"),
                   help="Base output dir. Default: outputs/meta_graph_eval")
    p.add_argument("--plot", dest="plot", action=argparse.BooleanOptionalAction, default=True,
                   help="Whether to write a PNG plot (default: on).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def infer_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Infer (edges, node2comms, metrics_csv, out_dir) from dataset layout or explicit args."""
    if args.edges and args.node2comms and args.metrics_file:
        key = args.dataset_key or Path(args.metrics_file).stem.split("_")[0]
        return args.edges, args.node2comms, args.metrics_file, args.out_dir / key

    if not args.data_dir or not args.dataset_key:
        raise ValueError("Provide either explicit file paths or --data-dir + --dataset-key.")

    key = args.dataset_key
    data_dir = Path(args.data_dir) / key
    edges = data_dir / f"{key}_network.txt"
    node2comms = data_dir / f"{key}_node_to_communities.csv"
    metrics_csv = Path(args.proc3_dir) / f"{key}_all_metrics.csv"
    out_dir = args.out_dir / key
    return edges, node2comms, metrics_csv, out_dir


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)
    np.random.seed(args.seed)

    try:
        edges_path, node2_path, metrics_path, out_dir = infer_paths(args)
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("Edges: %s", edges_path)
        log.info("Node→Communities: %s", node2_path)
        log.info("Per-node metrics: %s", metrics_path)
        log.info("Output dir: %s", out_dir)

        # Load inputs
        g = load_graph_ncol(edges_path)
        names = np.array(g.vs["name"], dtype=str)
        n = g.vcount()
        log.info("Graph loaded: n=%d, m=%d", n, g.ecount())

        node2comms = load_node2comms_safe(node2_path)

        dfm = pd.read_csv(metrics_path)
        if args.hv_col not in dfm.columns:
            raise KeyError(f"Column '{args.hv_col}' not found in metrics file: {metrics_path}")
        # Identify ID column
        if "Node ID" in dfm.columns:
            id_col = "Node ID"
        else:
            id_col = dfm.columns[0]
            log.warning("Using first column as Node ID: '%s'", id_col)

        hv_map = dict(zip(dfm[id_col].astype(str), dfm[args.hv_col].astype(float)))
        hv = np.array([hv_map.get(u, np.nan) for u in names], dtype=float)

        # Filter nodes missing hv
        ok = np.isfinite(hv)
        if ok.sum() < len(hv):
            keep_idx = np.where(ok)[0]
            g = g.subgraph(keep_idx.tolist())
            names = np.array(g.vs["name"], dtype=str)
            hv = hv[ok]
            n = g.vcount()
            log.warning("Filtered nodes without '%s'. Remaining: n=%d", args.hv_col, n)

        # Precompute inter weights (single pass)
        W0, inter_adj = precompute_inter_weights(
            g, names, node2comms, edge_sample_frac=args.edge_sample_frac
        )
        log.info("Initial W_inter (possibly estimated): %.2f", W0)

        # Grid
        p_grid = np.arange(0.0, 1.0 + 1e-12, args.step_p)

        # Strategy: low_hv (ascending)
        order_low = np.argsort(hv)
        df_low = evaluate_order_incremental(order_low, p_grid, W0, inter_adj, n)
        df_low["strategy"] = "low_hv"

        # Strategy: random replicates
        dfs = [df_low]
        for r in range(args.random_reps):
            order_rand = np.random.permutation(n)
            df_r = evaluate_order_incremental(order_rand, p_grid, W0, inter_adj, n)
            df_r["strategy"] = f"rand_{r+1}"
            dfs.append(df_r)

        df_all = pd.concat(dfs, ignore_index=True)
        csv_out = out_dir / f"meta_eval_{out_dir.name}.csv"
        df_all.to_csv(csv_out, index=False)
        log.info("CSV written: %s", csv_out)

        if args.plot:
            df_low_only = df_all[df_all["strategy"] == "low_hv"].copy()
            df_rand = df_all[df_all["strategy"].str.startswith("rand_")].copy()

            def band(df, col):
                ggroup = df.groupby("p")[col]
                return pd.DataFrame({
                    "p": ggroup.mean().index,
                    "median": ggroup.median().values,
                    "p5": ggroup.quantile(0.05).values,
                    "p95": ggroup.quantile(0.95).values,
                })

            band_w = band(df_rand, "W_inter")

            plt.figure(figsize=(6.8, 4.2))
            plt.fill_between(band_w["p"], band_w["p5"], band_w["p95"], alpha=0.4, label="Random P5–P95")
            plt.plot(band_w["p"], band_w["median"], lw=1.8, label="Random median")
            plt.plot(df_low_only["p"], df_low_only["W_inter"], lw=2.2, label="low_hv")
            plt.xlabel("Fraction removed (p)")
            plt.ylabel(r"$\sum w_{ij}$ (inter-community weight)")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()
            fig_out = out_dir / f"meta_eval_{out_dir.name}.png"
            plt.savefig(fig_out, dpi=240)
            plt.close()
            log.info("PNG written: %s", fig_out)

        return 0

    except Exception as e:
        log.exception("Fatal error: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
