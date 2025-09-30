#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Network input validator: sanity checks for edge lists and community alignment.

Overview
--------
Runs a set of integrity checks on a dataset's network file (NCOL, undirected) and
(optionally) basic alignment with node IDs used by igraph. Designed to "fail fast"
before running the pipeline.

Checks reported
---------------
- Node/edge counts from NetworKit and igraph.
- Number of isolated nodes (degree == 0, via NetworKit).
- Presence of self-loops and multi-edges (via igraph).
- Active node-ID alignment between NetworKit (deg>0) and igraph (simplified).
- Connected components (count) and giant-component size (via igraph).

Inputs
------
Either:
  --data-dir <dir> --dataset-key <key>
or:
  --edges <path/to/<key>_network.txt>

Outputs
-------
- CSV summary with one row of flags and counts:
    <out-dir>/<key>_network_check.csv
- Optional Markdown report:
    <out-dir>/<key>_network_check.md

Exit code is 0 unless --strict is set and any critical issue is found.

Examples
--------
Validate a dataset from standard layout:
    python check_networks.py --data-dir datasets --dataset-key LFR_high

Custom file and strict mode:
    python check_networks.py --edges datasets/so/so_network.txt --strict --out-dir outputs/validation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from igraph import Graph

from common_functions import load_graph_nk

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("check_networks")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="check_networks",
        description="Validate network input files (NCOL, undirected) and report common issues.",
    )
    p.add_argument("--data-dir", type=Path, help="Base directory with dataset folders.")
    p.add_argument("--dataset-key", type=str, help="Dataset key (e.g., so, dblp, lj, youtube, LFR_mid).")
    p.add_argument("--edges", type=Path, help="Explicit path to <key>_network.txt (NCOL).")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/validation"),
                   help="Directory for reports. Default: outputs/validation")
    p.add_argument("--strict", action="store_true",
                   help="Exit with non-zero status if any critical issue is detected.")
    p.add_argument("--markdown", action=argparse.BooleanOptionalAction, default=True,
                   help="Also write a Markdown report alongside the CSV (default: on).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def infer_paths(args: argparse.Namespace) -> Tuple[str, str]:
    if args.edges:
        edges = args.edges
        key = edges.stem.replace("_network", "")
        return key, str(edges)
    if not args.data_dir or not args.dataset_key:
        raise ValueError("Provide either --edges OR (--data-dir AND --dataset-key).")
    key = args.dataset_key
    edges = Path(args.data_dir) / key / f"{key}_network.txt"
    return key, str(edges)


def run_checks(edges_path: str) -> Dict[str, object]:
    # NetworKit loading
    Gk = load_graph_nk(edges_path, directed=False)
    n_nk = Gk.numberOfNodes()
    e_nk = Gk.numberOfEdges()
    deg0 = sum(1 for u in Gk.iterNodes() if Gk.degree(u) == 0)
    n_active = n_nk - deg0

    # igraph loading
    G = Graph.Read_Ncol(edges_path, directed=False)
    n_ig = G.vcount()
    e_ig = G.ecount()

    # simplify to inspect loops/multiedges
    Gs = G.simplify(multiple=True, loops=True)
    loops = int(G.ecount_loops()) if hasattr(G, "ecount_loops") else (Gs.ecount() - G.simplify(multiple=True, loops=False).ecount())
    multiedges = e_ig - Gs.ecount()

    comp = G.components()
    n_comps = len(comp)
    giant_size = int(max(len(c) for c in comp)) if n_comps else n_ig

    # active ID alignment
    active_nk = {u for u in Gk.iterNodes() if Gk.degree(u) > 0}
    names_ig = set(map(int, Gs.vs["name"])) if "name" in Gs.vs.attributes() else set(range(Gs.vcount()))
    miss_in_nk = len(names_ig - active_nk)
    miss_in_ig = len(active_nk - names_ig)

    return {
        "n_nodes_networkit": n_nk,
        "n_edges_networkit": e_nk,
        "n_isolated_networkit": deg0,
        "n_active_networkit": n_active,
        "n_nodes_igraph": n_ig,
        "n_edges_igraph": e_ig,
        "n_components": n_comps,
        "giant_component_size": giant_size,
        "has_self_loops": loops > 0,
        "n_self_loops": loops,
        "has_multiedges": multiedges > 0,
        "n_multiedges": multiedges,
        "missing_ids_in_networkit": miss_in_nk,
        "missing_ids_in_igraph": miss_in_ig,
    }


def write_markdown(path: Path, key: str, stats: Dict[str, object]) -> None:
    lines = [f"# Network check report — {key}", ""]
    for k, v in stats.items():
        lines.append(f"- **{k.replace('_', ' ')}**: {v}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    try:
        key, edges = infer_paths(args)
        out_dir = args.out_dir / key
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Checking %s", edges)

        stats = run_checks(edges)
        df = pd.DataFrame([{"dataset": key, **stats}])
        csv_out = out_dir / f"{key}_network_check.csv"
        df.to_csv(csv_out, index=False)
        log.info("CSV written: %s", csv_out)

        if args.markdown:
            md_out = out_dir / f"{key}_network_check.md"
            write_markdown(md_out, key, stats)
            log.info("Markdown written: %s", md_out)

        critical = (
            stats["n_self_loops"] > 0
            or stats["n_multiedges"] > 0
            or stats["missing_ids_in_networkit"] > 0
            or stats["missing_ids_in_igraph"] > 0
        )
        if args.strict and critical:
            log.error("Critical issues found under --strict. Exiting with code 1.")
            return 1
        return 0

    except Exception as e:
        log.exception("Fatal error: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
