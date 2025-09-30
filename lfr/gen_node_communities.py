#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate node→communities mapping from an LFR (or any) communities file.

Overview
--------
Reads:
- a network edge list (NCOL undirected, 0-based IDs) to enumerate all nodes
- a communities file (one community per line, nodes separated by tabs by default)

Outputs:
- <out-dir>/<key>_node_to_communities.csv with format:
    node_id,comm1,comm2,...
  where rows are sorted by node_id and community IDs are sorted per node.

This matches the repository's expected layout and is compatible with loaders that
interpret variable-length rows.

Usage (standard layout):
    python gen_node_communities.py --data-dir datasets --datasets LFR_low LFR_mid LFR_high

Usage (explicit files, single dataset):
    python gen_node_communities.py \
        --network-file datasets/LFR_low/LFR_low_network.txt \
        --communities-file datasets/LFR_low/LFR_low_communities.txt \
        --out-dir datasets --dataset-key LFR_low

Notes
-----
- The script is idempotent: it rewrites the CSV deterministically.
- Community lines that are empty are ignored.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gen_node_communities",
        description="Create node→communities CSV (node_id,comm1,comm2,...) from a communities file.",
    )
    # Batch mode (standard layout)
    p.add_argument("--data-dir", type=Path, help="Base path containing datasets/<key>/ directories.")
    p.add_argument("--datasets", nargs="+", help="Dataset keys to process (e.g., LFR_low LFR_mid LFR_high).")
    # Single dataset (explicit files)
    p.add_argument("--network-file", type=Path, help="NCOL undirected edge list (0-based IDs).")
    p.add_argument("--communities-file", type=Path, help="Communities file (one community per line).")
    p.add_argument("--dataset-key", type=str, help="Key used to name the output file, if using explicit files.")
    # Options
    p.add_argument("--communities-sep", type=str, default="\\t", help="Separator for nodes in a community line. Default: TAB")
    p.add_argument("--out-dir", type=Path, default=Path("datasets"), help="Output base directory. Default: datasets/")
    return p.parse_args()

def _read_all_nodes_from_network(path: Path) -> Set[int]:
    nodes: Set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: 
                continue
            a, b = s.split()
            u = int(a); v = int(b)
            nodes.add(u); nodes.add(v)
    return nodes

def _build_node2comms(communities_path: Path, sep: str) -> Dict[int, List[int]]:
    node2comms: Dict[int, List[int]] = {}
    with open(communities_path, "r", encoding="utf-8") as f:
        for cid, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            toks = [t for t in s.split(sep) if t != ""]
            for t in toks:
                u = int(t)
                node2comms.setdefault(u, []).append(cid)
    # sort memberships and de-duplicate
    for u, lst in node2comms.items():
        node2comms[u] = sorted(set(lst))
    return node2comms

def _write_csv(out_path: Path, nodes: Iterable[int], node2comms: Dict[int, List[int]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # header is optional; we omit for compactness and loader flexibility
        # w.writerow(["node_id", "comm1", "comm2", "..."])
        for u in sorted(nodes):
            row = [u] + node2comms.get(u, [])
            w.writerow(row)

def _process_single(network_file: Path, communities_file: Path, out_dir: Path, key: str, sep: str) -> Path:
    nodes = _read_all_nodes_from_network(network_file)
    node2comms = _build_node2comms(communities_file, sep)
    out_path = out_dir / key / f"{key}_node_to_communities.csv"
    _write_csv(out_path, nodes, node2comms)
    return out_path

def main() -> int:
    args = parse_args()
    # Single explicit files
    if args.network_file and args.communities_file:
        key = args.dataset_key or args.network_file.stem.replace("_network", "")
        out_path = _process_single(args.network_file, args.communities_file, args.out_dir, key, args.communities_sep)
        print(f"[✓] {key}: wrote {out_path}")
        return 0

    # Batch (standard layout)
    if not args.data_dir or not args.datasets:
        raise SystemExit("Provide either explicit --network-file/--communities-file or --data-dir + --datasets.")

    for key in args.datasets:
        base = args.data_dir / key
        net = base / f"{key}_network.txt"
        com = base / f"{key}_communities.txt"
        out_path = _process_single(net, com, args.data_dir, key, args.communities_sep)
        print(f"[✓] {key}: wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
