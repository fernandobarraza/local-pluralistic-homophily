#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Plot a small network with overlapping communities and optional edge homophily styling.

Overview
--------
Utility visualization (used for the toy example in the paper). Draws communities
as colored "shadows" (vertex cover) and styles nodes by the sign of hv_s1
(blue: assortative, red: disassortative, gray: near-zero). Optionally styles edges
by h_e (positive, negative, near-zero).

Inputs (standard layout or explicit)
-----------------------------------
--data-dir <dir> --dataset-key <key>
or:
--network-file, --communities-file
Optional:
--node-metrics <proc3>/<key>_all_metrics.csv (or hv_metrics) with columns:
    Node ID, hv_s1 (or pluralistic_homophily), num_communities (optional)
--edge-metrics  <supplemental>/<key>_he_metrics.csv with columns:
    Node 1 ID, Node 2 ID, he_s1

Outputs
-------
<out-file> (PNG/SVG/PDF). No display unless --show is passed.

Examples
--------
python plot_overlapping_communities.py \
  --data-dir datasets --dataset-key toy6 \
  --node-metrics outputs/proc3/toy6_all_metrics.csv \
  --edge-metrics outputs/supplemental/toy6/toy6_he_metrics.csv \
  --out-file outputs/viz/toy6_overlap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot_overlapping_communities",
        description="Plot overlapping communities with node hv_s1 and optional edge h_e styling.",
    )
    # Layout
    p.add_argument("--data-dir", type=Path, help="Base datasets directory.")
    p.add_argument("--dataset-key", type=str, help="Dataset key.")
    p.add_argument("--network-file", type=Path, help="NCOL undirected edge list.")
    p.add_argument("--communities-file", type=Path, help="Communities file.")
    # Metrics
    p.add_argument("--node-metrics", type=Path, help="Per-node metrics CSV (needs hv_s1).")
    p.add_argument("--edge-metrics", type=Path, help="Per-edge metrics CSV (needs he_s1).")
    # Options
    p.add_argument("--layout", default="kk", help="igraph layout name (kk, fr, rt, ...). Default: kk")
    p.add_argument("--eps", type=float, default=1e-3, help="Near-zero band for color neutral. Default: 1e-3")
    p.add_argument("--show", action="store_true", help="Display the figure window.")
    p.add_argument("--out-file", type=Path, required=True, help="Output image path.")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs. Default: 300")
    return p.parse_args()


def infer_paths(args: argparse.Namespace) -> Tuple[str, Path, Path]:
    if args.network_file and args.communities_file:
        key = args.dataset_key or args.network_file.stem.replace("_network", "")
        return key, args.network_file, args.communities_file
    if not args.data_dir or not args.dataset_key:
        raise ValueError("Provide either explicit files OR --data-dir + --dataset-key.")
    key = args.dataset_key
    base = Path(args.data_dir) / key
    return key, base / f"{key}_network.txt", base / f"{key}_communities.txt"


def _load_node_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize likely column names
    ren = {}
    if "Node ID" in df.columns: ren["Node ID"] = "Node"
    if "pluralistic_homophily" in df.columns: ren["pluralistic_homophily"] = "hv_s1"
    if "num_communities" in df.columns: ren["num_communities"] = "m"
    df = df.rename(columns=ren)
    return df


def _load_edge_metrics(path: Path) -> Dict[tuple, float]:
    df = pd.read_csv(path)
    ren = {}
    for c in df.columns:
        cc = c.strip().lower().replace(" ", "_")
        if c != cc: ren[c] = cc
    df = df.rename(columns=ren)
    need = {"node_1_id", "node_2_id", "he_s1"}
    if not need.issubset(df.columns):
        raise ValueError(f"Edge metrics must contain {need}")
    edge2he = {tuple(sorted((int(r["node_1_id"]), int(r["node_2_id"])))): float(r["he_s1"])
               for _, r in df.iterrows()}
    return edge2he


def _style_edges(g: ig.Graph, edge2he: Dict[tuple, float], eps: float):
    E = g.get_edgelist()
    he_vals = [edge2he.get(tuple(sorted(e)), np.nan) for e in E]
    a = np.array([0.0 if not np.isfinite(x) else abs(x) for x in he_vals], dtype=float)
    m = a.max() if a.size else 0.0
    widths = (1.2 + 2.6 * (a / m)) if m > 0 else np.full(len(E), 1.5)
    colors = []
    for h in he_vals:
        if not np.isfinite(h) or abs(h) <= eps:
            colors.append((0.45, 0.45, 0.45, 0.40))
        elif h > 0:
            colors.append((0.16, 0.38, 0.70, 0.95))
        else:
            colors.append((0.79, 0.20, 0.20, 0.95))
    g.es["width"] = widths
    g.es["color"] = colors


def main() -> int:
    args = parse_args()
    key, net_path, comms_path = infer_paths(args)

    g = ig.Graph.Read_Ncol(str(net_path), directed=False)

    # Default gray nodes
    g.vs["color"] = "grey"
    g.vs["frame_color"] = "#444444"
    g.vs["frame_width"] = 0.8
    g.vs["size"] = 26
    g.vs["label_color"] = "black"
    g.vs["label_size"] = 15

    # Optional node metrics (color by hv_s1 sign)
    if args.node_metrics and Path(args.node_metrics).exists():
        df = _load_node_metrics(args.node_metrics)
        # Map Node → hv_s1
        df["Node"] = df["Node"].astype(str)
        hv_map = df.set_index("Node")["hv_s1"].to_dict()
        for v in g.vs:
            name = str(v["name"])
            val = hv_map.get(name, np.nan)
            if not np.isfinite(val) or abs(val) <= args.eps:
                v["color"] = "#808080"
            elif val > 0:
                v["color"] = "#4c72b0"  # blue
            else:
                v["color"] = "#c44e52"  # red

    # Communities cover
    comms_raw: List[List[str]] = []
    with open(comms_path, "r", encoding="utf-8") as fh:
        for line in fh:
            toks = [t for t in line.strip().split("\t") if t != ""]
            if toks:
                comms_raw.append([t for t in toks if t in set(g.vs["name"])])

    clusters = [[g.vs.find(name=v).index for v in cl] for cl in comms_raw]
    cover = ig.VertexCover(g, clusters)
    pal = ig.RainbowPalette(n=len(clusters))

    # Optional edge styling by h_e
    if args.edge_metrics and Path(args.edge_metrics).exists():
        edge2he = _load_edge_metrics(args.edge_metrics)
        _style_edges(g, edge2he, args.eps)

    # Layout
    layout = g.layout(args.layout)

    # Plot
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ig.plot(
        cover, mark_groups=True, inline=False, layout=layout, palette=pal, target=ax,
        edge_color=(g.es["color"] if "color" in g.es.attributes() else (0.3, 0.3, 0.3, 0.25)),
        edge_width=(g.es["width"] if "width" in g.es.attributes() else 1.5),
        vertex_frame_color=g.vs["frame_color"], vertex_frame_width=g.vs["frame_width"],
    )
    plt.savefig(args.out_file, dpi=args.dpi, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()
    print(f"Figure written to {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
