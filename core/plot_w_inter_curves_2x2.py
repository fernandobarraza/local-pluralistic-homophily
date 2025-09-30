#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Plot a 2×2 grid of normalized W_inter(p) curves from community-graph evaluations.

Overview
--------
This script reads the per-dataset CSV files produced by `calc_W_inter_comm_graph.py`,
normalizes W_inter(p) by the initial random-median value, and renders a 2×2 grid
comparing the targeted strategy (low_hv) against the random band (P5–P95).

It is intended as a *visualization step* for the auxiliary analysis on the
community graph and is not part of the core evaluation pipeline.

Inputs
------
For each dataset key `<key>`, the script expects an input file at:
    <in-dir>/<key>/meta_eval_<key>.csv

Each CSV must contain columns: ["p", "W_inter", "strategy"],
where strategy is either "low_hv" or "rand_i".

Outputs
-------
A single image file with the 2×2 grid layout:
    <out-file>  (PNG by default; SVG optional via --out-svg)

Examples
--------
Default datasets and paths:
    python plot_W_inter_comm_graph_2x2.py \
        --in-dir outputs/meta_graph_eval \
        --out-file outputs/meta_graph_eval/meta_eval_grid_2x2_norm.png

Custom dataset list and titles:
    python plot_W_inter_comm_graph_2x2.py \
        --datasets dblp lj youtube so \
        --titles DBLP LiveJournal YouTube StackOverflow \
        --in-dir outputs/meta_graph_eval \
        --out-file outputs/meta_graph_eval/meta_eval_grid_2x2_norm.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot_w_inter_curves_2x2",
        description="Plot a 2×2 grid of normalized W_inter(p) curves from community-graph evaluations.",
    )
    p.add_argument(
        "--datasets", nargs="+",
        default=["dblp", "lj", "youtube", "so"],
        help="Dataset keys (order defines panel placement). Default: dblp lj youtube so",
    )
    p.add_argument(
        "--titles", nargs="+",
        default=["DBLP", "LiveJournal", "YouTube", "StackOverflow"],
        help="Panel titles corresponding to --datasets. Default matches the four defaults.",
    )
    p.add_argument(
        "--in-dir", type=Path, default=Path("outputs/meta_graph_eval"),
        help="Directory containing per-dataset CSVs from calc_W_inter_comm_graph.py.",
    )
    p.add_argument(
        "--out-file", type=Path, default=Path("outputs/meta_graph_eval/meta_eval_grid_2x2_norm.png"),
        help="Output image path (PNG recommended).",
    )
    p.add_argument(
        "--out-svg", action="store_true",
        help="Also write an SVG with the same basename.",
    )
    p.add_argument(
        "--ylim", nargs=2, type=float, default=[0.0, 1.0],
        help="Y-axis limits for normalized curves. Default: 0.0 1.0",
    )
    p.add_argument(
        "--share-axes", action="store_true",
        help="Share x/y axes across subplots.",
    )
    p.add_argument(
        "--legend", choices=["panel", "bottom", "none"], default="bottom",
        help="Legend placement: per-panel, bottom (shared), or none. Default: bottom",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Image resolution (DPI). Default: 300",
    )
    return p.parse_args()


def load_band(df: pd.DataFrame, col: str = "W_inter") -> pd.DataFrame:
    g = df.groupby("p")[col]
    return pd.DataFrame({
        "p": g.mean().index,
        "median": g.median().values,
        "p5": g.quantile(0.05).values,
        "p95": g.quantile(0.95).values,
    })


def normalize_to_initial(band: pd.DataFrame, df_low: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    init = float(band["median"].iloc[0])
    if init == 0.0:
        raise ValueError("Initial random-median value is zero; cannot normalize.")
    band_norm = band.copy()
    band_norm[["median", "p5", "p95"]] = band_norm[["median", "p5", "p95"]] / init
    df_low_norm = df_low.copy()
    df_low_norm["W_inter"] = df_low_norm["W_inter"] / init
    return band_norm, df_low_norm


def plot_panel(ax, band_norm: pd.DataFrame, df_low_norm: pd.DataFrame) -> None:
    ax.fill_between(
        band_norm["p"], band_norm["p5"], band_norm["p95"],
        alpha=0.35, label=r"Random $P_{5}$–$P_{95}$"
    )
    ax.plot(
        band_norm["p"], band_norm["median"],
        lw=1.8, label=r"Random median"
    )
    ax.plot(
        df_low_norm["p"], df_low_norm["W_inter"],
        lw=2.2, label=r"$h_v \uparrow$"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 1.0)


def main() -> int:
    args = parse_args()

    ds: List[str] = args.datasets
    titles: List[str] = args.titles
    if len(titles) != len(ds):
        raise ValueError("--titles must have the same length as --datasets")

    # Prepare grid
    n_panels = len(ds)
    if n_panels != 4:
        raise ValueError("This script renders a 2×2 grid; please pass exactly 4 datasets.")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7),
                             sharex=args.share-axes, sharey=args.share-axes)
    axes = axes.flatten()

    handles_first = None

    for i, (key, name) in enumerate(zip(ds, titles)):
        ax = axes[i]
        csv_path = args.in_dir / key / f"meta_eval_{key}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input file: {csv_path}")

        df = pd.read_csv(csv_path)

        # Split strategies
        df_low = df[df["strategy"] == "low_hv"]
        df_rand = df[df["strategy"].str.startswith("rand_")]
        if df_low.empty or df_rand.empty:
            raise ValueError(f"File {csv_path} lacks required strategies (low_hv and rand_*).")

        # Build band and normalize
        band = load_band(df_rand, col="W_inter")
        band_norm, df_low_norm = normalize_to_initial(band, df_low)

        # Panel content
        plot_panel(ax, band_norm, df_low_norm)
        ax.set_title(name, fontsize=11, fontweight="bold", y=1.06)

        # Axis labels on left/bottom panels
        if i % 2 == 0:
            ax.set_ylabel(r"$\widehat{W}_{\mathrm{inter}}(p)$")
        if i >= 2:
            ax.set_xlabel("Fraction removed (p)")

        ax.set_ylim(args.ylim[0], args.ylim[1])

        if args.legend == "panel":
            ax.legend(loc="best", frameon=False)

        if handles_first is None:
            handles_first, labels_first = ax.get_legend_handles_labels()

    if args.legend == "bottom":
        fig.legend(handles_first, labels_first, loc="lower center", ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0.10, 1, 0.96] if args.legend == "bottom" else None)

    # Write outputs
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_file, dpi=args.dpi, bbox_inches="tight")
    if args.out_svg:
        svg_path = args.out_file.with_suffix(".svg")
        plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
