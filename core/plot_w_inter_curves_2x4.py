#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Plot normalized W_inter(p) bands and targeted curves for 8 datasets (2x4 grid).

Overview
--------
This script reads per-dataset meta-graph evaluation CSV files (with columns like
'p', 'W_inter', and 'strategy'∈{low_hv, rand_*}), builds normalized W_inter(p)
curves by dividing each series by its initial value at p_min, and renders a 2x4
panel figure: the random band [P5,P95], its median, and the targeted curve
(low_hv). X ticks only on the bottom row; shared axes; global legend.

Inputs
------
For each dataset key 'k', this script expects a CSV at:
    <in-dir>/<k>/meta_eval_<k>.csv
with columns: p, W_inter, strategy.

Outputs
-------
- A PNG file with the 2x4 grid (default name: meta_eval_grid_4x2_norm.png).

Examples
--------
python core/plot_w_inter_curves_2x4.py \
  --datasets lj youtube dblp twitch amazon deezer so github \
  --in-dir outputs/meta_graph_eval \
  --out-file outputs/meta_graph_eval/meta_eval_grid_4x2_norm.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("plot_w_inter_curves_2x4")

# Friendly titles for known dataset keys
DEFAULT_TITLES: Dict[str, str] = {
    "lj": "LiveJournal",
    "youtube": "YouTube",
    "dblp": "DBLP",
    "twitch": "Twitch",
    "amazon": "Amazon",
    "deezer": "Deezer",
    "so": "StackOverflow",
    "github": "GitHub",
}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot_w_inter_curves_2x4",
        description="Render a 2x4 panel for normalized W_inter(p) across 8 datasets.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Exactly 8 dataset keys (order = panel order, row-major).",
    )
    p.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Directory containing per-dataset CSVs at <in-dir>/<key>/meta_eval_<key>.csv",
    )
    p.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <in-dir>/meta_eval_grid_4x2_norm.png",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args(argv)


def load_band(df: pd.DataFrame, col: str = "W_inter") -> pd.DataFrame:
    """Return median and central 90% band of 'col' grouped by p."""
    g = df.groupby("p")[col]
    out = pd.DataFrame(
        {
            "p": g.mean().index,  # group index
            "median": g.median().values,
            "p5": g.quantile(0.05).values,
            "p95": g.quantile(0.95).values,
        }
    )
    return out


def normalize_series(df: pd.DataFrame, col: str, by_value: float) -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col] / by_value
    return out


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    if len(args.datasets) != 8:
        raise ValueError("This script expects exactly 8 dataset keys for a 2x4 grid.")

    out_file = args.out_file or (args.in_dir / "meta_eval_grid_4x2_norm.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Figure size: taller to improve slope perception (≈ 17.0cm × 29.0cm)
    fig_w_in, fig_h_in = 6.69, 11.4
    fig, axes = plt.subplots(
        nrows=4, ncols=2, figsize=(fig_w_in, fig_h_in), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for i, key in enumerate(args.datasets):
        ax = axes[i]
        csv_path = args.in_dir / key / f"meta_eval_{key}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for dataset '{key}': {csv_path}")

        df = pd.read_csv(csv_path)
        if not {"p", "W_inter", "strategy"}.issubset(df.columns):
            raise ValueError(f"{csv_path} lacks required columns: p, W_inter, strategy")

        # Filter targeted and random strategies
        df_low = df[df["strategy"] == "low_hv"]
        df_rand = df[df["strategy"].str.startswith("rand_")]

        if df_low.empty or df_rand.empty:
            raise ValueError(f"[{key}] Missing series 'low_hv' or 'rand_*' in {csv_path}")

        # Compute random band and normalize all by initial median
        band = load_band(df_rand, col="W_inter")
        init = float(band["median"].iloc[0])
        if init == 0.0:
            raise ValueError(f"[{key}] Initial random median is zero; cannot normalize.")

        band_norm = band.copy()
        band_norm[["median", "p5", "p95"]] = band_norm[["median", "p5", "p95"]] / init

        df_low_norm = normalize_series(df_low[["p", "W_inter"]], "W_inter", init)

        # Plot
        ax.fill_between(
            band_norm["p"],
            band_norm["p5"],
            band_norm["p95"],
            color="#bbbbbb",
            alpha=0.4,
            label="Random $[P_5, P_{95}]$",
        )
        ax.plot(
            band_norm["p"],
            band_norm["median"],
            color="dimgray",
            lw=1.8,
            label="Random (median)",
        )
        ax.plot(
            df_low_norm["p"],
            df_low_norm["W_inter"],
            color="#1f77b4",
            lw=2.2,
            label=r"Targeted ($\tilde{h}_v$)",
        )

        title = DEFAULT_TITLES.get(key, key)
        ax.set_title(title, fontsize=11, fontweight="bold", y=1.04)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 0.95)
        ax.set_ylim(0.0, 1.02)

        # Hide X tick labels on the first 3 rows
        if i < 6:
            ax.tick_params(labelbottom=False)

    # Y label only on the first column
    for r in range(4):
        axes[r * 2].set_ylabel(r"$\widehat{W}_{\mathrm{inter}}(p)$", fontsize=10)

    # X label only on the last row
    axes[6].set_xlabel("Fraction removed (p)", fontsize=10)
    axes[7].set_xlabel("Fraction removed (p)", fontsize=10)

    # Global legend (take handles from first axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)

    plt.tight_layout(rect=[0.00, 0.065, 1.00, 0.985])
    plt.savefig(out_file, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    log.info("Figure written: %s", out_file)
    print(f"✔ Figure saved:\n  - {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
