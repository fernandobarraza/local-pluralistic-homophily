#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Forest plot of ΔIER_node-mean (hv_s1 − BridgeCC) at a fixed τ.

Overview
--------
Reads the consolidated standard CSV produced by calc_edge_ratio.py and builds a
monochrome forest plot ordered by |Δ| (pp). Optionally annotates ψ per dataset.

Inputs
------
--in-dir/summary_all_datasets_standard.csv
(or falls back to concatenating per-dataset edge_ratio_summary_<ds>.csv)

Outputs
-------
PNG and PDF forest plots. Optional LaTeX table (disabled by default).

Examples
--------
python core/forest_plot.py \
  --in-dir outputs/edge_ratio \
  --datasets dblp lj youtube so deezer github amazon twitch \
  --titles DBLP LiveJournal YouTube StackOverflow Deezer GitHub Amazon Twitch \
  --tau 0.3 \
  --out-file outputs/edge_ratio/figs/forest_meta_bw.pdf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("forest_plot")

TAU_ATOL = 5e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="forest_plot", description="Forest plot for ΔIER_node-mean at τ.")
    p.add_argument("--in-dir", type=Path, required=True, help="Directory with edge-ratio CSV outputs.")
    p.add_argument("--datasets", nargs="+", required=True, help="Dataset keys (order for the figure).")
    p.add_argument("--titles", nargs="+", required=False, help="Pretty titles per dataset key (same length as --datasets).")
    p.add_argument("--tau", type=float, default=0.3, help="τ for Jaccard threshold (default 0.3).")
    p.add_argument("--out-file", type=Path, default=None, help="Output file (PDF/PNG). If omitted, writes to <in-dir>/figs/.")
    p.add_argument("--annotate-threshold-pp", type=float, default=5.0, help="Annotate Δ text if |Δ| > threshold (pp).")
    p.add_argument("--no-psi", action="store_true", help="Do not show ψ in y-labels.")
    p.add_argument("--psi-file", type=Path, default=None, help="Optional CSV with columns: key,title,psi to override labels.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ----------------------------- Load & filter ----------------------------- #
def load_consolidated(in_dir: Path) -> Optional[pd.DataFrame]:
    for fname in ["summary_all_datasets_standard.csv"]:
        p = in_dir / fname
        if p.exists():
            return pd.read_csv(p)
    return None


def load_per_dataset(in_dir: Path, keys: List[str]) -> pd.DataFrame:
    dfs = []
    for k in keys:
        p = in_dir / k / f"edge_ratio_summary_{k}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True)


def filter_and_clean_tau(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    c_dataset = cols.get("dataset", "dataset")
    c_tau = cols.get("tau", "tau")
    c_str = cols.get("strategy", "strategy")

    df[c_tau] = pd.to_numeric(df[c_tau], errors="coerce")
    df = df[np.isfinite(df[c_tau]) & (np.abs(df[c_tau] - tau) < TAU_ATOL)].copy()
    df[c_str] = df[c_str].astype(str).lower().str.strip()

    return df.rename(columns={c_dataset: "dataset", c_tau: "tau", c_str: "strategy"})


# ----------------------------- Build Δ rows ----------------------------- #
def extract_delta_rows(df_tau: pd.DataFrame, keys: List[str], titles: List[str]) -> pd.DataFrame:
    rows = []
    for key, name in zip(keys, titles):
        sub = df_tau[df_tau["dataset"] == key]
        if sub.empty:
            raise ValueError(f"[{key}] no rows at τ≈ target.")

        hv = sub[sub["strategy"].isin(["hv_s1", "hv"])].iloc[0]
        br = sub[sub["strategy"].isin(["bridgecc", "bridge_cc", "br"])].iloc[0]

        mu_hv = float(hv["IER_node_mean"])
        mu_br = float(br["IER_node_mean"])
        delta = mu_hv - mu_br

        w_hv = (float(hv["IER_node_mean_CI_high"]) - float(hv["IER_node_mean_CI_low"])) / 2.0
        w_br = (float(br["IER_node_mean_CI_high"]) - float(br["IER_node_mean_CI_low"])) / 2.0

        se_delta = ((w_hv / 1.96) ** 2 + (w_br / 1.96) ** 2) ** 0.5
        hw_delta = 1.96 * se_delta
        ci_low = delta - hw_delta
        ci_high = delta + hw_delta

        rows.append(
            {
                "key": key,
                "name": name,
                "delta_pp": delta * 100.0,
                "ci_low_pp": ci_low * 100.0,
                "ci_high_pp": ci_high * 100.0,
                "nm_hv": mu_hv,
                "nm_br": mu_br,
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(by="delta_pp", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
    return out


# ----------------------------- Plot ----------------------------- #
def forest_meta_bw(
    df: pd.DataFrame,
    annotate_threshold_pp: float,
    show_psi: bool,
    psi_map: Optional[dict],
    out_base: Path,
):
    y = np.arange(len(df))
    x = df["delta_pp"].to_numpy()
    lo = df["ci_low_pp"].to_numpy()
    hi = df["ci_high_pp"].to_numpy()

    if show_psi and psi_map:
        ylabels = [f"{row['name']}\n(ψ={psi_map.get(row['key'], np.nan):.2f})" for _, row in df.iterrows()]
    else:
        ylabels = [row["name"] for _, row in df.iterrows()]

    fig, ax = plt.subplots(figsize=(8.6, 4.3))

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8, zorder=0)

    for i, (xi, l, h) in enumerate(zip(x, lo, hi)):
        if np.isfinite(l) and np.isfinite(h):
            ax.errorbar(
                xi,
                i,
                xerr=[[xi - l], [h - xi]],
                fmt="none",
                ecolor="black",
                elinewidth=2.2,
                capsize=4,
                zorder=2,
            )

    for i, xi in enumerate(x):
        if xi >= 0:
            ax.scatter([xi], [i], s=90, marker="o", facecolors="white", edgecolors="black", linewidths=1.5, zorder=3)
        else:
            ax.scatter([xi], [i], s=100, marker="s", facecolors="black", edgecolors="black", linewidths=1.5, zorder=3)
        if abs(xi) > annotate_threshold_pp:
            ax.text(
                xi + (0.9 if xi > 0 else -0.9),
                i,
                f"{xi:+.1f} pp",
                va="center",
                ha=("left" if xi > 0 else "right"),
                fontsize=10,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel(r'$\Delta$IER$_{\mathrm{node\text{-}mean}}$ ($\tilde{h}_v$ $-$ BridgeCC) [pp]', fontsize=11)

    xmin = min(-12, np.floor((x.min() - 3) / 5) * 5)
    xmax = max(28, np.ceil((x.max() + 3) / 5) * 5)
    ax.set_xlim(xmin, xmax)
    for xv in np.arange(np.floor(xmin / 5) * 5, np.ceil(xmax / 5) * 5 + 0.1, 10):
        if abs(xv) > 1e-9:
            ax.axvline(xv, color="0.88", linestyle=":", linewidth=0.8, zorder=0)

    ax.grid(False)
    ax.tick_params(axis="x", labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    if out_base.suffix.lower() in {".png", ".pdf"}:
        out_png = out_base.with_suffix(".png")
        out_pdf = out_base.with_suffix(".pdf")
    else:
        out_png = out_base.with_suffix(".png")
        out_pdf = out_base.with_suffix(".pdf")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s and %s", out_png, out_pdf)


# ----------------------------- Main ----------------------------- #
def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    # figure out titles & psi map
    keys = args.datasets
    if args.titles and len(args.titles) != len(keys):
        raise ValueError("--titles must have the same length as --datasets.")
    titles = args.titles or [k.capitalize() for k in keys]

    psi_map = None
    if args.psi_file and args.psi_file.exists():
        df_psi = pd.read_csv(args.psi_file)
        # expect columns: key,title,psi (title optional)
        psi_map = {str(r["key"]): float(r["psi"]) for _, r in df_psi.iterrows() if "key" in r and "psi" in r}

    # load consolidated or per-dataset
    df_all = load_consolidated(args.in_dir)
    if df_all is None:
        df_all = load_per_dataset(args.in_dir, keys)

    # filter by tau
    df_tau = filter_and_clean_tau(df_all, args.tau)

    # build Δ rows, ordered by |Δ|
    df_plot = extract_delta_rows(df_tau, keys, titles)

    # out base
    if args.out_file:
        out_base = args.out_file
        if out_base.suffix.lower() not in {".png", ".pdf"}:
            out_base = out_base.with_suffix(".pdf")
    else:
        out_base = args.in_dir / "figs" / "forest_meta_bw.pdf"

    forest_meta_bw(
        df_plot,
        annotate_threshold_pp=args.annotate_threshold_pp,
        show_psi=not args.no_psi,
        psi_map=psi_map,
        out_base=out_base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
