#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Plot sensitivity panels: $\tilde h(p)$ and (optionally) $|GCC(p)|/N$ for multiple datasets.

Overview
--------
This script renders a **two-row figure**:
- Panel A: homophily sensitivity $\tilde h(p)$ (or `h_hat` if provided).
- Panel B: giant component fraction $\rho_{\mathrm{GCC}}(p)=|GCC(p)|/N$ (optional).

It consumes the consolidated CSVs produced by `calc_sensitivity_curves.py`:
- `summary_curves.csv`  with columns at least: [p, h or h_hat, dataset]
- `summary_markers.csv` with columns: [dataset, m_mean, psi] (optional; used for legend)
- (optional) `gcc_curves.csv` with columns: [p, gcc_norm, dataset] or a generic schema
  containing GCC with a 'metric' column and 'metric_value'.

Hardcoded paths and Spanish comments were removed; the figure is configured via CLI flags.

Examples
--------
Default layout (GT datasets + one LFR family):
    python plot_sensitivity_with_gcc.py \
        --curves-file outputs/sensitivity/summary_curves.csv \
        --markers-file outputs/sensitivity/summary_markers.csv \
        --gcc-file outputs/sensitivity/gcc_curves.csv \
        --keep-one-lfr mid \
        --out-file outputs/sensitivity/sensitivity_panel_with_gcc.png

Custom datasets and titles:
    python plot_sensitivity_with_gcc.py \
        --curves-file outputs/sensitivity/summary_curves.csv \
        --datasets dblp lj youtube so \
        --titles DBLP LiveJournal YouTube StackOverflow \
        --out-file outputs/sensitivity/sensitivity_gt.png \
        --no-gcc

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot_sensitivity_with_gcc",
        description="Plot a two-row figure with $\u007E h(p)$ (or h_hat) and optionally |GCC(p)|/N.",
    )
    p.add_argument("--curves-file", type=Path, required=True,
                   help="Path to summary_curves.csv (from calc_sensitivity_curves.py).")
    p.add_argument("--markers-file", type=Path, default=None,
                   help="Path to summary_markers.csv (optional; adds <m> and psi in legend).")
    p.add_argument("--gcc-file", type=Path, default=None,
                   help="Path to GCC curves CSV (optional).")
    p.add_argument("--datasets", nargs="+",
                   default=["dblp", "lj", "youtube", "so", "LFR_low", "LFR_mid", "LFR_high"],
                   help="Datasets to plot (order also controls legend order).")
    p.add_argument("--titles", nargs="+", default=None,
                   help="Titles per dataset (same length as --datasets). Defaults to display names.")
    p.add_argument("--keep-one-lfr", choices=["low", "mid", "high", "none"], default="mid",
                   help="Which LFR to include (others hidden). Use 'none' to hide all LFR. Default: mid")
    p.add_argument("--out-file", type=Path, required=True,
                   help="Output figure path (PNG/SVG/PDF).")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI for raster formats. Default: 300")
    p.add_argument("--show-pstar", action="store_true",
                   help="Draw optional vertical p* lines if a 'p_star' col is present (markers/gcc markers).")
    return p.parse_args()


# ---------------- Utils ----------------
DISPLAY_DEFAULT = {
    "dblp": "DBLP",
    "lj": "LiveJournal",
    "youtube": "YouTube",
    "so": "StackOverflow",
    "LFR_low": "LFR",
    "LFR_mid": "LFR",
    "LFR_high": "LFR",
}

COLORS_DEFAULT = {
    "dblp":     "#1f77b4",  # blue
    "lj":       "#2ca02c",  # green
    "youtube":  "#d62728",  # red
    "so":       "#9467bd",  # purple
    "LFR_low":  "#000000",
    "LFR_mid":  "#000000",
    "LFR_high": "#000000",
}

LINEWIDTH = 2.2


def _pick_datasets(ds_list: List[str], keep_one_lfr: str) -> List[str]:
    if keep_one_lfr == "none":
        return [d for d in ds_list if not d.startswith("LFR_")]
    keep_tag = f"LFR_{keep_one_lfr}"
    return [d for d in ds_list if (not d.startswith("LFR_")) or d == keep_tag]


def _legend_label(name: str, df_marks: Optional[pd.DataFrame], ds: str) -> str:
    if df_marks is None:
        return name
    row = df_marks[df_marks["dataset"] == ds]
    if row.empty:
        return name
    m_mean = row["m_mean"].iloc[0]
    psi = row["psi"].iloc[0]
    return fr"{name}" + "\n" + fr"$\langle m\rangle={m_mean:.2f},\, \psi={psi:.2f}$"


def _load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        return None
    return pd.read_csv(path)


def _gcc_from_generic(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """If GCC data is embedded in a 'metric' column, filter and rename to [p, rho_gcc, dataset]."""
    if "metric" not in df.columns:
        return None
    sel = df["metric"].str.lower().str.contains("gcc")
    if not sel.any():
        return None
    tmp = df.loc[sel].copy()
    # Try common names
    p_col = "fraction_removed" if "fraction_removed" in tmp.columns else (
        next((c for c in tmp.columns if c.lower().startswith("p")), None)
    )
    v_col = "metric_value" if "metric_value" in tmp.columns else (
        next((c for c in tmp.columns if "value" in c.lower() or "gcc" in c.lower()), None)
    )
    if p_col is None or v_col is None:
        return None
    out = tmp.rename(columns={p_col: "p", v_col: "rho_gcc"})
    return out[["p", "rho_gcc", "dataset"]]


# ---------------- Main ----------------
def main() -> int:
    args = parse_args()

    df_curves = _load_csv(args.curves_file)
    if df_curves is None or df_curves.empty:
        raise FileNotFoundError(f"Curves file not found or empty: {args.curves_file}")
    df_marks = _load_csv(args.markers_file)

    # Either h or h_hat must be present
    ycol = "h" if "h" in df_curves.columns else "h_hat"
    if ycol not in df_curves.columns:
        raise ValueError("Curves file must contain a 'h' or 'h_hat' column.")

    # GCC (optional)
    df_gcc_raw = _load_csv(args.gcc_file)
    df_gcc: Optional[pd.DataFrame] = None
    if df_gcc_raw is not None and not df_gcc_raw.empty:
        # Accept standard [p, gcc_norm, dataset]
        if {"p", "gcc_norm", "dataset"}.issubset(df_gcc_raw.columns):
            df_gcc = df_gcc_raw.rename(columns={"gcc_norm": "rho_gcc"})[["p", "rho_gcc", "dataset"]]
        else:
            df_gcc = _gcc_from_generic(df_gcc_raw)

    # Dataset list and titles
    ds_all = _pick_datasets(args.datasets, args.keep_one_lfr)
    if args.titles is None:
        titles = [DISPLAY_DEFAULT.get(d, d) for d in ds_all]
    else:
        if len(args.titles) != len(ds_all):
            raise ValueError("--titles length must match the number of plotted datasets.")
        titles = args.titles

    # Figure (two rows)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.6, 8.6), sharex=True,
        gridspec_kw={"hspace": 0.16, "height_ratios": [1.0, 0.9]}
    )
    ax_bot.set_xticks(np.arange(0.0, 1.01, 0.10))
    ax_top.tick_params(axis="x", pad=6)
    ax_bot.tick_params(axis="x", pad=6)

    handles, labels = [], []
    for ds, title in zip(ds_all, titles):
        curv = df_curves[df_curves["dataset"] == ds].copy().sort_values("p")
        if curv.empty:
            continue
        color = COLORS_DEFAULT.get(ds, "#333333")
        ls = "--" if ds.startswith("LFR_") else "-"
        lbl = _legend_label(title, df_marks, ds)

        line, = ax_top.plot(curv["p"], curv[ycol], color=color, lw=LINEWIDTH, ls=ls, label=lbl)
        handles.append(line); labels.append(lbl)

    # Panel A cosmetics
    ax_top.set_ylabel(r"$\tilde h(p)$" if ycol == "h" else r"$\hat h(p)$")
    ax_top.set_ylim(-1.00, 1.05)
    ax_top.set_yticks(np.arange(-1.0, 1.01, 0.25))
    ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_top.grid(True, alpha=0.3)
    ax_top.margins(x=0)
    ax_top.text(0.01, 0.02, "A", transform=ax_top.transAxes,
                va="bottom", ha="left", fontsize=13, fontweight="bold")

    # Legend at right of Panel A
    ax_top.legend(
        handles, labels, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
        frameon=True, fancybox=True, framealpha=0.95,
        edgecolor="#dddddd", fontsize=10,
        labelspacing=0.6, handlelength=2.8, handletextpad=0.8
    )

    # Panel B (optional GCC)
    if df_gcc is not None and not df_gcc.empty:
        for ds in ds_all:
            curv = df_gcc[df_gcc["dataset"] == ds].copy().sort_values("p")
            if curv.empty:
                continue
            color = COLORS_DEFAULT.get(ds, "#333333")
            ls = "--" if ds.startswith("LFR_") else "-"
            ax_bot.plot(curv["p"], curv["rho_gcc"], color=color, lw=LINEWIDTH, ls=ls)

    ax_bot.set_xlabel("Fraction removed (p)")
    ax_bot.set_ylabel(r"$|GCC(p)|/N$")
    ax_bot.set_xlim(0.0, 1.0)
    ax_bot.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.set_yticks(np.arange(0.0, 1.01, 0.20))
    ax_bot.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_bot.grid(True, alpha=0.3)
    ax_bot.margins(x=0)
    ax_bot.text(0.01, 0.02, "B", transform=ax_bot.transAxes,
                va="bottom", ha="left", fontsize=13, fontweight="bold")

    # Save
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_file, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
