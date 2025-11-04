#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
Compute ΔAUC and p_sep over normalized W_inter(p) curves (meta-graph evaluation).

Overview
--------
For each dataset key:
  - Read <in-dir>/<key>/meta_eval_<key>.csv (produced by calc_w_inter_curves.py).
  - Build the random band across strategies with prefix 'rand_' (median, p5, p95) per p.
  - Normalize by the initial random median (as in the figure).
  - Compute:
      * ΔAUC = ∫ (target_norm - rand_mean_norm) dp  [trapezoidal rule, non-uniform grid]
      * p_sep: first p with ≥ K consecutive points where target_norm < rand_p5_norm
  - Append one row to a consolidated CSV.

Outputs
-------
- CSV with columns:
  dataset, name, target_strategy, p_separation, AUC_difference, AUC_depletion
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_auc_w_inter")


# ----------------------------- helpers ----------------------------- #
def need(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def load_band(df: pd.DataFrame, col: str = "W_inter") -> pd.DataFrame:
    """Aggregate random band per p: median, p5, p95."""
    g = df.groupby("p")[col]
    return pd.DataFrame(
        {
            "p": g.mean().index,
            "median": g.median().values,
            "p5": g.quantile(0.05).values,
            "p95": g.quantile(0.95).values,
        }
    )


def auc_diff_norm(g: pd.DataFrame) -> float:
    """ΔAUC (normalized): integrate target_norm - rand_mean_norm over p (non-uniform grid)."""
    g2 = g.sort_values("p")
    p = g2["p"].to_numpy()
    y = (g2["target_norm"] - g2["rand_mean_norm"]).to_numpy()
    return float(np.trapz(y, p))


def p_sep_norm(g: pd.DataFrame, k_consec: int) -> float:
    """First p where target_norm < rand_p5_norm for ≥ k_consec consecutive points."""
    g2 = g.sort_values("p")
    below = (g2["target_norm"].to_numpy() < g2["rand_p5_norm"].to_numpy())
    run = 0
    for p_val, ok in zip(g2["p"].to_numpy(), below):
        run = run + 1 if ok else 0
        if run >= k_consec:
            return float(p_val)
    return float("nan")


# ----------------------------- per-dataset ----------------------------- #
def process_dataset(
    key: str,
    title: str,
    in_dir: Path,
    target_strategy: str,
    random_prefix: str,
    k_consec: int,
) -> dict:
    csv_path = in_dir / key / f"meta_eval_{key}.csv"
    need(csv_path, f"meta-eval CSV for {key}")

    df = pd.read_csv(csv_path)
    for c in ["p", "strategy", "W_inter"]:
        if c not in df.columns:
            raise ValueError(f"[{key}] CSV {csv_path} missing column '{c}'")

    # Split target vs random
    df_target = df[df["strategy"].astype(str).str.strip().eq(target_strategy)].copy()
    df_rand = df[df["strategy"].astype(str).str.startswith(random_prefix)].copy()

    if df_target.empty:
        # fallback a denominaciones alternativas comunes (por compatibilidad)
        alt = ["hv_s1", "low_hv", "low_hv_s1"]
        df_target = df[df["strategy"].astype(str).isin(alt)].copy()
        if df_target.empty:
            raise ValueError(f"[{key}] No rows for target strategy '{target_strategy}' (nor common fallbacks).")

    if df_rand.empty:
        raise ValueError(f"[{key}] No random strategies with prefix '{random_prefix}' in {csv_path}")

    # Random band and normalization
    band = load_band(df_rand, col="W_inter")
    band = band.sort_values("p")
    init = float(band["median"].iloc[0])
    if init == 0.0:
        raise ValueError(f"[{key}] Initial random median is zero; cannot normalize.")

    band_norm = band.copy()
    band_norm[["median", "p5", "p95"]] = band_norm[["median", "p5", "p95"]] / init

    # Target mean per p and normalization
    tg = (
        df_target[["p", "W_inter"]]
        .groupby("p", as_index=False)["W_inter"]
        .mean()
        .rename(columns={"W_inter": "target"})
    )
    tg["target_norm"] = tg["target"] / init

    g = (
        tg[["p", "target_norm"]]
        .merge(
            band_norm.rename(
                columns={"median": "rand_mean_norm", "p5": "rand_p5_norm", "p95": "rand_p95_norm"}
            )[["p", "rand_mean_norm", "rand_p5_norm", "rand_p95_norm"]],
            on="p",
            how="inner",
        )
        .sort_values("p")
    )

    auc = auc_diff_norm(g)
    psep = p_sep_norm(g, k_consec=k_consec)

    return {
        "dataset": key,
        "name": title,
        "target_strategy": target_strategy,
        "p_separation": psep,
        "AUC_difference": auc,   # (target_norm - rand_mean_norm)
        "AUC_depletion": -auc,   # útil si prefieres “más depleción = positivo”
    }


# ----------------------------- CLI ----------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_auc_w_inter",
        description="ΔAUC and p_sep over normalized W_inter(p) curves (meta-graph evaluation).",
    )
    p.add_argument("--datasets", nargs="+", required=True, help="Dataset keys (order for table).")
    p.add_argument(
        "--titles",
        nargs="+",
        default=None,
        help="Pretty titles per dataset (same length as --datasets). Defaults to capitalized keys.",
    )
    p.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Directory that contains <key>/meta_eval_<key>.csv files.",
    )
    p.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Output CSV (default: <in-dir>/separation_metrics.csv).",
    )
    p.add_argument(
        "--target-strategy",
        default="low_hv",
        help="Target strategy name in meta_eval CSV (default: 'low_hv'). Fallbacks: hv_s1, low_hv_s1.",
    )
    p.add_argument(
        "--random-prefix",
        default="rand_",
        help="Prefix for random strategies aggregated into the band (default: 'rand_').",
    )
    p.add_argument(
        "--k-consec",
        type=int,
        default=3,
        help="Consecutive points below P5 to define p_sep (default: 3).",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    keys: List[str] = args.datasets
    titles: List[str] = args.titles or [k.capitalize() for k in keys]
    if len(titles) != len(keys):
        raise ValueError("--titles length must match --datasets.")

    out_csv = args.out_file or (args.in_dir / "separation_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, title in zip(keys, titles):
        log.info("Dataset: %s", key)
        row = process_dataset(
            key=key,
            title=title,
            in_dir=args.in_dir,
            target_strategy=args.target_strategy,
            random_prefix=args.random_prefix,
            k_consec=args.k_consec,
        )
        rows.append(row)

    table = pd.DataFrame(
        rows,
        columns=["dataset", "name", "target_strategy", "p_separation", "AUC_difference", "AUC_depletion"],
    )
    table.to_csv(out_csv, index=False)

    # Pretty print (console)
    with pd.option_context(
        "display.max_colwidth",
        None,
        "display.precision",
        6,
        "display.float_format",
        lambda x: f"{x:.6f}",
    ):
        print("\n== separation_metrics.csv ==")
        print(table.to_string(index=False))
    print(f"\nWritten: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
