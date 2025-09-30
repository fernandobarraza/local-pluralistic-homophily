#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Unify per-node metric tables into a single CSV for a given dataset.

Overview
--------
This script merges outputs produced by:
1) `calc_hv.py`    → local pluralistic homophily proxies (legacy hv_* columns)
2) `calc_ghv.py`   → global-mean local homophily vector (ghv_* columns)
3) (optional) a structural metrics table with centralities, etc.

The merge is performed on the "Node ID" column. Only non-duplicated columns
are added on each successive merge step.

Inputs
------
- One or more hv tables (default inferred from --proc1-dir).
- One ghv table (default inferred from --proc1-dir).
- Optional structural metrics table (default inferred from --proc3-dir).

Outputs
-------
- A single CSV with "Node ID" first, followed by all available metric columns.

Examples
--------
Minimal (infer default file names from dirs and dataset key):
    python unify_metrics.py \
        --dataset-key so \
        --proc1-dir outputs/proc1 \
        --proc3-dir outputs/proc3

Explicit output path and no structural metrics:
    python unify_metrics.py \
        --dataset-key so \
        --proc1-dir outputs/proc1 \
        --include-struct false \
        --out outputs/proc3/so_unified_metrics.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("unify_metrics")


def parse_bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="unify_metrics",
        description="Merge hv/ghv and (optionally) structural metrics into a unified per-node CSV.",
    )
    p.add_argument("--dataset-key", required=True, help="Dataset short key (e.g., so, dblp, lj, youtube, text, LFR_*).")
    p.add_argument("--proc1-dir", type=Path, default=Path("outputs/proc1"),
                   help="Directory where hv/ghv tables were written. Default: outputs/proc1")
    p.add_argument("--proc3-dir", type=Path, default=Path("outputs/proc3"),
                   help="Directory for structural metrics and unified outputs. Default: outputs/proc3")
    p.add_argument("--hv-files", nargs="*", default=None,
                   help="Optional explicit list of hv CSVs. If omitted, defaults to "
                        "[<key>_hv_metrics.csv, <key>_ghv_metrics.csv] under --proc1-dir.")
    p.add_argument("--struct-file", type=Path, default=None,
                   help="Optional explicit structural CSV path. If omitted, default is "
                        "<proc3-dir>/<key>_all_metrics_struct.csv.")
    p.add_argument("--include-struct", type=parse_bool, default=True,
                   help="Whether to merge structural metrics if available. Default: true")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path. Default: <proc3-dir>/<key>_unified_metrics.csv")
    p.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def read_csv_safe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Node ID" not in df.columns:
        # Assume first column is the node identifier and rename
        first = df.columns[0]
        df = df.rename(columns={first: "Node ID"})
    return df


def unify_metrics(args: argparse.Namespace) -> Path:
    key = args.dataset_key

    # Resolve defaults
    proc1 = args.proc1_dir
    proc3 = args.proc3_dir
    proc1.mkdir(parents=True, exist_ok=True)
    proc3.mkdir(parents=True, exist_ok=True)

    hv_files: List[Path]
    if args.hv_files:
        hv_files = [Path(p) for p in args.hv_files]
    else:
        hv_files = [
            proc1 / f"{key}_hv_metrics.csv",
            proc1 / f"{key}_ghv_metrics.csv",
        ]

    struct_file = args.struct_file if args.struct_file else (proc3 / f"{key}_all_metrics_struct.csv")
    out = args.out if args.out else (proc3 / f"{key}_unified_metrics.csv")

    # Check output existence
    if out.exists() and not args.force:
        raise FileExistsError(f"Output exists: {out}. Use --force to overwrite.")

    # 1) Load hv/ghv metrics
    dfs: List[pd.DataFrame] = []
    for f in hv_files:
        if f.exists():
            df = read_csv_safe(f)
            log.info("Loaded %s (%d rows, %d cols)", f.name, len(df), len(df.columns))
            dfs.append(df)
        else:
            log.warning("Missing hv/ghv file: %s", f)

    if not dfs:
        raise ValueError("No hv/ghv metrics files found. Provide --hv-files or ensure defaults exist.")

    df_hv = dfs[0]
    for df2 in dfs[1:]:
        # Only merge new columns not present yet
        new_cols = [c for c in df2.columns if c != "Node ID" and c not in df_hv.columns]
        df_hv = pd.merge(df_hv, df2[["Node ID"] + new_cols], on="Node ID", how="outer")

    df_all = df_hv

    # 2) Merge structural metrics (optional)
    if args.include_struct and struct_file.exists():
        df_struct = read_csv_safe(struct_file)
        new_cols = [c for c in df_struct.columns if c != "Node ID" and c not in df_all.columns]
        df_all = pd.merge(df_all, df_struct[["Node ID"] + new_cols], on="Node ID", how="outer")
        log.info("Structural metrics merged: %s", ", ".join(new_cols) if new_cols else "(no new columns)")
    elif args.include_struct:
        log.warning("Structural metrics not found: %s", struct_file)

    # 3) Save
    cols = ["Node ID"] + [c for c in df_all.columns if c != "Node ID"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out, columns=cols, index=False)
    log.info("Unified metrics saved: %s (%d nodes, %d columns)", out, len(df_all), len(df_all.columns))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)
    try:
        unify_metrics(args)
        return 0
    except Exception as e:
        log.exception("Fatal error: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
