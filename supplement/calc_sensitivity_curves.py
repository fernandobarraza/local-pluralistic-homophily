#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Prepare sensitivity curves for ĥ(p) and (optionally) ρ_GCC(p) without modifying existing inputs.

Overview
--------
This script ingests precomputed node-removal curves (produced elsewhere) and prepares:
1) A normalized homophily curve ĥ(p) for the targeted strategy (low_hv_s1), and
2) Early-cut markers at p0 (bridge-like signal at node level) plus simple overlap stats.

Optionally, if a GCC curve file is available, it is copied alongside ĥ(p) for plotting.

Inputs (per dataset key <key>)
------------------------------
- Processed homophily curves:
    <curves-dir>/processed_curves_homophily_<key>.csv
  (Fallback filename with suffix "_covgcc" is also attempted.)
  Expected columns include: ['strategy', 'metric', 'fraction_removed', 'metric_value'].

- Graph and memberships:
    <data-dir>/<key>/<key>_network.txt                 (NCOL undirected; integer IDs)
    <data-dir>/<key>/<key>_node_to_communities.csv     (node_id, comm1, comm2, ...)

- Per-node metrics:
    <proc3-dir>/<key>_all_metrics.csv                  (must include --hv-col; default: hv_s1)

Outputs (per dataset)
---------------------
- <out-dir>/<key>/h_curve_<key>.csv        with columns: [p, h, h_hat, dataset]
- <out-dir>/<key>/markers_<key>.csv        with columns: [dataset, p0, S_cut_mean, S_cut_median, ...]
- <out-dir>/summary_curves.csv             concatenation of h_curve_* across datasets
- <out-dir>/summary_markers.csv            concatenation of markers_* across datasets
- (Optional) if --include-gcc and a GCC file is available:
    <out-dir>/<key>/gcc_curve_<key>.csv    with columns: [p, rho_gcc, dataset]

Notes
-----
- ĥ(p) := h(p) / h(0), restricted to p ≤ --p-max.
- The early-cut marker at p0 computes, for the lowest-hv_s1 fraction (k = round(p0 * n)),
  the per-node ratio (# inter-community neighbors) / degree, then reports mean/median/std.
- Overlap summary includes: mean membership count (m_mean) and fraction psi of nodes with m>1.

Examples
--------
Default layout and datasets (dblp, lj, youtube, so, LFR_low, LFR_mid, LFR_high):
    python calc_sensitivity_curves.py \
        --data-dir datasets \
        --proc3-dir outputs/proc3 \
        --curves-dir outputs/node_removal_curves \
        --out-dir outputs/sensitivity

Custom subset and parameters:
    python calc_sensitivity_curves.py \
        --datasets so youtube \
        --p-max 1.0 \
        --p0 0.01 \
        --hv-col hv_s1 \
        --include-gcc \
        --data-dir datasets \
        --proc3-dir outputs/proc3 \
        --curves-dir outputs/node_removal_curves \
        --out-dir outputs/sensitivity
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import igraph as ig

from common_functions import load_node_to_communities_map

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_sensitivity_curves")

# Defaults aligned with the manuscript set
DEFAULT_DATASETS = ["dblp", "lj", "youtube", "so", "LFR_low", "LFR_mid", "LFR_high"]
NODE_ID_HINTS = ["node", "id", "node_id", "Node ID", "nid", "_id", "index", "name"]


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_sensitivity_curves",
        description="Prepare normalized ĥ(p) and markers (and optionally ρ_GCC(p)) for sensitivity plots.",
    )
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   help="Dataset keys to process. Default: dblp lj youtube so LFR_low LFR_mid LFR_high")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Base directory containing dataset folders.")
    p.add_argument("--proc3-dir", type=Path, required=True,
                   help="Directory containing per-node metrics CSVs (<key>_all_metrics.csv).")
    p.add_argument("--curves-dir", type=Path, required=True,
                   help="Directory containing processed curves CSVs for homophily (and optionally GCC).")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/sensitivity"),
                   help="Output base directory. Default: outputs/sensitivity")

    p.add_argument("--hv-col", type=str, default="hv_s1",
                   help="Column name in per-node metrics used for targeting (default: hv_s1).")
    p.add_argument("--p-max", type=float, default=1.0,
                   help="Maximum fraction removed to keep in ĥ(p). Default: 1.0")
    p.add_argument("--p0", type=float, default=0.01,
                   help="Early-cut fraction for the S_cut marker. Default: 0.01")
    p.add_argument("--include-gcc", action=argparse.BooleanOptionalAction, default=True,
                   help="If on, try to load and export ρ_GCC(p) curves when available (default: on).")

    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------- IO utils ----------------
def load_graph_ncol(path: Path) -> ig.Graph:
    g = ig.Graph.Read_Ncol(str(path), directed=False)
    if "name" not in g.vs.attributes():
        g.vs["name"] = [str(v.index) for v in g.vs]
    return g


def load_node2comms_safe(path_csv: Path, delimiters=(",", ";", "\t", " ")):
    last_err = None
    for d in delimiters:
        try:
            m = load_node_to_communities_map(str(path_csv), delimiter=d)
            if not m:
                raise ValueError("Empty mapping")
            norm = {str(u): frozenset(int(c) for c in cs if str(c).strip() != "") for u, cs in m.items()}
            if not norm:
                raise ValueError("Normalization failed")
            return norm
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read node→communities map: {path_csv}") from last_err


def find_id_column(df: pd.DataFrame) -> str:
    for c in NODE_ID_HINTS:
        if c in df.columns:
            return c
    # Fallback: first column after resetting index
    return df.reset_index().columns[0]


def is_inter(Cu: frozenset, Cv: frozenset) -> bool:
    if not Cu or not Cv:
        return False
    return Cu.isdisjoint(Cv)


def inter_degree_of_node(g: ig.Graph, u_idx: int, C_of: List[frozenset]) -> int:
    Cu = C_of[u_idx]
    if not Cu:
        return 0
    cnt = 0
    for v in g.neighbors(u_idx):
        if is_inter(Cu, C_of[v]):
            cnt += 1
    return cnt


def read_processed_curves(curves_dir: Path, ds: str) -> pd.DataFrame:
    candidates = [
        curves_dir / f"processed_curves_homophily_{ds}.csv",
        curves_dir / f"processed_curves_homophily_{ds}_covgcc.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No processed homophily curves for '{ds}' in: {', '.join(str(p) for p in candidates)}")


def maybe_read_gcc(curves_dir: Path, ds: str) -> pd.DataFrame | None:
    candidates = [
        curves_dir / f"processed_curves_gcc_{ds}.csv",
        curves_dir / f"processed_curves_gcc_{ds}_covgcc.csv",
        curves_dir / f"processed_curves_{ds}.csv",  # generic multi-metric file (if present)
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # Accept either dedicated file or generic with a 'metric' column
            if "metric" in df.columns:
                df = df[df["metric"].str.lower().str.contains("gcc")]
            return df if not df.empty else None
    return None


# ---------------- Core per-dataset ----------------
def prepare_dataset(
    ds: str,
    data_dir: Path,
    proc3_dir: Path,
    curves_dir: Path,
    out_base: Path,
    hv_col: str,
    p_max: float,
    p0: float,
    include_gcc: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1) ĥ(p) from processed curves (strategy 'low_hv_s1' + metric 'homophily')
    df_curves_raw = read_processed_curves(curves_dir, ds)
    sel = (df_curves_raw["strategy"].isin(["low_hv_s1", "low_hv"])) & (df_curves_raw["metric"].str.lower() == "homophily")
    df_low = df_curves_raw.loc[sel, ["fraction_removed", "metric_value"]].dropna().copy()
    if df_low.empty:
        raise ValueError(f"{ds}: missing rows for strategy low_hv_s1 in processed curves.")

    df_low.sort_values("fraction_removed", inplace=True)
    h0 = float(df_low.iloc[0]["metric_value"])
    if h0 == 0.0:
        raise ValueError(f"{ds}: h(0) is zero; cannot normalize.")
    df_low = df_low[df_low["fraction_removed"] <= p_max].copy()
    df_low["h_hat"] = df_low["metric_value"] / h0
    df_low["dataset"] = ds

    out_dir = out_base / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_out = out_dir / f"h_curve_{ds}.csv"
    df_low.rename(columns={"fraction_removed": "p", "metric_value": "h"}).to_csv(curve_out, index=False)
    log.info("[%s] curve → %s", ds, curve_out)

    # 2) Early-cut marker S_cut and overlap
    net_path = data_dir / ds / f"{ds}_network.txt"
    n2c_path = data_dir / ds / f"{ds}_node_to_communities.csv"
    met_path = proc3_dir / f"{ds}_all_metrics.csv"

    g = load_graph_ncol(net_path)
    names = np.array(g.vs["name"], dtype=str)
    node2comms = load_node2comms_safe(n2c_path)

    dfm = pd.read_csv(met_path)
    id_col = find_id_column(dfm)
    if hv_col not in dfm.columns:
        raise ValueError(f"{ds}: column {hv_col} not found in {met_path}")
    hv_map = dict(zip(dfm[id_col].astype(str), dfm[hv_col].astype(float)))
    hv = np.array([hv_map.get(u, np.nan) for u in names], dtype=float)

    ok = np.isfinite(hv)
    if ok.sum() < len(hv):
        g = g.subgraph(np.where(ok)[0])
        names = names[ok]
        hv = hv[ok]

    n = g.vcount()
    k = max(1, int(round(p0 * n)))
    order = np.argsort(hv)[:k]  # ascending hv: boundary-like nodes first

    C_of = [node2comms.get(u, frozenset()) for u in names]
    deg = np.array(g.degree(), dtype=int)

    ratios: List[float] = []
    for u in order:
        d = deg[u]
        if d <= 0:
            continue
        delta_inter = inter_degree_of_node(g, u, C_of)
        ratios.append(delta_inter / d)

    if ratios:
        arr = np.array(ratios, dtype=float)
        S_mean = float(arr.mean())
        S_median = float(np.median(arr))
        S_std = float(arr.std(ddof=1) if arr.size > 1 else 0.0)
    else:
        S_mean = S_median = S_std = np.nan

    m_arr = np.array([len(node2comms.get(u, frozenset())) for u in names], dtype=int)
    m_mean = float(m_arr.mean()) if m_arr.size else np.nan
    psi = float((m_arr > 1).mean()) if m_arr.size else np.nan

    # (Optional) attach ĥ(p0) for reference
    try:
        h_hat_p0 = float(df_low.loc[df_low["fraction_removed"] <= p0, "h_hat"].iloc[-1])
    except Exception:
        h_hat_p0 = np.nan

    marker_row = {
        "dataset": ds,
        "p0": p0,
        "S_cut_mean": S_mean,
        "S_cut_median": S_median,
        "S_cut_std": S_std,
        "m_mean": m_mean,
        "psi": psi,
        "n": int(g.vcount()),
        "e": int(g.ecount()),
        "h_hat_p0": h_hat_p0,
    }

    marker_out = out_dir / f"markers_{ds}.csv"
    pd.DataFrame([marker_row]).to_csv(marker_out, index=False)
    log.info("[%s] marker → %s", ds, marker_out)

    # 3) Optionally export GCC curve if present
    if include_gcc:
        df_gcc = maybe_read_gcc(curves_dir, ds)
        if df_gcc is not None and not df_gcc.empty:
            # Normalize schema to [p, rho_gcc, dataset]
            if {"fraction_removed", "metric_value"}.issubset(df_gcc.columns):
                df_gcc_out = df_gcc.loc[:, ["fraction_removed", "metric_value"]].rename(
                    columns={"fraction_removed": "p", "metric_value": "rho_gcc"}
                )
            else:
                # Generic fallback: try to detect columns
                p_col = [c for c in df_gcc.columns if "fraction" in c or c.lower().startswith("p")]
                v_col = [c for c in df_gcc.columns if "value" in c or "gcc" in c.lower()]
                if not p_col or not v_col:
                    log.warning("[%s] GCC file found but could not infer columns; skipping.", ds)
                    df_gcc_out = None
                else:
                    df_gcc_out = df_gcc.rename(columns={p_col[0]: "p", v_col[0]: "rho_gcc"})[["p", "rho_gcc"]]

            if df_gcc_out is not None:
                df_gcc_out["dataset"] = ds
                gcc_out = out_dir / f"gcc_curve_{ds}.csv"
                df_gcc_out.to_csv(gcc_out, index=False)
                log.info("[%s] gcc → %s", ds, gcc_out)
        else:
            log.info("[%s] GCC curve not found; skipping.", ds)

    # Return ready-to-concatenate frames
    df_curve_ready = pd.read_csv(curve_out)
    df_marker_ready = pd.read_csv(marker_out)
    return df_curve_ready, df_marker_ready


# ---------------- Main ----------------
def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    out_base = args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)

    all_curves = []
    all_markers = []
    for ds in args.datasets:
        try:
            c, m = prepare_dataset(
                ds=ds,
                data_dir=args.data_dir,
                proc3_dir=args.proc3_dir,
                curves_dir=args.curves_dir,
                out_base=out_base,
                hv_col=args.hv_col,
                p_max=args.p_max,
                p0=args.p0,
                include_gcc=args.include_gcc,
            )
            all_curves.append(c.assign(dataset=ds))
            all_markers.append(m.assign(dataset=ds))
        except Exception as e:
            log.exception("[%s] failed: %s", ds, e)

    if all_curves:
        df_curves_all = pd.concat(all_curves, ignore_index=True)
        df_curves_all.to_csv(out_base / "summary_curves.csv", index=False)
        log.info("SUMMARY curves → %s", out_base / "summary_curves.csv")
    else:
        log.warning("No curves were produced.")

    if all_markers:
        df_markers_all = pd.concat(all_markers, ignore_index=True)
        df_markers_all.to_csv(out_base / "summary_markers.csv", index=False)
        log.info("SUMMARY markers → %s", out_base / "summary_markers.csv")
    else:
        log.warning("No markers were produced.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
