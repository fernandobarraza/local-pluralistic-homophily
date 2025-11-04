#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
Compute rho_GCC-only trajectories (fraction of nodes in the Giant CC) for targeted
node-removal strategies, and summarize rho_GCC at specific removal budgets p.

Use cases
---------
- Produce raw trajectories ρ_GCC(p) up to p ≤ 0.10 (or user-specified).
- Summarize ρ_GCC at p ∈ {0.05, 0.10} (o los que indiques con --p-points).
- Default strategy: 'low_hv' (ascendente por hv_s1).

Inputs per dataset <key>
------------------------
<data-dir>/<key>/<key>_network.txt           (NCOL, undirected)
<proc3-dir>/<key>_all_metrics.csv            (must contain 'hv_s1' for low_hv)

Outputs (per dataset, under --out-dir)
--------------------------------------
<out-dir>/rho_only_<key>.csv                 (raw trajectory)
<out-dir>/rho_only_summary_<key>.csv         (summary at requested p)
and consolidated: <out-dir>/rho_only_summary_all.csv

Examples
--------
python aux/calc_rho_gcc.py \
  --datasets amazon twitch \
  --data-dir datasets \
  --proc3-dir outputs/proc3 \
  --out-dir outputs/node_removal \
  --strategies low_hv \
  --step-frac 0.01 --max-p 0.10 --p-points 0.05 0.10 \
  --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import igraph as ig
import numpy as np
import pandas as pd

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_rho_gcc")

# Column hints
HV_COL = "hv_s1"
NODE_ID_HINTS = ["node", "id", "node_id", "name", "Node ID", "nid", "_id", "index"]


# ----------------------------- I/O helpers ----------------------------- #
def need(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def load_graph_ncol(path: Path) -> ig.Graph:
    g = ig.Graph.Read_Ncol(str(path), directed=False)
    if "name" in g.vs.attributes() and g.vs["name"] is not None:
        g.vs["name"] = [str(n) for n in g.vs["name"]]
    else:
        g.vs["name"] = [str(v.index) for v in g.vs]
    return g


def find_id_column(df: pd.DataFrame) -> str:
    for c in NODE_ID_HINTS:
        if c in df.columns:
            return c
    # fallback razonable
    df = df.reset_index()
    return "index"


# -------------------------- Rankings (estrategias) -------------------------- #
def build_ranking(names: np.ndarray, dfm: pd.DataFrame, strategy: str, id_col: str) -> np.ndarray:
    """
    Devuelve el orden de eliminación (arreglo de nombres) para la estrategia pedida.
    Implementadas:
      - low_hv: ascendente por hv_s1 (más negativo primero).
      - bridgecc: descendente por 'bridgecc' si está disponible; si no, KeyError.
    """
    strategy = strategy.strip().lower()
    id_map = dfm[id_col].astype(str)

    if strategy == "low_hv":
        if HV_COL not in dfm.columns:
            raise KeyError(f"Missing column '{HV_COL}' required for strategy 'low_hv'.")
        hv_map: Dict[str, float] = dict(zip(id_map, dfm[HV_COL].astype(float)))
        hv_vals = np.array([hv_map.get(n, np.nan) for n in names], dtype=float)
        ok = np.isfinite(hv_vals)
        return names[ok][np.argsort(hv_vals[ok])]  # más negativos primero

    if strategy == "bridgecc":
        if "bridgecc" not in dfm.columns:
            raise KeyError("Missing 'bridgecc' column required for strategy 'bridgecc'.")
        br_map: Dict[str, float] = dict(zip(id_map, dfm["bridgecc"].astype(float)))
        br_vals = np.array([br_map.get(n, np.nan) for n in names], dtype=float)
        ok = np.isfinite(br_vals)
        return names[ok][np.argsort(-br_vals[ok])]  # descendente

    raise NotImplementedError(f"Estrategia no soportada: {strategy}")


# ----------------------------- ρ_GCC primitives ----------------------------- #
def rho_gcc(g: ig.Graph, N0: int) -> float:
    if g.vcount() == 0:
        return 0.0
    comps = g.connected_components(mode="WEAK")
    return comps.giant().vcount() / float(N0)


def evaluate_rho_trajectory(
    g_full: ig.Graph,
    ranking_names: Sequence[str],
    step_frac: float,
    max_p: float,
    label: str,
) -> pd.DataFrame:
    """
    Calcula ρ_GCC vs p removido, hasta max_p (incluido), en pasos de step_frac.
    """
    N = g_full.vcount()
    names = np.array(g_full.vs["name"])
    name_to_idx = {n: i for i, n in enumerate(names)}
    keep = np.ones(N, dtype=bool)

    # Alínea ranking al grafo y quita ausentes
    idx_seq = np.array([name_to_idx.get(n) for n in ranking_names if n in name_to_idx], dtype=float)
    idx_seq = idx_seq[np.isfinite(idx_seq)].astype(int)

    step_size = int(max(1, round(step_frac * N)))
    max_steps = int(round(max_p / step_frac))

    rows = []
    ptr = 0
    for step in range(max_steps + 1):
        curr_idx = np.where(keep)[0]
        g = g_full.subgraph(curr_idx)

        removed_frac = 1.0 - (g.vcount() / N)
        rows.append(
            {
                "strategy": label,
                "step": step,
                "removed_frac": float(np.round(removed_frac, 6)),
                "rho_GCC": float(rho_gcc(g, N0=N)),
                "nodes_left": int(g.vcount()),
                "edges_left": int(g.ecount()),
            }
        )

        if step == max_steps:
            break
        take_to = min(ptr + step_size, idx_seq.size)
        if ptr >= take_to:
            break
        batch = idx_seq[ptr:take_to]
        keep[batch] = False
        ptr = take_to

    return pd.DataFrame(rows)


# ----------------------------- per-dataset job ----------------------------- #
def process_dataset(
    key: str,
    data_dir: Path,
    proc3_dir: Path,
    out_dir: Path,
    strategies: List[str],
    step_frac: float,
    max_p: float,
    p_points: Iterable[float],
) -> pd.DataFrame:
    net_path = data_dir / key / f"{key}_network.txt"
    met_path = proc3_dir / f"{key}_all_metrics.csv"
    need(net_path, f"network for {key}")
    need(met_path, f"metrics for {key}")

    g = load_graph_ncol(net_path)
    names = np.array(g.vs["name"], dtype=str)
    dfm = pd.read_csv(met_path)
    id_col = find_id_column(dfm)

    dfs = []
    for st in strategies:
        order = build_ranking(names, dfm, st, id_col=id_col)
        traj = evaluate_rho_trajectory(
            g_full=g,
            ranking_names=order.tolist(),
            step_frac=step_frac,
            max_p=max_p,
            label=st,
        )
        dfs.append(traj)

    df_out = pd.concat(dfs, ignore_index=True)
    df_out.insert(0, "dataset", key)

    # per-dataset writes
    out_dir.mkdir(parents=True, exist_ok=True)
    out_raw = out_dir / f"rho_only_{key}.csv"
    df_out.to_csv(out_raw, index=False)

    # summary at requested p
    df_tmp = df_out.copy()
    # index with safe rounding consistent with step size
    decimals = max(0, int(round(-np.log10(step_frac)))) if step_frac < 1 else 0
    df_tmp["rf_round"] = df_tmp["removed_frac"].round(decimals)

    recs = []
    for st in strategies:
        sub = df_tmp[df_tmp["strategy"] == st]
        for p in sorted(set(round(float(x), decimals) for x in p_points)):
            r = sub[sub["rf_round"] == p]
            if not r.empty:
                rho = float(r.iloc[0]["rho_GCC"])
                recs.append({"dataset": key, "strategy": st, "p": p, "rho_GCC": rho})

    df_sum = pd.DataFrame.from_records(recs)
    out_sum = out_dir / f"rho_only_summary_{key}.csv"
    df_sum.to_csv(out_sum, index=False)

    log.info("[%s] written: %s ; %s", key, out_raw, out_sum)
    return df_sum


# ----------------------------- CLI ----------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calc_rho_gcc",
        description="rho_GCC trajectories and snapshots at given removal fractions p.",
    )
    p.add_argument("--datasets", nargs="+", required=True, help="Dataset keys to process.")
    p.add_argument("--data-dir", type=Path, required=True, help="Base datasets directory.")
    p.add_argument("--proc3-dir", type=Path, default=Path("outputs/proc3"), help="Directory with <key>_all_metrics.csv.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/node_removal"), help="Output directory.")
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["low_hv"],
        help="Node-removal strategies (e.g., low_hv, bridgecc). Default: low_hv.",
    )
    p.add_argument("--step-frac", type=float, default=0.01, help="Removal step fraction (default 0.01).")
    p.add_argument("--max-p", type=float, default=0.10, help="Max removal fraction inclusive (default 0.10).")
    p.add_argument(
        "--p-points",
        nargs="+",
        type=float,
        default=[0.05, 0.10],
        help="Removal fractions to summarize rho_GCC (default: 0.05 0.10).",
    )
    p.add_argument("--seed", type=int, default=123, help="RNG seed (reserved for random baselines; default 123).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)
    np.random.seed(args.seed)

    all_summaries = []
    for key in args.datasets:
        log.info("Dataset: %s", key)
        df_sum = process_dataset(
            key=key,
            data_dir=args.data_dir,
            proc3_dir=args.proc3_dir,
            out_dir=args.out_dir,
            strategies=list(args.strategies),
            step_frac=args.step_frac,
            max_p=args.max_p,
            p_points=args.p_points,
        )
        all_summaries.append(df_sum)

    if all_summaries:
        df_all = pd.concat(all_summaries, ignore_index=True)
        out_all = args.out_dir / "rho_only_summary_all.csv"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_all, index=False)
        log.info("Written consolidated: %s", out_all)

        # pretty print
        with pd.option_context(
            "display.max_colwidth",
            None,
            "display.precision",
            6,
            "display.float_format",
            lambda x: f"{x:.6f}",
        ):
            print("\n== rho_GCC summary ==")
            print(df_all.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
