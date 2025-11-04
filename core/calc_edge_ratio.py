#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compute Inter-Edge Ratio (IER) under a Jaccard τ-threshold across datasets.

Overview
--------
For each dataset and each τ in a given grid, this script:
  1) loads G (ncol), node→communities, and a metrics table with hv_s1 and (if present) bridgecc,
  2) builds targeted sets S_h (lowest \\tilde{h}_v) and S_b (highest bridgecc) of size k=⌊p·|V|⌋,
  3) computes IER (node-mean and stub) using inter-edge counts defined by Jaccard(C(u),C(v)) < τ,
  4) estimates 95% CIs via bootstrap, and two-sided permutation p-values under degree-stratified null,
  5) writes per-dataset CSVs and a multi-dataset consolidated file.

Inputs
------
--data-dir/<ds>/{ds}_network.txt, {ds}_node_to_communities.csv
--proc3-dir/{ds}_all_metrics.csv      (must contain 'hv_s1'; 'bridgecc' is preferred but can be proxied)

Outputs (under --out-dir)
-------------------------
Per dataset (in <out-dir>/<ds>/):
  - edge_ratio_<ds>.csv               (legacy mixed, kept for compatibility)
  - edge_ratio_summary_<ds>.csv       (standard: strategy-wise, CIs, p-values)
  - edge_ratio_by_degree_<ds>.csv     (stable: degree bins summary)
  - edge_ratio_deltas_<ds>.csv        (standard deltas hv - bridgecc by τ)

Consolidated (in <out-dir>/):
  - summary_all_datasets_standard.csv     (multi-dataset τ/strategy table)
  - summary_all_datasets.csv              (legacy consolidation if legacy files exist)

Examples
--------
python core/calc_edge_ratio.py \
  --datasets dblp lj youtube so deezer github amazon twitch \
  --data-dir datasets --proc3-dir outputs/proc3 --out-dir outputs/edge_ratio \
  --taus 0.0 0.1 0.2 0.3 --budget 0.10 --n-boot 300 --n-perm 500 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import igraph as ig
import numpy as np
import pandas as pd

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("calc_edge_ratio")

# Columns / hints
HV_COL = "hv_s1"
BRIDGE_COL = "bridgecc"
NODE_ID_HINTS = ["node", "id", "node_id", "Node ID", "nid", "_id", "index", "name"]


# ----------------------------- I/O helpers ----------------------------- #
def need(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def load_graph_ncol(path: Path) -> ig.Graph:
    g = ig.Graph.Read_Ncol(str(path), directed=False)
    # normalize names to str
    if "name" not in g.vs.attributes() or g.vs["name"] is None:
        g.vs["name"] = [str(v.index) for v in g.vs]
    else:
        g.vs["name"] = [str(x) for x in g.vs["name"]]
    return g


def find_id_column(df: pd.DataFrame) -> str:
    for c in NODE_ID_HINTS:
        if c in df.columns:
            return c
    return df.columns[0]


def load_node2comms_safe(path_csv: Path, delimiters=(",", ";", "\t", " ")):
    from common_functions import load_node_to_communities_map

    last_err = None
    for d in delimiters:
        try:
            m = load_node_to_communities_map(str(path_csv), delimiter=d)
            if not m:
                raise ValueError("Empty node→communities map")
            return {str(u): frozenset(int(c) for c in cs if str(c).strip() != "") for u, cs in m.items()}
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read node→communities from {path_csv}") from last_err


def ensure_bridgecc(g: ig.Graph, dfm: pd.DataFrame, id_col: str, names: np.ndarray) -> np.ndarray:
    """Return bridgecc vector; fallback proxy = deg*(1 - local clustering)."""
    if BRIDGE_COL in dfm.columns:
        m = dict(zip(dfm[id_col].astype(str), dfm[BRIDGE_COL].astype(float)))
        return np.array([m.get(u, np.nan) for u in names], dtype=float)
    deg = np.array(g.degree(), dtype=float)
    trans = np.array(g.transitivity_local_undirected(vertices=None, mode="zero"), dtype=float)
    return deg * (1.0 - trans)


# -------------------------- stratified permutations -------------------------- #
def degree_quartiles_labels(deg: np.ndarray) -> np.ndarray:
    qs = np.quantile(deg, (0.25, 0.5, 0.75))
    bins = np.digitize(deg, qs, right=True)  # 0..3
    labels = np.array(["Q1", "Q2", "Q3", "Q4"], dtype=object)
    return labels[bins]


def build_idx_by_bin(bins_all: np.ndarray) -> Dict[object, np.ndarray]:
    uniq = np.unique(bins_all)
    return {b: np.where(bins_all == b)[0] for b in uniq}


def counts_by_bin(bins_all: np.ndarray, S: np.ndarray) -> Dict[object, int]:
    uniq, counts = np.unique(bins_all[S], return_counts=True)
    return dict(zip(uniq.tolist(), counts.tolist()))


def sample_by_counts(rng, idx_by_bin: Dict[object, np.ndarray], target_counts: Dict[object, int]) -> np.ndarray:
    picks = []
    for b, cnt in target_counts.items():
        pool = idx_by_bin[b]
        repl = pool.size < cnt
        picks.append(rng.choice(pool, size=cnt, replace=repl))
    return np.concatenate(picks).astype(int)


# ----------------------------- IER primitives ----------------------------- #
def precompute_inter_deg_tau(g: ig.Graph, C_of: List[frozenset], tau: float) -> np.ndarray:
    inter_deg = np.zeros(g.vcount(), dtype=np.int32)
    if tau == 0.0:
        for (u, v) in g.get_edgelist():
            Cu, Cv = C_of[u], C_of[v]
            if Cu and Cv and Cu.isdisjoint(Cv):
                inter_deg[u] += 1
                inter_deg[v] += 1
        return inter_deg

    for (u, v) in g.get_edgelist():
        Cu, Cv = C_of[u], C_of[v]
        if not Cu or not Cv:
            continue
        inter = len(Cu & Cv)
        union = len(Cu | Cv)
        j = (inter / union) if union > 0 else 0.0
        if j < tau:
            inter_deg[u] += 1
            inter_deg[v] += 1
    return inter_deg


def IER_stub(inter_deg_sum: float, deg_sum: float) -> float:
    return (inter_deg_sum / deg_sum) if deg_sum > 0 else np.nan


def IER_node_mean(inter_deg: np.ndarray, deg: np.ndarray, S: np.ndarray) -> float:
    denom = np.maximum(1, deg[S])
    return float(np.mean(inter_deg[S] / denom))


def bootstrap_stub(inter_deg: np.ndarray, deg: np.ndarray, S: np.ndarray, rng, n_boot=300) -> Tuple[float, float]:
    if S.size == 0:
        return (np.nan, np.nan)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        Sb = rng.choice(S, size=S.size, replace=True)
        vals[i] = IER_stub(inter_deg[Sb].sum(), deg[Sb].sum())
    return (np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5))


def bootstrap_node_mean(inter_deg: np.ndarray, deg: np.ndarray, S: np.ndarray, rng, n_boot=300) -> Tuple[float, float]:
    if S.size == 0:
        return (np.nan, np.nan)
    denom_all = np.maximum(1, deg)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        Sb = rng.choice(S, size=S.size, replace=True)
        vals[i] = float(np.mean(inter_deg[Sb] / denom_all[Sb]))
    return (np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5))


def perm_values_stub_fast(
    rng, inter_deg: np.ndarray, deg: np.ndarray, idx_by_bin: Dict[object, np.ndarray], target_counts: Dict[object, int], n_perm: int
) -> np.ndarray:
    vals = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        Sperm = sample_by_counts(rng, idx_by_bin, target_counts)
        vals[i] = IER_stub(inter_deg[Sperm].sum(), deg[Sperm].sum())
    return vals


def perm_values_nm_fast(
    rng, inter_deg: np.ndarray, deg: np.ndarray, idx_by_bin: Dict[object, np.ndarray], target_counts: Dict[object, int], n_perm: int
) -> np.ndarray:
    denom = np.maximum(1, deg)
    vals = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        Sperm = sample_by_counts(rng, idx_by_bin, target_counts)
        vals[i] = float(np.mean(inter_deg[Sperm] / denom[Sperm]))
    return vals


def perm_pvalue_one_two_tailed(obs: float, null_vals: np.ndarray) -> Tuple[float, float]:
    arr = null_vals[np.isfinite(null_vals)]
    if arr.size == 0 or not np.isfinite(obs):
        return (np.nan, np.nan)
    geq = np.count_nonzero(arr >= obs)  # one-tailed (≥)
    p_one = (geq + 1) / (arr.size + 1)
    med = np.median(arr)  # two-tailed wrt median
    obs_dev = abs(obs - med)
    devs = np.abs(arr - med)
    geq2 = np.count_nonzero(devs >= obs_dev)
    p_two = (geq2 + 1) / (arr.size + 1)
    return (p_one, min(1.0, p_two))


# ----------------------------- per-dataset job ----------------------------- #
def process_dataset(
    ds: str,
    data_dir: Path,
    proc3_dir: Path,
    out_dir: Path,
    taus: List[float],
    budget_frac: float,
    n_boot: int,
    n_perm: int,
    rseed: int,
) -> pd.DataFrame:
    log.info("Dataset: %s", ds)

    net_path = data_dir / ds / f"{ds}_network.txt"
    n2c_path = data_dir / ds / f"{ds}_node_to_communities.csv"
    met_path = proc3_dir / f"{ds}_all_metrics.csv"

    need(net_path, f"network {ds}")
    need(n2c_path, f"node_to_communities {ds}")
    need(met_path, f"all_metrics {ds}")

    g = load_graph_ncol(net_path)
    names = np.array(g.vs["name"], dtype=str)
    node2comms = load_node2comms_safe(n2c_path)
    C_of = [node2comms.get(u, frozenset()) for u in names]

    dfm = pd.read_csv(met_path)
    id_col = find_id_column(dfm)
    if HV_COL not in dfm.columns:
        raise RuntimeError(f"[{ds}] missing '{HV_COL}' in {met_path}")

    hv_map = dict(zip(dfm[id_col].astype(str), dfm[HV_COL].astype(float)))
    hv = np.array([hv_map.get(u, np.nan) for u in names], dtype=float)
    br_vec = ensure_bridgecc(g, dfm, id_col, names)

    ok = np.isfinite(hv) & np.isfinite(br_vec)
    if ok.sum() < len(hv):
        g = g.subgraph(np.where(ok)[0].tolist())
        names = np.array(g.vs["name"], dtype=str)
        hv = hv[ok]
        br_vec = br_vec[ok]
        C_of = [C_of[i] for i in np.where(ok)[0]]

    n = g.vcount()
    k = max(1, int(round(budget_frac * n)))
    deg = np.array(g.degree(), dtype=np.int32)
    bins_all = degree_quartiles_labels(deg)
    idx_by_bin = build_idx_by_bin(bins_all)

    # Targets
    S_h = np.argsort(hv)[:k]  # tilde_hv (ascending)
    S_b = np.argsort(-br_vec)[:k]  # bridgecc (descending)
    tcounts_h = counts_by_bin(bins_all, S_h)
    tcounts_b = counts_by_bin(bins_all, S_b)

    rows_standard: List[dict] = []
    rows_bins_all: List[dict] = []
    all_rows_legacy = []

    ds_dir = out_dir / ds
    ds_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(rseed)

    for tau in taus:
        inter_deg = precompute_inter_deg_tau(g, C_of, tau)

        ier_h_stub = IER_stub(inter_deg[S_h].sum(), deg[S_h].sum())
        ier_b_stub = IER_stub(inter_deg[S_b].sum(), deg[S_b].sum())
        ier_h_nm = IER_node_mean(inter_deg, deg, S_h)
        ier_b_nm = IER_node_mean(inter_deg, deg, S_b)
        delta_stub = ier_h_stub - ier_b_stub
        delta_nm = ier_h_nm - ier_b_nm

        ci_h_stub = bootstrap_stub(inter_deg, deg, S_h, rng, n_boot=n_boot)
        ci_b_stub = bootstrap_stub(inter_deg, deg, S_b, rng, n_boot=n_boot)
        ci_h_nm = bootstrap_node_mean(inter_deg, deg, S_h, rng, n_boot=n_boot)
        ci_b_nm = bootstrap_node_mean(inter_deg, deg, S_b, rng, n_boot=n_boot)

        perm_vals_h_stub = perm_values_stub_fast(rng, inter_deg, deg, idx_by_bin, tcounts_h, n_perm)
        perm_vals_b_stub = perm_values_stub_fast(rng, inter_deg, deg, idx_by_bin, tcounts_b, n_perm)
        p1_h_stub, p2_h_stub = perm_pvalue_one_two_tailed(ier_h_stub, perm_vals_h_stub)
        p1_b_stub, p2_b_stub = perm_pvalue_one_two_tailed(ier_b_stub, perm_vals_b_stub)
        perm_vals_delta_stub = perm_vals_h_stub - perm_vals_b_stub
        p1_d_stub, p2_d_stub = perm_pvalue_one_two_tailed(delta_stub, perm_vals_delta_stub)

        perm_vals_h_nm = perm_values_nm_fast(rng, inter_deg, deg, idx_by_bin, tcounts_h, n_perm)
        perm_vals_b_nm = perm_values_nm_fast(rng, inter_deg, deg, idx_by_bin, tcounts_b, n_perm)
        p1_h_nm, p2_h_nm = perm_pvalue_one_two_tailed(ier_h_nm, perm_vals_h_nm)
        p1_b_nm, p2_b_nm = perm_pvalue_one_two_tailed(ier_b_nm, perm_vals_b_nm)
        perm_vals_delta_nm = perm_vals_h_nm - perm_vals_b_nm
        p1_d_nm, p2_d_nm = perm_pvalue_one_two_tailed(delta_nm, perm_vals_delta_nm)

        for label in ["Q1", "Q2", "Q3", "Q4"]:
            idx_h = S_h[bins_all[S_h] == label]
            idx_b = S_b[bins_all[S_b] == label]
            if idx_h.size:
                rows_bins_all.append(
                    {
                        "dataset": ds,
                        "tau": tau,
                        "strategy": "hv_s1",
                        "degree_bin": label,
                        "n_nodes": int(idx_h.size),
                        "sum_inter_deg": int(inter_deg[idx_h].sum()),
                        "sum_deg": int(deg[idx_h].sum()),
                        "IER_node_mean": IER_node_mean(inter_deg, deg, idx_h),
                        "IER_stub": IER_stub(inter_deg[idx_h].sum(), deg[idx_h].sum()),
                    }
                )
            if idx_b.size:
                rows_bins_all.append(
                    {
                        "dataset": ds,
                        "tau": tau,
                        "strategy": "bridgecc",
                        "degree_bin": label,
                        "n_nodes": int(idx_b.size),
                        "sum_inter_deg": int(inter_deg[idx_b].sum()),
                        "sum_deg": int(deg[idx_b].sum()),
                        "IER_node_mean": IER_node_mean(inter_deg, deg, idx_b),
                        "IER_stub": IER_stub(inter_deg[idx_b].sum(), deg[idx_b].sum()),
                    }
                )

        # 1) Standard rows (one per strategy)
        rows_standard += [
            {
                "dataset": ds,
                "tau": float(tau),
                "strategy": "hv_s1",
                "IER_node_mean": float(ier_h_nm),
                "IER_node_mean_CI_low": float(ci_h_nm[0]),
                "IER_node_mean_CI_high": float(ci_h_nm[1]),
                "p_perm_nm": float(p2_h_nm),
                "IER_stub": float(ier_h_stub),
                "IER_stub_CI_low": float(ci_h_stub[0]),
                "IER_stub_CI_high": float(ci_h_stub[1]),
                "p_perm_stub": float(p2_h_stub),
                "n_selected": int(k),
                "K_permutations": int(n_perm),
            },
            {
                "dataset": ds,
                "tau": float(tau),
                "strategy": "bridgecc",
                "IER_node_mean": float(ier_b_nm),
                "IER_node_mean_CI_low": float(ci_b_nm[0]),
                "IER_node_mean_CI_high": float(ci_b_nm[1]),
                "p_perm_nm": float(p2_b_nm),
                "IER_stub": float(ier_b_stub),
                "IER_stub_CI_low": float(ci_b_stub[0]),
                "IER_stub_CI_high": float(ci_b_stub[1]),
                "p_perm_stub": float(p2_b_stub),
                "n_selected": int(k),
                "K_permutations": int(n_perm),
            },
        ]

        # 2) Legacy mixed (compatibility)
        all_rows_legacy.append(
            pd.DataFrame(
                [
                    {
                        "dataset": ds,
                        "tau": tau,
                        "strategy": "hv_s1",
                        "k": int(k),
                        "IER_stub": float(ier_h_stub),
                        "IER_stub_ci_low": float(ci_h_stub[0]),
                        "IER_stub_ci_high": float(ci_h_stub[1]),
                        "IER_node_mean": float(ier_h_nm),
                        "IER_nm_ci_low": float(ci_h_nm[0]),
                        "IER_nm_ci_high": float(ci_h_nm[1]),
                        "p_one_stub": float(p1_h_stub),
                        "p_two_stub": float(p2_h_stub),
                        "p_one_nm": float(p1_h_nm),
                        "p_two_nm": float(p2_h_nm),
                    },
                    {
                        "dataset": ds,
                        "tau": tau,
                        "strategy": "bridgecc",
                        "k": int(k),
                        "IER_stub": float(ier_b_stub),
                        "IER_stub_ci_low": float(ci_b_stub[0]),
                        "IER_stub_ci_high": float(ci_b_stub[1]),
                        "IER_node_mean": float(ier_b_nm),
                        "IER_nm_ci_low": float(ci_b_nm[0]),
                        "IER_nm_ci_high": float(ci_b_nm[1]),
                        "p_one_stub": float(p1_b_stub),
                        "p_two_stub": float(p2_b_stub),
                        "p_one_nm": float(p1_b_nm),
                        "p_two_nm": float(p2_b_nm),
                    },
                    {
                        "dataset": ds,
                        "tau": tau,
                        "strategy": "delta(hv-bridgecc)",
                        "k": int(k),
                        "IER_stub": float(delta_stub),
                        "IER_stub_ci_low": np.nan,
                        "IER_stub_ci_high": np.nan,
                        "IER_node_mean": float(delta_nm),
                        "IER_nm_ci_low": np.nan,
                        "IER_nm_ci_high": np.nan,
                        "p_one_stub": float(p1_d_stub),
                        "p_two_stub": float(p2_d_stub),
                        "p_one_nm": float(p1_d_nm),
                        "p_two_nm": float(p2_d_nm),
                    },
                ]
            )
        )

        log.info("[%s τ=%.3f] IER_stub hv=%.4f br=%.4f Δ=%.4f (p2=%.3g) | IER_nm hv=%.4f br=%.4f Δ=%.4f (p2=%.3g)",
                 ds, tau, ier_h_stub, ier_b_stub, delta_stub, p2_d_stub, ier_h_nm, ier_b_nm, delta_nm, p2_d_nm)

    # Writes (per-dataset)
    df_mixed = pd.concat(all_rows_legacy, ignore_index=True)
    out_mixed = ds_dir / f"edge_ratio_{ds}.csv"
    df_mixed.to_csv(out_mixed, index=False)

    df_std = pd.DataFrame(
        rows_standard,
        columns=[
            "dataset",
            "tau",
            "strategy",
            "IER_node_mean",
            "IER_node_mean_CI_low",
            "IER_node_mean_CI_high",
            "p_perm_nm",
            "IER_stub",
            "IER_stub_CI_low",
            "IER_stub_CI_high",
            "p_perm_stub",
            "n_selected",
            "K_permutations",
        ],
    )
    out_std = ds_dir / f"edge_ratio_summary_{ds}.csv"
    df_std.to_csv(out_std, index=False)

    df_bins_all = pd.DataFrame(
        rows_bins_all,
        columns=[
            "dataset",
            "tau",
            "strategy",
            "degree_bin",
            "n_nodes",
            "sum_inter_deg",
            "sum_deg",
            "IER_node_mean",
            "IER_stub",
        ],
    )
    out_bins = ds_dir / f"edge_ratio_by_degree_{ds}.csv"
    df_bins_all.to_csv(out_bins, index=False)

    # Deltas (standard)
    deltas = []
    for tau in sorted(set(df_std["tau"].tolist())):
        hv_row = df_std[(df_std["tau"] == tau) & (df_std["strategy"] == "hv_s1")].iloc[0]
        br_row = df_std[(df_std["tau"] == tau) & (df_std["strategy"] == "bridgecc")].iloc[0]
        deltas.append(
            {
                "dataset": ds,
                "tau": float(tau),
                "Delta_IER_node_mean": float(hv_row["IER_node_mean"] - br_row["IER_node_mean"]),
                "p_perm_nm_delta": np.nan,
                "Delta_IER_stub": float(hv_row["IER_stub"] - br_row["IER_stub"]),
                "p_perm_stub_delta": np.nan,
            }
        )
    df_delta = pd.DataFrame(
        deltas,
        columns=["dataset", "tau", "Delta_IER_node_mean", "p_perm_nm_delta", "Delta_IER_stub", "p_perm_stub_delta"],
    )
    out_delta = ds_dir / f"edge_ratio_deltas_{ds}.csv"
    df_delta.to_csv(out_delta, index=False)

    return df_std


# ----------------------------- CLI ----------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="calc_edge_ratio", description="Inter-Edge Ratio analysis with Jaccard τ-grid")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--proc3-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/edge_ratio"))
    p.add_argument("--taus", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    p.add_argument("--budget", type=float, default=0.10, help="Node budget fraction (default 0.10)")
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    std_blocks = []
    for ds in args.datasets:
        df_std = process_dataset(
            ds=ds,
            data_dir=args.data_dir,
            proc3_dir=args.proc3_dir,
            out_dir=args.out_dir,
            taus=args.taus,
            budget_frac=args.budget,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            rseed=args.seed,
        )
        std_blocks.append(df_std)

    # Consolidated (standard)
    df_all_std = pd.concat(std_blocks, ignore_index=True)
    out_all_std = args.out_dir / "summary_all_datasets_standard.csv"
    df_all_std.to_csv(out_all_std, index=False)
    log.info("Written: %s", out_all_std)

    # Consolidated (legacy)
    mixtos = []
    for ds in args.datasets:
        p = args.out_dir / ds / f"edge_ratio_{ds}.csv"
        if p.exists():
            mixtos.append(pd.read_csv(p))
    if mixtos:
        df_all_mixto = pd.concat(mixtos, ignore_index=True)
        out_all_mixto = args.out_dir / "summary_all_datasets.csv"
        df_all_mixto.to_csv(out_all_mixto, index=False)
        log.info("Written: %s", out_all_mixto)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
