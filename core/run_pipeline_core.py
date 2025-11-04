#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Core pipeline runner for the paper (main figures).

Overview
--------
This runner executes the main analysis steps that produce the primary figures:
1) hv_s1 per dataset (needed by subsequent steps).
2) Meta-graph evaluation: compute W_inter(p) curves and render the 2×4 (or 2×2) panel.
3) IER analysis (BridgeCC vs \\tilde{h}_v): compute per-dataset CSVs and render the forest plot.

Notes
-----
- Supplemental analyses (e.g., functional sensitivity panels) are intentionally excluded
  from this runner; see `supplement/run_pipeline_supplement.py`.
- This script assumes preprocessed datasets in `--data-dir`, following:
    <data-dir>/<key>/<key>_network.txt
    <data-dir>/<key>/<key>_communities.txt
    <data-dir>/<key>/<key>_node_to_communities.csv   (auto-generated if missing)
- The hv_s1 table is expected at `<proc3-dir>/<key>_hv_metrics.csv` (created by calc_hv.py).

Examples
--------
python core/run_pipeline_core.py \
  --datasets lj youtube dblp twitch amazon deezer so github \
  --data-dir datasets \
  --proc3-dir outputs/proc3 \
  --meta-dir outputs/meta_graph_eval \
  --curves-dir outputs/curves \
  --edge-ratio-dir outputs/edge_ratio \
  --taus 0.0 0.1 0.2 0.3 \
  --budget 0.10 \
  --n-boot 300 --n-perm 500 --seed 42 \
  --log-level INFO
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import logging

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger("run_pipeline_core")


# ----------------------------- CLI ----------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pipeline_core",
        description="Execute the core pipeline: hv → W_inter curves (Fig.1) → IER + forest (main figure).",
    )
    # datasets & base dirs
    p.add_argument("--datasets", nargs="+", required=True, help="Dataset keys in desired order.")
    p.add_argument("--data-dir", type=Path, required=True, help="Base directory of datasets.")
    p.add_argument("--proc3-dir", type=Path, default=Path("outputs/proc3"))
    p.add_argument("--meta-dir", type=Path, default=Path("outputs/meta_graph_eval"))
    p.add_argument("--curves-dir", type=Path, default=Path("outputs/curves"))
    p.add_argument("--edge-ratio-dir", type=Path, default=Path("outputs/edge_ratio"))

    # plotting toggles
    p.add_argument("--no-winter-plot", action="store_true", help="Skip the W_inter grid plot.")
    p.add_argument("--no-forest-plot", action="store_true", help="Skip forest plot.")

    # IER params
    p.add_argument("--taus", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3], help="τ grid for Jaccard threshold.")
    p.add_argument("--budget", type=float, default=0.10, help="Node budget fraction (default 0.10).")
    p.add_argument("--n-boot", type=int, default=300, help="Bootstrap iterations.")
    p.add_argument("--n-perm", type=int, default=500, help="Permutation iterations (degree-stratified).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # logging
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ----------------------------- small utils ----------------------------- #
def run(cmd):
    """Echo and run a subprocess command, raising on failure."""
    print(">>", " ".join(map(str, cmd)))
    subprocess.check_call([str(x) for x in cmd])


def ensure_node2comms(data_dir: Path, key: str) -> Path:
    """Generate node→communities CSV if missing (requires communities.txt)."""
    ddir = data_dir / key
    n2c = ddir / f"{key}_node_to_communities.csv"
    if n2c.exists():
        return n2c
    comms = ddir / f"{key}_communities.txt"
    edges = ddir / f"{key}_network.txt"
    if not comms.exists():
        raise FileNotFoundError(
            f"Missing both node→communities and communities for {key}: {n2c} / {comms}"
        )
    run(
        [
            "python",
            "lfr/gen_node_communities.py",
            "--network-file",
            edges,
            "--communities-file",
            comms,
            "--out-dir",
            data_dir,
            "--dataset-key",
            key,
        ]
    )
    if not n2c.exists():
        raise RuntimeError(f"Failed to generate node→communities for {key}.")
    return n2c


# ----------------------------- main steps ----------------------------- #
def step_hv(args: argparse.Namespace):
    """1) Compute hv_s1 per dataset → <proc3-dir>/<key>_hv_metrics.csv"""
    for key in args.datasets:
        run(
            [
                "python",
                "core/calc_hv.py",
                "--data-dir",
                args.data_dir,
                "--dataset-key",
                key,
                "--results-dir",
                args.proc3_dir,
                "--methods",
                "s1",
                "--log-level",
                args.log_level,
            ]
        )


def step_winter_curves(args: argparse.Namespace):
    """2) Compute W_inter curves and (optionally) plot the grid."""
    # 2a) Curves per dataset
    for key in args.datasets:
        ddir = args.data_dir / key
        edges = ddir / f"{key}_network.txt"
        n2c = ensure_node2comms(args.data_dir, key)
        hv_tab = args.proc3_dir / f"{key}_hv_metrics.csv"
        run(
            [
                "python",
                "core/calc_w_inter_curves.py",
                "--edges",
                edges,
                "--node2comms",
                n2c,
                "--metrics-file",
                hv_tab,
                "--out-dir",
                args.meta_dir,
                "--random-reps",
                "20",
                "--log-level",
                args.log_level,
                "--no-plot",
            ]
        )

    # 2b) Grid plot (Fig. 1)
    if args.no_winter_plot:
        log.info("Skipping W_inter grid plot (--no-winter-plot).")
        return

    if len(args.datasets) == 8:
        run(
            [
                "python",
                "core/plot_w_inter_curves_2x4.py",
                "--datasets",
                *args.datasets,
                "--in-dir",
                args.meta_dir,
                "--out-file",
                args.meta_dir / "meta_eval_grid_4x2_norm.png",
                "--dpi",
                "300",
                "--log-level",
                args.log_level,
            ]
        )
    elif len(args.datasets) == 4:
        # keep optional compatibility if present in repo
        run(
            [
                "python",
                "core/plot_w_inter_curves_2x2.py",
                "--datasets",
                *args.datasets,
                "--in-dir",
                args.meta_dir,
                "--out-file",
                args.meta_dir / "meta_eval_grid_2x2_norm.png",
                "--dpi",
                "300",
                "--log-level",
                args.log_level,
            ]
        )
    else:
        log.warning("W_inter grid plot expects 4 or 8 datasets; got %d. Skipping.", len(args.datasets))


def step_metrics_and_ier(args: argparse.Namespace):
    """3) IER prerequisites (metrics), IER computation, and forest plot."""
    args.edge_ratio_dir.mkdir(parents=True, exist_ok=True)

    # 3a) Ensure structural metrics (BridgeCC, etc.) per dataset
    for key in args.datasets:
        run(
            [
                "python",
                "core/calc_all_metrics.py",
                "--data-dir",
                args.data_dir,
                "--dataset-key",
                key,
                "--proc3-dir",
                args.proc3_dir,
                "--metrics",
                "bridgecc",
                "--force",
                "--log-level",
                args.log_level,
            ]
        )

    # 3b) IER compute and consolidate
    run(
        [
            "python",
            "core/calc_edge_ratio.py",
            "--datasets",
            *args.datasets,
            "--data-dir",
            args.data_dir,
            "--proc3-dir",
            args.proc3_dir,
            "--out-dir",
            args.edge_ratio_dir,
            "--taus",
            *[f"{t:.2f}" for t in args.taus],
            "--budget",
            f"{args.budget:.2f}",
            "--n-boot",
            str(args.n_boot),
            "--n-perm",
            str(args.n_perm),
            "--seed",
            str(args.seed),
            "--log-level",
            args.log_level,
        ]
    )

    # 3c) Forest plot (main figure)
    if args.no_forest_plot:
        log.info("Skipping forest plot (--no-forest-plot).")
        return

    run(
        [
            "python",
            "core/forest_plot.py",
            "--in-dir",
            args.edge_ratio_dir,
            "--datasets",
            *args.datasets,
            "--tau",
            f"{max(args.taus):.2f}",
            "--out-file",
            args.edge_ratio_dir / "figs" / "forest_meta_bw.pdf",
            "--log-level",
            args.log_level,
        ]
    )


# ----------------------------- entrypoint ----------------------------- #
def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FMT)

    # Ensure output dirs exist
    args.proc3_dir.mkdir(parents=True, exist_ok=True)
    args.meta_dir.mkdir(parents=True, exist_ok=True)
    args.curves_dir.mkdir(parents=True, exist_ok=True)
    args.edge_ratio_dir.mkdir(parents=True, exist_ok=True)

    log.info("Datasets: %s", ", ".join(args.datasets))
    log.info("Data dir: %s", args.data_dir)

    # Steps
    step_hv(args)                # 1) hv_s1 per dataset
    step_winter_curves(args)     # 2) W_inter curves + grid plot (Fig. 1)
    step_metrics_and_ier(args)   # 3) IER + forest (main result)

    log.info("Pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
