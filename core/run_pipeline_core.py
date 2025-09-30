#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Core pipeline runner (paper main):
1) calc_hv.py
2) calc_w_inter_curves.py  → (auto-generates node→communities if missing)
3) calc_sensitivity_curves.py

Datasets are assumed preprocessed (Zenodo). If `<key>_node_to_communities.csv`
is missing but `<key>_communities.txt` exists, this runner will invoke:
    python lfr/gen_node_communities.py --network-file ... --communities-file ... --out-dir datasets --dataset-key <key>
"""

from __future__ import annotations
import argparse, subprocess
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(prog="run_pipeline_core",
        description="Execute the core pipeline: hv → W_inter curves → sensitivity curves + plots.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--proc3-dir", type=Path, default=Path("outputs/proc3"))
    p.add_argument("--meta-dir", type=Path, default=Path("outputs/meta_graph_eval"))
    p.add_argument("--sens-dir", type=Path, default=Path("outputs/sensitivity"))
    p.add_argument("--curves-dir", type=Path, required=True)
    p.add_argument("--random-reps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--no-winter-plot", action="store_true")
    p.add_argument("--no-sens-plot", action="store_true")
    return p.parse_args()

def run(cmd):
    print(">>", " ".join(map(str, cmd)))
    subprocess.check_call([str(x) for x in cmd])

def ensure_node2comms(data_dir: Path, key: str):
    ddir = data_dir / key
    n2c = ddir / f"{key}_node_to_communities.csv"
    if n2c.exists():
        return n2c
    comms = ddir / f"{key}_communities.txt"
    edges = ddir / f"{key}_network.txt"
    if not comms.exists():
        raise FileNotFoundError(f"Missing {n2c} and {comms}. Provide communities or precomputed node→comms for {key}.")
    run([
        "python", "lfr/gen_node_communities.py",
        "--network-file", edges,
        "--communities-file", comms,
        "--out-dir", data_dir,
        "--dataset-key", key,
    ])
    if not n2c.exists():
        raise RuntimeError(f"Failed to generate node→communities for {key}.")
    return n2c

def main():
    args = parse_args()
    args.proc3_dir.mkdir(parents=True, exist_ok=True)
    args.meta_dir.mkdir(parents=True, exist_ok=True)
    args.sens_dir.mkdir(parents=True, exist_ok=True)

    # 1) hv_s1 per dataset
    for key in args.datasets:
        run([
            "python", "core/calc_hv.py",
            "--data-dir", args.data_dir, "--dataset-key", key,
            "--results-dir", args.proc3_dir,
            "--methods", "s1",
            "--log-level", args.log_level
        ])

    # 2) W_inter curves per dataset (ensure node→comms)
    for key in args.datasets:
        ddir = args.data_dir / key
        edges = ddir / f"{key}_network.txt"
        n2c = ensure_node2comms(args.data_dir, key)
        hv_tab = args.proc3_dir / f"{key}_hv_metrics.csv"
        run([
            "python", "core/calc_w_inter_curves.py",
            "--edges", edges,
            "--node2comms", n2c,
            "--metrics-file", hv_tab,
            "--out-dir", args.meta_dir,
            "--random-reps", args.random_reps,
            "--log-level", args.log_level,
            "--no-plot"
        ])

    if not args.no_winter_plot and len(args.datasets) == 4:
        run([
            "python", "core/plot_w_inter_curves_2x2.py",
            "--datasets", *args.datasets,
            "--in-dir", args.meta_dir,
            "--out-file", args.meta_dir / "meta_eval_grid_2x2_norm.png",
        ])

    # 3) Sensitivity curves (reads processed homophily/GCC curves; attaches hv metrics by template)
    run([
        "python", "core/calc_sensitivity_curves.py",
        "--datasets", *args.datasets,
        "--data-dir", args.data_dir,
        "--proc3-dir", args.proc3_dir,
        "--curves-dir", args.curves_dir,
        "--out-dir", args.sens_dir,
        "--metrics-template", str(args.proc3_dir / "{key}_hv_metrics.csv"),
        "--log-level", args.log_level
    ])

    if not args.no_sens_plot:
        run([
            "python", "core/plot_sensitivity_with_gcc.py",
            "--curves-file", args.sens_dir / "summary_curves.csv",
            "--markers-file", args.sens_dir / "summary_markers.csv",
            "--out-file", args.sens_dir / "sensitivity_panel_with_gcc.png",
        ])

if __name__ == "__main__":
    main()
