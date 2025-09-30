#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Auxiliary pipeline runner (comparatives): ghv and structural baselines.

If `overlap_neigh` is requested and `<key>_node_to_communities.csv` is missing,
this runner will auto-generate it using `lfr/gen_node_communities.py` provided
`<key>_communities.txt` exists.
"""

from __future__ import annotations
import argparse, subprocess
from pathlib import Path
from typing import List

def parse_args():
    p = argparse.ArgumentParser(prog="run_pipeline_aux",
        description="Compute ghv (optional) and append structural baselines onto per-node tables.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--proc1-dir", type=Path, default=Path("outputs/proc1"))
    p.add_argument("--proc3-dir", type=Path, default=Path("outputs/proc3"))
    p.add_argument("--with-ghv", action="store_true")
    p.add_argument("--metrics", nargs="+", default=["bridgecc"])
    p.add_argument("--ci-radius", type=int, default=2)
    p.add_argument("--metrics-template", type=str, default=None)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()

def run(cmd: List[str]):
    print(">>", " ".join(map(str, cmd)))
    subprocess.check_call([str(x) for x in cmd])

def ensure_node2comms_if_needed(data_dir: Path, key: str, need: bool):
    if not need:
        return
    ddir = data_dir / key
    n2c = ddir / f"{key}_node_to_communities.csv"
    if n2c.exists():
        return
    comms = ddir / f"{key}_communities.txt"
    edges = ddir / f"{key}_network.txt"
    if not comms.exists():
        raise FileNotFoundError(f"Missing {n2c} and {comms}; cannot compute overlap_neigh for {key}.")
    run([
        "python", "lfr/gen_node_communities.py",
        "--network-file", edges,
        "--communities-file", comms,
        "--out-dir", data_dir,
        "--dataset-key", key,
    ])

def main():
    args = parse_args()
    args.proc1_dir.mkdir(parents=True, exist_ok=True)
    args.proc3_dir.mkdir(parents=True, exist_ok=True)

    need_overlap = ("overlap_neigh" in args.metrics)

    for key in args.datasets:
        ddir = args.data_dir / key
        edges = ddir / f"{key}_network.txt"
        comms = ddir / f"{key}_communities.txt"

        # Optional ghv
        if args.with_ghv:
            run([
                "python", "aux/calc_ghv.py",
                "--data-dir", args.data_dir, "--dataset-key", key,
                "--results-dir", args.proc1_dir,
                "--methods", "s2",
                "--force",
                "--log-level", args.log_level
            ])

        # Ensure node→comms if needed
        ensure_node2comms_if_needed(args.data_dir, key, need_overlap)

        # Structural baselines on top of hv table
        base_csv = (args.metrics_template.format(key=key) if args.metrics_template
                    else str(args.proc3_dir / f"{key}_hv_metrics.csv"))
        cmd = [
            "python", "aux/calc_all_metrics.py",
            "--data-dir", args.data_dir,
            "--dataset-key", key,
            "--proc3-dir", args.proc3_dir,
            "--metrics-file", base_csv,
            "--network-file", edges,
            "--metrics", *args.metrics,
            "--log-level", args.log_level,
            "--force",
        ]
        if "overlap_neigh" in args.metrics:
            cmd += ["--communities-file", comms]
        if "ci" in args.metrics:
            cmd += ["--ci-radius", str(args.ci_radius)]
        run(cmd)

if __name__ == "__main__":
    main()
