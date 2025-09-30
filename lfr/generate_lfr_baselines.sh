#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
if [[ $# -lt 2 ]]; then echo "Usage: $0 <LFR_BENCHMARK_EXE> <OUT_DIR>"; exit 1; fi
LFR_BIN="$1"; OUT_DIR="$2"; mkdir -p "$OUT_DIR"
N=10000; K=20; MAXK=50; T1=2.5; T2=1.5; MINC=40; MAXC=400
SEED=424242; echo "$SEED" > time_seed.dat
declare -a NAMES=("LFR_low" "LFR_mid" "LFR_high")
declare -a MU=(0.15 0.35 0.65)
declare -a ON=(1500 4000 7000)
declare -a OM=(2 2 3)
for idx in 0 1 2; do
  NAME="${NAMES[$idx]}"
  echo "[*] Generating ${NAME} (mu=${MU[$idx]}, on=${ON[$idx]}, om=${OM[$idx]})"
  "$LFR_BIN" -N $N -k $K -maxk $MAXK -mu ${MU[$idx]} -t1 $T1 -t2 $T2 -minc $MINC -maxc $MAXC -on ${ON[$idx]} -om ${OM[$idx]}
  mv -f network.dat    "${OUT_DIR}/${NAME}_network.dat"
  mv -f community.dat  "${OUT_DIR}/${NAME}_communities.dat"
  mv -f statistics.dat "${OUT_DIR}/${NAME}_statistics.dat"
done
echo "[✓] Done. Raw LFR baselines in: $OUT_DIR"
