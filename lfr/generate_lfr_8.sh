#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
if [[ $# -lt 2 ]]; then echo "Usage: $0 <LFR_BENCHMARK_EXE> <OUT_DIR>"; exit 1; fi
LFR_BIN="$1"; OUT_DIR="$2"; mkdir -p "$OUT_DIR"
N=10000; K=12; MAXK=50; MU=0.15; T1=2.5; T2=1.5; MINC=20; MAXC=200; OM=5
START=5000; END=7000; STEP=250
SEED=123456; echo "$SEED" > time_seed.dat
i=0
for ON in $(seq $START $STEP $END); do
  i=$((i+1)); NAME="LFR${i}"
  echo "[*] Generating $NAME (on=$ON, om=$OM, mu=$MU)"
  "$LFR_BIN" -N $N -k $K -maxk $MAXK -mu $MU -t1 $T1 -t2 $T2 -minc $MINC -maxc $MAXC -on $ON -om $OM
  mv -f network.dat    "${OUT_DIR}/${NAME}_network.dat"
  mv -f community.dat  "${OUT_DIR}/${NAME}_communities.dat"
  mv -f statistics.dat "${OUT_DIR}/${NAME}_statistics.dat"
done
echo "[✓] Done. Raw LFR files in: $OUT_DIR"
