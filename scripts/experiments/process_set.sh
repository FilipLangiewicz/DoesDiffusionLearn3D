#!/bin/bash
# Unpack the Kaggle features of one data set, fit all probes and draw the scan.
# Usage: scripts/experiments/process_set.sh <set>, with set v1, v2, v2r, v2rns or v2rso.
# For v2pos, append its features to v2 and run the v2 scan again (test_pos split).
set -euo pipefail
SET="$1"
PY=.venv/bin/python
FEAT="data/features/$SET"
OUT="reports/experiments/$SET"

case "$SET" in
  v1) SPLITS=data/splits; IMAGES=data/final/images; EXTRA=(--scenes data/final/scenes) ;;
  v2) SPLITS=data/v2/splits; IMAGES=data/v2/images; EXTRA=() ;;
  *) SPLITS="data/$SET/splits"; IMAGES="data/$SET/images"; EXTRA=() ;;
esac

if [ ! -d "$FEAT" ]; then
  $PY tools/kaggle/collect.py runs/kaggle/extract "dd3d-extract-$SET-" "$FEAT" --merge
  # The chunk files are in the merged files now; keep only the merged files.
  find "$FEAT" \( -name 'chunk_*.npy' -o -name 'chunk_*_ids.txt' \) -delete
fi
RUNS=$(find "$FEAT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | paste -sd,)
$PY scripts/experiments/feature_scan.py --features "$FEAT" --runs "$RUNS" --splits "$SPLITS" \
  --images "$IMAGES" "${EXTRA[@]}" --out "$OUT/scan"
$PY scripts/experiments/plot_scan.py --scan "$OUT/scan/scan.csv" --title "$SET" --out "$OUT/scan_summary"
