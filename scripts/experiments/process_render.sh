#!/bin/bash
# Unpack the Kaggle renders of one data set and run the pixel controls on it.
# Usage: scripts/experiments/process_render.sh <set>, for example v2r or v2ns.
# The splits use the same rules and the same seed as v2, so image i is in the same split.
set -euo pipefail
SET="$1"
PY=.venv/bin/python
DATA="data/$SET"
OUT="reports/experiments/$SET"

if [ ! -f "$DATA/labels.csv" ]; then
  $PY tools/kaggle/collect.py runs/kaggle/render "dd3d-render-$SET-" "$DATA"
  mv "$DATA/images/labels.csv" "$DATA/labels.csv"
fi
for s in Sphere SmoothCylinder SmoothCube_v2; do
  echo "$s: $(find "$DATA/images/$s" -name '*.png' | wc -l) images"
done
$PY scripts/data/make_v2_splits.py --labels "$DATA/labels.csv" --pos-labels none --out "$DATA/splits"
$PY scripts/experiments/feature_scan.py --features "data/features/$SET" --runs none --splits "$DATA/splits" \
  --images "$DATA/images" --out "$OUT/scan" | grep -v '^skip'
$PY scripts/experiments/pixel_cues.py --images "$DATA/images" --scenes none --splits "$DATA/splits" \
  --out "$OUT/pixel_cues"
