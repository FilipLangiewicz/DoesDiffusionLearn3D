#!/bin/bash
# Process each render set and each feature set when all its Kaggle jobs are done.
# The script checks every 5 minutes and runs one set at a time, because the laptop has little RAM.
# A marker file .processed in the report folder stops a second run of the same set.
# Start it detached from the session, with the log in runs/kaggle/logs:
#   setsid nohup scripts/experiments/auto_process.sh >> runs/kaggle/logs/auto_process.log 2>&1 &
# After a failure, the marker .failed_<kind> stops the next tries. Delete it to try again.
cd "$(dirname "$0")/../.."

all_done() {
  # $1 = job folder root, $2 = slug prefix, $3 = number of jobs
  [ "$(find "$1" -maxdepth 2 -path "*/$2*/.done" | wc -l)" -ge "$3" ]
}

while true; do
  work=0
  for set in v2ns v2r v2rns v2rso; do
    mark="reports/experiments/$set/.processed_render"
    if [ ! -f "$mark" ] && [ ! -f "reports/experiments/$set/.failed_render" ] && all_done runs/kaggle/render "dd3d-render-$set-" 6; then
      echo "[$(date '+%F %T')] render set $set: start"
      if scripts/experiments/process_render.sh "$set"; then
        mkdir -p "$(dirname "$mark")" && date > "$mark"
        echo "[$(date '+%F %T')] render set $set: done"
      else
        mkdir -p "reports/experiments/$set" && date > "reports/experiments/$set/.failed_render"
        echo "[$(date '+%F %T')] render set $set: FAILED"
      fi
      work=1
    fi
  done
  for set in v1 v2 v2r v2rns v2rso; do
    mark="reports/experiments/$set/.processed_features"
    # A feature set needs its renders first (except v1, which has its own images).
    if [ "$set" != v1 ] && [ "$set" != v2 ] && [ ! -f "reports/experiments/$set/.processed_render" ]; then
      continue
    fi
    if [ ! -f "$mark" ] && [ ! -f "reports/experiments/$set/.failed_features" ] && all_done runs/kaggle/extract "dd3d-extract-$set-" 6; then
      echo "[$(date '+%F %T')] feature set $set: start"
      if scripts/experiments/process_set.sh "$set"; then
        mkdir -p "$(dirname "$mark")" && date > "$mark"
        echo "[$(date '+%F %T')] feature set $set: done"
      else
        mkdir -p "reports/experiments/$set" && date > "reports/experiments/$set/.failed_features"
        echo "[$(date '+%F %T')] feature set $set: FAILED"
      fi
      work=1
    fi
  done
  [ "$work" = 0 ] && sleep 300
done
