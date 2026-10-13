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
  # Pixel transfer tests, when their render sets are ready.
  done_r() { [ -f "reports/experiments/$1/.processed_render" ]; }
  if done_r v2r && [ ! -f reports/experiments/transfer/.pos_pixels ]; then
    echo "[$(date '+%F %T')] transfer v2 -> v2r (pixels): start"
    .venv/bin/python scripts/experiments/transfer.py --fit v2 --tests v2r --out reports/experiments/transfer/position_pixels \
      && date > reports/experiments/transfer/.pos_pixels
    work=1
  fi
  if done_r v2r && done_r v2rns && done_r v2rso && [ ! -f reports/experiments/transfer/.cue_pixels ]; then
    echo "[$(date '+%F %T')] transfer v2r -> v2rns, v2rso (pixels): start"
    .venv/bin/python scripts/experiments/transfer.py --fit v2r --tests v2rns,v2rso --out reports/experiments/transfer/cue_pixels \
      && date > reports/experiments/transfer/.cue_pixels
    work=1
  fi
  # Causal test on v2 (block 10, t = 400). It starts only after the test job of the
  # pipeline wrote its images, so a broken job never uses GPU time twice.
  test_tar=runs/kaggle/intervene/dd3d-intervene-test/output/test.tar
  if [ -f reports/experiments/v2/.processed_features ] && [ -f "$test_tar" ] \
    && [ "$(stat -c %s "$test_tar")" -gt 1000000 ] && [ ! -f reports/experiments/causal/.v2b10_queued ]; then
    echo "[$(date '+%F %T')] causal test v2 block 10: plan and job"
    mkdir -p reports/experiments/causal
    if .venv/bin/python scripts/experiments/make_intervention_plan.py \
        --features data/features/v2/sana/block_10/t_400 --n-images 60 --out runs/kaggle/plans/v2_b10_t400 \
      && cp runs/kaggle/plans/v2_b10_t400/plan.npz runs/kaggle/plans/dataset/plan_v2_b10_t400.dat \
      && /home/flang/tools/kag datasets version -p runs/kaggle/plans/dataset -m "plan v2 block 10 t 400" -q \
      && sleep 180 \
      && .venv/bin/python tools/kaggle/make_jobs.py intervene --name v2b10 --plan plan_v2_b10_t400.dat \
        $(for i in 0 1 2 3 4 5; do printf -- " --kernel jankombajn/dd3d-render-v2-%s" "$i"; done) \
        --config '{"block": 10, "sigma0": 0.7, "steps": 20}' --ref "$(git rev-parse --short origin/feat/kaggle-pipeline)" \
        --out-root runs/kaggle/intervene; then
      # The causal test goes right after v2pos, before the long v2r runs.
      sed -i 's|^\(runs/kaggle/extract/dd3d-extract-v2pos-0 GPU.*\)$|\1\nruns/kaggle/intervene/dd3d-intervene-v2b10 GPU|' runs/kaggle/queue.txt
      date > reports/experiments/causal/.v2b10_queued
      echo "[$(date '+%F %T')] causal test v2 block 10: queued"
    else
      date > reports/experiments/causal/.v2b10_queued
      echo "[$(date '+%F %T')] causal test v2 block 10: FAILED"
    fi
    work=1
  fi
  [ "$work" = 0 ] && sleep 300
done
