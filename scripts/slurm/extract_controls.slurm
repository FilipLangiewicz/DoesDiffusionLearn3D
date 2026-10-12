#!/bin/bash
# Controls on Athena: random weights and the old noise rule (10 tasks each).
#SBATCH -A plgzzsn2026-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 01:00:00
#SBATCH --array=0-19
#SBATCH -o logs/extract_controls_%A_%a.log

set -euo pipefail
cd "${PROJECT:-$SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
eval "$(conda shell.bash hook)"
conda activate "${DD3D_ENV:-diffusion_3d_env}"
if [ "$SLURM_ARRAY_TASK_ID" -lt 10 ]; then
  dd3d extract --config configs/extract_random_init.yaml --task-id "$SLURM_ARRAY_TASK_ID"
else
  dd3d extract --config configs/extract_legacy_noise.yaml --task-id "$((SLURM_ARRAY_TASK_ID - 10))"
fi
