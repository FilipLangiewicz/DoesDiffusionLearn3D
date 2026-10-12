#!/bin/bash
# Corrected extraction on Athena: 30 array tasks of 100 images each.
#SBATCH -A plgzzsn2026-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 01:30:00
#SBATCH --array=0-29
#SBATCH -o logs/extract_v1_%A_%a.log

set -euo pipefail
cd "${PROJECT:-$SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
eval "$(conda shell.bash hook)"
conda activate "${DD3D_ENV:-diffusion_3d_env}"
dd3d extract --config configs/extract.yaml
