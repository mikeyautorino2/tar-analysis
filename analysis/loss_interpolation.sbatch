#!/bin/bash
# 1-D loss interpolation between official TAR-Bio-v2 (alpha=0) and the perturbation model (alpha=1).
#SBATCH --job-name=loss-interp
#SBATCH --account=ddp477
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=exp-19-14
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .env
ulimit -c 0

python -u analysis/loss_interpolation.py \
  --model_a lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2 --name_a "TAR-Bio-v2 official" \
  --model_b "$SAVE_MODELS_DIR/Llama-3-8B-Instruct-TAR-Bio-v2-perturb_rms1pct_v1" --name_b "perturb-rms1pct-v1" \
  --out_dir results/loss_interpolation/official_vs_perturb "$@"
