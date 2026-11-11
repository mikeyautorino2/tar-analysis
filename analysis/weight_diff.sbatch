#!/bin/bash
# Per-layer weight differences between two checkpoints (CPU work; the shared GPU partition is the only one this account can use).
#SBATCH --job-name=weight-diff
#SBATCH --account=ddp477
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .env
ulimit -c 0

python -u analysis/weight_diff.py "$@"
