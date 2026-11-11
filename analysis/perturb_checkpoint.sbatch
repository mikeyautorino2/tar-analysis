#!/bin/bash
# Perturbed copies of official TAR-Bio-v2 and diverse-params v1 (0.1% and 1% of each tensor's RMS)
# for the Qi et al. attack evaluation. CPU work; the shared GPU partition is the only one this account can use.
#SBATCH --job-name=perturb-ckpt
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

M=/expanse/lustre/projects/ddp477/mautorino/eval_models
for model in TAR-Bio-v2-official TAR-Bio-diverse-params-v1; do
  if [ "$model" = TAR-Bio-v2-official ]; then
    source_ckpt=lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2
  else
    source_ckpt=$SAVE_MODELS_DIR/Llama-3-8B-Instruct-TAR-Bio-diverse_sft_diverse_params_v1
  fi
  for scale in 0.001 0.01; do
    tag=$([ "$scale" = 0.001 ] && echo 0p1pct || echo 1pct)
    python -u analysis/perturb_checkpoint.py \
      --source "$source_ckpt" --link_from "$M/$model" \
      --out_dir "$M/$model-noise$tag" --scale $scale --seed 0
  done
done
ls -la $M
