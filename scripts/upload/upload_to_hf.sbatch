#!/bin/bash
# Convert the merged TAR checkpoint to bf16, add tokenizer + model card, sanity-check, and upload to HF.
#SBATCH --job-name=hf-upload
#SBATCH --account=ddp477
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .env
ulimit -c 0

# Defaults reproduce the diverse-params v1 upload. Override, e.g.:
#   NAME=<hf repo name> MODEL_DIR=<merged model dir> EXTRA="<files/dirs>" sbatch scripts/upload/upload_to_hf.sbatch
NAME=${NAME:-Llama-3-8B-Instruct-TAR-Bio-diverse-params-v1}
MODEL_DIR=${MODEL_DIR:-$SAVE_MODELS_DIR/Llama-3-8B-Instruct-TAR-Bio-diverse_sft_diverse_params_v1}
python scripts/upload/upload_to_hf.py \
  --model_dir "$MODEL_DIR" \
  --out_dir "/expanse/lustre/projects/ddp477/$USER/hf_upload/$NAME" \
  --repo_id "mikeyautorino2/$NAME" \
  --readme "model_cards/$NAME.md" \
  ${EXTRA:+--extra $EXTRA}
