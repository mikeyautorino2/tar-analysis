#!/bin/bash
# TAR-Bio-v2 on 1 node x 4 H100 (gradient checkpointing keeps it within memory; NVLink keeps FSDP fast).
#SBATCH --job-name=tar-bio-v2
#SBATCH --account=ddp477
#SBATCH --partition=nairr-gpu
#SBATCH --qos=nairr-gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # one accelerate launcher; it spawns 4 workers
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=exp-19-14   # CUDA failed to initialize there

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .env
ulimit -c 0   # no core dumps: a crashed 8B run writes tens of GB into the repo folder in home

export ExACCEL_CONFIG="$SLURM_SUBMIT_DIR/configs/accel_config_4_gpu.yaml"
export ACCEL_LAUNCH_ARGS=""
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Submit from the repo root. RUN_SCRIPT picks the config (default scripts/train/run_tar_bio.sh), e.g.
#   RUN_SCRIPT=scripts/train/run_tar_bio_perturb.sh sbatch scripts/train/run_tar_bio_1x4.sbatch
# Extra arguments override it, e.g. sbatch scripts/train/run_tar_bio_1x4.sbatch --max_steps 3 --expname test
RUN_SCRIPT=${RUN_SCRIPT:-scripts/train/run_tar_bio.sh}
echo "Config: $RUN_SCRIPT $*"
sh "$RUN_SCRIPT" "$@"

# Merge the per-rank FSDP shards into a regular HF checkpoint (CPU only).
# The last --new_model_name / --expname wins, so check overrides before the script's defaults.
NAME=$( { grep -oP '(?<=--new_model_name )\S+' "$RUN_SCRIPT"; echo "$*" | grep -oP '(?<=--new_model_name )\S+' || true; } | tail -1)
EXP=$( { echo latest; grep -oP '(?<=--expname )\S+' "$RUN_SCRIPT"; echo "$*" | grep -oP '(?<=--expname )\S+' || true; } | tail -1)
OUT="$SAVE_MODELS_DIR/${NAME}_${EXP}"
echo "Merging FSDP shards in $OUT"
accelerate merge-weights "$OUT/pytorch_model_fsdp_0" "$OUT"
ls -lh "$OUT"

# Weight plots: per-attack displacement (if tracked) and final model vs its starting checkpoint
if [ -f "$OUT/attack_displacement.csv" ]; then
  python analysis/plot_attack_displacement.py "$OUT" || echo "Attack displacement plotting failed (rerun analysis/plot_attack_displacement.py $OUT)"
fi
BASE=$(grep -oP '(?<=--base_model_name )\S+' "$RUN_SCRIPT")
python -u analysis/weight_diff.py --base "$BASE" --tar "$OUT" --out_dir "$OUT/plots" || echo "Weight diff plotting failed (rerun analysis/weight_diff.py)"
