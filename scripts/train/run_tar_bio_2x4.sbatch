#!/bin/bash
# TAR-Bio-v2 on 2 nodes x 4 H100 (8 GPUs total, same global batch as the 8-GPU paper setup).
#SBATCH --job-name=tar-bio-v2
#SBATCH --account=ddp477
#SBATCH --partition=nairr-gpu
#SBATCH --qos=nairr-gpu-normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # one accelerate launcher per node; it spawns 4 workers
#SBATCH --gpus=8                   # Expanse only accepts total --gpus (4 per node)
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .env

export ExACCEL_CONFIG="$SLURM_SUBMIT_DIR/configs/accel_config_2x4_gpu.yaml"
export MAIN_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MAIN_PORT=29500
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# $SLURM_NODEID must be expanded on each node, hence the single quotes.
srun --kill-on-bad-exit=1 bash -c '
  export ACCEL_LAUNCH_ARGS="--num_machines 2 --num_processes 8 --machine_rank $SLURM_NODEID --main_process_ip $MAIN_IP --main_process_port $MAIN_PORT"
  echo "[$(hostname)] node $SLURM_NODEID -> $ACCEL_LAUNCH_ARGS"
  sh scripts/train/run_tar_bio.sh
'

# Merge the per-rank FSDP shards into a regular HF checkpoint (CPU only, runs once on node 0).
NAME=$(grep -oP '(?<=--new_model_name )\S+' scripts/train/run_tar_bio.sh)
EXP=$(grep -oP '(?<=--expname )\S+' scripts/train/run_tar_bio.sh || echo latest)
OUT="$SAVE_MODELS_DIR/${NAME}_${EXP}"
echo "Merging FSDP shards in $OUT"
accelerate merge-weights "$OUT/pytorch_model_fsdp_0" "$OUT"
ls -lh "$OUT"
