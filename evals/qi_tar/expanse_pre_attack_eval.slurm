#!/bin/bash
#SBATCH --job-name=tar-pre-attack
#SBATCH --account=ddp477
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclude=exp-19-14
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out

# Pre-attack reference: WMDP bio/chem/cyber (same eval as the attack scripts) and MMLU (the repo's
# utility eval) for each model before any fine-tuning attack.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/tar-analysis/.env
export PATH=/expanse/lustre/projects/ddp477/mautorino/envs/durable/bin:$PATH
export MKL_THREADING_LAYER=GNU
# DeepSpeed 0.14.4 checks the CUDA toolkit (nvcc) on import; CUDA 12.x toolkit, same major as torch cu121
export CUDA_HOME=/cm/shared/apps/spack/0.21.2/gpu/dev/opt/spack/linux-rocky8-cascadelake/gcc-13.3.0/cuda-12.6.3-guub3z6c6k7zgb3qv3vdngxr6zu5mhzy
export VLLM_GPU_MEMORY_UTILIZATION=0.5   # default 0.9 OOMs: prompt_logprobs over the 128K vocab need ~20+ GB; answers are 1 token, so little KV cache is needed
ulimit -c 0

M=/expanse/lustre/projects/ddp477/mautorino/eval_models
tokenizer_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
model_family="llama3"

for model_name in TAR-Bio-v2-official TAR-Bio-diverse-params-v1 TAR-Bio-v2-perturb-rms1pct-v1
do
    for safety_bench in "wmdp_bio" "wmdp_chem" "wmdp_cyber"
    do
    python eval_safety_vllm.py \
        --model_path $M/$model_name \
        --model_name "${model_name}_pre_attack" \
        --tokenizer_name_or_path $tokenizer_name_or_path \
        --safety_bench $safety_bench \
        --model_family $model_family \
        --drop_system_prompt \
        --evaluator 'wmdp' \
        --num_gpus 1 \
        --save_path 'logs/fine-tuning-attack/safety_eval/tar/pre_attack/score' \
        --QA_save_path 'logs/fine-tuning-attack/safety_eval/tar/pre_attack/raw/' \
        --seed 1
    done

    python inference_utility_vllm.py \
        --model_path=$M/$model_name \
        --model_name="${model_name}_pre_attack" \
        --tokenizer_name_or_path=$tokenizer_name_or_path \
        --dataset=mmlu \
        --drop_system_prompt \
        --num_gpus=1 \
        --model_family=$model_family \
        --save_path="logs/fine-tuning-attack/utility_eval/tar/raw/"
    python eval_utility_vllm.py \
        --model-name "${model_name}_pre_attack-nosys" \
        --bench mmlu \
        --save-path "logs/fine-tuning-attack/utility_eval/tar/raw/" \
        --output-path "logs/fine-tuning-attack/utility_eval/tar/score"
done
