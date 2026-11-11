#!/bin/bash
#SBATCH --job-name=tar-redteam-full
#SBATCH --account=ddp477
#SBATCH --partition=nairr-gpu
#SBATCH --qos=nairr-gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --exclude=exp-19-14          # CUDA failed to initialize there
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out

# Expanse version of launch_ft_safety_eval.slurm. Attack/eval settings are unchanged; only the cluster setup differs.
# Usage: MODEL_PATH=<eval_models dir> MODEL_NAME=<name> sbatch expanse_ft_safety_eval.slurm
#        (smoke test: add SEEDS="1" MAX_STEPS=10)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/tar-analysis/.env                      # HF_HOME / HF_TOKEN / caches on project storage
export PATH=/expanse/lustre/projects/ddp477/mautorino/envs/durable/bin:$PATH
export MKL_THREADING_LAYER=GNU
# DeepSpeed 0.14.4 checks the CUDA toolkit (nvcc) on import; CUDA 12.x toolkit, same major as torch cu121
export CUDA_HOME=/cm/shared/apps/spack/0.21.2/gpu/dev/opt/spack/linux-rocky8-cascadelake/gcc-13.3.0/cuda-12.6.3-guub3z6c6k7zgb3qv3vdngxr6zu5mhzy
export VLLM_GPU_MEMORY_UTILIZATION=0.5   # default 0.9 OOMs: prompt_logprobs over the 128K vocab need ~20+ GB; answers are 1 token, so little KV cache is needed
ulimit -c 0
: "${MODEL_PATH:?set MODEL_PATH}" "${MODEL_NAME:?set MODEL_NAME}"

# dataset_name="pile_bio_subset" # For fine-tuning on the forget set
# dataset_name="mixed_pile_retain_magpie_align" # For fine-tuning on the mixture of Pile-bio retain and Magpie Align, which corresponds to the "Retain" set in our main section
dataset_name="pile_bio_retain" # For fine-tuning on the Pile-bio Retain Set, which corresponds to the "Retain" set used in Appedix F.

tokenizer_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
model_name="$MODEL_NAME"
model_path="$MODEL_PATH"
model_family="llama3"
lr=2e-5
max_num_samples=-1
max_steps=${MAX_STEPS:-1000}
# lr_scheduler_type="constant"
# lr_scheduler_type="constant_with_warmup"
lr_scheduler_type="cosine"
warmup_steps=100
per_device_train_batch_size=8
optim="adamw_torch"
weight_decay=0.01
batch_size=$((per_device_train_batch_size * 8))

for i in ${SEEDS:-1 2 3 4 5}
do
output_dir="${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_sample_${max_num_samples}_${max_steps}_${warmup_steps}_${batch_size}_${lr_scheduler_type}_${weight_decay}/${i}"
accelerate launch --config_file=accelerate_configs/deepspeed_zero3_gc2.yaml \
    --main_process_port=12024 \
    --num_processes 4 \
     finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family=$model_family \
    --learning_rate=$lr \
    --lr_scheduler_type=$lr_scheduler_type \
    --warmup_steps=$warmup_steps \
    --optim=$optim \
    --max_steps=$max_steps \
    --weight_decay=$weight_decay \
    --ft_seed $i \
    --max_num_samples=$max_num_samples \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=2 \
    --output_dir=$output_dir \
    --logging_steps=1 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

# # safety_eval
ft_model_path=$output_dir
ft_model_name="${model_name}_ft_${dataset_name}_${lr}_sample_${max_num_samples}_steps_${max_steps}_warmup_steps_${warmup_steps}_batch_size_${batch_size}_scheduler_${lr_scheduler_type}_weight_decay_${weight_decay}"
for safety_bench in "wmdp_bio" "wmdp_chem" "wmdp_cyber"
do
python eval_safety_vllm.py \
    --model_path $ft_model_path \
    --model_name $ft_model_name \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --safety_bench $safety_bench \
    --model_family $model_family \
    --drop_system_prompt \
    --evaluator 'wmdp' \
    --num_gpus 1 \
    --save_path 'logs/fine-tuning-attack/safety_eval/tar/score' \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/tar/raw/' \
    --seed $i
done
# if [ $i != 1 ]
# then
    # delete checkpoints to save disk space
rm -rf $ft_model_path
# fi
done



