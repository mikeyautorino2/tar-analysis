CHANGES FROM ORIGINAL TAR (as of 2026-10-14)

Goal
----
Test whether randomly sampling the simulated fine-tuning attack (the
"Sample attack ~ A_train" step of TAR, Algorithm 1) from a more diverse set
of hyperparameters gives better tamper resistance than baseline TAR.
Evaluation: the fine-tuning attacks from Qi et al. (2024), "On Evaluating the
Durability of Safeguards for Open-Weight LLMs".


1. Diverse adversary hyperparameters
------------------------------------
Every simulated attack now samples each hyperparameter independently
(all GPUs get the same values). Format is "value:weight,...".

  Hyperparameter    Baseline TAR          Diverse run
  ----------------  --------------------  -----------------------------------
  Optimizer         AdamW                 AdamW
  LR                2e-6, 2e-5, 4e-5      2e-6, 1e-5, 2e-5, 3e-5, 4e-5, 6e-5, 1e-4
  LR schedule       constant              constant, linear decay, cosine decay
  Warmup steps      0                     0, 10 (linear warmup)
  Attack steps      64                    64 (fixed, same as baseline)
  Global adv batch  64 (4 x 2 accum x 8)  64 (4 x 4 accum x 4 GPUs, fixed, same as baseline)

Unchanged: adversary data types (pile-bio, camel-bio, retain_forget_switch),
switching-point distribution, TR loss (max entropy), retain loss, outer
optimizer and learning rate.

New flags (tar.py). Each is optional; leaving it out keeps the baseline
behaviour.
  --adversary_warmup_steps      e.g. 0:0.5,10:0.5
  --adversary_inner_loop_steps  e.g. 32:0.34,64:0.33,128:0.33 (not used now; steps
                                default to --tar_inner_loop_steps 64)
  --adversary_batch_sizes       per-GPU; must be multiples of --tar_adversary_batch_size
                                (not used now; defaults to 4 x accum steps)
  --adversary_lr_schedulers     now also accepts linear and cosine

Implementation: modules/training.py
  - _sample_adversary_config(): samples optimizer, warmup, steps and batch size
  - _adversary_lr_lambda(): linear warmup, then constant, linear or cosine decay
  - Everything that used the fixed step count now uses the sampled one: the
    TR loss weighting schedule, the switching point and the progress bar.
  - The sampled settings for each attack are printed to the log and sent to wandb.
  - SGD support (--adversary_optimizers, --adversary_sgd_lr_samples) is still in
    the code but not used in the current run.


2. Fixes and infrastructure (not part of the experiment)
--------------------------------------------------------
- modules/dataloaders.py: the adversary's retain data (adv_retain) now uses
  --tar_adversary_batch_size (4) instead of --batch_size (8). This makes the
  sampled batch sizes correct for retain_forget_switch attacks and halves
  activation memory. The outer retain loss is unchanged.
- modules/training.py: after each attack, the adversary optimizer is now freed
  directly. The original load_state_dict(state_dict -> "cpu") round trip cast
  the AdamW state back onto the GPU, briefly doubling it, and caused an OOM at
  outer step 24 on 4 GPUs. No effect on training.
- Attack displacement tracking (--track_attack_displacement, modules/layer_tracking.py):
  after every simulated attack, before the weights are restored, measures how far
  each decoder layer / module (and embed, final norm, lm_head) moved during that
  attack: ||theta_after - theta_before||_2, absolute and relative. It compares
  against the pre-attack copy TAR already keeps, so it needs no extra memory and
  takes about 1-2 s per outer step. Results are written to
  <output_dir>/attack_displacement.csv (one row per attack x module, with the
  attack's sampled settings) and logged to wandb as attack_disp/*. Plot them with
  analysis/plot_attack_displacement.py <output_dir>. Read-only, so it does not
  affect training.
- Weight perturbation (--weight_perturb_scale, default 0 = off): before each
  simulated attack, every weight tensor W gets Gaussian noise
  N(0, (scale * rms(W))^2), so the attack starts from theta + eps. The TR loss is
  computed on the perturbed and attacked model, and the clean weights are restored
  from the existing pre-attack snapshot afterwards, so the outer model is never
  noised. Noise is drawn per rank (seed 1234 + rank). With tracking on, the same
  noise is regenerated, so attack displacement measures only the SFT movement from
  the perturbed start point. Config: scripts/train/run_tar_bio_perturb.sh (the repo's default
  run_tar_bio.sh plus 1-node changes: accumulation 2 on 4 GPUs = repo's 1 on 8,
  gradient checkpointing, wandb, tracking, --weight_perturb_scale 0.01).
- scripts/train/run_tar_bio_1x4.sbatch: RUN_SCRIPT selects the config, e.g.
  `RUN_SCRIPT=scripts/train/run_tar_bio_perturb.sh sbatch scripts/train/run_tar_bio_1x4.sbatch`. After the
  merge, the job writes plots: attack displacement (if tracked) and
  analysis/weight_diff.py of the final model vs its --base_model_name, saved to
  <output_dir>/plots. Excludes node exp-19-14 (CUDA failed to initialize there).
- scripts/train/run_tar_bio.sh / run_tar_bio_1x4.sbatch: extra sbatch arguments are passed
  through and override the script, e.g.
  `sbatch scripts/train/run_tar_bio_1x4.sbatch --max_steps 3 --expname test`.
- tar.py: added --gradient_checkpointing so training fits on a single 4-GPU
  node. One node is about 3x faster than two nodes, because FSDP traffic stays
  on NVLink.
- tar.py: the final save writes a sharded FSDP checkpoint (each rank writes its
  own shard). The previous full-state-dict gather onto rank 0 hung on 2 nodes.
  The sbatch script then merges the shards into model.safetensors with
  `accelerate merge-weights`.
- scripts/train/run_tar_bio_1x4.sbatch: 1 node x 4 H100 (the current setup).
  scripts/train/run_tar_bio_2x4.sbatch: 2 nodes x 4 H100 (works, but is slower).
- scripts/train/run_tar_bio.sh: 750 steps, --expname diverse_params_v1, --wandb.


3. Notes for comparison
-----------------------
- Baseline TAR should be retrained in the same setup (1 node / 4 GPUs,
  gradient checkpointing, 750 steps, same seed). The released TAR-Bio-v2 was
  trained on 8 GPUs with a different global batch.
- Attack steps are fixed at 64, as in baseline. The higher LRs (6e-5, 1e-4)
  let a 64-step attack move the weights further, as a rough stand-in for
  longer attacks; this is not equivalent to taking more steps.
- Attack steps (64) and global batch sizes match the official 8-GPU TAR-Bio-v2
  run: --gradient_accumulation_steps 4 on 4 GPUs gives outer retain / TR batches
  of 128 and an adversary batch of 64, as with accum 2 on 8 GPUs. Only the
  attack optimizer settings (LR, schedule, warmup) differ.
- Timing: the 3-step test (accum 2) took about 74 s per outer step on 1 node.
  With accum 4 the full run is estimated at about 30-40 h (48 h limit, and no
  intermediate checkpoints).
