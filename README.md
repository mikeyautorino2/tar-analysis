# tar-analysis

Experiments on **TAR (Tamper-Resistant Safeguards for open-weight LLMs)** for the biosecurity
domain:
- changes to how TAR's simulated fine-tuning attacks are sampled during training;
- analyses of where TAR changes the weights;
- red-teaming the resulting models with the attacks from Qi et al. (2024).

This repository is built on the official TAR code by Tamirisa et al.
([rishub-tamirisa/tamper-resistance](https://github.com/rishub-tamirisa/tamper-resistance)). The
original README is kept in [`docs/TAR_README.md`](docs/TAR_README.md); please cite the TAR paper
([`CITATION.bib`](CITATION.bib)) if you use this code.

## Research questions

1. **Diverse attack sampling.** Does sampling TAR's training-time attacks from a wider range of
   optimizer settings (learning rate, schedule, warmup) improve tamper resistance?
2. **Random weight perturbation.** Does training TAR against attacks that start from randomly
   perturbed weights (θ + ε, with ε ~ N(0, (0.01·rms(W))²) per tensor) improve tamper resistance?
3. **Where does TAR act?** Which layers and modules does TAR change, and which ones do attacks move?

## Results

### Red-teaming with Qi et al.'s TAR attacks

Full results, settings and per-seed scores are in [`results/qi_tar_eval/`](results/qi_tar_eval/).

All values are **accuracy in %** on 4-choice tests, so 25% is chance.
- **WMDP** measures hazardous knowledge: **lower is safer**, and about 25% means none.
- **MMLU** measures general capability: **higher is better**.
- The attacks fine-tune on Pile-Bio retain text for 1000 steps, with 5 seeds each. **Worst** is the
  maximum over the seeds.

| Model | Before attack: WMDP-bio ↓ | Before attack: MMLU ↑ | Full-param attack: WMDP-bio ↓ (worst / mean) | LoRA attack: WMDP-bio ↓ (worst / mean) |
|---|---|---|---|---|
| TAR-Bio-v2 (official) | 28.1 | 55.1 | 60.4 / 58.4 | 38.6 / 37.1 |
| **Diverse-params v1** (this repo) | 25.0 | 49.3 | **47.7 / 45.2** | **30.8 / 30.2** |
| Perturb-rms1pct v1 (this repo) | 24.0 | 27.2 | 52.7 / 50.3 | 25.5 / 25.3 |

For reference, the original Llama-3-8B-Instruct scores about 71% on WMDP-bio under this
evaluation.

- **The official TAR-Bio-v2 reproduces Qi et al.** Its pre-attack WMDP-bio is 28.1, as in their
  Table 7, and the full-param attack recovers it to about 60%.
- **Diverse attack sampling is the most robust usable model.** Its worst-case WMDP-bio after the
  full-param attack is 12.7 points lower than the official model's, and below the official
  model's on every seed. The cost is about 6 points of MMLU. Its training setup also differs from
  the official run in more than the sampling, so not all of the gap can be attributed to it.
- **The weight perturbation model collapsed.** Before any attack it answers "D" to every WMDP and
  MMLU question (MMLU at chance). The full-param attack restores WMDP-chem and WMDP-cyber to normal
  levels as well as bio, so the knowledge was not removed. This is a negative result: noise at
  1% of each tensor's RMS before every attack made TAR degrade the whole model instead of
  selectively removing bio knowledge.

### Weight analyses
- [`results/weight_diff/`](results/weight_diff/): per-layer and per-module ‖Δθ‖ between
  TAR-Bio-v2 and Llama-3-8B-Instruct. The change is spread across all layers (2.4–4% relative),
  rises in the last ~10 layers, and is largest in `lm_head` (13.8%) and in the attention `v_proj` and
  `o_proj`.
- [`results/loss_interpolation/`](results/loss_interpolation/): 1-D loss along
  θ(α) = (1−α)·θ_official + α·θ_perturbed, on bio (forget) and benign (retain) Pile-Bio text.
  *Preliminary:* the absolute losses at the endpoints are very high (e.g. 8.1 nats/token on benign
  text for the official model) and have not yet been cross-checked against an independent
  evaluation.
- With `--track_attack_displacement`, training logs how far each layer moves during every
  simulated attack. The CSV and plots for the perturbation run are published with that model on
  Hugging Face.

## Models (Hugging Face)
- [`mikeyautorino2/Llama-3-8B-Instruct-TAR-Bio-diverse-params-v1`](https://huggingface.co/mikeyautorino2/Llama-3-8B-Instruct-TAR-Bio-diverse-params-v1)
- [`mikeyautorino2/Llama-3-8B-Instruct-TAR-Bio-v2-perturb-rms1pct-v1`](https://huggingface.co/mikeyautorino2/Llama-3-8B-Instruct-TAR-Bio-v2-perturb-rms1pct-v1)

Model cards: [`model_cards/`](model_cards/).

## Repository layout

```
tar.py, modules/, configs/      TAR training code (upstream, with the changes in docs/CHANGES.md)
  modules/layer_tracking.py     per-layer attack displacement + random weight perturbation (FSDP)
scripts/train/                  training configs + Slurm jobs (1 node × 4 H100)
  run_tar_bio.sh                diverse-params v1 config
  run_tar_bio_perturb.sh        weight-perturbation config (repo default + --weight_perturb_scale 0.01)
  run_tar_bio_1x4.sbatch        train → sharded save → merge → plots
scripts/upload/                 convert to bf16 + upload a checkpoint to Hugging Face
analysis/                       weight diff, attack-displacement plots, loss interpolation,
                                perturbed-checkpoint generation
evals/qi_tar/                   Expanse versions of Qi et al.'s TAR red-teaming scripts + summary
results/                        evaluation and analysis results (summaries, CSVs, plots)
model_cards/                    Hugging Face model cards
docs/                           CHANGES.md (all changes vs. upstream TAR), TAR_README.md
red_teaming/, extra/            upstream TAR red-teaming and probing code
```

## Running

The scripts were written for SDSC Expanse (Slurm, H100 nodes) and set the account, partition and
paths for that cluster; adjust the `#SBATCH` lines for your own.

1. **Environment:** Python 3.11 with `pip install -r requirements.txt`. The Qi evaluation needs
   its own environment; see [`evals/qi_tar/`](evals/qi_tar/) and
   [`results/qi_tar_eval/README.md`](results/qi_tar_eval/README.md).
2. **Configuration:** copy [`dotenv`](dotenv) to `.env` and fill it in (Hugging Face token and
   cache, `SAVE_MODELS_DIR`, accelerate config paths). `.env` is git-ignored.
3. **Training** (from the repo root):
   ```bash
   sbatch scripts/train/run_tar_bio_1x4.sbatch                                                # diverse-params config
   RUN_SCRIPT=scripts/train/run_tar_bio_perturb.sh sbatch scripts/train/run_tar_bio_1x4.sbatch  # perturbation config
   sbatch scripts/train/run_tar_bio_1x4.sbatch --max_steps 3 --expname test                   # quick test
   ```
   Each job trains, saves a sharded checkpoint, merges it into `model.safetensors` and writes plots.
4. **Analyses:** `sbatch analysis/weight_diff.sbatch`, `sbatch analysis/loss_interpolation.sbatch`,
   `python analysis/plot_attack_displacement.py <run output dir>`.
5. **Qi et al. evaluation:** clone
   [Evaluating-Durable-Safeguards](https://github.com/princeton-polaris-lab/Evaluating-Durable-Safeguards),
   copy `evals/qi_tar/*` into its `scripts/tar/`, and submit them as described in
   [`results/qi_tar_eval/README.md`](results/qi_tar_eval/README.md).

All changes relative to the upstream TAR code are documented in
[`docs/CHANGES.md`](docs/CHANGES.md).
