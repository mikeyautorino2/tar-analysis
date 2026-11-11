---
license: llama3
base_model: lapisrocks/Llama-3-8B-Instruct-Random-Mapped-Bio
library_name: transformers
tags:
- tamper-resistance
- tar
- safety
- unlearning
- llama-3
---

# Llama-3-8B-Instruct-TAR-Bio-v2-perturb-rms1pct-v1

A research checkpoint of Llama-3-8B-Instruct trained with
[TAR (Tamper-Resistant Safeguards)](https://arxiv.org/abs/2408.00761) for the
biosecurity domain. It uses the default TAR-Bio configuration from the TAR repository with
one change: **a random weight perturbation is applied before every simulated attack**.

**Research question:** does training TAR against attacks that start from randomly perturbed
weights improve tamper resistance? The planned evaluation uses the fine-tuning attacks from
Qi et al. (2024),
[On Evaluating the Durability of Safeguards for Open-Weight LLMs](https://arxiv.org/abs/2412.07097).

> **Status:** not yet evaluated. Treat any robustness claims as unverified.

## The change: random weight perturbation

At the start of each simulated fine-tuning attack in TAR's inner loop, every weight tensor `W`
gets Gaussian noise:

`W ← W + ε`, with `ε ~ N(0, (0.01 · rms(W))²)`, i.e. noise at 1% of each tensor's RMS.

The attack then fine-tunes from the perturbed weights, and the tamper-resistance loss is
computed on the perturbed and attacked model. Afterwards the clean weights are restored, as
TAR already does after every attack, so the trained model itself is never noised. Noise is
drawn independently per GPU shard (seed `1234 + rank`).

Everything else follows the repository's default `run_tar_bio.sh`:

| Setting | Value |
|---|---|
| Start checkpoint | `lapisrocks/Llama-3-8B-Instruct-Random-Mapped-Bio` |
| Outer steps / LR | 750, 2e-5 (schedule-free AdamW, 32 warmup steps) |
| Attack | AdamW; LR from {2e-6, 2e-5, 4e-5}; constant schedule; 64 steps; global batch 32 |
| Attack data | Pile-Bio, Camel-Bio, retain→forget switch (⅓ each) |
| TR loss / retain loss | max entropy / representation-engineering retain loss |
| Global batch (retain, TR) | 64, the same as the repository's 8-GPU default |

## Training setup

- 1 node × 4 H100 80GB with FSDP and gradient checkpointing; gradient accumulation 2
  (equivalent to the repository's accumulation 1 on 8 GPUs). Wall-clock time: about 23.5 hours.
- Trained in float32 with bf16 mixed precision. The weights here are cast to bfloat16.
- During training, the movement of each layer during each attack was logged, measured from the
  perturbed starting point so that the noise is excluded. The data is in
  `attack_displacement.csv` and summarized in the `attack_disp_*.png` plots. `plots/`
  compares the final weights with the starting checkpoint.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "mikeyautorino2/Llama-3-8B-Instruct-TAR-Bio-v2-perturb-rms1pct-v1"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="bfloat16", device_map="auto")
```

## Intended use

For research on tamper-resistant safeguards and their evaluation. It is not intended for
deployment. Like other TAR models, it is meant to resist fine-tuning that recovers hazardous
biology knowledge, but this checkpoint has not yet been red-teamed.

## Citation

Tamirisa et al., *Tamper-Resistant Safeguards for Open-Weight LLMs*, 2024.
Qi et al., *On Evaluating the Durability of Safeguards for Open-Weight LLMs*, 2024.
