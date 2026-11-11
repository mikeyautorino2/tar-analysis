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

# Llama-3-8B-Instruct-TAR-Bio-diverse-params-v1

A research checkpoint of Llama-3-8B-Instruct trained with
[TAR (Tamper-Resistant Safeguards)](https://arxiv.org/abs/2408.00761) for the
biosecurity domain. It differs from the official
[`lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2`](https://huggingface.co/lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2)
only in how TAR's simulated fine-tuning attacks are sampled during training.

**Research question:** does randomly sampling the training-time attacks (the
"Sample attack ~ A_train" step of TAR, Algorithm 1) from a more diverse set of
hyperparameters improve tamper resistance? The planned evaluation uses the
fine-tuning attacks from Qi et al. (2024),
[On Evaluating the Durability of Safeguards for Open-Weight LLMs](https://arxiv.org/abs/2412.07097).

> **Status:** not yet evaluated. Treat any robustness claims as unverified.

## What changed versus baseline TAR

Each simulated attack samples its optimizer settings independently:

| Attack hyperparameter | Baseline TAR | This model |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 2e-6, 2e-5, 4e-5 | 2e-6, 1e-5, 2e-5, 3e-5, 4e-5, 6e-5, 1e-4 |
| LR schedule | constant | constant, linear decay, cosine decay |
| Warmup steps | 0 | 0, 10 (linear) |
| Attack steps | 64 | 64 |
| Global attack batch | 64 | 64 |

The attack step count and batch size match baseline, so both models use the
same attack compute. The higher learning rates (6e-5, 1e-4) let a 64-step attack
move the weights further, as a rough and not equivalent stand-in for longer
attacks. The adversary data types (Pile-Bio, Camel-Bio, retain→forget switch),
the tamper-resistance loss (max entropy), the representation-engineering retain
loss and the outer optimizer are unchanged.

## Training setup

- Initialized from `lapisrocks/Llama-3-8B-Instruct-Random-Mapped-Bio`, the same
  starting point as TAR-Bio-v2.
- 750 outer steps, outer LR 2e-5 (schedule-free AdamW, 32 warmup steps).
- 1 node × 4 H100 80GB with FSDP and gradient checkpointing.
- Gradient accumulation 4, so the global batch sizes match the official 8-GPU
  run: 128 for retain and tamper-resistance, 64 for the adversary.
- The adversary's retain-data micro-batch is 4 instead of 8 (to save memory).
  The total adversary batch is unchanged.
- Wall-clock time: about 30 hours.
- Trained in float32 with bf16 mixed precision. The weights here are cast to
  bfloat16.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "mikeyautorino2/Llama-3-8B-Instruct-TAR-Bio-diverse-params-v1"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="bfloat16", device_map="auto")
```

## Intended use

For research on tamper-resistant safeguards and their evaluation. It is not
intended for deployment. Like other TAR models, it is meant to resist
fine-tuning that recovers hazardous biology knowledge, but this checkpoint has
not yet been red-teamed.

## Citation

Tamirisa et al., *Tamper-Resistant Safeguards for Open-Weight LLMs*, 2024.
Qi et al., *On Evaluating the Durability of Safeguards for Open-Weight LLMs*, 2024.
