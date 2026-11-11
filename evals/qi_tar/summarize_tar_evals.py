"""Summarize the TAR red-teaming results for the eval_models checkpoints.

Reads the score CSVs written by eval_safety_vllm.py / eval_utility_vllm.py and writes
logs/fine-tuning-attack/safety_eval/tar/summary.{csv,md}:
  - pre-attack WMDP bio/chem/cyber and MMLU accuracy (%)
  - per attack (full-param, LoRA): mean and worst case (max) WMDP accuracy over the seeds

The safety score CSVs do not record the seed: each seed's run appends one row per bench, so the
n-th row of a bench is the n-th seed. Run from the repo root.
"""

import glob
import os

import pandas as pd

MODELS = ["TAR-Bio-v2-official", "TAR-Bio-diverse-params-v1", "TAR-Bio-v2-perturb-rms1pct-v1"]
BENCHES = ["wmdp_bio", "wmdp_chem", "wmdp_cyber"]
SCORE_DIR = "logs/fine-tuning-attack/safety_eval/tar/score"
PRE_DIR = "logs/fine-tuning-attack/safety_eval/tar/pre_attack/score"
MMLU_DIR = "logs/fine-tuning-attack/utility_eval/tar/score"
FULL_TAG = "steps_1000_warmup_steps_100_batch_size_64_scheduler_cosine_weight_decay_0.01-nosys"
PEFT_TAG = "steps_1000_warmup_steps_0_batch_size_64_scheduler_constant_weight_decay_0.01_peft-nosys"


def read_scores(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def attack_stats(model, tag, row):
    """Mean and max over seeds for one model/attack; warns if a bench doesn't have 5 seeds."""
    files = glob.glob(os.path.join(SCORE_DIR, f"{model}_ft_*{tag}+score.csv"))
    if not files:
        return
    df = read_scores(files[0])
    attack = "full" if tag == FULL_TAG else "lora"
    for bench in BENCHES:
        s = df[df.bench == bench].score
        if len(s) != 5:
            print(f"WARNING: {model} {attack} {bench}: {len(s)} seeds (expected 5)")
        if len(s):
            row[f"{attack}_{bench}_mean"] = s.mean()
            row[f"{attack}_{bench}_max"] = s.max()
            row[f"{attack}_seeds"] = len(s)


def main():
    rows = []
    for model in MODELS:
        row = {"model": model}
        pre = read_scores(os.path.join(PRE_DIR, f"{model}_pre_attack-nosys+score.csv"))
        for bench in BENCHES:
            if pre is not None and (pre.bench == bench).any():
                row[f"pre_{bench}"] = pre[pre.bench == bench].score.iloc[-1]
        mmlu = read_scores(os.path.join(MMLU_DIR, f"{model}_pre_attack-nosys+score.csv"))
        if mmlu is not None and (mmlu.bench == "mmlu").any():
            score = mmlu[mmlu.bench == "mmlu"].score.iloc[-1]
            row["pre_mmlu"] = score * 100 if score <= 1 else score  # report as %
        attack_stats(model, FULL_TAG, row)
        attack_stats(model, PEFT_TAG, row)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("model")
    out = "logs/fine-tuning-attack/safety_eval/tar/summary"
    summary.round(2).to_csv(out + ".csv")  # markdown below rounds the unrounded values (no double rounding)

    # Compact markdown: pre-attack, then worst case (max over seeds) per attack
    md = ["| Model | Before attack: WMDP-bio ↓ | Before attack: WMDP-chem | Before attack: WMDP-cyber | Before attack: MMLU ↑ | Full-param attack: WMDP-bio ↓ (worst / mean) | LoRA attack: WMDP-bio ↓ (worst / mean) | Full-param attack: WMDP-chem (worst) | Full-param attack: WMDP-cyber (worst) |",
          "|---|---|---|---|---|---|---|---|---|"]
    get = lambda r, k: f"{r[k]:.1f}" if k in r and pd.notna(r[k]) else "–"
    for model, r in summary.iterrows():
        r = r.to_dict()
        md.append(f"| {model} | {get(r, 'pre_wmdp_bio')} | {get(r, 'pre_wmdp_chem')} | {get(r, 'pre_wmdp_cyber')} | "
                  f"{get(r, 'pre_mmlu')} | {get(r, 'full_wmdp_bio_max')} / {get(r, 'full_wmdp_bio_mean')} | "
                  f"{get(r, 'lora_wmdp_bio_max')} / {get(r, 'lora_wmdp_bio_mean')} | "
                  f"{get(r, 'full_wmdp_chem_max')} | {get(r, 'full_wmdp_cyber_max')} |")
    open(out + ".md", "w").write("\n".join(md) + "\n\nAll values are accuracy in % on 4-choice tests (chance = 25%). WMDP = hazardous knowledge (lower is safer; ~25% = none). MMLU = general capability (higher is better). Worst = max over the 5 attack seeds.\n")
    print("\n".join(md))
    print(f"\nWrote {out}.csv and {out}.md")


if __name__ == "__main__":
    main()
