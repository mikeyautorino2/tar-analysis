| Model | Before attack: WMDP-bio ↓ | Before attack: WMDP-chem | Before attack: WMDP-cyber | Before attack: MMLU ↑ | Full-param attack: WMDP-bio ↓ (worst / mean) | LoRA attack: WMDP-bio ↓ (worst / mean) | Full-param attack: WMDP-chem (worst) | Full-param attack: WMDP-cyber (worst) |
|---|---|---|---|---|---|---|---|---|
| TAR-Bio-v2-official | 28.1 | 42.6 | 43.2 | 55.1 | 60.4 / 58.4 | 38.6 / 37.1 | 46.3 | 45.7 |
| TAR-Bio-diverse-params-v1 | 25.0 | 40.4 | 40.0 | 49.3 | 47.7 / 45.2 | 30.8 / 30.2 | 49.0 | 47.1 |
| TAR-Bio-v2-perturb-rms1pct-v1 | 24.0 | 26.5 | 24.6 | 27.2 | 52.7 / 50.3 | 25.5 / 25.3 | 47.3 | 43.6 |

All values are accuracy in % on 4-choice tests (chance = 25%). WMDP = hazardous knowledge (lower is safer; ~25% = none). MMLU = general capability (higher is better). Worst = max over the 5 attack seeds.
