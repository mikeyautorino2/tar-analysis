"""1-D loss interpolation between two checkpoints.

theta(alpha) = (1 - alpha) * theta_A + alpha * theta_B,   f(alpha) = L(theta(alpha))

L is the mean next-token cross-entropy (nats/token) on a fixed sample of Pile-Bio text, measured
separately on the forget split (hazardous bio, label=True) and the retain split (benign,
label=False). Both endpoints' weights are kept on the GPU in bf16. Each interpolation is done in
fp32, one tensor at a time, and written into a single bf16 model.
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_diff import AXIS, INK, INK_2, MUTED, SERIES, SURFACE, style, tensor_locations  # noqa: E402

SERIES_2 = "#eb6834"  # categorical slot 2 (reference palette)


def load_state(repo, device):
    """All parameters of a checkpoint as bf16 tensors on `device`."""
    state = {}
    for name, path in tensor_locations(repo).items():
        with safe_open(path, "pt", device="cpu") as f:
            state[name] = f.get_tensor(name).to(device=device, dtype=torch.bfloat16)
    return state


def sample_texts(split_label, n, seed):
    ds = load_dataset("lapisrocks/pile-bio", split="train")
    ds = ds.filter(lambda x: x["label"] == split_label).shuffle(seed=seed).select(range(n))
    return [x["txt_chunk"] for x in ds]


@torch.no_grad()
def mean_loss(model, batches):
    total, tokens = 0.0, 0
    for input_ids, attention_mask in batches:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
        targets = input_ids[:, 1:]
        mask = attention_mask[:, 1:].bool()
        loss = torch.nn.functional.cross_entropy(logits[mask], targets[mask], reduction="sum")
        total += loss.item()
        tokens += mask.sum().item()
    return total / tokens


def plot(rows, args, out_png):
    a = np.array([r["alpha"] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    style(ax)
    for key, label, color in [("loss_forget", "Bio / forget text", SERIES),
                              ("loss_retain", "Benign / retain text", SERIES_2)]:
        y = np.array([r[key] for r in rows])
        ax.plot(a, y, color=color, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1, label=label, zorder=3)
        ax.annotate(label, (a[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    for x, name in [(0, args.name_a), (1, args.name_b)]:
        ax.axvline(x, color=AXIS, linestyle=":", linewidth=1, zorder=1)
        ax.annotate(f"α={x}: {name}", (x, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color=INK_2)
    if args.log_y:
        ax.set_yscale("log")
    ax.set_xlabel("α   (θ(α) = (1−α)·θ_A + α·θ_B)", color=INK_2, fontsize=9)
    ax.set_ylabel("Cross-entropy loss (nats / token)" + (" — log scale" if args.log_y else ""),
                  color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center")
    fig.suptitle(f"1-D loss interpolation: {args.name_a} (α=0) → {args.name_b} (α=1)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", required=True, help="theta at alpha=0 (hub id or local dir)")
    p.add_argument("--model_b", required=True, help="theta at alpha=1 (hub id or local dir)")
    p.add_argument("--name_a", default="model A")
    p.add_argument("--name_b", default="model B")
    p.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--alpha_min", type=float, default=-0.5)
    p.add_argument("--alpha_max", type=float, default=1.5)
    p.add_argument("--alpha_step", type=float, default=0.05)
    p.add_argument("--n_texts", type=int, default=128)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_y", action="store_true")
    p.add_argument("--plot_only", action="store_true", help="Redraw from the saved CSV")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "loss_interpolation.csv")
    out_png = os.path.join(args.out_dir, "loss_interpolation.png")

    if args.plot_only:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(out_csv))]
        plot(rows, args, out_png)
        return

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.pad_token = tok.eos_token
    batches = {}
    for split, label in [("forget", True), ("retain", False)]:
        texts = sample_texts(label, args.n_texts, args.seed)
        enc = tok(texts, truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt")
        batches[split] = [(enc.input_ids[i:i + args.batch_size].to(device),
                           enc.attention_mask[i:i + args.batch_size].to(device))
                          for i in range(0, args.n_texts, args.batch_size)]
        print(f"{split}: {args.n_texts} texts, {int(enc.attention_mask.sum())} tokens", flush=True)

    theta_a, theta_b = load_state(args.model_a, device), load_state(args.model_b, device)
    model = AutoModelForCausalLM.from_pretrained(args.model_a, torch_dtype=torch.bfloat16).to(device).eval()
    params = dict(model.named_parameters())
    assert params.keys() == theta_a.keys() == theta_b.keys(), "Parameter names differ"

    alphas = np.round(np.arange(args.alpha_min, args.alpha_max + 1e-9, args.alpha_step), 4)
    rows = []
    for alpha in alphas:
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(((1 - alpha) * theta_a[name].float() + alpha * theta_b[name].float()).to(param.dtype))
        row = {"alpha": float(alpha),
               "loss_forget": mean_loss(model, batches["forget"]),
               "loss_retain": mean_loss(model, batches["retain"])}
        rows.append(row)
        print(f"alpha={alpha:+.2f}  forget={row['loss_forget']:.4f}  retain={row['loss_retain']:.4f}", flush=True)
        with open(out_csv, "w", newline="") as f:  # rewrite each step, so partial results survive
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerows(rows)

    plot(rows, args, out_png)
    print(f"Wrote {out_csv} and {out_png}")


if __name__ == "__main__":
    main()
