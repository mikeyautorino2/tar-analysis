"""Per-layer weight differences between a base model and a TAR checkpoint.

Each block (embeddings, decoder layer i, final norm, lm_head) is flattened into one
long vector, and we report ||theta_tar - theta_base||_2 and the relative change
||delta||_2 / ||theta_base||_2. Tensors are streamed one at a time, so neither model
is ever fully loaded.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download
from matplotlib.colors import LinearSegmentedColormap
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.layer_tracking import block_of  # noqa: E402

# Reference palette (dataviz skill): series-1 blue, sequential blue ramp, chart chrome
SERIES = "#2a78d6"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "input_layernorm", "post_attention_layernorm",
]


def tensor_locations(repo):
    """Map each parameter name to the safetensors shard that holds it (hub ID or local directory)."""
    path = repo if os.path.isdir(repo) else snapshot_download(repo, local_files_only=True)
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        weight_map = json.load(open(index))["weight_map"]
    else:
        with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    return {k: os.path.join(path, v) for k, v in weight_map.items()}


def compute(base_repo, tar_repo):
    base_loc, tar_loc = tensor_locations(base_repo), tensor_locations(tar_repo)
    assert base_loc.keys() == tar_loc.keys(), "Parameter names differ between checkpoints"

    # sums of squares per (block, module): ||delta||^2, ||base||^2, numel
    stats = defaultdict(lambda: np.zeros(3))
    handles = {}

    def get(path, name):
        if path not in handles:
            handles[path] = safe_open(path, "pt", device="cpu")
        return handles[path].get_tensor(name).to(torch.float32)

    for name in sorted(base_loc):
        base, tar = get(base_loc[name], name), get(tar_loc[name], name)
        assert base.shape == tar.shape, f"Shape mismatch for {name}"
        delta = tar - base
        stats[block_of(name)] += [
            delta.pow(2).sum().item(), base.pow(2).sum().item(), base.numel()
        ]
    return stats


def per_block(stats):
    order = ["embed"] + list(range(32)) + ["norm", "lm_head"]
    rows = []
    for block in order:
        d2 = sum(v[0] for (b, _), v in stats.items() if b == block)
        b2 = sum(v[1] for (b, _), v in stats.items() if b == block)
        n = sum(v[2] for (b, _), v in stats.items() if b == block)
        rows.append({
            "block": block, "numel": int(n), "l2_diff": d2 ** 0.5,
            "rel_diff": (d2 / b2) ** 0.5, "rms_diff": (d2 / n) ** 0.5,
        })
    return rows


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)


def plot_layers(rows, title, out_png):
    # Decoder layers get their own panel; embed / final norm / lm_head differ in kind and
    # scale (lm_head dwarfs everything), so they sit in a separate narrow panel.
    layers = [r for r in rows if str(r["block"]).isdigit()]
    others = [r for r in rows if not str(r["block"]).isdigit()]
    other_labels = {"embed": "embed", "norm": "final norm", "lm_head": "lm_head"}
    x = np.arange(len(layers))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [6, 1]}, sharex="col")
    panels = [
        ("l2_diff", "‖Δθ‖₂  (whole block as one vector)"),
        ("rel_diff", "‖Δθ‖₂ / ‖θ_base‖₂  (relative change)"),
    ]
    for (ax, ax_o), (key, ylabel) in zip(axes, panels):
        y = np.array([float(r[key]) for r in layers])
        style(ax)
        ax.plot(x, y, color=SERIES, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.set_ylim(0, y.max() * 1.15)
        for i in (int(np.argmax(y)), int(np.argmin(y))):
            ax.annotate(f"layer {i}: {y[i]:.3g}", (x[i], y[i]), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)

        yo = [float(r[key]) for r in others]
        style(ax_o)
        ax_o.scatter(range(len(others)), yo, s=40, color=SERIES, edgecolors=SURFACE,
                     linewidths=1.5, zorder=3)
        for i, v in enumerate(yo):
            ax_o.annotate(f"{v:.3g}", (i, v), xytext=(0, 7), textcoords="offset points",
                          ha="center", fontsize=8, color=INK)
        ax_o.set_ylim(0, max(yo) * 1.2)
        ax_o.set_xlim(-0.6, len(others) - 0.4)
    axes[1][0].set_xticks(x)
    axes[1][0].set_xticklabels(x, fontsize=8)
    axes[1][0].set_xlabel("Decoder layer", color=INK_2, fontsize=9)
    axes[1][1].set_xticks(range(len(others)))
    axes[1][1].set_xticklabels([other_labels[r["block"]] for r in others], fontsize=8, rotation=30, ha="right")
    axes[0][1].set_title("Other blocks (own scale)", color=INK_2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_heatmap(stats, title, out_png):
    grid = np.array([[(stats[(i, m)][0] / stats[(i, m)][1]) ** 0.5 for i in range(32)] for m in MODULES])
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(len(MODULES)))
    ax.set_yticklabels([m.split(".")[-1] for m in MODULES], fontsize=8, color=INK_2)
    ax.set_xticks(range(32))
    ax.set_xticklabels(range(32), fontsize=8, color=INK_2)
    ax.set_xlabel("Decoder layer", color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("‖Δθ‖₂ / ‖θ_base‖₂", color=INK_2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK_2)
    cb.outline.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--tar", default="lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2")
    parser.add_argument("--out_dir", default="results/weight_diff")
    parser.add_argument("--plot_only", action="store_true", help="Redraw plots from existing CSVs")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.tar.split('/')[-1]}_vs_{args.base.split('/')[-1]}"

    if args.plot_only:
        # Redraw from saved CSVs without re-reading the checkpoints
        rows = list(csv.DictReader(open(os.path.join(args.out_dir, f"{tag}_layers.csv"))))
        stats = {}
        for r in csv.DictReader(open(os.path.join(args.out_dir, f"{tag}_modules.csv"))):
            block = int(r["block"]) if r["block"].isdigit() else r["block"]
            stats[(block, r["module"])] = (float(r["rel_diff"]) ** 2, 1.0, 0)
        title = f"Weight change: {args.tar.split('/')[-1]} vs {args.base.split('/')[-1]}"
        plot_layers(rows, title, os.path.join(args.out_dir, f"{tag}_layers.png"))
        plot_heatmap(stats, title + " — relative change per module",
                     os.path.join(args.out_dir, f"{tag}_modules_heatmap.png"))
        return

    stats = compute(args.base, args.tar)
    rows = per_block(stats)

    with open(os.path.join(args.out_dir, f"{tag}_layers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, f"{tag}_modules.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "module", "numel", "l2_diff", "rel_diff"])
        for (b, m), (d2, b2, n) in sorted(stats.items(), key=lambda kv: str(kv[0])):
            w.writerow([b, m, int(n), d2 ** 0.5, (d2 / b2) ** 0.5])

    title = f"Weight change: {args.tar.split('/')[-1]} vs {args.base.split('/')[-1]}"
    plot_layers(rows, title, os.path.join(args.out_dir, f"{tag}_layers.png"))
    plot_heatmap(stats, title + " — relative change per module",
                 os.path.join(args.out_dir, f"{tag}_modules_heatmap.png"))
    for r in rows:
        print(f"{str(r['block']):>8}  l2={r['l2_diff']:.4f}  rel={r['rel_diff']:.5f}")


if __name__ == "__main__":
    main()
