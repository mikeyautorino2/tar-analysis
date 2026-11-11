"""Plot per-layer attack displacement logged during TAR training (--track_attack_displacement).

Reads <run_dir>/attack_displacement.csv and writes, next to it:
  attack_disp_share_heatmap.png   decoder layer x outer step: each layer's share of the attack's movement
  attack_disp_profile.png         mean relative displacement per layer, early vs late in training
  attack_disp_modules_heatmap.png per-module relative displacement averaged over all attacks
  attack_disp_by_lr.png           mean relative displacement per layer for each sampled attack LR
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from weight_diff import AXIS, GRID, INK, INK_2, MODULES, MUTED, SEQUENTIAL, SERIES, SURFACE, style

SERIES_2 = "#eb6834"  # categorical slot 2 (reference palette)
LR_STEPS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
OTHER = ["embed", "norm", "lm_head"]


def load(path):
    df = pd.read_csv(path)
    df["d2"] = df["l2_diff"] ** 2
    df["b2"] = (df["l2_diff"] / df["rel_diff"].replace(0, np.nan)) ** 2
    df["attack"] = df["outer_step"].astype(str) + "_" + df["attack_idx"].astype(str)
    return df


def per_block(df):
    """Aggregate modules into blocks: relative displacement per (attack, block)."""
    g = df.groupby(["attack", "outer_step", "lr", "block"], as_index=False)[["d2", "b2"]].sum()
    g["rel"] = np.sqrt(g["d2"] / g["b2"])
    return g


def heat(ax, grid, xlabel, ylabels, cbar_label, fig):
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest", origin="lower")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7, color=INK_2)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(cbar_label, color=INK_2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK_2)
    cb.outline.set_visible(False)


def share_heatmap(blocks, out, title):
    layers = blocks[blocks["block"].str.isdigit()].copy()
    layers["layer"] = layers["block"].astype(int)
    step = layers.groupby(["outer_step", "layer"])["d2"].mean().unstack("layer")
    share = step.div(step.sum(axis=1), axis=0)  # fraction of the attack's squared movement
    window = max(1, len(share) // 50)
    share = share.rolling(window, min_periods=1, center=True).mean()
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    heat(ax, share.T.values, f"Outer training step (rolling mean over {window})",
         [str(i) for i in share.columns], "Share of attack's ‖Δθ‖² in layer", fig)
    xt = np.linspace(0, len(share) - 1, min(len(share), 11)).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels(share.index[xt], fontsize=8, color=INK_2)
    ax.set_ylabel("Decoder layer", color=INK_2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def profile(blocks, out, title):
    steps = sorted(blocks["outer_step"].unique())
    k = max(1, len(steps) // 10)
    phases = [(f"first {k} steps", SERIES, steps[:k]), (f"last {k} steps", SERIES_2, steps[-k:])]
    fig, (ax, ax_o) = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [6, 1]})
    style(ax)
    style(ax_o)
    for p, (label, color, sel) in enumerate(phases):
        sub = blocks[blocks["outer_step"].isin(sel)].groupby("block")["rel"].mean()
        layers = sub[[b for b in sub.index if b.isdigit()]]
        layers.index = layers.index.astype(int)
        layers = layers.sort_index()
        ax.plot(layers.index, layers.values, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.annotate(label, (layers.index[-1], layers.values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
        other = [sub.get(b, np.nan) for b in OTHER]
        ax_o.scatter(np.arange(3) + (p - 0.5) * 0.25, other, s=40, color=color,
                     edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.set_xlabel("Decoder layer", color=INK_2, fontsize=9)
    ax.set_ylabel("Mean ‖Δθ‖₂ / ‖θ‖₂ during attack", color=INK_2, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(32))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    ax_o.set_xticks(range(3))
    ax_o.set_xticklabels(["embed", "final norm", "lm_head"], fontsize=8, rotation=30, ha="right")
    ax_o.set_ylim(bottom=0)
    ax_o.set_title("Other blocks", color=INK_2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def module_heatmap(df, out, title):
    layers = df[df["block"].str.isdigit()].copy()
    layers["layer"] = layers["block"].astype(int)
    grid = layers.groupby(["module", "layer"])["rel_diff"].mean().unstack("layer").reindex(MODULES)
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    heat(ax, grid.values, "Decoder layer", [m.split(".")[-1] for m in MODULES],
         "Mean ‖Δθ‖₂ / ‖θ‖₂ during attack", fig)
    ax.invert_yaxis()
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(grid.columns, fontsize=8, color=INK_2)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def by_lr(blocks, out, title):
    layers = blocks[blocks["block"].str.isdigit()].copy()
    layers["layer"] = layers["block"].astype(int)
    lrs = sorted(layers["lr"].unique())
    colors = LR_STEPS[-len(lrs):] if len(lrs) <= len(LR_STEPS) else None
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=SURFACE)
    style(ax)
    for lr, color in zip(lrs, colors or [SERIES] * len(lrs)):
        prof = layers[layers["lr"] == lr].groupby("layer")["rel"].mean()
        n = layers[layers["lr"] == lr]["attack"].nunique()
        ax.plot(prof.index, prof.values, color=color, linewidth=2, label=f"lr {lr:g} (n={n})")
    ax.set_yscale("log")
    ax.set_xlabel("Decoder layer", color=INK_2, fontsize=9)
    ax.set_ylabel("Mean ‖Δθ‖₂ / ‖θ‖₂ during attack (log)", color=INK_2, fontsize=9)
    ax.set_xticks(range(32))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=min(len(lrs), 4), loc="upper left")
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run output dir containing attack_displacement.csv")
    args = parser.parse_args()
    df = load(os.path.join(args.run_dir, "attack_displacement.csv"))
    df["block"] = df["block"].astype(str)
    blocks = per_block(df)
    name = os.path.basename(os.path.normpath(args.run_dir))
    n = df["attack"].nunique()
    share_heatmap(blocks, os.path.join(args.run_dir, "attack_disp_share_heatmap.png"),
                  f"Where attacks move the weights over training — {name}")
    profile(blocks, os.path.join(args.run_dir, "attack_disp_profile.png"),
            f"Per-layer attack displacement, early vs late training — {name}")
    module_heatmap(df, os.path.join(args.run_dir, "attack_disp_modules_heatmap.png"),
                   f"Per-module attack displacement (mean over {n} attacks) — {name}")
    by_lr(blocks, os.path.join(args.run_dir, "attack_disp_by_lr.png"),
          f"Attack displacement by sampled LR — {name}")
    print(f"Wrote plots for {n} attacks to {args.run_dir}")


if __name__ == "__main__":
    main()
