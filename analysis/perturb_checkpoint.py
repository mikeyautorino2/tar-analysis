"""Make a randomly perturbed copy of a checkpoint for the Qi et al. attack evaluation.

Every tensor W gets W + eps, with eps ~ N(0, (scale * rms(W))^2): the same form as
--weight_perturb_scale in training (modules/layer_tracking.perturb_weights), applied once and
offline to whole tensors. The result is saved as an fp32 model.safetensors. Config, generation
config and tokenizer files are symlinked from an existing eval_models folder, so finetune.py and
vLLM can load it like the clean model.

The report gives the realised relative noise ||eps|| / ||theta|| in fp32 and after a bf16 cast.
The attack loads weights in bf16 (about 0.4% precision per weight), so a small perturbation is
partly rounded away there.
"""

import argparse
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_diff import tensor_locations  # noqa: E402

LINKED_FILES = ["config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Checkpoint to perturb (hub id or local dir)")
    p.add_argument("--link_from", required=True, help="Existing eval_models dir with config + tokenizer files")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scale", type=float, required=True, help="Noise std as a fraction of each tensor's RMS")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    generator = torch.Generator().manual_seed(args.seed)
    locations = tensor_locations(args.source)
    state, sums = {}, {"w2": 0.0, "eps2": 0.0, "bf16_eps2": 0.0}
    for name in sorted(locations):  # fixed order, so the same seed gives the same noise
        with safe_open(locations[name], "pt", device="cpu") as f:
            w = f.get_tensor(name).float()
        rms = w.pow(2).mean().sqrt()
        eps = torch.randn(w.shape, generator=generator) * (args.scale * rms)
        noisy = w + eps
        sums["w2"] += w.pow(2).sum(dtype=torch.float64).item()
        sums["eps2"] += eps.pow(2).sum(dtype=torch.float64).item()
        sums["bf16_eps2"] += (noisy.bfloat16().float() - w.bfloat16().float()).pow(2).sum(dtype=torch.float64).item()
        state[name] = noisy.contiguous()

    save_file(state, os.path.join(args.out_dir, "model.safetensors"), metadata={"format": "pt"})
    for fname in LINKED_FILES:
        src = os.path.realpath(os.path.join(args.link_from, fname))
        dst = os.path.join(args.out_dir, fname)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    rel = (sums["eps2"] / sums["w2"]) ** 0.5
    rel_bf16 = (sums["bf16_eps2"] / sums["w2"]) ** 0.5
    print(f"{os.path.basename(args.out_dir)}: scale={args.scale} seed={args.seed} "
          f"tensors={len(state)}  ||eps||/||theta|| fp32={rel:.5f}  after bf16 cast={rel_bf16:.5f}")


if __name__ == "__main__":
    main()
