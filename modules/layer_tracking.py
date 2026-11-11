"""Per-layer / per-module weight displacement for FSDP v1 models.

Used to measure how far each layer moves during a simulated attack in the TAR inner loop,
by comparing the live sharded parameters against the pre-attack snapshot already held in
FSDPModelStorage. Relies on FSDP v1 internals (FlatParameter._shard_param_infos and
_get_param_to_fqns), which match the pinned torch 2.4.
"""

import re
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed.fsdp._common_utils import _get_param_to_fqns

from modules.fsdp_v1_utils import FSDPModelStorage, fsdp_v1_model_params


def block_of(name: str):
    """Return (block, module) for a parameter name, e.g. (5, "mlp.down_proj") or ("lm_head", "lm_head")."""
    m = re.match(r"model\.layers\.(\d+)\.(.+)\.weight$", name)
    if m:
        return int(m.group(1)), m.group(2)
    if name.startswith("model.embed_tokens"):
        return "embed", "embed_tokens"
    if name.startswith("model.norm"):
        return "norm", "norm"
    if name.startswith("lm_head"):
        return "lm_head", "lm_head"
    raise ValueError(f"Unexpected parameter {name}")


class ShardIndex:
    """Maps each FSDP flat-param shard on this rank back to (block, module) slices."""

    def __init__(self, model: torch.nn.Module):
        param_to_fqns = _get_param_to_fqns(model)
        self.entries: List[Tuple] = []  # (block, module), same order on every rank
        entry_ids: Dict[Tuple, int] = {}
        # slices[i] = [(entry_id, offset_in_shard, numel_in_shard), ...] for flat param i
        self.slices: List[List[Tuple[int, int, int]]] = []
        for param in fsdp_v1_model_params(model):
            fqns = param_to_fqns[param]
            infos = getattr(param, "_shard_param_infos", None)
            param_slices = []
            if infos is None:  # not FSDP-managed: whole tensor is local and unsharded
                infos = [(True, 0, param.numel())]
            for fqn, info in zip(fqns, infos):
                key = block_of(fqn)
                if key not in entry_ids:
                    entry_ids[key] = len(self.entries)
                    self.entries.append(key)
                if info[0]:  # in_shard
                    param_slices.append((entry_ids[key], info[1], info[2]))
            self.slices.append(param_slices)


@torch.no_grad()
def perturb_weights(
    model: torch.nn.Module, index: ShardIndex, scale: float, generator: torch.Generator
):
    """
    Add Gaussian noise to every weight tensor in place: W += N(0, (scale * rms(W))^2).

    Each (block, module) entry is one weight tensor (Llama has no biases), so its RMS is
    reduced over the shards on all ranks. Returns (perturbation, rel): perturbation holds
    what attack_displacement needs to regenerate the same noise, and rel is ||eps|| / ||W||
    over the whole model.
    """
    device = next(model.parameters()).device
    sq = torch.zeros(len(index.entries), 2, dtype=torch.float64, device=device)
    for i, param in enumerate(fsdp_v1_model_params(model)):
        flat = param.data.view(-1)
        for entry_id, offset, numel in index.slices[i]:
            sq[entry_id, 0] += flat[offset : offset + numel].float().pow(2).sum(dtype=torch.float64)
            sq[entry_id, 1] += numel
    if dist.is_initialized():
        dist.all_reduce(sq, op=dist.ReduceOp.SUM)
    rms = (sq[:, 0] / sq[:, 1]).sqrt().float()

    generator_state = generator.get_state()
    noise_sq = torch.zeros(1, dtype=torch.float64, device=device)
    for i, param in enumerate(fsdp_v1_model_params(model)):
        flat = param.data.view(-1)
        for entry_id, offset, numel in index.slices[i]:
            noise = torch.randn(numel, generator=generator, device=device) * (scale * rms[entry_id])
            flat[offset : offset + numel] += noise.to(flat.dtype)
            noise_sq += noise.pow(2).sum(dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(noise_sq, op=dist.ReduceOp.SUM)
    rel = (noise_sq / sq[:, 0].sum()).sqrt().item()
    perturbation = {"generator": generator, "state": generator_state, "rms": rms, "scale": scale}
    return perturbation, rel


@torch.no_grad()
def attack_displacement(
    model: torch.nn.Module,
    model_storage: FSDPModelStorage,
    index: ShardIndex,
    perturbation: dict = None,
) -> List[dict]:
    """
    Compare the live parameters with the pre-attack snapshot in model_storage.

    Returns one dict per (block, module) with numel, l2_diff = ||theta - theta_ref||_2 and
    rel_diff = l2_diff / ||theta_ref||_2, reduced over all ranks. Works slice by slice,
    so the extra GPU memory is at most one module's local shard.

    If the weights were perturbed before the attack (perturb_weights), pass its returned
    perturbation: the same noise is regenerated so theta_ref is the perturbed starting
    point, i.e. only the movement caused by the attack itself is measured.
    """
    device = next(model.parameters()).device
    if perturbation is not None:
        perturbation["generator"].set_state(perturbation["state"])
    sums = torch.zeros(len(index.entries), 3, dtype=torch.float64, device=device)
    for i, param in enumerate(fsdp_v1_model_params(model)):
        ref_full = model_storage.storage_dict["params"][i]
        live = param.data.view(-1)
        ref_flat = ref_full.view(-1)
        for entry_id, offset, numel in index.slices[i]:
            ref = ref_flat[offset : offset + numel].to(device, dtype=torch.float32)
            if perturbation is not None:
                ref = ref + torch.randn(  # out of place: never modify the stored snapshot
                    numel, generator=perturbation["generator"], device=device
                ) * (perturbation["scale"] * perturbation["rms"][entry_id])
            cur = live[offset : offset + numel].float()
            sums[entry_id, 0] += (cur - ref).pow(2).sum(dtype=torch.float64)
            sums[entry_id, 1] += ref.pow(2).sum(dtype=torch.float64)
            sums[entry_id, 2] += numel
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    sums = sums.cpu()
    results = []
    for (block, module), (d2, b2, n) in zip(index.entries, sums.tolist()):
        results.append({
            "block": block,
            "module": module,
            "numel": int(n),
            "l2_diff": d2 ** 0.5,
            "rel_diff": (d2 / b2) ** 0.5 if b2 > 0 else 0.0,
            "_d2": d2,
            "_b2": b2,
        })
    return results


def per_block_summary(results: List[dict]) -> Dict[str, float]:
    """Aggregate module results into one relative displacement per block (plus total)."""
    blocks: Dict = {}
    for r in results:
        d2, b2 = blocks.get(r["block"], (0.0, 0.0))
        blocks[r["block"]] = (d2 + r["_d2"], b2 + r["_b2"])
    summary = {}
    for block, (d2, b2) in blocks.items():
        key = f"layer_{block:02d}" if isinstance(block, int) else block
        summary[key] = (d2 / b2) ** 0.5 if b2 > 0 else 0.0
    total_d2 = sum(d2 for d2, _ in blocks.values())
    total_b2 = sum(b2 for _, b2 in blocks.values())
    summary["total"] = (total_d2 / total_b2) ** 0.5 if total_b2 > 0 else 0.0
    return summary
