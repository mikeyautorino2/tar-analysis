"""Convert a merged TAR checkpoint to bf16 HF shards, add the tokenizer and model card, and upload it."""

import argparse
import os
import shutil

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--readme", required=True)
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra files/dirs to upload alongside the model")
    args = parser.parse_args()

    # Convert fp32 -> bf16 and re-save as 5GB safetensors shards
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.config.use_cache = True  # was disabled for gradient checkpointing during training
    model.config._name_or_path = args.repo_id
    model.save_pretrained(args.out_dir, max_shard_size="5GB", safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(args.out_dir)
    shutil.copy(args.readme, os.path.join(args.out_dir, "README.md"))
    for path in args.extra:
        dest = os.path.join(args.out_dir, os.path.basename(path.rstrip("/")))
        if os.path.isdir(path):
            shutil.copytree(path, dest, dirs_exist_ok=True)
        else:
            shutil.copy(path, dest)

    # Sanity check: reload from the staged folder and generate a few tokens
    model = AutoModelForCausalLM.from_pretrained(
        args.out_dir, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.out_dir)
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    output = model.generate(inputs, max_new_tokens=30, do_sample=False)
    print("Sanity generation:", tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True))

    print("Staged files:", sorted(os.listdir(args.out_dir)))
    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=args.out_dir,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload {args.repo_id.split('/')[-1]} (bf16)",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
