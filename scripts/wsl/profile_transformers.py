#!/usr/bin/env python3
"""In-process CUDA operator profile when Nsight counters are unavailable."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/peter/kairo-models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--trace", type=Path, default=Path(".kairo-local/transformers-profile.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text = "Kairo profiler token. " * (args.prompt_tokens // 4 + 4)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.prompt_tokens)
    input_ids = encoded.input_ids.to("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda").eval()

    # Profile the real prefill or KV-cache decode path; each iteration uses a
    # distinct final token to avoid measuring a framework shortcut.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.inference_mode():
            if args.mode == "prefill":
                for step in range(args.steps):
                    batch = input_ids.clone()
                    batch[:, -1] = (int(batch[0, -1]) + step) % tokenizer.vocab_size
                    model(input_ids=batch, use_cache=False)
                    torch.cuda.synchronize()
            else:
                warm = model(input_ids=input_ids, use_cache=True)
                past = warm.past_key_values
                token = input_ids[:, -1:]
                for step in range(args.steps):
                    out = model(input_ids=token, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    token = out.logits[:, -1:].argmax(dim=-1)
                    torch.cuda.synchronize()

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.trace))
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    print(f"trace={args.trace} mode={args.mode} input_tokens={input_ids.shape[-1]} steps={args.steps}")


if __name__ == "__main__":
    main()
