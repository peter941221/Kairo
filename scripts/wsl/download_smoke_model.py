"""Download a public Kairo model into the WSL filesystem."""

import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = args.output or f"/home/peter/kairo-models/{args.model.split('/')[-1]}"
    path = snapshot_download(
        args.model,
        local_dir=output,
    )
    print(path)
