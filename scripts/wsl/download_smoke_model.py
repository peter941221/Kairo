"""Download the small, public smoke-test model into the WSL filesystem."""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    path = snapshot_download(
        "Qwen/Qwen2.5-0.5B-Instruct",
        local_dir="/home/peter/kairo-models/Qwen2.5-0.5B-Instruct",
    )
    print(path)
