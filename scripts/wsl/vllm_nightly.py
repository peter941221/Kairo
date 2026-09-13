#!/usr/bin/env python3
"""Launch the isolated vLLM nightly wheel with the shared GPU dependencies.

The nightly wheel and Torch live in ``venv-vllm-nightly`` while compatible
Python dependencies are reused from ``venv-gpu``. Import Torch first, then put
the shared site-packages after the nightly site so the nightly vLLM package is
never shadowed by the pinned vLLM 0.29.0 package.
"""

from __future__ import annotations

import sys
from pathlib import Path


GPU_SITE = Path("/home/peter/venv-gpu/lib/python3.12/site-packages")


def main() -> None:
    if str(GPU_SITE) not in sys.path:
        nightly_site = next(
            (
                index
                for index, entry in enumerate(sys.path)
                if "venv-vllm-nightly" in entry
                and Path(entry).name == "site-packages"
            ),
            len(sys.path) - 1,
        )
        sys.path.insert(nightly_site + 1, str(GPU_SITE))
    # Put shared dependencies ahead of system packages before Torch imports
    # typing_extensions/pydantic; the nightly Torch package remains first.
    import torch  # noqa: F401  # must resolve from the nightly environment

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
