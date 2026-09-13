# WSL environment contract

The canonical development environment is Ubuntu 24.04 under WSL2, with the host
RTX 5090 exposed through NVIDIA's WSL driver integration.

## Current local probe

- GPU: NVIDIA GeForce RTX 5090, 32 GiB
- Driver: 596.36
- WSL distribution: Ubuntu 24.04
- System CUDA compiler: 12.0

The last item is a blocker for compiling native RTX 5090-targeted CUDA kernels.
CUDA 12.8 or newer must be installed in WSL before Phase 0 custom-kernel work.
The Python bootstrap uses a CUDA 12.8 PyTorch wheel only for framework baseline
profiling; it does not upgrade `nvcc`.

## Bootstrap baseline Python

```bash
cd /mnt/c/Projects/Kairo
bash scripts/wsl/bootstrap_python.sh
bash scripts/wsl/doctor.sh
```

## Environment acceptance gate

Before publishing any result, record the outputs of `doctor.sh`, the container or
Python package versions, GPU driver, model revision, and Git revision. Do not
mix framework runtime CUDA with a system compiler that cannot target RTX 5090
when building custom extensions.
