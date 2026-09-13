# WSL environment contract

The canonical development environment is Ubuntu 24.04 under WSL2, with the host
RTX 5090 exposed through NVIDIA's WSL driver integration.

## Current local probe

- GPU: NVIDIA GeForce RTX 5090, 32 GiB
- Driver: 596.36
- WSL distribution: Ubuntu 24.04
- CUDA toolchains: 12.0 is still the default `PATH` compiler; CUDA 12.8 and
  CUDA 13.0 are also installed, and Kairo's WSL doctor/CLI prefer CUDA 13.0.

CUDA 12.8 or newer is required for compiling native RTX 5090-targeted CUDA
kernels; the existing CUDA 13.0 toolchain satisfies that version gate.
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
