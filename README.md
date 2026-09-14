# Kairo

> Evidence-driven Blackwell inference research: measure a workload, prove a
> result, and route only what was measured.

[Phase 1 results](docs/PHASE_1_RESULTS.md) ·
[Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) ·
[Phase 1 article](blog/001-when-cuda-graphs-actually-help.md) ·
[Reproduce Phase 1](docs/REPRODUCE_PHASE1.md) ·
[Blog](https://peter941221.github.io/Kairo/) ·
[Contributing](CONTRIBUTING.md) ·
[Apache-2.0](LICENSE)

## Why Kairo

On a single RTX 5090, CUDA Graph replay can improve NVFP4 serving throughput by
more than 2x for some workloads—and by far less for another valid workload.
The useful conclusion is not “always enable Graphs.” Kairo captures the
conditions behind a result, passes a correctness gate, and promotes only exact
measured workload buckets into a fail-closed runtime policy.

Kairo is a research workbench and the beginning of a path toward native
Blackwell inference components. It is not yet a general-purpose inference
engine or a replacement for vLLM, CUTLASS, or cuBLAS.

## Phase 1 at a glance

Local measurements on one RTX 5090 32GB, using pinned model and software
revisions. Read the linked protocols before comparing values.

| Measured workload | Graph | Eager | Result |
|---|---:|---:|---:|
| Qwen3.8-27B-NVFP4, c32, prompt 512, context 1K | 645.08 tok/s | 308.51 tok/s | 2.09x |
| Qwen3.8-27B-NVFP4, c8, prompt 2048, context 4K | 322.78 tok/s | 123.22 tok/s | 2.62x |
| Qwen3.8-27B-NVFP4, c16, prompt 2048, context 4K | 260.07 tok/s | 199.52 tok/s | 1.30x |
| Qwen3-8B-NVFP4, c16, prompt 512, context 1K | 2177.49 tok/s | 1073.63 tok/s | 2.03x |

The 1.30x result is intentional context: CUDA Graphs are not promoted as a
global default. [Phase 1 results](docs/PHASE_1_RESULTS.md) includes the scope,
evidence links, and claims Kairo does not make.

## What Kairo does

- Defines versioned workload and experiment protocols.
- Verifies declarative kernel blueprints before a build or launch.
- Records cache, correctness, artifact, and launch timing separately.
- Selects serving profiles only for exact measured workload buckets.
- Fails closed to `manual` when no measured route covers a request.
- Probes native SM120 data movement and matrix-compute paths, including
  correctness-aware negative results.

## Quick start

Clone the repository:

```bash
git clone https://github.com/peter941221/Kairo.git
cd Kairo
```

The portable contracts do not require a GPU:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -q
```

Validate the sample kernel blueprint:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && bash scripts/wsl/run_lab.sh validate-blueprint --file experiments/blueprints/fp16-tma-wmma.yaml'
```

Ask the runtime policy about a measured route:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Projects/Kairo && bash scripts/wsl/run_lab.sh recommend-runtime --model qwen38 --concurrency 32 --prompt-tokens 512 --context-tokens 1024 --generation-tokens 128'
```

GPU-serving paths require a compatible RTX 5090 environment, local model
weights, and the pinned runtime described in [the environment contract](docs/ENVIRONMENT.md).
Set model and environment paths through the documented `KAIRO_*` variables;
do not commit local paths or credentials.

For the full evidence-to-serving-route procedure, including the expected route
and the exact c32/1K reproduction command, see [Reproduce Phase 1](docs/REPRODUCE_PHASE1.md).

## Evidence and reproducibility

Every public result should answer: under which model, quantization, workload,
hardware, software revision, and correctness tolerance did it win?

- [Phase 1 results](docs/PHASE_1_RESULTS.md) is the concise public result index.
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) defines controls and the
  promotion gate.
- [Baseline status](docs/BASELINE_STATUS.md) is the detailed chronological lab
  record and includes both promoted and rejected experiments.
- [`experiments/protocols/`](experiments/protocols) contains machine-readable
  result and method records.
- [`evidence/`](evidence) contains reviewed, redacted per-repeat public records;
  `python scripts/render_phase1_results.py` regenerates the published table and
  SVG chart in [`docs/generated/`](docs/generated).
- [Public release checklist](docs/PUBLIC_RELEASE.md) lists the artifact,
  review, and repository gates before public visibility changes.

## Repository map

```text
blog/                 publication drafts and article sources
docs/                 result index, protocol, environment, and release guides
experiments/          versioned workloads, protocols, and kernel blueprints
scripts/wsl/          local RTX 5090 probes, runners, and analyzers
src/kairo_lab/        dependency-light validation, cache, dispatch, and policy
tests/                portable contract tests
.kairo-local/         ignored local credentials, logs, and machine facts
```

## Roadmap

**Phase 1 — complete research baseline.** Controlled protocols, correctness
gates, measured Graph/eager routes, route selection, and native capability
probes.

**Phase 2 — native components.** Use Phase 1 evidence to choose narrowly
scoped data movement, layout, subgraph, or execution-path work. A candidate is
not promoted without correctness, boundary coverage, repeatability, and a fair
end-to-end comparison.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In brief: do not extrapolate measured
wins, do not commit credentials or private artifacts, and retain negative
results that establish a meaningful boundary.

## License

Copyright 2026 Kairo contributors. Licensed under the
[Apache License 2.0](LICENSE). The license grants broad reuse rights, including
an express patent grant for contributed work; it does not grant trademark rights.
