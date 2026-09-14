---
layout: default
title: Kairo Notes
---

<div class="hero">

<div class="eyebrow">BLACKWELL / INFERENCE / EXPERIMENTS</div>

# Kairo Notes

Evidence-driven Blackwell inference research: measure a workload, prove a
result, and route only what was measured.

<p class="status"><span class="dot"></span> PHASE 1 COMPLETE · PHASE 2 IN PROGRESS</p>

</div>

## The work

<div class="grid">

<div class="card"><div class="eyebrow">SERVING</div><div class="metric">2.09–2.62x</div><p>Measured CUDA Graph throughput gain across selected Qwen NVFP4 workload buckets.</p></div>

<div class="card"><div class="eyebrow">NATIVE PATH</div><div class="metric">SM120</div><p>TMA + WMMA correctness baseline on a consumer Blackwell GPU.</p></div>

</div>

## Phase 1

### [Building Kairo: From Measured Routes to Native Blackwell Components](001-when-cuda-graphs-actually-help.html)

Measured CUDA Graph routing on an RTX 5090, the limits of the current native
TMA + WMMA path, and the Phase 2 direction toward native Blackwell components.

- [Kairo repository](https://github.com/peter941221/Kairo)
- [Phase 1 results](https://github.com/peter941221/Kairo/blob/main/docs/PHASE_1_RESULTS.md)
- [Public evidence](https://github.com/peter941221/Kairo/blob/main/evidence/phase1/serving-benchmark-repeats.jsonl)

The repository is the source of code and evidence; this page is the readable
article index.
