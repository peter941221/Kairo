---
layout: default
title: Kairo Labs
---

<p class="eyebrow">Blackwell · LLM inference · RTX 5090</p>

# Notes from an inference lab.

<p class="lede">
Kairo is an evidence-driven workbench for LLM inference on consumer Blackwell
GPUs. Phase 1 measured CUDA Graph routing on an RTX 5090: gains ranged from
1.30x to 2.62x depending on the exact workload, and only routes that passed a
paired benchmark and a correctness gate were promoted.
</p>

<p class="status">
  <span class="dot on"></span> phase 1 complete
  <span class="sep">·</span>
  <span class="dot off"></span> phase 2 in progress
</p>

## Notes

<ol class="notes">
  <li>
    <span class="note-date">2026-09-14 · Phase 1</span>
    <a class="note-title" href="001-when-cuda-graphs-actually-help.html">Building Kairo: from measured routes to native Blackwell components</a>
    <p>The flagship write-up: paired Graph/eager benchmarks, a fail-closed
    routing policy, and the first TMA + WMMA baselines on SM120.</p>
  </li>
  <li>
    <span class="note-date">Draft</span>
    <a class="note-title" href="002-benchmarks-that-can-say-no.html">Benchmarks that can say no</a>
    <p>The evidence contract behind every Kairo number, and why negative
    results stay in the record.</p>
  </li>
  <li>
    <span class="note-date">Draft</span>
    <a class="note-title" href="003-tma-wmma-on-consumer-blackwell.html">TMA + WMMA on consumer Blackwell</a>
    <p>A correctness-first native CUDA baseline, and the shared-memory layout
    constraint that defines Phase 2.</p>
  </li>
</ol>

## Elsewhere

- [Kairo repository](https://github.com/peter941221/Kairo) — code, protocols, runners
- [Phase 1 results](https://github.com/peter941221/Kairo/blob/main/docs/PHASE_1_RESULTS.md) — the measured table behind the numbers
- [Public evidence](https://github.com/peter941221/Kairo/blob/main/evidence/phase1/serving-benchmark-repeats.jsonl) — per-repeat benchmark records
- [Reproduce Phase 1](https://github.com/peter941221/Kairo/blob/main/docs/REPRODUCE_PHASE1.md) — run the paired benchmark on your own RTX 5090

The Markdown source lives in
[`blog/`](https://github.com/peter941221/Kairo/tree/main/blog) in the repository.
