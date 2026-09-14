---
layout: default
title: "Benchmarks That Can Say No"
description: "The evidence contract behind Kairo's Blackwell inference measurements."
---

# Benchmarks That Can Say No

GPU performance claims fail when the comparison silently changes. Kairo treats
the evidence behind a claim as an interface.

## The six required fields

```text
+----------------------+----------------------------------------------+
| field                | required identity                            |
+----------------------+----------------------------------------------+
| model                | checkpoint and weight revision               |
| environment          | GPU, driver, CUDA, framework, source commit  |
| workload             | actual input/output tokens and concurrency   |
| variable             | the one setting changed between A and B      |
| correctness           | deterministic numerical or semantic gate    |
| artifact              | raw repeats, command, and derived table      |
+----------------------+----------------------------------------------+
```

An unknown shape is not a nearby known shape. Kairo returns `manual` instead of
silently extrapolating a measured winner.

## Negative results are data

The 27B long-context c16 route produced only a 1.30x Graph lead. Raising the
SGLang Mamba budget helped short prompts but not the 2K prompt. A TMA swizzle
variant compiled but failed the correctness gate with 18.52 maximum error.

Each result narrows the next experiment. A protocol should be able to reject a
claim.

See [Reproduce Phase 1](../docs/REPRODUCE_PHASE1.md), the public JSONL evidence,
and the rendering script in the [Kairo repository](https://github.com/peter941221/Kairo).
