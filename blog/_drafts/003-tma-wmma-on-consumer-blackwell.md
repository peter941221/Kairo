---
layout: default
title: "TMA + WMMA on Consumer Blackwell"
description: "A correctness-first native CUDA baseline on an RTX 5090."
---

# TMA + WMMA on Consumer Blackwell

Kairo's native-kernel work starts with a verified data-movement path, not a
performance headline.

## Capability boundary

```text
+-------------------------+------------------------------+
| probe                   | result                       |
+-------------------------+------------------------------+
| 2D TMA copy             | correct, max error 0         |
| tcgen05 path            | unavailable in current stack|
| TMA + WMMA, 1K          | 35.2 TFLOP/s, correct       |
| TMA + WMMA, 4K          | 41.6 TFLOP/s, correct       |
| m256 swizzle            | rejected, max error 18.52   |
+-------------------------+------------------------------+
```

The current implementation is not a cuBLAS replacement. Its value is that it
closes the SM120 TMA + WMMA path and identifies the next constraint: the TMA
producer's shared-memory layout must match the WMMA consumer.

## Phase 2 target

Measure occupancy, K-tile staging, synchronization, and compatible swizzled
loads. Promote a native component only after correctness, boundary coverage,
repeated measurements, and a fair end-to-end comparison pass.

See the [TMA protocol](../experiments/protocols/tma-wmma-gemm-phase1.yaml) and
the [Kairo repository](https://github.com/peter941221/Kairo).
