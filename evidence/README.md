# Public evidence bundles

This directory contains reviewed, redacted benchmark records that support public
claims. It never contains credentials, model weights, local paths, server logs,
or request text.

`phase1/serving-benchmark-repeats.jsonl` contains one record per measured
serving repeat. Every record identifies the pinned model revision, workload,
runtime mode, throughput, correctness result, request success count, and source
protocol. Regenerate the derived table and SVG with:

```powershell
python scripts/render_phase1_results.py
```

The generated files live in `docs/generated/`. A change to the JSONL must be
reviewed against its local raw artifact and protocol before publication.
