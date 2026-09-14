# Contributing to Kairo

Kairo is a measurement-driven Blackwell inference research workbench. Useful
contributions make an experiment more reproducible, a decision boundary more
explicit, or a measured execution path more correct and effective.

## Before opening a pull request

Run the portable contract suite from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -q
```

GPU-dependent probes are local by design. Do not report a GPU result as a
project result unless its protocol records the relevant hardware, software,
model, workload, correctness method, repeat policy, and fair baseline.

## Evidence standard

Performance changes must:

1. Keep model weights, actual token counts, workload, warm-up policy, and
   relevant runtime settings controlled.
2. Pass the declared numerical or semantic correctness gate before a speed claim.
3. Record the result in a versioned experiment protocol.
4. Preserve raw output locally and publish a reviewed, redacted artifact when a
   public headline depends on it.
5. Retain negative results when they establish a useful boundary or reject an
   invalid design.

Unmeasured workload shapes must not be promoted by extrapolation. Kairo's
default is to fail closed to manual selection.

## Repository hygiene

- Never commit checkpoints, access tokens, credentials, private logs, or machine
  notes. `.kairo-local/` is reserved for those local materials.
- Keep generated traces and large artifacts out of Git unless they are a
  reviewed public evidence bundle.
- Do not add unrelated editorial material to `blog/`.
- Add tests for changes to portable Python behavior and update the corresponding
  protocol or documentation when a decision changes.

## Pull requests

Keep each pull request narrow. Explain the hypothesis, the changed evidence or
code path, the validation command, and any known limitation. A result that
cannot yet satisfy Kairo's promotion gate is welcome when it is clearly labeled
as exploratory.

## License

By submitting a contribution, you agree that it is licensed under the
[Apache License 2.0](LICENSE), consistent with Section 5 of that license.
