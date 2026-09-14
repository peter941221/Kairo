# Public release checklist

Kairo should be published from this repository, preserving the existing
experiment history. A separate public repository is unnecessary unless a future
history audit finds credentials, private data, or copyrighted material that
cannot be removed safely.

## Repository boundary

The public repository contains source, tests, declarative protocols, concise
aggregated results, and publication drafts. Local logs, checkpoints, traces,
credentials, and machine notes stay in `.kairo-local/` or other ignored output
directories. The unrelated MuseProbe editorial reference is explicitly ignored
and must not be staged for Kairo.

## Required before changing repository visibility

- [x] Apache-2.0 license and `NOTICE` added.
- [ ] Add a short `CONTRIBUTING.md` with the test command and evidence standard.
- [x] Export headline per-repeat records as a reviewed, redacted public JSONL
      bundle; do not publish credentials, local network addresses, or server logs.
- [x] Add a dependency-free script that regenerates the published table and SVG
      chart from that bundle.
- [x] Replace public quick-start and primary serving-runner personal paths with
      `KAIRO_*` environment variables; legacy probe scripts remain local tools.
- [ ] Re-run portable checks: `PYTHONPATH=src python -m unittest discover -s tests -q`.
- [ ] Inspect both the staged diff and the full Git history for secrets before
      publishing.

## Recommended release sequence

1. Complete this checklist and tag the first public Phase 1 release.
2. Publish the repository and the Phase 1 article from `blog/001-when-cuda-graphs-actually-help.md`.
3. Submit a regular technical article to Hacker News if the evidence bundle is
   readable; use `Show HN` only when visitors can run or inspect the project
   without a private GPU setup.
4. Publish short derivative posts elsewhere that link back to the canonical
   repository and article.
