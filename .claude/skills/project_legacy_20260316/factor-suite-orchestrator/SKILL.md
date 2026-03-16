---
name: factor-suite-orchestrator
description: Run factor mining across benchmark suites, identifiers, seeds, and manual checkpoints in a structured order. Use when Codex needs to stage experiments, narrow task lists, compare identifiers, or turn one large sweep into a reproducible research loop.
---

# Factor Suite Orchestrator

Run factor mining as a staged research loop rather than as one opaque sweep.

## Order
1. Smoke tasks to verify implementation.
2. Slow-fast friendly tasks to verify factor semantics.
3. Chaotic/high-dimensional tasks to check whether the same factors survive sparse observation.
4. Multi-seed reruns for shortlisted factors only.
5. Human review before promotion into a persistent factor library.

## Workflow
1. Fix the config in `configs/factor_mining.yaml`.
2. Run `scripts/run_factor_mining.py` on a narrow task set first.
3. Compare identifier families separately before merging conclusions.
4. Save per-task artifacts before building cross-task summaries.
5. Use the upstream dynamics-research skills for deeper evidence grading and experiment redesign.

## Guardrails
- Do not jump directly to large suites before the smoke suite passes.
- Separate candidate generation, validation, and narrative writing.
- Treat cross-task consistency as stronger evidence than a single-task gain.
