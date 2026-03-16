---
name: factor-review-archivist
description: Archive mined factors, candidate tables, and validation runs cleanly for human review or later comparison. Use when Codex is packaging outputs, writing review queues, checking manifest completeness, or preserving the evidence trail behind selected factors.
---

# Factor Review Archivist

Keep every mining run reproducible, reviewable, and easy to diff.

## Required artifacts per run
- `candidate_scores.csv`
- `selected_factor_library.json`
- `metrics.json`
- `manual_review.md`
- `run_summary.md`
- `manifest.txt`
- `finance_to_dynamics_translation.md`

## Workflow
1. Check whether selected factors are mechanistically distinct.
2. Downgrade factors that only help one-step error but harm rollout stability.
3. Flag all physics-identifier-dependent factors for human interpretation.
4. Promote only stable factors into curated libraries.

## Guardrails
- Never overwrite a run directory without preserving a manifest.
- Keep human-readable markdown next to machine-readable JSON/CSV.
- Archive the translation table so reviewers can trace finance language back to dynamics language.
