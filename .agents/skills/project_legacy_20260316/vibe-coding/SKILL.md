---
name: vibe-coding
description: Maintain, debug, refactor, and extend the DynamicAlpha Lab codebase with repo-aware file placement and minimal-risk validation. Use when Codex needs to make general code changes in this repository, especially to wire new models, factors, identifiers, configs, CLI flags, docs, or research-loop behavior into the existing `scripts/` + `src/fsrc_sindy/` architecture without bypassing registries, review artifacts, or smoke checks.
---

# Vibe Coding

Keep edits native to the repository. Follow a short plan -> execute -> verify loop, prefer existing registries and orchestrators, and leave the codebase easier for the next agent to resume.

## First Pass
1. Read `scripts/README.md`, `src/README.md`, and `src/fsrc_sindy/README.md` before large edits.
2. Run `git status --short` and protect unrelated user changes.
3. Classify the request as entrypoint, model/benchmark, factor-mining, coordinate-analysis, research-loop, or docs/config work.
4. Read `references/repo-map.md` whenever file placement or validation scope is unclear.

## Plan -> Execute -> Verify
1. Restate the target behavior and the narrowest file set likely to change.
2. Prefer extending existing registries, suites, and manifests over creating parallel scripts or duplicate logic.
3. Keep CLI parsing in `scripts/` and reusable logic in `src/fsrc_sindy/`.
4. Run the smallest relevant check from `references/repo-map.md`.
5. Report changed files, verification, generated output paths, and any remaining risk.

## Placement Rules
- Put reusable implementation in `src/fsrc_sindy/`; keep `scripts/` thin wrappers.
- Wire new models through `models/`, `selection.py`, and any benchmark surface that exposes them.
- Wire new factors or features through the factor bank, screening/selection, and review/archive outputs together.
- Keep new identifier backends compatible with the factor-mining flow instead of branching around it.
- Attach research-loop changes to `research/loop.py` and the corresponding run script instead of adding a second top-level workflow.
- Update the nearest README or config example when user-visible behavior changes.

## Guardrails
- Do not overwrite historical outputs in `runs/`, `archive/`, or `data/` unless the user explicitly asks.
- Do not hardcode machine-specific paths outside repo-relative config/output locations.
- Do not bypass `selection.py`, `factors/identifiers.py`, `factors/factor_bank.py`, or `research/loop.py` with ad hoc one-off logic when a registry already exists.
- Prefer smoke suites, targeted CLI `--help`, or narrow compile checks before expensive runs.
- Keep `confidence_report.json`, `expert_review_template.md`, review queues, and manifests intact when touching research-loop or archive code.

## Pair With Specialized Skills
- Use this skill alone for generic maintenance, bug fixes, refactors, CLI plumbing, docs sync, and repo cleanup.
- Pair it with the project research skills when the change is specifically about factor design, identifier swaps, experiment orchestration, or theory interpretation.
