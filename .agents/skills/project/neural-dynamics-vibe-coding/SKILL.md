---
name: neural-dynamics-vibe-coding
description: Maintain, debug, refactor, and extend the Neural Dynamic System repository with repo-aware file placement and minimal-risk validation. Use when Codex needs to make general code changes, wire CLI flags, adjust configs, update training/data/model logic, or sync documentation without breaking the q/h slow-memory workflow.
---

# Neural Dynamics Vibe Coding

Use this skill for general repository work in the current `Neural Dynamic System` project.

## First Pass
1. Read `README.md` and [references/repo-map.md](references/repo-map.md) before large edits.
2. Run `git status --short` and protect unrelated user changes.
3. Classify the request as one of:
   - CLI or run orchestration
   - config or hyperparameter plumbing
   - trajectory or label data handling
   - latent architecture or training loss changes
   - result summarization or docs
4. Keep changes native to the existing `scripts/` plus `neural_dynamic_system/` layout.

## Plan -> Execute -> Verify
1. Restate the target behavior and the narrowest file set likely to change.
2. Prefer reusing the existing dataclasses, CLI flags, and model/training contracts over adding parallel code paths.
3. Keep `scripts/run_neural_dynamic_system.py` as the thin entrypoint and put reusable logic under `neural_dynamic_system/`.
4. Run the smallest relevant check from [references/repo-map.md](references/repo-map.md).
5. Report changed files, verification, generated run paths, and any remaining risk.

## Placement Rules
- Put argument parsing and output writing in `neural_dynamic_system/cli.py`.
- Put config surface changes in `neural_dynamic_system/config.py`.
- Put trajectory loading and dataset behavior in `neural_dynamic_system/data.py`.
- Put synthetic generators in `neural_dynamic_system/synthetic.py`.
- Put latent architecture and flow logic in `neural_dynamic_system/model.py`.
- Put training-loop and loss changes in `neural_dynamic_system/training.py`.
- Update `README.md` when user-facing behavior or recommended commands change.

## Guardrails
- Do not overwrite existing run directories under `runs/` unless the user explicitly asks.
- Do not add a flag in `cli.py` without plumbing it into the matching config or implementation layer.
- Do not bypass `ModelConfig`, `LossConfig`, `TrainConfig`, or `SupervisionConfig` with ad hoc dictionaries.
- Prefer `python scripts/run_neural_dynamic_system.py --help`, `python -m compileall neural_dynamic_system scripts`, or a tiny synthetic run before expensive training.

## Pair With Specialized Skills
- Use `$neural-dynamics-training-runner` for experiment launches or reruns.
- Use `$trajectory-label-prepper` for input-shape, label-alignment, or file-format work.
- Use `$latent-qh-architect` for q/h architecture and loss-coupling changes.
- Use `$run-artifact-summarizer` when the main task is interpreting completed runs.
