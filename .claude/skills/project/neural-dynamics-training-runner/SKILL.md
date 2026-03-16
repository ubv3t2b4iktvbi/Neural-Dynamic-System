---
name: neural-dynamics-training-runner
description: Train or rerun the q/h slow-memory latent dynamics model on synthetic or file-based trajectories, choose safe CLI settings, and preserve run artifacts under `runs/neural_dynamic_system/`. Use when Codex needs to launch experiments, benchmark a config change, tune training flags, or collect outputs such as `summary.json`, `history.csv`, and probe reports.
---

# Neural Dynamics Training Runner

Use this skill when the task is primarily to launch, rerun, or stage training experiments.

## Workflow
1. Decide the data source first:
   - synthetic run with `--synthetic_kind`
   - external file with `--data_path`
2. Start with the smallest convincing run before a long experiment.
3. Choose the latent surface explicitly:
   - `q_dim`
   - `h_dim`
   - `koopman_dim`
   - `latent_scheme`
   - `koopman_input_mode`
   - `hidden_coordinate_mode`
4. Choose the curriculum and horizons before changing loss weights.
5. Write outputs to a fresh directory under `runs/neural_dynamic_system/`.
6. Read `summary.json` first, then `history.csv`, then any probe outputs.

## Preferred Run Order
1. `python scripts/run_neural_dynamic_system.py --help`
2. tiny smoke run on synthetic data
3. focused rerun with the changed flags only
4. longer run only after the smoke output is sane

## Guardrails
- Do not reuse an existing `--out_dir` unless the user explicitly wants overwrite behavior.
- Do not mix many architecture, data, and loss changes in the first run.
- Do not claim a change helped unless `summary.json` or `history.csv` shows it.
- When `--data_path` is used, verify trajectory and label alignment with `$trajectory-label-prepper` first if shapes are uncertain.

## References
- Artifact inventory and smoke-command patterns: [references/run-artifacts.md](references/run-artifacts.md)
