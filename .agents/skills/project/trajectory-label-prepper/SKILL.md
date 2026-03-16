---
name: trajectory-label-prepper
description: Prepare, inspect, and validate trajectory or label inputs for Neural Dynamic System across `.npy`, `.npz`, and `.csv` sources, including multi-episode arrays and supervision labels. Use when Codex needs to ingest external time series, debug shape or array-key issues, align labels with episodes, or decide q/h label index mappings before training.
---

# Trajectory Label Prepper

Use this skill when the task is about getting data into the repository cleanly.

## Workflow
1. Identify whether the source is trajectory data, labels, or both.
2. Confirm the file format:
   - `.npy`
   - `.npz`
   - `.csv`
3. Normalize the mental model to episodes first, even if the input is a single array.
4. Verify the trajectory can satisfy `context_len + max(horizon)` before training.
5. If labels are present, verify:
   - same episode count as the trajectory
   - same step count inside each episode
   - label columns match the intended `q` and `h` supervision indices
6. Only then wire the paths and index flags into the training command.

## Data Rules
- Treat 1D input as one feature over time.
- Treat 2D input as one episode with shape `[T, D]`.
- Treat 3D input as multiple episodes with shape `[E, T, D]`.
- For object arrays or episode lists, each episode must be rank-2 after coercion.
- Label arrays follow the same episode structure as trajectory arrays.

## Guardrails
- Do not assume `.npz` key names; use `--array_key` or `--label_array_key` when needed.
- Do not pass label indices before confirming the actual label column order.
- Do not start a long run if the dataset barely satisfies the window and horizon requirements.
- If the user only needs a quick demo or regression check, prefer synthetic data over inventing a file-conversion pipeline.

## References
- Supported shapes, file rules, and supervision mapping notes: [references/data-contract.md](references/data-contract.md)
