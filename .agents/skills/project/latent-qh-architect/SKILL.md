---
name: latent-qh-architect
description: Modify or analyze the q/h latent architecture, Koopman spectrum settings, hidden-state parameterization, encoder options, and loss couplings without breaking the current training contract. Use when Codex needs to change `q_dim`, `h_dim`, `latent_scheme`, `koopman_input_mode`, `hidden_coordinate_mode`, model internals, or architecture-facing CLI/config plumbing.
---

# Latent QH Architect

Use this skill when the user is changing the model's q/h latent structure or the training assumptions tied to it.

## Workflow
1. Map the requested change onto the code surface before editing:
   - config dataclasses
   - CLI arguments
   - model implementation
   - training losses and summaries
2. Preserve the current contract between `cli.py`, `config.py`, `model.py`, and `training.py`.
3. Make architecture changes in the narrowest layer that can hold them.
4. If a config value changes the user-facing surface, keep the CLI, dataclass validation, serialization, and saved summaries aligned.
5. Verify with `--help`, `compileall`, and a tiny synthetic run when behavior changed.

## Common Surfaces
- Latent sizing: `q_dim`, `h_dim`, `koopman_dim`
- Modal behavior: `latent_scheme`, `modal_dim`, `modal_temperature`
- Encoder behavior: `encoder_type`, `encoder_levels`, `encoder_kernel_size`, `hidden_dim`, `depth`
- Koopman wiring: `koopman_input_mode`, VAMP settings, rate constraints
- Hidden-state geometry: `hidden_coordinate_mode`, `hidden_rank`, hidden SSM parameters
- Training couplings: semigroup, RG, contraction, separation, metric, and supervision losses

## Guardrails
- Do not change only the CLI or only the dataclass; keep both in sync.
- Preserve alias compatibility for `h_dim` and `m_dim`, and for hidden-memory loss or supervision aliases when those surfaces are touched.
- Respect existing config invariants such as `koopman_dim >= q_dim`, positive rates, and positive RG temperature.
- Do not add a new architectural branch without deciding how it is serialized into `config.json` and summarized in `summary.json`.

## References
- Cross-file coupling map for architecture changes: [references/architecture-map.md](references/architecture-map.md)
