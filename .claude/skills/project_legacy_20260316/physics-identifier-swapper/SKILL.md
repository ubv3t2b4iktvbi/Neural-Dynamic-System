---
name: physics-identifier-swapper
description: Change, compare, or extend the physics-identification backbone without rewriting the RC screening pipeline. Use when Codex is swapping identifier families, adding a new identifier backend, or checking that identifier outputs stay compatible with the factor bank and review flow.
---

# Physics Identifier Swapper

Swap the physics-identification layer while keeping the factor bank and RC proxy pipeline stable.

## Current backbones
- `sindy_slow`: sparse slow-manifold drift identification
- `spline_kan_like`: spline-basis additive surrogate with KAN-like flavor and no external KAN dependency
- `none`: no explicit physics identifier

## Workflow
1. State the target mechanism to identify: slow drift, local closure, regime residual, or nonlinear control response.
2. Keep the identifier interface unchanged:
   - `fit(features)`
   - `batch_outputs(features)`
   - `step_outputs(ctx)`
3. Expose at least these outputs:
   - `id_drift_pred`
   - `id_drift_pred_norm`
   - `id_drift_surprise`
   - `id_drift_surprise_norm`
   - `id_drift_alignment`
4. Re-run `scripts/run_factor_mining.py` with both the old and new identifiers.
5. Queue the changed factors for manual review.

## Guardrails
- A new identifier must improve interpretation or search space, not merely add parameters.
- If a true external KAN package is unavailable, label the fallback honestly as KAN-like rather than KAN.
- Keep one-step screening cheap enough for broad candidate search.
