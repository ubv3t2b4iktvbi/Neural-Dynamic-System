---
name: dynamical-theory-reasoner
description: Analyze ablations and coordinate experiments through Markov closure, spectral preservation, separability, and theory-guided follow-up design. Use when Codex needs to interpret benchmark or factor-mining results, compare raw versus delay versus fast-slow versus factor coordinates, or turn empirical patterns into targeted dynamical-systems experiments.
---

# Dynamical Theory Reasoner

Turn experiment tables into mechanistic hypotheses instead of stopping at RMSE.

## Workflow
1. Run `scripts/run_coordinate_analysis.py` on the relevant task or suite before changing architecture.
2. Read `coordinate_summary.csv` and `coordinate_summary.md` first.
3. Interpret each coordinate using three primary lenses:
   - Markov closure: does `z_t` already explain `z_{t+1}`, or does `z_{t-1}` add material value?
   - Local spectral preservation: does the coordinate retain local expansion and contraction patterns?
   - Dynamical separability: do coordinate components weakly couple in their one-step updates?
4. Attach an explicit confidence tier to every interpretation: `high`, `medium`, or `low`.
5. Mark every conclusion as provisional until a human dynamics expert reviews it.
6. Map the observed pattern to theory and choose the next experiment.
7. Preserve both the numeric table and the written interpretation in the run directory.

## Pattern guide
- If delay coordinates win the Markov test, treat missing memory or Mori-Zwanzig-style closure as the leading explanation.
- If fast-slow coordinates improve separability but degrade spectral preservation, treat them as denoisers that may distort attractor geometry.
- If factor coordinates improve both Markov closure and separability, treat them as candidates for invariant or slow-manifold-aligned coordinates.
- If short-horizon prediction looks good but spectral preservation is poor, distrust long-rollout stability claims.

## Review gate
- Never treat an LLM interpretation as self-validating.
- Always produce a confidence tier and route the conclusion through the project expert-review artifact before promoting it into design decisions.

## References
- For Koopman-oriented interpretation, read `references/koopman-factor-view.md`.
- For theory-to-pattern mappings, read `references/patterns.md`.
- For follow-up experiment suggestions, reuse the outputs of `scripts/run_coordinate_analysis.py` instead of inventing new diagnostics from scratch.
