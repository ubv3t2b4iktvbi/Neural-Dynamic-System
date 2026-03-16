---
name: dynamics-literature-factor-updater
description: Search primary dynamics literature or canonical models, extract reusable mechanistic motifs, translate them into this project's causal fast-slow, RG, and closure-aware factors, and plan validation runs. Use when Codex needs to mine factors from bifurcation theory, slow-fast systems, phase reduction, coarse-graining, critical transitions, or memory-closure literature rather than from an external quant repo.
---

# Dynamics Literature Factor Updater

Use primary-source dynamics papers and canonical models as a factor-idea engine. Keep each pass narrow: one motif family, a few source papers, a few factors, and one smoke validation loop.

## Workflow
1. Start from the current strongest coordinates.
   - Read the latest factor or benchmark artifacts before searching.
   - If `RG` or `fast-slow` is already strong, prefer literature that sharpens those coordinates instead of branching into an unrelated representation family.
2. Search primary sources, not blog summaries.
   - Prefer journal pages, arXiv, proceedings, or classic textbook sections that expose the mechanism clearly.
   - Record the source, year, model class, governing variables, and the observable that carries the mechanism.
3. Translate the mechanism into local causal objects.
   - Map the idea into one of the local evidence families: order parameter, phase recovery, energy-control injection, RG coarse-graining, criticality, or memory-closure.
   - Prefer the existing base quantities in `src/fsrc_sindy/factors/feature_engine.py`, especially `slow_level_norm`, `timescale_separation`, `slow_manifold_alignment`, `adiabatic_coherence`, `closure_stress`, `critical_window`, `rg_order_parameter`, `rg_control_parameter`, `rg_noise_scale`, and `rg_beta_flow`.
   - Read [references/literature-motif-map.md](references/literature-motif-map.md) for the current motif families and factor sketches.
4. Implement conservatively.
   - Add only `FactorSpec` when the idea is a causal composition of existing base quantities.
   - Extend `DynamicsFeatureEngine` only when one new short-memory, manifold, or recovery variable is essential to preserve the mechanism.
   - Update `src/fsrc_sindy/factors/finance_to_dynamics.py` so run artifacts keep the literature-to-factor evidence trail.
5. Validate on the existing research loop.
   - Start with `python scripts/run_factor_mining.py --suite smoke --tasks ...`.
   - Use `fastslow_smoke` or `fastslow_theory` first when the motif depends on scale separation or adiabatic structure.
   - Run `python scripts/run_coordinate_analysis.py ...` before claiming a memory or closure factor improved Markov quality.
6. Promote cautiously.
   - Read `candidate_scores.csv`, `selected_factor_library.json`, and `manual_review.md` before treating a factor as reusable.
   - Treat theory-heavy interpretations as provisional until they survive more than one task, seed, or coordinate family.

## Guardrails
- Do not import equations verbatim when only one observable or coupling survives translation.
- Do not add hidden variables, future information, or noncausal smoothing just because a paper uses them.
- Do not let theory language outrun what `feature_engine.py` can compute causally.
- Do not claim validation from one-step RC screening alone.

## Resources
- Motif families, source hints, and local factor sketches: [references/literature-motif-map.md](references/literature-motif-map.md)
