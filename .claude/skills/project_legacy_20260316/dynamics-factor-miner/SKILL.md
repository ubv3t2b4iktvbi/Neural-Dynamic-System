---
name: dynamics-factor-miner
description: Translate finance-inspired or theory-inspired motifs into causal dynamical factors, screen them quickly with RC, and build a curated factor library for later validation. Use when Codex is designing new factors, extending the factor bank, or turning qualitative signal ideas into reviewable factor specs for `scripts/run_factor_mining.py`.
---

# Dynamics Factor Miner

Convert ideas such as slow-fast separation, KDJ-like low retracement, energy injection, multiscale collapse, or breakout logic into causal factor formulas that can be screened rapidly with a reservoir-computing readout.

## Workflow
1. Restate the financial or intuitive signal in dynamical language.
2. Decide whether it belongs to one of the core evidence families:
   - order parameter / slow-fast gap
   - phase / retracement / valley upturn
   - energy / control parameter
   - multiscale collapse / coarse-graining quality
   - physics-identifier surprise
3. Encode it as either:
   - a base feature from `DynamicsFeatureEngine`, or
   - a `FactorSpec` built from base features.
4. Ask whether the factor can be interpreted as an approximate Koopman observable or part of a Koopman-invariant subspace.
5. Record both the finance origin and the mechanistic dynamics meaning.
6. Run `scripts/run_factor_mining.py` on at least one smoke or common task.
7. Review `candidate_scores.csv`, `selected_factor_library.json`, and `manual_review.md` before promoting the factor.

## Guardrails
- All factors must be causal.
- Do not introduce volume-specific semantics unless a general control/energy proxy exists.
- Do not add a factor unless you can state its mechanistic meaning.
- Do not confuse short-horizon predictive gain with Koopman quality; prefer factors that remain interpretable and dynamically stable.
- Use RC for screening speed; reserve slower models for follow-up validation.
