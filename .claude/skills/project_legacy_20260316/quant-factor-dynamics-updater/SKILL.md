---
name: quant-factor-dynamics-updater
description: Inspect external quant factor libraries or alpha-mining repositories such as QuantaAlpha, extract the dynamical essence of their factor motifs or hypothesis trajectories, translate them into this project's causal dynamical factors, and design validation experiments along the current RC, coordinate-analysis, and research-loop workflow. Use when Codex needs to mine new ideas from an outside quant repo, update `src/fsrc_sindy/factors/`, or turn external alpha semantics into reviewable `FactorSpec` candidates and experiment plans.
---

# Quant Factor Dynamics Updater

Bridge external quant-alpha repositories into this project's dynamical factor workflow. Start from the source repo's factor semantics or evolution traces, restate them as dynamical objects, implement only the causal pieces that survive translation, and validate them with the local screening and theory artifacts.

## Workflow
1. Intake the external repo.
   - Clone or open the source repo locally.
   - Run `scripts/repo_factor_manifest.py <repo_path> --format md` to rank factor, hypothesis, evolution, config, and backtest files.
   - Read the top-ranked files plus the repo README before opening deeper files.
   - Use [references/external-repo-intake.md](references/external-repo-intake.md) when the repo mixes factor definitions, evolution logic, and backtests.
2. Reduce external quant semantics to dynamical mechanisms.
   - Record the source formula or description, finance intent, candidate dynamical object, causal state variables required, and expected failure modes.
   - Use [references/translation-rubric.md](references/translation-rubric.md) to classify the motif into one or more local evidence families:
     - order parameter / slow-fast gap
     - phase / retracement / recovery
     - energy / control injection
     - multiscale collapse / compression
     - regime surprise / physics-identifier mismatch
     - memory / closure / delay dependence
     - composite interaction gates
   - Reject motifs that depend on unavailable exogenous semantics unless you can restate them as causal local dynamics or explicitly add a new base feature.
3. Bind the motif to the local implementation surface.
   - Prefer existing base quantities from `src/fsrc_sindy/factors/feature_engine.py`.
   - Prefer existing `FactorSpec` ops and families from `src/fsrc_sindy/factors/base.py` and `src/fsrc_sindy/factors/factor_bank.py`.
   - Update `src/fsrc_sindy/factors/finance_to_dynamics.py` when a source motif adds a reusable translation pattern.
   - Add a new base feature only when composition from existing features would hide the actual mechanism.
   - Every new factor must carry both `finance_origin` and `dynamics_meaning`.
4. Validate with the current research loop.
   - Start with narrow smoke runs in `scripts/run_factor_mining.py`.
   - Run `scripts/run_coordinate_analysis.py` when the claim involves closure, spectrum, separability, or memory.
   - Use `scripts/run_research_loop.py` when several translated factors should be evaluated as a coherent family or when the source repo implies an `identify` workflow.
   - Use [references/validation-protocol.md](references/validation-protocol.md) for command patterns and promotion gates.
5. Archive and promote conservatively.
   - Read `candidate_scores.csv`, `selected_factor_library.json`, `manual_review.md`, `metrics.json`, and coordinate or loop summaries before claiming success.
   - Promote only factors that stay causal, mechanistically interpretable, and stable across more than one task, seed, or diagnostic.
   - Mark all theory-heavy interpretations as pending human dynamics review.

## Default decisions
- Use `accumulate` mode when porting a curated source library into the existing factor bank.
- Use `identify` mode when the outside repo is better treated as a source of mechanism families or hypotheses than as a fixed formula list.
- Treat external backtest metrics as idea priors, not proof that the translated dynamical factor is valid in this project.
- Keep the first pass small: one repo, one motif family, one smoke task, and one coordinate comparison.

## Guardrails
- Do not copy formulas verbatim into `factor_bank.py` without a dynamical restatement.
- Do not let market-specific heuristics outrun causal observability in the local benchmark state.
- Do not confuse better one-step screening with better rollout geometry or Koopman quality.
- Do not promote factors that only help because of physics-identifier leakage or overfit composite interactions.
- Keep the evidence trail reviewable: source repo note, translation table, code change, smoke result, and follow-up experiment plan.

## Resources
- Intake and repo triage: [references/external-repo-intake.md](references/external-repo-intake.md)
- Translation template and factor-family mapping: [references/translation-rubric.md](references/translation-rubric.md)
- Validation matrix and command recipes: [references/validation-protocol.md](references/validation-protocol.md)
- Repo scanning helper: `scripts/repo_factor_manifest.py`
