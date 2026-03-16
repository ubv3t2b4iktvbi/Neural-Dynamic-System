# Translation Rubric

Translate the external quant idea into a dynamical statement before you write code.

## Family mapping

| Source motif | Finance reading | Dynamical object | Local base signals | Candidate ops | Theory tags | First validation |
|---|---|---|---|---|---|---|
| momentum / MA breakout | trend continuation after escape | slow-manifold departure under drive | `m_norm`, `energy_ratio`, `breakout_strength` | `identity`, `product`, `pos_gate_product` | `slow_fast`, `energy`, `breakout` | factor mining smoke + coordinate analysis |
| oversold rebound / reversal | selloff exhaustion then recovery | phase-bottom recovery | `resid_norm`, `dm_norm`, `phase_bottom_score`, `support_recovery` | `identity`, `product` | `phase`, `support`, `recovery` | factor mining smoke |
| volatility compression then release | range contraction before expansion | multiscale collapse plus activation | `compression_ratio`, `collapse_quality`, `energy_ratio`, `energy_release` | `ratio`, `product` | `multiscale`, `criticality`, `activation` | factor mining + research loop |
| regime shift / anomaly | known model stops explaining motion | physics-identifier surprise or changepoint | `id_drift_surprise_norm`, `critical_window`, `physics_alignment` | `identity`, `product`, `ratio` | `physics_identifier`, `changepoint` | coordinate analysis + factor mining |
| long-memory mean reversion | current state is not Markov enough | closure gap or delayed latent structure | compare `raw`, `delay`, `factor` coordinates first | experiment before code | `closure`, `delay`, `memory` | coordinate analysis first |
| multi-factor conjunction | alpha requires two conditions jointly | gated interaction | any two causal base features | `product`, `pos_gate_product`, `neg_gate_product` | combined tags from parents | factor mining smoke |

## Implementation decision tree
1. If the motif already matches an existing base signal in `feature_engine.py`, add or revise only `FactorSpec`.
2. If the motif is a causal composition of existing base signals, prefer a composite factor in `factor_bank.py`.
3. If the motif needs a new causal state variable, extend `feature_engine.py` and add the translation note in `finance_to_dynamics.py`.
4. If the motif is not yet representable without hidden variables or lookahead, keep it as a hypothesis and design an experiment instead of coding it immediately.

## Required translation fields

Record these fields for every imported motif:

| Field | Requirement |
|---|---|
| source_name | original factor or hypothesis name |
| source_formula | formula, pseudocode, or concise description |
| finance_origin | why quant researchers use it |
| dynamics_object | order parameter, phase, energy, multiscale, surprise, memory, or composite |
| local_mapping | existing base signals or new causal state variable needed |
| proposed_factor | candidate `FactorSpec` name and formula sketch |
| dynamics_meaning | one mechanistic sentence |
| failure_mode | where the translation could be misleading |
| validation_claim | what experiment should succeed if the idea is real |

## Quality bar
- Keep the translated factor causal.
- Preserve the mechanism, not the original syntax.
- Prefer one interpretable factor family at a time over a large composite import.
- If the source repo uses price-volume semantics, restate volume only as a control or energy proxy unless the local system genuinely exposes an equivalent observable.
