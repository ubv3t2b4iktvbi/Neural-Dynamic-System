# Literature Motif Map

Use this reference after the skill triggers. Pick one motif family, extract the mechanism from primary sources, then translate only the causal observable that can survive inside `feature_engine.py`.

## Motif Families

| Motif family | Primary-source hints | Local anchors | Candidate factor patterns | First validation |
|---|---|---|---|---|
| critical slowing down | early-warning signals near bifurcation, recovery rate collapse, rising lag-1 memory | `critical_window`, `energy_ratio`, short-lag autocorrelation | `critical_memory_gate`, `critical_slowing_pressure` | factor mining smoke plus coordinate analysis |
| phase-amplitude reduction | isostables, amplitude relaxation, return toward a cycle or slow manifold | `phase_bottom_score`, `band_position`, residual-gap restoring force | `isostable_relaxation`, `isostable_bottom_recovery` | factor mining smoke |
| adiabatic slow-fast structure | adiabatic invariance, slow-manifold tracking, fast relaxation under slow drift | `timescale_separation`, `slow_manifold_alignment`, `adiabatic_coherence`, `rg_noise_scale` | `adiabatic_noise_shield`, `isostable_adiabatic_support` | fastslow factor mining plus coordinate analysis |
| noise-assisted escape | Kramers barrier crossing, metastability, escape under agitation | `rg_noise_scale`, `rg_critical_balance`, `rg_control_parameter` | `kramers_escape_pressure` | factor mining plus theory review |
| memory-closure pressure | Mori-Zwanzig memory kernel, generalized Langevin closure, unresolved fast modes | `lag1_autocorr`, `closure_stress`, `rg_noise_scale` | `memory_closure_load` | coordinate analysis first, then factor mining |

## Translation Fields

Capture these fields for each literature-derived motif:

| Field | Requirement |
|---|---|
| source_title | paper or model name |
| source_year | publication year or canonical era |
| model_class | Hopf, saddle-node, slow-fast, RG, metastable diffusion, generalized Langevin, etc. |
| mechanism | one sentence on what physically changes |
| local_mapping | existing base signals or one new causal state variable |
| candidate_factor | factor name and formula sketch |
| dynamics_meaning | one mechanistic sentence suitable for `FactorSpec` |
| failure_mode | where the translation could be misleading |
| first_check | smoke factor mining, coordinate analysis, or full research loop |

## Decision Rules

1. If the paper's mechanism already matches an existing base signal, add or revise only `FactorSpec`.
2. If the mechanism needs one new causal variable such as short-lag memory or restoring-force strength, add exactly one state quantity to `DynamicsFeatureEngine` and keep the composite factors small.
3. If the paper depends on hidden state, noncausal filtering, or a full PDE field that the benchmark does not expose, write it down as a hypothesis and design an experiment instead of coding it immediately.

## Current Preferred Search Order

1. critical transitions and early-warning signals
2. slow-fast and adiabatic invariants
3. phase-amplitude reduction and isostables
4. metastability and Kramers escape
5. Mori-Zwanzig memory and closure

## Suggested Primary Sources

- Scheffer et al., "Early-warning signals for critical transitions" (Nature, 2009): <https://www.nature.com/articles/nature08227>
- Mauroy et al., "Global phase-amplitude description of limit-cycle dynamics" (arXiv): <https://arxiv.org/abs/1208.6309>
- "Parameterizing unresolved small-scale processes within the Mori-Zwanzig formalism" (arXiv): <https://arxiv.org/abs/2108.12145>
- For adiabatic slow-fast structure and Kramers escape, start from a canonical review or textbook section, then follow the citations into the exact model class that matches the current benchmark.
