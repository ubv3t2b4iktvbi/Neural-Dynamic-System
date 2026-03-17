# Closure-First Flag-Structured Information Blueprint

## Status

This document records the current theory-first design target for the next model family.
It is a blueprint only.
No implementation details in this note should be read as already present in `neural_dynamic_system/`.

## One-Line Summary

Learn an over-complete closure state first, organize it into routed flag-ordered blocks, then promote only the blocks with high long-horizon incremental predictive value into `q`, while keeping short-horizon closure blocks in `h` and pruning redundant blocks.

## Scope

This blueprint is intended to guide:

- future architecture changes,
- loss-library design,
- modular ablation studies,
- smoke experiments on synthetic systems.

It does not claim:

- exact slow-manifold recovery,
- exact Koopman eigenfunction recovery,
- exact basin discovery,
- theorem-level identifiability of the true physical coordinates.

## Core Objects

We start from a closure-first latent state:

```math
W_t = (x_{t-L+1}, \dots, x_t), \qquad s_t = E(W_t) \in \mathbb{R}^{d_s}.
```

The primary predictive model is:

```math
s_{t+1} = \sum_{k=1}^K c_{t,k} \, U_k \, \Phi_k(U_k^\top s_t; a_k) + \varepsilon_t,
```

with:

- `c_t` as chart or basin routing weights,
- `U_k` as chart-local flag-structured frames,
- `a_k` as block occupancy and shrinkage controls,
- `Phi_k` as chart-local block dynamics.

At the beginning of training, the model should behave as if the entire latent is closure-like.
`q` is not treated as a primitive object at initialization.

## Why the Flag Structure Exists

The flag geometry is used to represent an ordered decomposition of latent space into subspaces with different roles.
This is stronger than a single subspace method and more meaningful than a plain coordinate permutation.

In this project, the flag structure is used to support:

- ordered persistent blocks,
- one-dimensional real decay directions,
- two-dimensional oscillatory planes,
- a residual closure subspace.

It is therefore a geometric object for "how to cut the latent space into ordered blocks", not a direct criterion for "which blocks are useful".

## Block Coordinates

For chart `k`, define:

```math
y_t^{(k)} = U_k^\top s_t = \bigl(b_t^{(k,1)}, \dots, b_t^{(k,M)}, r_t^{(k)}\bigr).
```

Each block `b_t^{(k,j)}` has dimension `1` or `2`.

- `1D` blocks represent real decay or slow recovery directions.
- `2D` blocks represent local rotational or Hopf-like planes.
- `r_t^{(k)}` is residual closure state.

## Chart-Local Dynamics

The block dynamics should support:

- stable real blocks,
- rotational blocks,
- amplitude saturation for limit-cycle-like behavior,
- small residual corrections.

A typical chart-local form is:

```math
y_{t+1}^{(k)} = A_k(a_k) y_t^{(k)} + \Gamma_k(y_t^{(k)}) + \delta_k(y_t^{(k)}),
```

where:

- `A_k(a_k)` is block upper-triangular,
- `Gamma_k` contains nonlinear amplitude control for oscillatory blocks,
- `delta_k` is a small expressive residual.

Block shrinkage happens through:

- occupancy going to zero,
- oscillation frequency collapsing toward zero,
- decay becoming strong enough that the block becomes transient.

## Emergent q/h Semantics

After block statistics become meaningful, define a soft promotion operator:

```math
P_t = \sum_k c_{t,k} U_k \Pi_k U_k^\top,
```

where `Pi_k` is block-diagonal and stores promotion weights.

Then:

```math
q_t = P_t s_t, \qquad h_t = (I - P_t) s_t.
```

Interpretation:

- `q_t` is the promoted long-horizon predictive subspace,
- `h_t` is the remaining closure and memory subspace.

This means the modeling sequence is:

```text
all closure state -> blockified closure state -> emergent q/h split
```

## Closure Backbone Reuse

The closure model should be reusable.
The recommended training ladder is:

1. Train a closure backbone with `encode / step / flow / decode`.
2. Reuse the learned state `s_t`, encoder, decoder, and rollout statistics.
3. Add chart, flag, and block promotion layers on top.
4. Fine-tune only the smallest affected surfaces first.

The closure backbone should export a unified interface:

- `encode_window(W_t) -> s_t`
- `step(s_t, dt, ctx) -> s_{t+1}`
- `flow(s_t, dt, H) -> s_{t+H}`
- `decode(s_t) -> x_t`
- `local_linearize(s_t) -> A_t, b_t` when available

## Backbone Candidates

The target architecture should keep the backbone choice modular.

### Preferred main backbone

Structured discrete latent SSM:

```math
s_{t+1} = A_\theta(s_t) s_t + r_\theta(s_t).
```

Reason:

- easiest to align with block promotion,
- natural explicit state semantics,
- easiest surface for stability and block constraints,
- best fit for the current repository contracts.

### Secondary candidates

- Neural ODE:
  good for continuous-time interpretation and irregular sampling, but not the default main backbone.
- Reservoir computing:
  strong smoke baseline and memory scout, but weak final semantic interpretability for `q/h`.
- Mamba or selective SSM:
  acceptable closure backbone candidate, but not the first theory-facing model family.
- Transformer:
  strong sequence learner or teacher model, but not the preferred reduced-model core.

## Approximation Claim

The correct expressivity claim is limited:

- universal approximation belongs mainly to the closure backbone plus residual family,
- chart, flag, and block constraints are structural priors,
- removing the residual family turns the model into a structured approximation family rather than a general approximator.

So the safe claim is:

"the closure backbone can be chosen from a family with strong finite-window approximation properties; the routed flag-structured layer is added for structure discovery and compression, not as the primary source of universality."

## Information-Theoretic Promotion Principle

For each block `b_j`, define:

```math
U_j^L = I(b_j; F_t^L \mid b_{<j}, c_t),
```

the incremental long-horizon predictive information.

```math
U_j^S = I(b_j; F_t^S \mid q_t, c_t),
```

the short-horizon closure value.

```math
R_j = I(b_j; b_{<j} \mid F_t^L, c_t),
```

the conditional redundancy.

Let `kappa_j` be a block capacity cost.
Then define:

```math
S_j = U_j^L - \lambda_R R_j - \lambda_C \kappa_j.
```

Decision logic:

- promote to `q` when `S_j` is high,
- keep in `h` when short-horizon value is high but long-horizon value is modest,
- prune when both values are low and redundancy or cost is high.

## Fast-Slow Geometry Losses

The following theory-facing losses are considered useful as soft constraints.

| Theory | Quantity | Candidate loss | Intended effect |
| --- | --- | --- | --- |
| fast-slow spectral gap | block time scales | `L_gap` | separate persistent and transient blocks |
| normal contraction | symmetric transverse Jacobian | `L_trans` | keep closure directions contractive |
| tracked section invariance | approximate driven section | `L_inv_sec` | make closure coordinates follow a stable section |
| local Koopman regularity | promoted slow observables | `L_koop_q` | stabilize persistent blocks and oscillatory blocks |
| predictive information | long-horizon block utility | `L_promote` | decide promotion into `q` |
| conditional redundancy | repeated future utility | `L_red` | discourage duplicate promotion |
| coarse relevance | retained coarse predictive value | `L_rg_rel` | bias `q` toward relevant variables |
| chart persistence | sticky chart assignments | `L_sticky` | stabilize multi-attractor routing |
| flag geometry | orthogonality and block nesting | `L_flag` | keep blocks meaningful as ordered subspaces |

Important caution:
none of these losses should be described as proving exact slow manifolds or exact fibers.

## Minimal Loss Stack

The first practical loss stack should be:

```text
L_rec + L_pred + L_promote + L_red + L_gap + L_trans
```

Then add in later steps:

```text
L_koop_q, L_flag, L_sticky, L_rg_rel
```

## Merge vs Coexist

Strong correlation alone is not enough to decide whether two blocks should merge.
The relevant question is whether they remain jointly useful after conditioning on the future target and the already promoted blocks.

A useful pairwise proxy is:

```math
Delta_{ij}
=
I((b_i, b_j); F_t^L \mid q_t, c_t)
- I(b_i; F_t^L \mid q_t, c_t)
- I(b_j; F_t^L \mid q_t, c_t).
```

Interpretation:

- high redundancy with low synergy suggests merge or prune,
- high redundancy with high synergy suggests coexist or joint higher-dimensional blocks,
- low redundancy suggests coexist.

## Modular Ablation Ladder

The recommended ablation order is:

1. `A0`: closure backbone with reconstruction and prediction only
2. `A1`: add promotion and redundancy losses
3. `A2`: add gap and transverse-contraction losses
4. `A3`: add Koopman-style slow-block regularization
5. `A4`: add flag geometry penalties
6. `A5`: add sticky chart routing
7. `A6`: add coarse relevance losses
8. `A7`: full model

Each new stage should answer one structural question before adding the next family of losses.

## Smoke Experiments

### 1. Hidden-memory linear

Goal:

- verify basic `q/h/prune` separation,
- test whether short-term closure blocks stay in `h`,
- test whether long-horizon predictive blocks are promoted.

Main metrics:

- promotion precision and recall,
- short-vs-long block utility gap,
- one-step and long-horizon prediction.

### 2. van der Pol with duplicate channels and nuisance noise

Goal:

- verify one dominant 2D oscillatory `q` block,
- test whether duplicated observation channels are merged or not repeatedly promoted,
- test whether nuisance noise stays in `h` or is pruned.

Main metrics:

- number of active promoted oscillatory dimensions,
- frequency and phase error,
- false promotion rate on duplicate channels and nuisance channels.

### 3. Two stable limit cycles

Goal:

- test whether chart routing is learned,
- test whether the model avoids collapsing two basins into one oversized oscillatory block.

Main metrics:

- chart purity,
- block simplicity inside each chart,
- cross-basin rollout failure rate.

### 4. Equilibrium plus oscillation mixed episodes

Goal:

- test whether one chart becomes real-decay dominant while another becomes rotation dominant.

Main metrics:

- chart specialization,
- oscillatory frequency distribution by chart,
- equilibrium non-oscillation score.

## Implementation Order

The recommended implementation order is:

1. closure backbone interface,
2. block utilities and promotion scores,
3. gap and contraction losses,
4. oscillatory and real block family,
5. flag-structured block layer,
6. chart router,
7. coarse relevance diagnostics,
8. full smoke suite.

This order keeps the easiest failure modes observable early and avoids mixing too many speculative ideas at once.

## Current Recommendation

Before code changes:

- treat this document as the design target,
- keep all implementation claims below this level,
- confirm the loss library and smoke metrics first,
- then build the model incrementally through the ablation ladder.
